#!/usr/bin/env python3
"""
Captures the Chrome Web Store screenshots (1280x800, 24-bit PNG) and, optionally, a demo video.

Setup (once):
    pip install -U playwright pillow && playwright install chromium

Run:
    python scripts/store_screenshots.py                      # all four screenshots
    python scripts/store_screenshots.py --only badges        # just one
    python scripts/store_screenshots.py --only video         # playground demo (.webm)

Put 4-6 images you own or that are free to use in docs/store/samples/ first.
The API must be running; use a real key (AON_API_KEY) to avoid the demo rate limit.
"""

import argparse
import base64
import json
import mimetypes
import os
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

W, H = 1280, 800
BG = (13, 13, 15)
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}

GALLERY = """<!doctype html><meta charset="utf-8"><title>Gallery</title>
<style>body{margin:0;background:#101012;color:#e4e4e7;font:16px system-ui;padding:48px 50px}
h1{font:italic 40px Georgia,serif;margin:0 0 6px}p{color:#8a8a93;margin:0 0 28px}
.grid{display:grid;grid-template-columns:repeat(3,380px);gap:20px}
figure{margin:0}img{width:380px;height:250px;object-fit:cover;border-radius:8px;display:block}</style>
<h1>Photographs, paintings and everything between</h1>
<p>A page the way the extension meets it: ordinary images, no labels.</p>
<div class="grid">{figures}</div>"""


# ---------- image helpers (no browser needed) ----------
def finalize(path: Path) -> None:
    """The Web Store wants exactly 1280x800 and no alpha channel."""
    img = Image.open(path).convert("RGB")
    if img.size != (W, H):
        raise SystemExit(
            f"{path.name} is {img.size[0]}x{img.size[1]}, expected {W}x{H}"
        )
    img.save(path)


def compose(shot: Path, out: Path, target_h: int = 720) -> None:
    """Centre a tall screenshot (popup or result window) on the dark 1280x800 canvas."""
    img = Image.open(shot).convert("RGB")
    scale = min(target_h / img.height, (W - 160) / img.width)
    img = img.resize(
        (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
    )
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    canvas.save(out)
    finalize(out)


def start_gallery(images: list[Path]) -> ThreadingHTTPServer:
    """Serves a plain gallery page so the content script meets real http images."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path == "/":
                    figures = "".join(
                        f'<figure><img src="/img/{i}"></figure>'
                        for i in range(len(images))
                    )
                    body, ctype = (
                        GALLERY.replace("{figures}", figures).encode(),
                        "text/html; charset=utf-8",
                    )
                elif self.path.startswith("/img/"):
                    p = images[int(self.path.rsplit("/", 1)[1])]
                    body, ctype = (
                        p.read_bytes(),
                        mimetypes.guess_type(p.name)[0] or "image/jpeg",
                    )
                else:
                    raise IndexError
            except (IndexError, ValueError):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ---------- scenes ----------
def shot_playground(p, args, sample: Path, out: Path) -> Path:
    browser = p.chromium.launch()
    try:
        page = browser.new_page(viewport={"width": W, "height": H})
        page.add_init_script(
            f"localStorage.setItem('aon_key', {json.dumps(args.api_key)}); sessionStorage.setItem('aon_hero', '1');"
        )
        page.goto(args.api_url, wait_until="networkidle")
        page.set_input_files("input[type=file]", str(sample))
        page.wait_for_selector(
            ".broadside", timeout=120_000
        )  # a sleeping server can take a minute
        page.wait_for_timeout(2200)  # let the confidence marker settle
        page.locator(".act-label", has_text="ACT IV").scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        path = out / "1-playground-verdict.png"
        page.screenshot(path=str(path), scale="css")
        finalize(path)
        return path
    finally:
        browser.close()


def run_extension_scenes(
    p, args, samples: list[Path], out: Path, scenes: set[str]
) -> list[Path]:
    tmp = tempfile.mkdtemp(prefix="aon-ext-")
    ext = str(Path(args.ext).resolve())
    ctx = p.chromium.launch_persistent_context(
        tmp,
        channel="chromium",
        headless=not args.headed,
        viewport={"width": W, "height": H},
        device_scale_factor=2,
        args=[f"--disable-extensions-except={ext}", f"--load-extension={ext}"],
    )
    made = []
    try:
        sw = (
            ctx.service_workers[0]
            if ctx.service_workers
            else ctx.wait_for_event("serviceworker", timeout=20_000)
        )
        ext_id = sw.url.split("/")[2]
        base = {
            "apiUrl": args.api_url,
            "apiKey": args.api_key,
            "mode": "explain",
            "passiveBadge": False,
        }
        sw.evaluate("s => chrome.storage.sync.set(s)", base)

        if "popup" in scenes:
            page = ctx.new_page()
            page.set_viewport_size({"width": 320, "height": 640})
            page.goto(f"chrome-extension://{ext_id}/popup.html")
            try:
                page.wait_for_function(
                    "document.getElementById('accessLine').textContent.includes('ready')",
                    timeout=90_000,
                )
            except Exception:
                print(
                    "  note: the API was not ready, so the popup is captured in its waiting state"
                )
            raw = Path(tmp) / "popup.png"
            page.locator("body").screenshot(path=str(raw))
            made.append(out / "2-extension-popup.png")
            compose(raw, made[-1])
            page.close()

        if "results" in scenes:
            sample = samples[0]
            mime = mimetypes.guess_type(sample.name)[0] or "image/jpeg"
            resp = ctx.request.post(
                f"{args.api_url}/v1/inference/explain",
                headers={"X-API-Key": args.api_key},
                timeout=120_000,
                multipart={
                    "file": {
                        "name": sample.name,
                        "mimeType": mime,
                        "buffer": sample.read_bytes(),
                    }
                },
            )
            if not resp.ok:
                raise SystemExit(f"The API answered {resp.status}: {resp.text()[:200]}")
            data_url = (
                f"data:{mime};base64,{base64.b64encode(sample.read_bytes()).decode()}"
            )
            sw.evaluate(
                "r => chrome.storage.session.set(r)",
                {
                    "analysis:demo": {
                        "status": "success",
                        "imageUrl": data_url,
                        "filename": sample.name,
                        "result": resp.json(),
                        "apiUrl": args.api_url,
                        "mode": "explain",
                    }
                },
            )
            page = ctx.new_page()
            page.set_viewport_size({"width": 460, "height": 640})
            page.goto(f"chrome-extension://{ext_id}/results.html?id=demo")
            page.wait_for_selector("#view-result", state="visible", timeout=15_000)
            page.wait_for_timeout(2200)
            raw = Path(tmp) / "results.png"
            page.screenshot(path=str(raw), full_page=True)
            made.append(out / "3-extension-verdict.png")
            compose(raw, made[-1])
            page.close()

        if "badges" in scenes:
            gallery_images = samples[:6]
            server = start_gallery(gallery_images)
            sw.evaluate(
                "s => chrome.storage.sync.set(s)",
                {**base, "passiveBadge": True, "mode": "predict"},
            )
            page = ctx.new_page()
            page.set_viewport_size({"width": W, "height": H})
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            want = min(4, len(gallery_images))
            page.wait_for_function(
                f"document.querySelectorAll('[data-aon-badge]').length >= {want}",
                timeout=120_000,
            )
            page.wait_for_timeout(900)  # badges fade in over 350 ms
            made.append(out / "4-passive-badges.png")
            page.screenshot(path=str(made[-1]), scale="css")
            finalize(made[-1])
            server.shutdown()
    finally:
        ctx.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return made


def record_demo(p, args, sample: Path, out: Path) -> Path:
    tmp = out / "video-tmp"
    browser = p.chromium.launch()
    ctx = browser.new_context(
        viewport={"width": 1280, "height": 720},
        record_video_dir=str(tmp),
        record_video_size={"width": 1280, "height": 720},
    )
    page = ctx.new_page()
    page.add_init_script(
        f"localStorage.setItem('aon_key', {json.dumps(args.api_key)});"
    )
    page.goto(args.api_url, wait_until="networkidle")
    page.wait_for_timeout(1800)  # linger on the hero
    page.locator(".act-label", has_text="ACT II").scroll_into_view_if_needed()
    page.wait_for_timeout(800)
    page.set_input_files("input[type=file]", str(sample))
    page.wait_for_selector(".broadside", timeout=120_000)
    page.wait_for_timeout(2500)
    page.locator(".slider").scroll_into_view_if_needed()
    box = page.locator(".slider").bounding_box()
    y = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + box["width"] * 0.5, y)
    page.mouse.down()
    sweep = [0.5 - 0.4 * i / 30 for i in range(31)] + [
        0.1 + 0.8 * i / 50 for i in range(51)
    ]
    for t in sweep:
        page.mouse.move(box["x"] + box["width"] * t, y)
        page.wait_for_timeout(25)
    page.mouse.up()
    page.wait_for_timeout(1500)
    video = Path(page.video.path())
    ctx.close()
    browser.close()
    target = out / "demo.webm"
    shutil.move(str(video), target)
    shutil.rmtree(tmp, ignore_errors=True)
    return target


# ---------- entry point ----------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--api-url", default=os.getenv("AON_API_URL", "http://localhost:8000")
    )
    ap.add_argument("--api-key", default=os.getenv("AON_API_KEY", "demo-public"))
    ap.add_argument(
        "--ext", default="extension", help="folder of the unpacked extension"
    )
    ap.add_argument("--samples", default="docs/store/samples")
    ap.add_argument("--out", default="docs/store/screenshots")
    ap.add_argument(
        "--only",
        nargs="+",
        choices=["playground", "popup", "results", "badges", "video"],
        help="default: playground popup results badges",
    )
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    args = ap.parse_args(argv)
    args.api_url = args.api_url.rstrip("/")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    scenes = set(args.only or ["playground", "popup", "results", "badges"])
    samples = (
        sorted(
            f for f in Path(args.samples).glob("*") if f.suffix.lower() in IMAGE_TYPES
        )
        if Path(args.samples).is_dir()
        else []
    )
    if not samples:
        raise SystemExit(f"Put at least one .jpg or .png in {args.samples} first.")
    if "badges" in scenes and len(samples) < 4:
        print(
            f"note: only {len(samples)} sample image(s); the badges shot looks better with 4-6"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "Run: pip install -U playwright pillow && playwright install chromium"
        )

    made = []
    with sync_playwright() as p:
        if "playground" in scenes:
            made.append(shot_playground(p, args, samples[0], out))
        ext_scenes = scenes & {"popup", "results", "badges"}
        if ext_scenes:
            made += run_extension_scenes(p, args, samples, out, ext_scenes)
        if "video" in scenes:
            made.append(record_demo(p, args, samples[0], out))
    print("\nDone:")
    for m in made:
        print("  ", m)
    if "video" in scenes:
        print(
            '\nGIF:  ffmpeg -i demo.webm -vf "fps=12,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" demo.gif'
        )


if __name__ == "__main__":
    main()
