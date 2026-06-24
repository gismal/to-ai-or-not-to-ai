const CONFIG = {
    MIN_IMAGE_PX: 150,   // ignore icons and avatars
    MAX_PER_PAGE: 5,     // max images analyzed per page load
    REQUEST_DELAY: 2000,  // ms between API calls (rate limit safety)
};

const analyzed = new Set();
const queue = [];
let running = false;


// ── Badge rendering ───────────────────────────────────────────────────────────

function createBadge(label, confidence) {
    const isAI = label.includes("AI_GENERATED");
    const isUncertain = label.includes("UNCERTAIN");

    const color = isAI ? "#6C63FF" : isUncertain ? "#FF9F43" : "#22D3A5";
    const icon = isAI ? "🤖" : isUncertain ? "❓" : "✓";
    const pct = (confidence * 100).toFixed(0);

    const badge = document.createElement("div");
    badge.dataset.aiDetector = "badge";
    badge.title = `AI Detector: ${label.replace(/_/g, " ")} (${(confidence * 100).toFixed(1)}%)`;
    badge.style.cssText = [
        "position: absolute",
        "top: 8px",
        "right: 8px",
        "z-index: 2147483647",
        `background: ${color}`,
        "color: white",
        "padding: 3px 9px",
        "border-radius: 99px",
        "font-size: 11px",
        "font-family: -apple-system, BlinkMacSystemFont, sans-serif",
        "font-weight: 700",
        "letter-spacing: 0.02em",
        "box-shadow: 0 2px 8px rgba(0,0,0,0.35)",
        "pointer-events: none",
        "line-height: 1.6",
        "white-space: nowrap",
        "opacity: 0",
        "transition: opacity 0.3s ease",
    ].join(";");

    badge.textContent = `${icon} ${pct}%`;

    // Fade in
    requestAnimationFrame(() => {
        badge.style.opacity = "0.92";
    });

    return badge;
}

function attachBadge(img, label, confidence) {
    // Skip if already badged
    if (img.parentElement?.querySelector("[data-ai-detector='badge']")) return;

    // Ensure parent is positioned so absolute child renders correctly
    const parent = img.parentElement;
    if (parent && getComputedStyle(parent).position === "static") {
        parent.style.position = "relative";
    }

    const badge = createBadge(label, confidence);
    parent.appendChild(badge);
}


// ── Single image analysis ─────────────────────────────────────────────────────

async function analyzeImage(img, apiKey, apiUrl) {
    const src = img.currentSrc || img.src;
    if (!src || analyzed.has(src)) return;
    analyzed.add(src);

    try {
        const imgRes = await fetch(src);
        if (!imgRes.ok) return;

        const blob = await imgRes.blob();
        const formData = new FormData();
        formData.append("file", blob, "image.jpg");

        const res = await fetch(`${apiUrl}/v1/inference/predict`, {
            method: "POST",
            headers: { "X-API-Key": apiKey },
            body: formData,
        });

        if (!res.ok) return;

        const data = await res.json();
        attachBadge(img, data.prediction, data.confidence);

    } catch {
        // Silently fail — badge mode should be invisible when blocked by CORS or network
    }
}


// ── Queue processor ───────────────────────────────────────────────────────────

async function processQueue(apiKey, apiUrl) {
    if (running) return;
    running = true;

    while (queue.length > 0) {
        const img = queue.shift();
        await analyzeImage(img, apiKey, apiUrl);
        await new Promise((r) => setTimeout(r, CONFIG.REQUEST_DELAY));
    }

    running = false;
}

function enqueue(img, apiKey, apiUrl) {
    if (queue.length >= CONFIG.MAX_PER_PAGE) return;
    queue.push(img);
    processQueue(apiKey, apiUrl);
}


// ── Image filtering ───────────────────────────────────────────────────────────

function isQualifying(img) {
    if (!img.src && !img.currentSrc) return false;
    if (img.src.startsWith("data:")) return false;  // inline data URIs — skip
    if (analyzed.has(img.currentSrc || img.src)) return false;

    const rect = img.getBoundingClientRect();
    return rect.width >= CONFIG.MIN_IMAGE_PX && rect.height >= CONFIG.MIN_IMAGE_PX;
}


// ── Initialisation ────────────────────────────────────────────────────────────

async function init() {
    const { passiveBadge, apiKey, apiUrl } =
        await chrome.storage.sync.get(["passiveBadge", "apiKey", "apiUrl"]);

    if (!passiveBadge || !apiKey) return;

    const base = (apiUrl || "http://localhost:8000").replace(/\/$/, "");

    // Scan images already in the DOM
    const existing = Array.from(document.querySelectorAll("img"))
        .filter(isQualifying)
        .slice(0, CONFIG.MAX_PER_PAGE);

    existing.forEach((img) => enqueue(img, apiKey, base));

    // Watch for images added after page load (infinite scroll, SPAs, etc.)
    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            for (const node of mutation.addedNodes) {
                if (node.nodeName === "IMG" && isQualifying(node)) {
                    enqueue(node, apiKey, base);
                }
                // Also check children of added nodes
                if (node.querySelectorAll) {
                    node.querySelectorAll("img").forEach((img) => {
                        if (isQualifying(img)) enqueue(img, apiKey, base);
                    });
                }
            }
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });
}

// Wait for DOM to be ready
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
} else {
    init();
}

