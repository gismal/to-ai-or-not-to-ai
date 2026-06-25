from pathlib import Path


def prepare(base_dir: Path) -> None:
    for split in ("train", "val", "test"):
        fake_dir = base_dir / split / "FAKE"
        ai_dir = base_dir / split / "AI_GENERATED"
        if fake_dir.exists():
            fake_dir.rename(ai_dir)
            print(f"Rename {fake_dir} -> {ai_dir}")


prepare(Path("data"))
