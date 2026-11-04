import re
from pathlib import Path


def test_local_markdown_links_exist():
    root = Path(__file__).resolve().parents[1]
    files = [root / "README.md"]
    for name in ("README.it.md",):
        candidate = root / name
        if candidate.exists():
            files.append(candidate)
    files.extend(sorted((root / "docs").glob("*.md")))

    for md in files:
        text = md.read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if link.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = (md.parent / link.split("#")[0]).resolve()
            assert target.exists(), f"{md}: broken link {link}"
