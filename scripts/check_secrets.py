"""
Quick scanner for secrets in repo files.
Fails CI if suspicious strings are found.
"""

import re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SELF = pathlib.Path(__file__).resolve()

# Paths (relative to repo root) to always ignore
IGNORE_DIRS = {
    ".git", "venv", ".venv", "__pycache__", "server/artifacts",
    "training/data", "notebooks/.ipynb_checkpoints", ".github",
}
IGNORE_FILES = {
    str(SELF.relative_to(ROOT)),  # skip this script itself
}

# Suspicious patterns
PATTERNS = [
    r"AIza[0-9A-Za-z\-_]{35}",                   # Google API keys
    r"-----BEGIN(?:.*?)PRIVATE KEY-----",        # PEM private keys
    r"projects/\d+/secrets/",                    # GCP Secret Manager refs
]

compiled = [re.compile(p, re.DOTALL) for p in PATTERNS]
bad = []

for p in ROOT.rglob("*"):
    if p.is_dir():
        # skip ignored dirs by name
        parts = {*p.parts}
        if parts & IGNORE_DIRS:
            continue
        continue

    rel = str(p.relative_to(ROOT))

    # skip known files and binary-ish assets
    if rel in IGNORE_FILES:
        continue
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ipynb", ".ico"}:
        continue

    # also skip big files (>1MB) to keep CI fast
    try:
        if p.stat().st_size > 1_000_000:
            continue
    except Exception:
        continue

    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue

    for rx in compiled:
        if rx.search(text):
            bad.append((rel, rx.pattern))
            break  # one hit per file is enough

if bad:
    print("⚠️ Potential secrets found:")
    for rel, pat in bad:
        print(f" - {rel} :: matched {pat}")
    sys.exit(2)

print("✅ No obvious secrets found")
