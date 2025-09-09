"""
Quick scanner for secrets in repo files.
Fails CI if suspicious strings are found.
"""

import re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

PATTERNS = [
    r"AIza[0-9A-Za-z\-_]{35}",                   # Google API keys
    r"-----BEGIN(.*?)PRIVATE KEY-----",          # PEM private keys
    r"projects/\d+/secrets/",                    # GCP secrets
]

bad = []
for p in ROOT.rglob("*"):
    if p.is_dir():
        continue
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ipynb"}:
        continue
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    for pat in PATTERNS:
        if re.search(pat, text, flags=re.DOTALL):
            bad.append((p, pat))

if bad:
    print("⚠️ Potential secrets found:")
    for p, pat in bad:
        print(f" - {p} :: matched {pat}")
    sys.exit(2)

print("✅ No obvious secrets found")
