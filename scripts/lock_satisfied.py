"""Exit 0 if the venv already matches requirements-lock.txt, 1 otherwise (prints what differs).

Lets install_kimodo.bat skip pip, which otherwise re-clones the git-pinned packages on every run.
"""

import re
import sys
from importlib import metadata
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "requirements-lock.txt"


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


installed = {norm(d.metadata["Name"]): d.version for d in metadata.distributions()}
diffs = []
for line in LOCK.read_text(encoding="utf-8").splitlines():
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        continue
    if " @ " in line:  # direct URL pin: present is enough
        name, want = line.split(" @ ", 1)[0], None
    else:
        name, want = line.split("==", 1)
    have = installed.get(norm(name.strip()))
    if have is None or (want is not None and have != want.strip()):
        diffs.append(f"{name.strip()}: lock {want or 'URL'}, installed {have or 'none'}")

for d in diffs:
    print(f"  differs: {d}")
sys.exit(1 if diffs else 0)
