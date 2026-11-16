"""Collects the chosen takes into one folder: outputs/<name>/<clip>.{fbx,mp4,npz} + README.md.

    venv311\\Scripts\\python.exe scripts\\kimodo_finalize.py outputs\\<name>\\picks.json

picks.json:
    {"name": "<name>", "note": "...",
     "picks": [{"row": "1", "name": "hike_walk", "from": "<batch>/<clip>/<clip>_s2",
                "why": "why this take", "left": "what is still wrong, or empty"}]}
"from" is relative to outputs/ and has no extension. The prompt is read from that batch's manifest.json.
"""

import json
import shutil
import sys
from pathlib import Path

picks_path = Path(sys.argv[1]).resolve()
out_dir = picks_path.parent
outputs = out_dir.parent
picks = json.loads(picks_path.read_text(encoding="utf-8"))

rows = []
for p in picks["picks"]:
    src = outputs / p["from"]
    batch, clip = p["from"].split("/")[:2]
    manifest = json.loads((outputs / batch / "manifest.json").read_text(encoding="utf-8"))
    for ext in (".fbx", ".mp4", ".npz"):
        if not src.with_suffix(ext).is_file():
            if ext == ".fbx":  # no Mixamo export for this take (FBX SDK or target character missing)
                continue
            sys.exit(f"missing {src.with_suffix(ext)}")
        shutil.copy2(src.with_suffix(ext), out_dir / f"{p['name']}{ext}")
    rows.append(f"| {p.get('row', '')} | [{p['name']}]({p['name']}.mp4) | {manifest['clips'][clip]['prompt']} | "
                f"`{p['from']}` | {p.get('why', '')} | {p.get('left', '')} |")

head = [f"# {picks['name']}", "",
        "One chosen take per row: `<name>.fbx` (Mixamo rig, animated), `<name>.mp4` (preview), `<name>.npz` (raw "
        "Kimodo motion). Why each take was chosen and what is still wrong: `picks.json`. Every take is in the batch "
        "folders next to this one.", "",
        "| # | clip | prompt used | source | why | left |", "|---|---|---|---|---|---|"]
(out_dir / "README.md").write_text("\n".join(head + rows) + "\n", encoding="utf-8")
print(f"{len(rows)} clips -> {out_dir}")
