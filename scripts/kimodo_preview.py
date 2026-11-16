"""Previews for a kimodo_batch output folder: an mp4 per FBX and a contact sheet per clip, plus README.md.

    venv311\\Scripts\\python.exe scripts\\kimodo_preview.py outputs\\<name> [--force]

Renders each take in Blender (Workbench, fixed 3/4 camera) on Kimodo's own SOMA body, straight from <clip>_s<k>.npz
(the motion itself, not its retarget; a take with no .npz is shown as its Mixamo FBX), encodes <clip>_s<k>.mp4 with ffmpeg,
writes <clip>_sheet.png (one row of 6 frames per sample) and outputs/<name>/README.md listing every clip.
Existing previews are kept unless --force.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BLENDER = os.environ.get("BLENDER", r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
COLS, THUMB = 6, 256
SKIN = HERE.parent / "kimodo" / "kimodo" / "assets" / "skeletons" / "somaskel77" / "skin_standard.npz"


def render(fbx: Path, force: bool) -> Path:
    mp4 = fbx.with_suffix(".mp4")
    if mp4.is_file() and not force:
        return mp4
    frames = fbx.parent / f"_frames_{fbx.stem}"
    shutil.rmtree(frames, ignore_errors=True)
    npz = fbx.with_suffix(".npz")
    r = subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(HERE / "blender_render_take.py"), "--",
                        str(npz if npz.is_file() else fbx), str(frames), "512", str(SKIN)], capture_output=True, text=True)
    m = re.search(r"FRAMES (\d+) (\d+) ([\d.]+)", r.stdout)
    if r.returncode or not m:
        sys.exit(f"Blender failed on {fbx}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    first, fps = int(m.group(1)), m.group(3)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", fps, "-start_number", str(first),
                    "-i", str(frames / "f_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    str(mp4)], check=True)
    return mp4


def sheet(clip_dir: Path, name: str, samples: list[Path], prompt: str) -> Path:
    font = ImageFont.truetype("arial.ttf", 18)
    small = ImageFont.truetype("arial.ttf", 14)
    head = 34
    img = Image.new("RGB", (COLS * THUMB, head + len(samples) * THUMB), "white")
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), f"{name}: {prompt}", fill="black", font=font)
    for row, fbx in enumerate(samples):
        frames = sorted((fbx.parent / f"_frames_{fbx.stem}").glob("f_*.png"))
        picks = [frames[round(i * (len(frames) - 1) / (COLS - 1))] for i in range(COLS)]
        for col, f in enumerate(picks):
            thumb = Image.open(f).convert("RGB").resize((THUMB, THUMB))
            x, y = col * THUMB, head + row * THUMB
            img.paste(thumb, (x, y))
            draw.text((x + 6, y + 4), f"s{row + 1}  f{int(f.stem[2:])}", fill="black", font=small)
    out = clip_dir / f"{name}_sheet.png"
    img.save(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch_dir", type=Path)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    args.batch_dir = args.batch_dir.resolve()  # Blender resolves relative render paths against the drive root
    manifest = json.loads((args.batch_dir / "manifest.json").read_text(encoding="utf-8"))

    rows = []
    for name, clip in manifest["clips"].items():
        fbxs = [args.batch_dir / f for f in clip["files"]]
        for fbx in fbxs:
            render(fbx, args.force)
            print(f"preview {fbx.name}")
        sh = sheet(fbxs[0].parent, name, fbxs, clip["prompt"])
        for fbx in fbxs:
            shutil.rmtree(fbx.parent / f"_frames_{fbx.stem}", ignore_errors=True)
        rows.append((name, clip, fbxs, sh))

    lines = [f"# {manifest['name']}", "",
             f"Model {manifest.get('model', '?')}, {manifest.get('fps', 30):g} fps. "
             f"Each clip has {len(rows[0][2]) if rows else 0} sample(s): `.npz` raw SOMA motion, `.mp4` preview of it on "
             f"Kimodo's SOMA body, `<clip>_s<k>.fbx` the same motion retargeted onto `{Path(manifest['target_fbx']).name}` "
             "(Mixamo rig); "
             "`<clip>_sheet.png` shows each sample as one row.", "",
             "| clip | shot | s | prompt | files |", "|---|---|---|---|---|"]
    for name, clip, fbxs, sh in rows:
        files = " ".join(f"[s{i + 1}]({f.relative_to(args.batch_dir).as_posix().replace(' ', '%20')})"
                         for i, f in enumerate(fbxs))
        lines.append(f"| `{name}` | {clip.get('note', '')} | {clip['seconds']:g} | {clip['prompt']} | "
                     f"{files}, [sheet]({sh.relative_to(args.batch_dir).as_posix()}) |")
    (args.batch_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"README: {args.batch_dir / 'README.md'}")


if __name__ == "__main__":
    main()
