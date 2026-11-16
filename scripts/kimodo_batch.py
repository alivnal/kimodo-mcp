"""Text prompts -> Kimodo motions -> Mixamo-retargeted FBX, in one run (the model loads once).

    venv311\\Scripts\\python.exe -u scripts\\kimodo_batch.py <spec.json> [--only a,b] [--force] [--samples N] [--reexport]

spec.json:
    {
      "name": "walk_cycle",                  # output folder: outputs/<name>/
      "target_fbx": "assets/characters/Y Bot.fbx",   # optional, Mixamo-rigged character (relative to the install)
      "seed": 1000,                          # optional; clip i uses seed + 100*i (same spec -> same motions)
      "samples": 2,                          # optional, variants per clip
      "in_place": false,                     # optional, strip horizontal root motion
      "diffusion_steps": 100,                # optional
      "clips": [ {"name": "hike_walk", "prompt": "A person walks ...", "seconds": 5, "note": "shot 3"} ]
    }
A clip may override seed, samples, in_place and diffusion_steps.

Constraints (Kimodo's own: the prompt says what, a constraint says where). A clip may have:
    "path": {"speed": 1.6}                       # the root goes straight ahead at 1.6 m/s for the whole clip
    "path": {"speed": 1.6, "stop_at": 3.0}       # ... and has come to rest by second 3 (slows over the last 0.6 s)
    "path": {"points": [[0, 0, 0], [2.5, 0, 3.0], [5, 1.0, 5.0]]}   # [second, x, z]: the root's place on the ground
                                                 # at those times (metres, +z ahead, +x to the character's left), a line between
    "constraints": "specs/my_constraints.json"   # a Kimodo constraints file (or the list itself): key poses, hand and
                                                 # foot targets, as saved by the demo's Save Constraints
    "constraint_weight": 2.0                     # how hard the constraints pull against the text (Kimodo's default 2.0)
    "poses": [...], "hands": [...]               # a pose taken from another take, a wrist target in space
    "ground": {"rise": 0.12}                     # the finished take fitted to sloping land (see kimodo_author.py for these)
Speeds are the SOMA person's (1.75 m tall): on a creature the retarget scales the travel by its hip height.
What was sent is saved next to the takes as <clip>/constraints.json; how far the take's hands, feet and poses ended from
their targets is printed and kept in the manifest ("pin_cm").

One continuous motion from several prompts (Kimodo's own transition overlap, no crossfade by eye):
    {"name": "walk_then_wave", "segments": [{"prompt": "A person walks forward.", "seconds": 3},
                                            {"prompt": "A person stops and waves.", "seconds": 3}], "transition": 5}
"transition" is the overlap in frames (default 5). A path, poses and hands run over the whole clip's time.

Writes outputs/<name>/<clip>/<clip>_s<k>.fbx, <clip>_s<k>.npz (raw SOMA motion) and outputs/<name>/manifest.json.
Clips whose FBX files all exist are skipped unless --force, so a stopped run resumes.
--reexport rebuilds the FBX files from the saved .npz motions without the model (after a retarget fix, or
after changing target_fbx / in_place in the spec); it never regenerates motion.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import kimodo_author as author

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "assets" / "characters" / "Y Bot.fbx"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spec", type=Path)
    p.add_argument("--force", action="store_true", help="regenerate clips that already have output")
    p.add_argument("--only", default="", help="comma-separated clip names to run")
    p.add_argument("--samples", type=int, default=None, help="override samples for every clip")
    p.add_argument("--reexport", action="store_true", help="rebuild FBX from the saved .npz, no generation")
    p.add_argument("--out", type=Path, default=ROOT / "outputs", help="outputs root (default: <install>/outputs)")
    return p.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


class CachedTextEncoder(torch.nn.Module):
    """Encodes every prompt once up front, then moves the 5.4 GB encoder off the GPU.

    Kimodo otherwise keeps the encoder in VRAM during diffusion; on an 8 GB card the diffusion model
    then spills into shared memory (10 s per step instead of about 1 s). Texts not seen up front
    are encoded on demand and cached too.
    """

    def __init__(self, encoder, texts):
        super().__init__()
        self.__dict__["encoder"] = encoder  # not a submodule: model.to(...) must not pull it back to the GPU
        self.cache = {}
        self._encode(sorted(set(texts)))

    def _encode(self, texts):
        if not texts:
            return
        feats, _ = self.encoder(list(texts))
        for t, f in zip(texts, feats):
            self.cache[t] = f.detach().to("cpu")
        if hasattr(self.encoder, "unload"):
            self.encoder.unload()
        torch.cuda.empty_cache()

    def forward(self, text):
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        self._encode([t for t in dict.fromkeys(texts) if t not in self.cache])
        feats = torch.stack([self.cache[t] for t in texts])
        return (feats[0], 1) if single else (feats, [1] * len(texts))


def soma77(skeleton):
    skeleton = skeleton.somaskel77 if hasattr(skeleton, "somaskel77") else skeleton
    return dict(joint_names=list(skeleton.bone_order_names),
                joint_parents=skeleton.joint_parents.detach().cpu().tolist(),
                neutral_joints=skeleton.neutral_joints.detach().cpu().numpy())


def skeleton_arrays(skel):
    return {"joint_names": np.array(skel["joint_names"]), "joint_parents": np.array(skel["joint_parents"]),
            "neutral_joints": np.asarray(skel["neutral_joints"], float)}


def segments(clip):
    """A clip's prompt segments: [(prompt, seconds)]; one for a plain clip."""
    if "segments" in clip:
        return [(s["prompt"], float(s["seconds"])) for s in clip["segments"]]
    return [(clip["prompt"], float(clip["seconds"]))]


def clip_constraints(clip, frames, fps, skel):
    """A clip's constraints as Kimodo's JSON list: its "path" as a root2d constraint on every frame, its "poses" and
    "hands", then its "constraints" (a file or a list) as they are."""
    out = []
    xz = None
    path = clip.get("path")
    if path:
        t = np.arange(frames) / fps
        if "points" in path:
            pts = np.array(sorted(path["points"]), float)  # [second, x, z]
            if len(pts) < 2:
                sys.exit(f"{clip['name']}: a path needs at least two points")
            xz = np.stack([np.interp(t, pts[:, 0], pts[:, 1]), np.interp(t, pts[:, 0], pts[:, 2])], 1)
        else:
            speed = np.full(frames, float(path["speed"]))
            if "stop_at" in path:  # slows to rest over the last 0.6 s before stop_at, then stands
                speed *= np.clip((float(path["stop_at"]) - t) / 0.6, 0.0, 1.0)
            xz = np.stack([np.zeros(frames), np.concatenate([[0.0], np.cumsum(speed[:-1]) / fps])], 1)
        k = max(1, int(round(0.3 * fps)) | 1)  # corners rounded over 0.3 s: the model's root is a smoothed one
        h = k // 2  # padded along the path's own direction at each end, so the ends keep their speed
        pad = np.concatenate([xz[0] - (xz[1] - xz[0]) * np.arange(h, 0, -1)[:, None], xz, xz[-1] + (xz[-1] - xz[-2]) * np.arange(1, h + 1)[:, None]])
        xz = np.stack([np.convolve(pad[:, c], np.ones(k) / k, mode="valid") for c in (0, 1)], 1)
        xz -= xz[0]
        xz = xz.round(5)
        out.append({"type": "root2d", "frame_indices": list(range(frames)), "smooth_root_2d": xz.tolist()})
    out += author.authored(clip, frames, fps, xz, skel, resolve)
    extra = clip.get("constraints")
    if isinstance(extra, str):
        extra = json.loads(resolve(extra).read_text(encoding="utf-8"))
    for c in extra or []:
        if max(c["frame_indices"]) >= frames:
            sys.exit(f"{clip['name']}: a constraint is on frame {max(c['frame_indices'])}, the clip has {frames}")
        out.append(c)
    return out


def export_fbx(skel, posed, rots, fps, target, fbx_path, in_place):
    from kimodo.exports import mixamo_retarget

    motion = SimpleNamespace(output_dict={"posed_joints": posed[None], "global_rot_mats": rots[None]},
                             fps=float(fps), batch_size=1, skeleton_name="SOMA77", **skel)
    mixamo_retarget.export_kimodo_fbx(motion, str(target), str(fbx_path), animate_in_place=in_place)


def main():
    args = parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    clips = spec["clips"]
    names = [c["name"] for c in clips]
    if len(set(names)) != len(names):
        sys.exit(f"duplicate clip names in {args.spec}")
    only = {n.strip() for n in args.only.split(",") if n.strip()}
    if only - set(names):
        sys.exit(f"--only names not in the spec: {sorted(only - set(names))}")

    target = resolve(spec.get("target_fbx", DEFAULT_TARGET))
    if not target.is_file():
        sys.exit(f"target character not found: {target}")
    out_dir = args.out / spec["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    from kimodo.exports import mixamo_retarget

    if not mixamo_retarget.HAS_FBX_SDK:
        sys.exit("FBX SDK (python 'fbx' module) missing in the venv; Mixamo export needs it.")

    # Work out what to run before loading the model (loading takes a few minutes).
    jobs = []
    for i, clip in enumerate(clips):
        if only and clip["name"] not in only:
            continue
        n = args.samples or clip.get("samples", spec.get("samples", 1))
        clip_dir = out_dir / clip["name"]
        fbx = [clip_dir / f"{clip['name']}_s{s + 1}.fbx" for s in range(n)]
        if args.reexport:
            missing = [f.with_suffix(".npz").name for f in fbx if not f.with_suffix(".npz").is_file()]
            if missing:
                sys.exit(f"--reexport: {clip['name']} has no saved motion {missing}; generate it first")
        elif not args.force and all(f.is_file() for f in fbx):
            print(f"skip {clip['name']}: output exists (use --force to redo)", flush=True)
            continue
        jobs.append((i, clip, n, clip_dir, fbx))

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"clips": {}}
    manifest.update(name=spec["name"], target_fbx=str(target), spec=str(args.spec.resolve()))

    def record(clip, i, n, fbx, frames, seed, steps, in_place, measured=None):
        manifest["clips"][clip["name"]] = {
            "prompt": " | ".join(p for p, _ in segments(clip)), "seconds": sum(t for _, t in segments(clip)), "frames": frames,
            "note": clip.get("note", ""), "seed": seed, "diffusion_steps": steps, "in_place": in_place,
            **{k: clip[k] for k in ("path", "poses", "hands", "ground", "segments") if clip.get(k)},
            **({"constraints": True} if clip.get("constraints") else {}), **(measured or {}),
            "files": [f.relative_to(out_dir).as_posix() for f in fbx],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if jobs and args.reexport:
        from kimodo.skeleton.definitions import SOMASkeleton77

        skel = soma77(SOMASkeleton77())
        for i, clip, n, clip_dir, fbx in jobs:
            in_place = clip.get("in_place", spec.get("in_place", False))
            for f in fbx:
                d = np.load(f.with_suffix(".npz"))
                export_fbx(skel, d["posed_joints"], d["global_rot_mats"], float(d["fps"]), target, f, in_place)
            entry = manifest["clips"].get(clip["name"], {})
            record(clip, i, n, fbx, int(d["posed_joints"].shape[0]), entry.get("seed", int(d["seed"])),
                   entry.get("diffusion_steps", spec.get("diffusion_steps", 100)), in_place)
            print(f"re-exported {clip['name']}: {n} sample(s)", flush=True)

    elif jobs:
        from kimodo import load_model
        from kimodo.constraints import load_constraints_lst
        from kimodo.model.registry import get_model_info
        from kimodo.tools import seed_everything

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model, resolved = load_model(spec.get("model", "kimodo-soma-rp"), device=device,
                                     default_family="Kimodo", return_resolved_name=True)
        info = get_model_info(resolved)
        manifest.update(model=info.display_name if info else resolved, fps=float(model.fps))
        skel, fps = soma77(model.skeleton), float(model.fps)
        print(f"model {manifest['model']}, {fps:g} fps, target {target.name}, {len(jobs)} clip(s)", flush=True)
        t0 = time.time()
        if device != "cpu":
            torch.cuda.reset_peak_memory_stats()
        texts = [p for _, clip, *_ in jobs for p, _ in segments(clip)]
        model.text_encoder = CachedTextEncoder(model.text_encoder, texts)
        peak = f", GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if device != "cpu" else ""
        print(f"encoded {len(set(texts))} prompt(s) in {time.time() - t0:.0f} s, encoder moved off the GPU{peak}", flush=True)

        for i, clip, n, clip_dir, fbx in jobs:
            t0 = time.time()
            segs = segments(clip)
            seg_frames = [int(round(t * fps)) for _, t in segs]
            frames = sum(seg_frames)
            seed = clip.get("seed", spec.get("seed", 1000) + 100 * i)
            steps = clip.get("diffusion_steps", spec.get("diffusion_steps", 100))
            in_place = clip.get("in_place", spec.get("in_place", False))
            seed_everything(seed)
            cons = clip_constraints(clip, frames, fps, skel)
            extra = {}
            if cons:
                extra["constraint_lst"] = load_constraints_lst(cons, model.skeleton, device=device)
                extra["cfg_weight"] = [2.0, float(clip.get("constraint_weight", 2.0))]
            if device != "cpu":
                torch.cuda.reset_peak_memory_stats()
            if len(segs) > 1:  # one motion through all the prompts, each segment started inside the last one's end
                out = model([p for p, _ in segs], seg_frames, num_denoising_steps=steps, num_samples=n, multi_prompt=True,
                            num_transition_frames=int(clip.get("transition", 5)), post_processing=True, return_numpy=True,
                            seeds=[seed + j for j in range(len(segs))], **extra)
            else:
                out = model(segs[0][0], frames, num_denoising_steps=steps, num_samples=n,
                            post_processing=True, return_numpy=True,
                            seeds=[seed], **extra)  # per prompt segment; samples share it and differ by batch noise
            clip_dir.mkdir(parents=True, exist_ok=True)
            if cons:
                (clip_dir / "constraints.json").write_text(json.dumps(cons), encoding="utf-8")
            measured = {}
            for s in range(n):
                # the whole model output for this sample (foot contacts, root path...) and the skeleton it is on, so the
                # npz is a complete source for retargeting onto another rig without Kimodo
                take = {k: v[s] for k, v in out.items() if hasattr(v, "shape") and v.ndim and v.shape[0] == n}
                pins = author.residual(take["posed_joints"], cons, skel)  # before the ground: the targets are on the flat floor
                if pins:
                    measured.setdefault("pin_cm", []).append(pins)
                if clip.get("ground"):
                    take, fit = author.apply_ground(take, clip["ground"], skel, fps)
                    measured.setdefault("ground_fit", []).append(fit)
                np.savez_compressed(fbx[s].with_suffix(".npz"), **take, fps=fps, prompt=" | ".join(p for p, _ in segs), seed=seed,
                                    sample=s, **skeleton_arrays(skel))
                export_fbx(skel, take["posed_joints"], take["global_rot_mats"], fps, target, fbx[s], in_place)
            record(clip, i, n, fbx, frames, seed, steps, in_place, measured)
            peak = f", GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if device != "cpu" else ""
            print(f"done {clip['name']}: {n} sample(s), {frames} frames, {time.time() - t0:.0f} s{peak}"
                  + "".join(f", {k} {v}" for k, v in measured.items()), flush=True)

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"outputs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
