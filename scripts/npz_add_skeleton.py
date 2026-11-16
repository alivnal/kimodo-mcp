"""Adds the SOMA77 skeleton (joint_names, joint_parents, neutral_joints) to Kimodo take .npz files written before
kimodo_batch.py stored it, so they can be retargeted without Kimodo. Files that have it already are left alone.

    venv311\\Scripts\\python.exe scripts\\npz_add_skeleton.py <file.npz | folder> ...
"""
import sys
from pathlib import Path

import numpy as np

from kimodo.skeleton.definitions import SOMASkeleton77

skel = SOMASkeleton77()
extra = {"joint_names": np.array(list(skel.bone_order_names)),
         "joint_parents": np.array(skel.joint_parents.tolist()),
         "neutral_joints": skel.neutral_joints.detach().cpu().numpy().astype(float)}
files = [f for a in sys.argv[1:] for f in (sorted(Path(a).rglob("*.npz")) if Path(a).is_dir() else [Path(a)])]
for f in files:
    d = dict(np.load(f, allow_pickle=False))
    if "joint_names" in d:
        print(f"has it: {f}")
        continue
    if d["posed_joints"].shape[1] != len(extra["joint_names"]):
        print(f"skipped, not SOMA77: {f}")
        continue
    np.savez_compressed(f, **d, **extra)
    print(f"added: {f}")
