"""Kimodo take .npz -> SOMA BVH (Kimodo's own exporter, standard T-pose rest), e.g. for NVIDIA soma-retargeter.

    venv311\\Scripts\\python.exe scripts\\npz_to_bvh.py <take.npz> <out.bvh> [--bones-seed-rest]
"""
import sys

import numpy as np
import torch

from kimodo.exports.bvh import save_motion_bvh
from kimodo.skeleton import global_rots_to_local_rots
from kimodo.skeleton.definitions import SOMASkeleton77

src, out = sys.argv[1], sys.argv[2]
d = np.load(src)
skel = SOMASkeleton77()
joints = torch.from_numpy(d["posed_joints"]).float()
rots = torch.from_numpy(d["global_rot_mats"]).float()
local = global_rots_to_local_rots(rots, skel)
save_motion_bvh(out, local, joints[:, skel.root_idx, :], skeleton=skel, fps=float(d["fps"]),
                standard_tpose="--bones-seed-rest" not in sys.argv)
print(f"wrote {out}: {len(joints)} frames at {float(d['fps']):g} fps")
