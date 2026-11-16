"""What kimodo_batch.py writes for a clip beyond its text and path: key poses, hand targets, ground. numpy and scipy only.

All of it is on the SOMA77 skeleton, in Kimodo's frame: metres, Y up, +Z ahead, +X to the character's left. In the rest
pose (a T-pose) every joint's global rotation is the identity.

    "poses": [{"at": 2.0, "from": "outputs/x/look_far/look_far_s1.npz", "frame": 45}]
        the body is in that take's pose at second 2. Optional: "until": 3.0 (held to second 3), "part": "left-hand" |
        "right-hand" | "left-foot" | "right-foot" (only that limb's end is fixed; default "fullbody"), "mirror": true
        (left and right swapped: a right-arm wave becomes a left-arm one). The pose stands where it stood in its own take;
        on a clip with a "path" it stands on the path.
    "hands": [{"at": 1.5, "hand": "left", "pos": [0.3, 1.7, 0.15]}]
        the wrist is at that point at second 1.5: [to the character's left, above the ground, ahead of the hips].
        Optional "until". Kimodo fixes the hips with every hand target, so the character stands at its start point (or
        on the path) at those times, upright (hips HIP_HEIGHT above the ground).
    "ground": {"rise": 0.12}                         # the land climbs 12 cm per metre ahead (negative: downhill)
    "ground": {"profile": [[0, 0], [2, 0], [4, 0.4]]}   # [metres ahead, height]: level for 2 m, then a climb
        after generation the take is fitted to that land: the hips ride over it, each foot is lifted by the land under
        it and tilted to it, the legs are solved again. The take is made on a flat floor; nothing else changes.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HIP_HEIGHT = 0.98  # of an upright SOMA person (0.96-0.99 in generated walks; 1.005 with straight legs)
ARMS_DOWN = np.radians(80)
LIMBS = {"left-hand": ("LeftArm", "LeftForeArm", "LeftHand"), "right-hand": ("RightArm", "RightForeArm", "RightHand"),
         "left-foot": ("LeftLeg", "LeftShin", "LeftFoot"), "right-foot": ("RightLeg", "RightShin", "RightFoot")}
MIRROR = np.diag([-1.0, 1.0, 1.0])


def fk(local, root, parents, neutral):
    """Local rotations [F, J, 3, 3] and hip positions [F, 3] -> global rotations and joint positions."""
    local = np.asarray(local, float)
    G, P = np.empty_like(local), np.empty(local.shape[:2] + (3,))
    for j, p in enumerate(parents):
        if p < 0:
            G[:, j], P[:, j] = local[:, j], root
        else:
            G[:, j] = G[:, p] @ local[:, j]
            P[:, j] = P[:, p] + G[:, p] @ (neutral[j] - neutral[p])
    return G, P


def rot_between(a, b):
    """The shortest rotation that turns direction a onto direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def two_bone(local, G, P, parents, chain, target, pole, end_rot=None):
    """One frame: turns the upper and lower bone of `chain` (three joint indices) so its end is at `target`, bending in
    the plane it bends in now (or towards `pole` when it is straight). The end joint keeps its rotation relative to the
    lower bone, or takes the global rotation `end_rot`. Edits local [J, 3, 3] in place; returns how far short it fell."""
    a, b, c = chain
    pa, pb, pc = P[a], P[b], P[c]
    la, lb = np.linalg.norm(pb - pa), np.linalg.norm(pc - pb)
    t = target - pa
    dist = np.linalg.norm(t)
    d = float(np.clip(dist, abs(la - lb) + 1e-4, la + lb - 1e-4))
    u = t / max(dist, 1e-9)
    w = (pb - pa) - ((pb - pa) @ u) * u
    if np.linalg.norm(w) < 0.02 * la:
        w = pole - (pole @ u) * u
    w /= np.linalg.norm(w)
    cos_a = (la * la + d * d - lb * lb) / (2 * la * d)
    nb = pa + la * (cos_a * u + np.sqrt(max(0.0, 1 - cos_a * cos_a)) * w)
    ra = rot_between(pb - pa, nb - pa)
    ga = ra @ G[a]
    rb = rot_between(ra @ (pc - pb), pa + d * u - nb)
    gb = rb @ ra @ G[b]
    gc = end_rot if end_rot is not None else rb @ ra @ G[c]
    local[a], local[b], local[c] = G[parents[a]].T @ ga, ga.T @ gb, gb.T @ gc
    return abs(dist - d)


def _span(item, frames, fps):
    first = int(round(float(item["at"]) * fps))
    last = int(round(float(item.get("until", item["at"])) * fps))
    if not 0 <= first <= last or first >= frames:
        raise SystemExit(f"a pose or hand target at {item['at']} s is outside the clip")
    return list(range(first, min(last, frames - 1) + 1))


def _mirror(local, root, names):
    swap = [names.index(n.replace("Left", "\0").replace("Right", "Left").replace("\0", "Right")) for n in names]
    return MIRROR @ local[swap] @ MIRROR, root * [-1.0, 1.0, 1.0]


def _constraint(kind, idx, local, root, smooth):
    n = len(idx)
    rv = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(n, -1, 3)
    return {"type": kind, "frame_indices": idx, "local_joints_rot": rv.round(6).tolist(),
            "root_positions": np.asarray(root).round(5).tolist(), "smooth_root_2d": np.asarray(smooth).round(5).tolist()}


def authored(clip, frames, fps, xz, skel, resolve):
    """The clip's "poses" and "hands" as Kimodo constraint dicts. xz: the path's [x, z] on every frame, or None."""
    names, parents = list(skel["joint_names"]), list(skel["joint_parents"])
    neutral = np.asarray(skel["neutral_joints"], float)
    out = []
    for pose in clip.get("poses", []):
        src = np.load(resolve(pose["from"]))
        f = int(pose.get("frame", 0))
        if not 0 <= f < len(src["local_rot_mats"]):
            raise SystemExit(f"{clip['name']}: {pose['from']} has no frame {f}")
        local, root = src["local_rot_mats"][f].astype(float), src["root_positions"][f].astype(float)
        if pose.get("mirror"):
            local, root = _mirror(local, root, names)
        idx = _span(pose, frames, fps)
        roots = np.tile(root, (len(idx), 1))
        if xz is not None:
            roots[:, [0, 2]] = xz[idx]
        kind = pose.get("part", "fullbody")
        if kind != "fullbody" and kind not in LIMBS:
            raise SystemExit(f"{clip['name']}: unknown part {kind!r}")
        out.append(_constraint(kind, idx, np.tile(local, (len(idx), 1, 1, 1)), roots, roots[:, [0, 2]]))
    # hand targets: one standing pose per frame with every hand that has a target on it. Two targets on one frame built as
    # two poses left Kimodo's post-processing one of them only, and the other hand was put back where that pose had it:
    # hanging (shrug: the left hand 43 cm from its target)
    wanted = {}
    for hand in clip.get("hands", []):
        kind = f"{hand['hand']}-hand"
        if kind not in LIMBS:
            raise SystemExit(f"{clip['name']}: hand must be left or right")
        for f in _span(hand, frames, fps):
            wanted.setdefault(f, {})[kind] = hand["pos"]
    posed = {}
    for f, goals in wanted.items():
        ground, heading = np.zeros(2), 0.0
        if xz is not None:
            ground = xz[f]
            step = xz[min(f + 1, frames - 1)] - xz[max(f - 1, 0)]
            if np.linalg.norm(step) > 1e-4:
                heading = float(np.arctan2(step[0], step[1]))
        turn = rot_y(heading)
        local = np.tile(np.eye(3), (len(names), 1, 1))
        local[0] = turn
        for arm, sign in (("LeftArm", -1), ("RightArm", 1)):  # arms down from the T-pose
            local[names.index(arm)] = Rotation.from_euler("z", sign * ARMS_DOWN).as_matrix()
        root = np.array([ground[0], HIP_HEIGHT, ground[1]])
        G, P = fk(local[None], root[None], parents, neutral)
        for kind, (x, y, z) in goals.items():
            target = turn @ np.array([x, 0.0, z]) + [ground[0], y, ground[1]]
            short = two_bone(local, G[0], P[0], parents, [names.index(n) for n in LIMBS[kind]], target, turn @ np.array([0.0, -1.0, -0.3]))
            if short > 0.01:
                raise SystemExit(f"{clip['name']}: the {kind} cannot reach {[x, y, z]} ({short * 100:.0f} cm short)")
        posed[f] = (local, root)
    for kind in ("left-hand", "right-hand"):
        idx = sorted(f for f, goals in wanted.items() if kind in goals)
        if idx:
            roots = np.array([posed[f][1] for f in idx])
            out.append(_constraint(kind, idx, np.array([posed[f][0] for f in idx]), roots, roots[:, [0, 2]]))
    return out


def targets(cons, skel):
    """Where each pose or limb constraint puts its joints: [(type, frames, joint indices, positions [n, k, 3])]."""
    names, parents = list(skel["joint_names"]), list(skel["joint_parents"])
    neutral = np.asarray(skel["neutral_joints"], float)
    out = []
    for c in cons:
        if c["type"] == "root2d" or "local_joints_rot" not in c:
            continue
        rv = np.asarray(c["local_joints_rot"], float)
        if rv.shape[1] != len(names):
            continue  # a file saved on another skeleton: not measured
        local = Rotation.from_rotvec(rv.reshape(-1, 3)).as_matrix().reshape(rv.shape[0], -1, 3, 3)
        _, P = fk(local, np.asarray(c["root_positions"], float), parents, neutral)
        joints = list(range(len(names))) if c["type"] == "fullbody" else [names.index(LIMBS[c["type"]][2])] if c["type"] in LIMBS else None
        if joints:
            out.append((c["type"], list(c["frame_indices"]), joints, P[:, joints]))
    return out


def residual(posed, cons, skel):
    """How far the take's joints are from each constraint's, worst frame, in cm: {"left-hand": 0.4, ...}."""
    worst = {}
    for kind, idx, joints, pos in targets(cons, skel):
        err = np.linalg.norm(posed[idx][:, joints] - pos, axis=-1).max() * 100
        worst[kind] = round(max(worst.get(kind, 0.0), float(err)), 2)
    return worst


def _height(ground):
    if "profile" in ground:
        pts = np.array(sorted(ground["profile"]), float)
        if len(pts) < 2:
            raise SystemExit("a ground profile needs at least two [metres ahead, height] points")
    else:
        rise = float(ground["rise"])
        pts = np.array([[-1000.0, -1000.0 * rise], [1000.0, 1000.0 * rise]])
    spread = np.linspace(-0.15, 0.15, 7)  # corners of the profile rounded over 30 cm: a foot is 25 cm long

    def h(z):
        z = np.asarray(z, float)
        return np.interp(z[..., None] + spread, pts[:, 0], pts[:, 1]).mean(-1)

    return h, lambda z: (h(np.asarray(z, float) + 0.05) - h(np.asarray(z, float) - 0.05)) / 0.1


def apply_ground(take, ground, skel, fps):
    """Fits one take (the batch's dict of arrays, made on a flat floor) to the land of `ground`. Returns the new dict
    and what was measured on it."""
    names, parents = list(skel["joint_names"]), list(skel["joint_parents"])
    neutral = np.asarray(skel["neutral_joints"], float)
    h, slope = _height(ground)
    local, root = take["local_rot_mats"].astype(float), take["root_positions"].astype(float)
    G0, P0 = fk(local, root, parents, neutral)
    lift = h(take["smooth_root_pos"][:, 2]) if "smooth_root_pos" in take else h(root[:, 2])
    legs = [[names.index(n) for n in LIMBS[k]] for k in ("left-foot", "right-foot")]
    goal = {}
    over = np.zeros(len(root))
    for chain in legs:  # each ankle goes up by the land under it
        a, b, c = chain
        goal[c] = P0[:, c] + np.c_[np.zeros(len(root)), h(P0[:, c, 2]), np.zeros(len(root))]
        reach = np.linalg.norm(neutral[b] - neutral[a]) + np.linalg.norm(neutral[c] - neutral[b])
        hip = P0[:, a] + np.c_[np.zeros(len(root)), lift, np.zeros(len(root))]
        over = np.maximum(over, np.linalg.norm(goal[c] - hip, axis=1) - 0.995 * reach)
    k = max(3, int(round(0.3 * fps)) | 1)  # where a leg would be too short for its foot the hips come down, eased over 0.3 s
    pad = np.pad(np.maximum(over, 0.0), k // 2, mode="edge")
    drop = np.array([pad[i:i + k].max() for i in range(len(root))])
    drop = np.convolve(np.pad(drop, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    root[:, 1] += lift - drop
    G, P = fk(local, root, parents, neutral)
    for f in range(len(root)):
        for chain in legs:
            c = chain[2]
            tilt = Rotation.from_euler("x", -np.arctan(float(slope(P0[f, c, 2])))).as_matrix()
            knee = rot_y(np.arctan2(G0[f, 0][0, 2], G0[f, 0][2, 2])) @ np.array([0.0, 0.0, 1.0])  # ahead of the hips
            two_bone(local[f], G[f], P[f], parents, chain, goal[c][f], knee, end_rot=tilt @ G0[f, c])
    G, P = fk(local, root, parents, neutral)
    out = {k_: v for k_, v in take.items() if k_ != "posed_joints_norootpos"}  # that array is the flat take's; nothing reads it
    out.update(local_rot_mats=local.astype(np.float32), global_rot_mats=G.astype(np.float32), posed_joints=P.astype(np.float32),
               root_positions=root.astype(np.float32))
    if "smooth_root_pos" in take:
        smooth = take["smooth_root_pos"].copy()
        smooth[:, 1] += lift - drop
        out["smooth_root_pos"] = smooth
    feet_err = max(float(np.linalg.norm(P[:, c] - goal[c], axis=1).max()) for c in goal) * 100
    return out, {"climb_m": round(float(lift[-1] - lift[0]), 3), "foot_off_land_cm": round(feet_err, 2), "hips_lowered_cm": round(float(drop.max()) * 100, 2)}
