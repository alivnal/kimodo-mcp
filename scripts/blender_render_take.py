"""Blender (run with -b -P): renders a take to numbered PNG frames.

    blender -b -P blender_render_take.py -- <take.npz | take.fbx> <out_dir> [size] [skin.npz]

A .npz take (kimodo_batch's raw SOMA motion) is shown on Kimodo's own SOMA body: the skin file is posed with the
take's joint rotations and positions, so the picture is the motion itself and not a retarget of it. A .fbx is shown
as it is (the Mixamo character). 3/4 front camera that follows the hips over the ground (smoothed), framed on the
body's largest pose, a 0.5 m floor grid so travel stays visible, Workbench render. Prints "FRAMES <first> <last> <fps>".
"""

import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out_dir = Path(argv[0]).resolve(), Path(argv[1]).resolve()  # relative render paths land on the drive root
size = int(argv[2]) if len(argv) > 2 else 512
out_dir.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

if src.suffix == ".npz":
    take, skin = np.load(src), np.load(argv[3])
    if [str(n) for n in take["joint_names"]] != [str(n) for n in skin["rig_joint_names"]]:
        sys.exit(f"{src.name} is not on the skin's skeleton")
    to_blender = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # SOMA: Y up, +Z ahead -> Z up, facing -Y
    joints = take["posed_joints"].astype(np.float64)
    pose = np.tile(np.eye(4), joints.shape[:2] + (1, 1))
    pose[..., :3, :3], pose[..., :3, 3] = take["global_rot_mats"], joints
    move = (pose @ np.linalg.inv(skin["bind_rig_transform"].astype(np.float64)))[..., :3, :]  # [F, J, 3, 4]
    bind = np.c_[skin["bind_vertices"], np.ones(len(skin["bind_vertices"]))]
    idx, wgt = skin["lbs_indices"], skin["lbs_weights"].astype(np.float64)
    verts = np.zeros((len(joints), len(bind), 3))
    for k in range(idx.shape[1]):  # linear blend skinning, one influence column at a time
        verts += np.einsum("fvij,vj->fvi", move[:, idx[:, k]], bind) * wgt[:, k, None]
    verts, joints = verts @ to_blender.T, joints @ to_blender.T
    mesh = bpy.data.meshes.new("soma")
    mesh.from_pydata(verts[0].tolist(), [], skin["faces"].tolist())
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    body = bpy.data.objects.new("soma", mesh)
    scene.collection.objects.link(body)
    mat = bpy.data.materials.new("soma")
    mat.diffuse_color = (0.33, 0.55, 0.62, 1)
    mesh.materials.append(mat)
    f0, f1 = 1, len(joints)
    scene.render.fps = int(round(float(take["fps"])))
    points = {f: [Vector(p) for p in joints[f - f0]] for f in range(f0, f1 + 1)}
    hip = {f: points[f][0] for f in points}

    def show(f):
        mesh.vertices.foreach_set("co", verts[f - f0].ravel())
        mesh.update()
else:
    bpy.ops.import_scene.fbx(filepath=str(src))
    arm = next(o for o in scene.objects if o.type == "ARMATURE")
    action = arm.animation_data.action if arm.animation_data else None
    if action is None:
        sys.exit(f"no animation in {src}")
    f0, f1 = (int(round(v)) for v in action.frame_range)
    points, hip = {}, {}
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        hip[f] = arm.matrix_world @ arm.pose.bones[0].head
        points[f] = [arm.matrix_world @ pb.head for pb in arm.pose.bones] + [arm.matrix_world @ pb.tail for pb in arm.pose.bones]

    def show(f):
        scene.frame_set(f)

frames = list(range(f0, f1 + 1))

# Per-frame hips position and the body's extent around it.
d = Vector((0.55, -0.83, 0.18)).normalized()  # from the subject towards the camera (the character faces -Y)
right = Vector((0, 0, 1)).cross(d).normalized()
up = d.cross(right).normalized()
hip_xy = {f: Vector((hip[f].x, hip[f].y, 0.0)) for f in frames}
span_x = [(p - hip_xy[f]).dot(right) for f in frames for p in points[f]]
span_y = [(p - hip_xy[f]).dot(up) for f in frames for p in points[f]]
floor = min(0.0, min(p.z for f in frames for p in points[f]))

# Follow the hips with a 15-frame moving average so steps do not shake the frame.
follow = {}
for i, f in enumerate(frames):
    win = [hip_xy[g] for g in frames[max(0, i - 7): i + 8]]
    follow[f] = sum(win, Vector()) / len(win)

# Floor: grey plane plus a 0.5 m grid of thin bars, covering the whole path.
xs = [v.x for v in hip_xy.values()]
ys = [v.y for v in hip_xy.values()]
cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
extent = max(max(xs) - min(xs), max(ys) - min(ys)) + 40.0  # wide enough that no edge shows behind the body
mat_floor = bpy.data.materials.new("floor")
mat_floor.diffuse_color = (0.62, 0.62, 0.62, 1)
mat_grid = bpy.data.materials.new("grid")
mat_grid.diffuse_color = (0.35, 0.35, 0.35, 1)
bpy.ops.mesh.primitive_plane_add(size=extent, location=(cx, cy, floor))
bpy.context.object.data.materials.append(mat_floor)
n = int(extent / 0.5)
bpy.ops.mesh.primitive_grid_add(x_subdivisions=n, y_subdivisions=n, size=n * 0.5, location=(cx, cy, floor + 0.002))
grid = bpy.context.object
grid.data.materials.append(mat_grid)
wf = grid.modifiers.new("wire", "WIREFRAME")
wf.thickness = 0.012

world = bpy.data.worlds.new("bg")
world.color = (0.86, 0.87, 0.9)
scene.world = world

# Camera: orthographic, framed on the largest pose plus a margin, following the hips.
cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = max(max(span_x) - min(span_x), max(span_y) - min(span_y)) * 1.2 + 0.3
cam_data.clip_end = 200
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
# Spans are measured from the hips' ground point, so aiming at that point plus the spans' middle
# keeps the body centred.
mid = right * (max(span_x) + min(span_x)) / 2 + up * (max(span_y) + min(span_y)) / 2
scene.camera = cam

scene.render.engine = "BLENDER_WORKBENCH"
shading = scene.display.shading
shading.light = "STUDIO"
shading.color_type = "MATERIAL"
shading.show_shadows = True
shading.show_cavity = True
scene.render.resolution_x = scene.render.resolution_y = size
scene.render.image_settings.file_format = "PNG"
for f in frames:
    show(f)
    cam.location = follow[f] + mid + d * 30
    scene.render.filepath = str(out_dir / f"f_{f:04d}.png")
    bpy.ops.render.render(write_still=True)
print(f"FRAMES {f0} {f1} {scene.render.fps / scene.render.fps_base:g}")
