"""Checks that a Kimodo environment can run kimodo-mcp's jobs. Run it with the Python of your Kimodo install:

    <kimodo python> kimodo_doctor.py

Required: Python 3.10+, kimodo importable, torch with CUDA, numpy and scipy. Optional: the Mixamo FBX export
(kimodo.exports.mixamo_retarget with the FBX SDK) and the default target character, Blender and ffmpeg for previews.
Exit code 0 when every required check passes.
"""

import importlib.util
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ.get("KIMODO_ROOT") or Path.cwd())
BLENDER = os.environ.get("BLENDER", r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
failed = []


def check(name, fn, required=True):
    try:
        detail = fn()
        print(f"  OK        {name}" + (f": {detail}" if detail else ""))
    except Exception as e:  # report every check, not the first failure only
        print(f"  {'FAIL' if required else 'OPTIONAL'}  {name}: {e}")
        if required:
            failed.append(name)


def python():
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python {sys.version.split()[0]}; Kimodo needs 3.10+")
    return f"{sys.version.split()[0]} at {sys.executable}"


def kimodo():
    spec = importlib.util.find_spec("kimodo")
    if spec is None:
        raise RuntimeError("kimodo is not installed in this Python (set KIMODO_PYTHON to your Kimodo env's python)")
    return str(Path(spec.origin).parent)


def torch_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__} sees no CUDA device")
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}"


def libs():
    import numpy
    import scipy

    return f"numpy {numpy.__version__}, scipy {scipy.__version__}"


def skin():
    spec = importlib.util.find_spec("kimodo")
    f = Path(spec.origin).parent / "assets" / "skeletons" / "somaskel77" / "skin_standard.npz"
    if not f.is_file():
        raise RuntimeError(f"no {f} (previews draw the SOMA body from it)")
    return "SOMA body for previews"


def fbx_export():
    from kimodo.exports import mixamo_retarget

    if not mixamo_retarget.HAS_FBX_SDK:
        raise RuntimeError("the FBX SDK (python 'fbx' module) is missing: takes are written as .npz only")
    target = ROOT / "assets" / "characters" / "Y Bot.fbx"
    return "Mixamo export ready" + ("" if target.is_file() else f"; default target {target} missing (set target_fbx)")


def blender():
    if not Path(BLENDER).is_file() and not shutil.which(BLENDER):
        raise RuntimeError(f"Blender not found at {BLENDER} (set BLENDER); preview needs it")
    return BLENDER


def ffmpeg():
    f = shutil.which("ffmpeg")
    if not f:
        raise RuntimeError("ffmpeg not on PATH; preview and view need it")
    return f


print(f"kimodo-mcp doctor (workspace {ROOT})")
check("python", python)
check("kimodo", kimodo)
check("torch + CUDA", torch_cuda)
check("numpy, scipy", libs)
check("SOMA skin asset", skin)
check("Mixamo FBX export", fbx_export, required=False)
check("Blender", blender, required=False)
check("ffmpeg", ffmpeg, required=False)
print("ready" if not failed else f"not ready: {', '.join(failed)}")
sys.exit(1 if failed else 0)
