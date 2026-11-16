"""Checks a Kimodo install end to end. Exit code 0 = ready, 1 = something the demo needs is broken.

Run with the venv's Python: venv311\\Scripts\\python.exe scripts\\kimodo_check.py
"""

import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "kimodo"
NF4_DIR = ROOT / "KIMODO-Meta3_llm2vec_NF4"
VISER_BUILD = REPO / "kimodo-viser" / "src" / "viser" / "client" / "build" / "index.html"

failures = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  OK    {name}" + (f": {detail}" if detail else ""))
    except Exception as e:  # report every failure, not just the first
        failures.append(name)
        print(f"  FAIL  {name}: {e}")
        if "-v" in sys.argv:
            traceback.print_exc()


def python_version():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Python {sys.version.split()[0]}, expected 3.11")
    return sys.version.split()[0]


def torch_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__} sees no CUDA device")
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}"


def triton_kernel():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

    a = torch.rand(4096, device="cuda")
    out = torch.empty_like(a)
    add[(triton.cdiv(a.numel(), 1024),)](a, a, out, a.numel(), BLOCK=1024)
    err = (out - 2 * a).abs().max().item()
    if err != 0:
        raise RuntimeError(f"kernel result off by {err}")
    return f"triton {triton.__version__}, test kernel exact"


def nf4_model():
    missing = [f for f in ("config.json", "model.safetensors", "tokenizer.json") if not (NF4_DIR / f).is_file()]
    if missing:
        raise RuntimeError(f"{NF4_DIR} lacks {', '.join(missing)}")
    return str(NF4_DIR)


def viser_client():
    if not VISER_BUILD.is_file():
        raise RuntimeError(f"{VISER_BUILD} missing (viser client not built)")
    return None


def motion_correction():
    import motion_correction  # noqa: F401

    return None


def kimodo_import():
    import kimodo
    import kimodo.demo  # noqa: F401
    from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder

    enc = LLM2VecEncoder("unused", None, "bfloat16", 4096)  # resolves and checks the NF4 path, loads nothing
    if Path(enc.custom_dir).resolve() != NF4_DIR.resolve():
        print(f"        note: text encoder from KIMODO_NF4_DIR = {enc.custom_dir}")
    return str(Path(kimodo.__file__).parents[1])


def repo_branch():
    out = subprocess.run(["git", "-C", str(REPO), "branch", "--show-current"], capture_output=True, text=True)
    branch = out.stdout.strip()
    if branch != "local":
        raise RuntimeError(
            f"kimodo repo is on '{branch}', not 'local' (the branch with the NVIDIA sync, Mixamo export and fixes)"
        )
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout
    return "branch local" + (", uncommitted changes" if dirty.strip() else ", clean")


print("Kimodo install check")
check("Python 3.11", python_version)
check("PyTorch + CUDA", torch_cuda)
check("Triton", triton_kernel)
check("NF4 text encoder", nf4_model)
check("Viser client build", viser_client)
check("motion_correction", motion_correction)
check("kimodo.demo import", kimodo_import)
check("kimodo repo", repo_branch)

if failures:
    print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("\nAll checks passed. Start with start_kimodo.bat")
