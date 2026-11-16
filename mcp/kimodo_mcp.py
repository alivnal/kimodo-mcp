"""MCP server for a local NVIDIA Kimodo install: specs, generation, previews, measures, picks and an optional retarget
step, as tools.

The server itself never loads Kimodo or torch. Every GPU or Blender job runs the install's own scripts
(scripts/kimodo_batch.py, scripts/kimodo_preview.py, the retarget script if one is set) as a separate process, so the
model is loaded per job and the GPU is released when the job's process exits: an 8 GB card stays free for Blender,
renders and other tools between jobs. A generation is refused while another Kimodo process uses the GPU.

Jobs are detached: they outlive the Claude session that started them. Each job is a record in
outputs/_logs/_mcp_jobs/<id>.json, written by a small runner (this file with --run-job) that waits for the process and
stores its exit code, so any session can read any job's state.

    <install>/mcp/.venv/Scripts/python.exe <install>/mcp/kimodo_mcp.py        (stdio MCP server; see README.md)

Author: Ali Unal (https://github.com/alivnal). Copyright (c) 2026 Ali Unal.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

WINDOWS = os.name == "nt"
ROOT = Path(os.environ.get("KIMODO_ROOT") or Path(__file__).resolve().parents[1])  # the install: scripts/, specs/, outputs/
REPO = ROOT / "kimodo"
PY = Path(os.environ.get("KIMODO_PYTHON") or ROOT / "venv311" / ("Scripts/python.exe" if WINDOWS else "bin/python"))
SCRIPTS = ROOT / "scripts"
SPECS = ROOT / "specs"
OUTPUTS = ROOT / "outputs"
LOGS = OUTPUTS / "_logs"
JOBS_DIR = LOGS / "_mcp_jobs"
# Optional retarget step: a bash script with the interface <script> <mesh> <run> [clip ...] and the env CLIPS_DIR,
# RIG_FROM, JOBS, FAST, BASELINE, TUBE_ARMS (e.g. scripts/mesh_to_clips.sh of a SkinTokens pipeline). The retarget tool
# exists only when it is set. It runs in KIMODO_RETARGET_CWD (default: the parent of the script's folder) and writes
# anim/<run>/report.md there.
RETARGET_SCRIPT = os.environ.get("KIMODO_RETARGET_SCRIPT")
RETARGET_CWD = Path(os.environ.get("KIMODO_RETARGET_CWD")
                    or (Path(RETARGET_SCRIPT).resolve().parents[1] if RETARGET_SCRIPT else ROOT))
RETARGET_NAME = Path(RETARGET_SCRIPT).name if RETARGET_SCRIPT else "mesh_to_clips.sh"
BASH = os.environ.get("GIT_BASH") or (r"C:\Program Files\Git\bin\bash.exe" if WINDOWS else "bash")
VRAM_WARN_GB = 5.5  # the NF4 text encoder peaks at 5.4 GB while it encodes the prompts
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

CREATE_NO_WINDOW = 0x08000000 if WINDOWS else 0
CREATE_NEW_PROCESS_GROUP = 0x00000200 if WINDOWS else 0
CREATE_BREAKAWAY_FROM_JOB = 0x01000000 if WINDOWS else 0

# An MCP host may start the server with a minimal Windows environment. Without these, nvidia-smi fails ("Failed to
# initialize NVML") and PIL cannot find arial.ttf in the jobs (measured 16 Nov 2026). Jobs inherit this environment.
_SYSROOT = os.environ.get("SYSTEMROOT", r"C:\Windows")
for _k, _v in {} if not WINDOWS else {"WINDIR": _SYSROOT, "ProgramFiles": r"C:\Program Files", "ProgramW6432": r"C:\Program Files",
               "ProgramFiles(x86)": r"C:\Program Files (x86)", "CommonProgramFiles": r"C:\Program Files\Common Files",
               "ProgramData": r"C:\ProgramData", "ALLUSERSPROFILE": r"C:\ProgramData",
               "ComSpec": _SYSROOT + r"\System32\cmd.exe", "PUBLIC": r"C:\Users\Public",
               "NUMBER_OF_PROCESSORS": str(os.cpu_count() or 1)}.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v


# ---------------------------------------------------------------------------------------------------------------------
# job records and the runner


def _write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_job(record_path: str):
    """Runner: start the job's command with its output in the log, wait, store the exit code in the record."""
    path = Path(record_path)
    rec = _read_json(path)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **rec.get("env", {})}
    with open(rec["log"], "wb") as log:
        p = subprocess.Popen(rec["cmd"], cwd=rec["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        rec["child_pid"] = p.pid
        _write_json(path, rec)
        code = p.wait()
    rec = _read_json(path)
    rec.update(exit_code=code, ended=datetime.now().isoformat(timespec="seconds"))
    _write_json(path, rec)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--run-job":
    run_job(sys.argv[2])
    sys.exit(0)

# ---------------------------------------------------------------------------------------------------------------------
# the server (imports kept below the runner so a runner starts fast)

import numpy as np  # noqa: E402
import psutil  # noqa: E402
from fastmcp import FastMCP  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402
from fastmcp.utilities.types import Image  # noqa: E402
from PIL import Image as PILImage, ImageDraw, ImageFont  # noqa: E402

mcp = FastMCP(
    "kimodo",
    version="0.1.0",
    website_url="https://github.com/alivnal",
    instructions=(
        "Kimodo text-to-motion on this machine: write_spec -> generate (background job, model loaded per job, GPU "
        "freed after) -> job (progress) -> preview (background job) -> view / measure each take -> finalize the "
        "picks -> retarget (only when configured). Paths in arguments are relative to the install's outputs folder "
        "unless stated. Prompts work best as 'A person ...': one action, the body not the story, the side named for "
        "one-arm actions. A clip's path key fixes walking speed; segments join several prompts into one take."
    ),
)


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _proc(pid, create_time):
    """The live process with this pid and start time, or None (pids are reused on Windows)."""
    try:
        p = psutil.Process(pid)
        return p if abs(p.create_time() - create_time) < 1 else None
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _job_state(rec):
    if rec.get("state") == "cancelled":
        return "cancelled"
    if "exit_code" in rec:
        return "done" if rec["exit_code"] == 0 else "failed"
    if _proc(rec["pid"], rec["create_time"]):
        return "running"
    return "stopped"  # the runner died without an exit code (machine restart, killed from outside)


def _records():
    if not JOBS_DIR.is_dir():
        return []
    recs = []
    for f in JOBS_DIR.glob("*.json"):
        try:
            recs.append(_read_json(f))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(recs, key=lambda r: r["started"], reverse=True)


def _record(job_id):
    f = JOBS_DIR / f"{job_id}.json"
    return _read_json(f) if f.is_file() else None


def _start_job(kind, name, cmd, cwd, log, env=None, extra=None):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"{kind}-{name}-{datetime.now():%Y%m%d-%H%M%S}"
    path = JOBS_DIR / f"{job_id}.json"
    rec = {"id": job_id, "kind": kind, "name": name, "cmd": [str(c) for c in cmd], "cwd": str(cwd), "log": str(log),
           "env": env or {}, "started": _now(), **(extra or {})}
    Path(log).write_bytes(b"")  # the last run's log is gone before anyone reads this job's progress
    _write_json(path, rec)
    runner = [sys.executable, str(Path(__file__).resolve()), "--run-job", str(path)]
    flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    kw = dict(cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not WINDOWS:  # its own session, so the job survives the MCP host closing
        p = subprocess.Popen(runner, start_new_session=True, **kw)
    else:
        try:  # out of the host's process tree (job object), so the job survives the session
            p = subprocess.Popen(runner, creationflags=flags | CREATE_BREAKAWAY_FROM_JOB, **kw)
        except OSError:
            p = subprocess.Popen(runner, creationflags=flags, **kw)
    rec.update(pid=p.pid, create_time=psutil.Process(p.pid).create_time())
    _write_json(path, rec)
    return rec


# ---------------------------------------------------------------------------------------------------------------------
# GPU and other processes


def _gpu():
    out = ""
    for _ in range(3):  # nvidia-smi now and then prints a lone status line under heavy load; ask again
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15,
                                 stdin=subprocess.DEVNULL).stdout
            line = next(l for l in out.splitlines() if l.count(",") == 3)
            name, used, total, util = [s.strip() for s in line.split(",")]
            used, total = float(used) / 1024, float(total) / 1024
            return {"gpu": name, "used_gb": round(used, 1), "total_gb": round(total, 1),
                    "free_gb": round(total - used, 1), "utilization_pct": int(util) if util.isdigit() else None}
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    return {"error": f"nvidia-smi gave no reading: {out.strip()[:200]!r}"}


def _scan():
    """Kimodo processes on the GPU (batch generation, the web demo), other heavy GPU apps, and running retargets."""
    kimodo, other, retarget = [], [], []
    procs = {}
    for p in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time"]):
        try:
            info = p.info
            cmd = info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        line = " ".join(cmd)
        low = line.lower()
        name = (info["name"] or "").lower()
        if "--run-job" in low:
            continue
        if "kimodo_batch.py" in low and "--reexport" not in low:
            procs[info["pid"]] = ("batch", info, cmd)
        elif "kimodo.demo" in low:
            procs[info["pid"]] = ("demo", info, cmd)
        elif name in ("bash.exe", "bash") and any(c.replace("\\", "/").endswith(RETARGET_NAME) for c in cmd):
            procs[info["pid"]] = ("retarget", info, cmd)
        elif name in ("blender.exe", "husk.exe", "houdini.exe", "houdinifx.exe", "hython.exe", "cascadeur.exe"):
            other.append({"pid": info["pid"], "name": info["name"],
                          "started": datetime.fromtimestamp(info["create_time"]).isoformat(timespec="seconds")})
    parents = {(procs[i][1]["ppid"], procs[i][0]) for i in procs}
    for pid, (kind, info, cmd) in procs.items():
        if (pid, kind) in parents:
            continue  # keep the innermost process: not a shell wrapper or the venv launcher around it
        entry = {"pid": pid, "kind": kind, "command": " ".join(cmd)[-220:],
                 "started": datetime.fromtimestamp(info["create_time"]).isoformat(timespec="seconds")}
        if kind == "batch":
            spec = next((c for c in cmd if c.lower().endswith(".json")), None)
            entry["spec"] = spec
            try:
                entry["batch"] = _read_json(Path(spec) if Path(spec).is_absolute() else Path(info.get("cwd") or REPO) / spec)["name"]
            except Exception:  # noqa: BLE001
                entry["batch"] = Path(spec).stem if spec else None
        if kind == "retarget":
            args = [c for c in cmd if not c.lower().endswith(("bash.exe", "bash"))]
            i = next((k for k, c in enumerate(args) if c.replace("\\", "/").endswith(RETARGET_NAME)), None)
            entry["run"] = args[i + 2] if i is not None and len(args) > i + 2 else None
            retarget.append(entry)
        else:
            kimodo.append(entry)
    return kimodo, other, retarget


def _running(kind, name=None):
    return [r for r in _records() if r["kind"] == kind and (name is None or r["name"] == name) and _job_state(r) == "running"]


def _gpu_guard():
    """Refuses while another Kimodo process uses the GPU; returns a warning when free VRAM is low."""
    kimodo, _, _ = _scan()
    busy = [f"{k['kind']} pid {k['pid']}" + (f" ({k.get('batch')})" if k.get("batch") else "") for k in kimodo]
    if not kimodo:  # a job just started whose batch process is not up yet
        busy += [f"MCP job {r['id']}" for r in _running("generate")]
    if busy:
        raise ToolError("Not started: Kimodo is already on the GPU: " + ", ".join(busy) +
                        ". Wait until it finishes (status / job), then start again. Never stop a Kimodo process this "
                        "server did not start.")
    gpu = _gpu()
    if "free_gb" in gpu and gpu["free_gb"] < VRAM_WARN_GB:
        return (f"Warning: only {gpu['free_gb']} GB of {gpu['total_gb']} GB VRAM free (the text encoder peaks at 5.4 GB); "
                "the encode may spill into shared memory and run slowly. Other GPU users: see status.")
    return None


class _Lock:
    """One start at a time across server processes (each Claude session runs its own server)."""

    def __init__(self, name):
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = JOBS_DIR / f".{name}.lock"

    def __enter__(self):
        for _ in range(100):
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except FileExistsError:
                if time.time() - self.path.stat().st_mtime > 60:
                    self.path.unlink(missing_ok=True)
                time.sleep(0.1)
        raise ToolError("Another session is starting a job right now; try again in a few seconds.")

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------------------------
# paths


def _batch_dir(batch):
    if not batch or "/" in batch.strip("/") or "\\" in batch or ".." in batch:
        raise ToolError(f"'{batch}' is not an output folder name (e.g. demo_showcase); list_outputs shows them.")
    d = OUTPUTS / batch
    if not d.is_dir():
        raise ToolError(f"No output folder outputs/{batch}. list_outputs shows the folders.")
    return d


def _inside(path: Path, base: Path):
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _take_npz(take):
    """'batch/clip/clip_s2', 'picks_folder/clip' (with or without .npz), or an absolute .npz path."""
    p = Path(take)
    if not p.is_absolute():
        p = OUTPUTS / take
    if p.suffix != ".npz":
        p = p.with_name(p.name + ".npz") if p.suffix not in (".fbx", ".mp4") else p.with_suffix(".npz")
    if not p.is_file():
        raise ToolError(f"No take {p}. Give it as '<batch>/<clip>/<clip>_s<k>' or '<picks folder>/<clip>' "
                        "(list_outputs with a batch shows the files).")
    return p


def _log_lines(path, limit_bytes=4_000_000):
    """The log's lines with progress-bar redraws dropped (tqdm writes with carriage returns)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - limit_bytes))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = []
    for raw in text.split("\n"):
        parts = [s for s in raw.split("\r") if s.strip()]
        if parts:
            lines.append(parts[-1].rstrip())
    return [l for l in lines if not re.search(r"\d+it/s\]|it/s, |s/it\]", l)]


# ---------------------------------------------------------------------------------------------------------------------
# spec validation

CLIP_KEYS = {"name", "prompt", "seconds", "segments", "transition", "note", "seed", "samples", "in_place",
             "diffusion_steps", "path", "poses", "hands", "ground", "constraints", "constraint_weight"}
SPEC_KEYS = {"name", "target_fbx", "seed", "samples", "in_place", "diffusion_steps", "clips", "model", "note"}
PARTS = {"left-hand", "right-hand", "left-foot", "right-foot"}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate(spec):
    errors, warnings = [], []
    if not isinstance(spec, dict):
        return ["the spec must be a JSON object"], []
    name = spec.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        errors.append("name: lowercase snake_case, e.g. walk_cycle_v1")
    for k in set(spec) - SPEC_KEYS:
        warnings.append(f"unknown spec key '{k}' (ignored by kimodo_batch.py)")
    if "target_fbx" in spec:
        t = Path(spec["target_fbx"])
        if not (t if t.is_absolute() else ROOT / t).is_file():
            errors.append(f"target_fbx not found: {spec['target_fbx']}")
    for k in ("seed", "samples", "diffusion_steps"):
        if k in spec and not isinstance(spec[k], int):
            errors.append(f"{k}: an integer")
    if isinstance(spec.get("samples"), int) and not 1 <= spec["samples"] <= 8:
        errors.append("samples: 1-8")
    clips = spec.get("clips")
    if not isinstance(clips, list) or not clips:
        errors.append("clips: a non-empty list")
        return errors, warnings
    seen = set()
    for i, c in enumerate(clips):
        where = f"clips[{i}]"
        if not isinstance(c, dict):
            errors.append(f"{where}: an object")
            continue
        cn = c.get("name")
        where = f"clip '{cn}'" if cn else where
        if not isinstance(cn, str) or not NAME_RE.match(cn):
            errors.append(f"{where}: name must be lowercase snake_case")
        elif cn in seen:
            errors.append(f"{where}: duplicate name")
        seen.add(cn)
        for k in set(c) - CLIP_KEYS:
            warnings.append(f"{where}: unknown key '{k}' (ignored)")
        if "segments" in c:
            if "prompt" in c or "seconds" in c:
                errors.append(f"{where}: give either prompt+seconds or segments, not both")
            segs = c["segments"]
            if not isinstance(segs, list) or len(segs) < 2:
                errors.append(f"{where}: segments is a list of at least two {{prompt, seconds}}")
                segs = []
        else:
            segs = [{"prompt": c.get("prompt"), "seconds": c.get("seconds")}]
        total = 0.0
        for s in segs:
            if not isinstance(s, dict) or not isinstance(s.get("prompt"), str) or not s["prompt"].strip():
                errors.append(f"{where}: every prompt must be a non-empty string")
                continue
            if not _num(s.get("seconds")) or s["seconds"] <= 0:
                errors.append(f"{where}: seconds must be a positive number")
                continue
            if s["seconds"] > 10:
                errors.append(f"{where}: {s['seconds']} s for one prompt; Kimodo makes at most 10 s a prompt "
                              "(split into clips, or use segments)")
            elif s["seconds"] < 2 and "segments" not in c:
                warnings.append(f"{where}: {s['seconds']} s is short; 2-10 s is the tested range")
            if not s["prompt"].startswith("A person"):
                warnings.append(f"{where}: prompt does not start with 'A person' (the form Kimodo follows best)")
            total += float(s["seconds"])
        for k in ("seed", "samples", "diffusion_steps", "transition"):
            if k in c and not isinstance(c[k], int):
                errors.append(f"{where}: {k} must be an integer")
        path = c.get("path")
        if path is not None:
            if not isinstance(path, dict):
                errors.append(f"{where}: path is an object")
            elif "points" in path:
                pts = path["points"]
                if not isinstance(pts, list) or len(pts) < 2 or not all(
                        isinstance(p, list) and len(p) == 3 and all(_num(v) for v in p) for p in pts):
                    errors.append(f"{where}: path.points is a list of at least two [second, x, z]")
                elif max(p[0] for p in pts) > total + 1e-6:
                    errors.append(f"{where}: a path point is after the clip's end ({total:g} s)")
            elif _num(path.get("speed")):
                if "stop_at" in path and (not _num(path["stop_at"]) or not 0 < path["stop_at"] <= total):
                    errors.append(f"{where}: path.stop_at must be a second inside the clip (0-{total:g})")
            else:
                errors.append(f"{where}: path needs speed (m/s) or points")
        for k, pose in enumerate(c.get("poses") or []):
            w = f"{where} poses[{k}]"
            if not isinstance(pose, dict) or not _num(pose.get("at")) or not isinstance(pose.get("from"), str):
                errors.append(f"{w}: needs at (second) and from (a take's .npz)")
                continue
            src = Path(pose["from"])
            if not (src if src.is_absolute() else ROOT / src).is_file():
                errors.append(f"{w}: from not found: {pose['from']} (relative to the install folder)")
            if not 0 <= pose["at"] <= total or ("until" in pose and not (_num(pose["until"]) and pose["at"] <= pose["until"] <= total)):
                errors.append(f"{w}: at/until must be inside the clip (0-{total:g} s)")
            if "frame" in pose and not isinstance(pose["frame"], int):
                errors.append(f"{w}: frame must be an integer")
            if "part" in pose and pose["part"] not in PARTS:
                errors.append(f"{w}: part is one of {sorted(PARTS)}")
        for k, h in enumerate(c.get("hands") or []):
            w = f"{where} hands[{k}]"
            if not isinstance(h, dict) or not _num(h.get("at")) or h.get("hand") not in ("left", "right") or not (
                    isinstance(h.get("pos"), list) and len(h["pos"]) == 3 and all(_num(v) for v in h["pos"])):
                errors.append(f"{w}: needs at (second), hand (left/right) and pos [x, y, z]")
            elif not 0 <= h["at"] <= total:
                errors.append(f"{w}: at must be inside the clip (0-{total:g} s)")
        if "ground" in c:
            g = c["ground"]
            if not isinstance(g, dict) or not (_num(g.get("rise")) or isinstance(g.get("profile"), list)):
                errors.append(f"{where}: ground is {{\"rise\": 0.12}} or {{\"profile\": [[metres ahead, height], ...]}}")
        if isinstance(c.get("constraints"), str):
            f = Path(c["constraints"])
            if not (f if f.is_absolute() else ROOT / f).is_file():
                errors.append(f"{where}: constraints file not found: {c['constraints']}")
    return errors, warnings


def _spec_prompts(spec):
    return {c["name"]: " | ".join(s["prompt"] for s in c["segments"]) if "segments" in c else c.get("prompt")
            for c in spec.get("clips", []) if isinstance(c, dict)}


# ---------------------------------------------------------------------------------------------------------------------
# tools


@mcp.tool(annotations={"title": "Kimodo status", "readOnlyHint": True, "openWorldHint": False})
def status(install_check: bool = False) -> dict:
    """GPU memory, Kimodo processes on the GPU (batch generation or the web demo, from any session or the CLI), other
    heavy GPU apps (Blender, Houdini husk, Cascadeur), running SkinTokens retargets and the latest MCP jobs.
    install_check=true also runs scripts/kimodo_check.py (8 install checks, about a minute; exit 0 = ready)."""
    kimodo, other, retarget = _scan()
    out = {"gpu": _gpu(), "kimodo_on_gpu": kimodo, "other_gpu_apps": other, "retargets_running": retarget,
           "jobs": [{"id": r["id"], "state": _job_state(r), "started": r["started"], "ended": r.get("ended")}
                    for r in _records()[:10]],
           "generate_allowed": not kimodo and not _running("generate")}
    if install_check:
        r = subprocess.run([str(PY), str(SCRIPTS / "kimodo_check.py")], cwd=str(ROOT), capture_output=True, text=True, stdin=subprocess.DEVNULL,
                           encoding="utf-8", errors="replace", timeout=600)
        out["install_check"] = {"exit_code": r.returncode, "output": (r.stdout + r.stderr)[-4000:]}
    return out


@mcp.tool(annotations={"title": "Write a Kimodo spec", "readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": True, "openWorldHint": False})
def write_spec(spec: dict[str, Any], overwrite: bool = False) -> dict:
    """Validate a batch spec and write it to specs/<name>.json in the install. Same format as for
    kimodo_batch.py: {name, target_fbx?, seed?, samples?, in_place?, diffusion_steps?, clips: [{name, prompt, seconds,
    note?} or {name, segments: [{prompt, seconds}], transition?}, with optional path, poses, hands, ground,
    constraints, constraint_weight, seed, samples, in_place, diffusion_steps]}. Nothing is written when there are
    errors; warnings are returned. An existing spec with other content, or an output folder of that name made with
    other prompts, needs overwrite=true (a new name such as <name>_v2 is the usual way). Does not start generation:
    use generate."""
    errors, warnings = _validate(spec)
    if errors:
        raise ToolError("Spec not written:\n- " + "\n- ".join(errors) + ("\nWarnings:\n- " + "\n- ".join(warnings) if warnings else ""))
    path = SPECS / f"{spec['name']}.json"
    text = json.dumps(spec, indent=2, ensure_ascii=False) + "\n"
    conflicts = []
    if path.is_file() and _read_json(path) != spec:
        conflicts.append(f"specs/{path.name} exists with other content")
    manifest = OUTPUTS / spec["name"] / "manifest.json"
    if manifest.is_file():
        old = {k: v["prompt"] for k, v in _read_json(manifest).get("clips", {}).items()}
        new = _spec_prompts(spec)
        changed = [k for k in new if k in old and old[k] != new[k]]
        if changed:
            conflicts.append(f"outputs/{spec['name']} has takes of {changed} made from other prompts "
                             "(generate skips clips whose files exist, so they would need force=true)")
    if conflicts and not overwrite:
        raise ToolError("Spec not written: " + "; ".join(conflicts) + ". Use a new name (e.g. "
                        f"{spec['name']}_v2), or overwrite=true.")
    path.write_text(text, encoding="utf-8")
    clips = spec["clips"]
    secs = sum(sum(s["seconds"] for s in c["segments"]) if "segments" in c else c["seconds"] for c in clips)
    samples = sum(c.get("samples", spec.get("samples", 1)) for c in clips)
    return {"written": str(path), "clips": len(clips), "takes": samples, "seconds_total": secs,
            "warnings": warnings + [f"overwrote: {c}" for c in conflicts]}


@mcp.tool(annotations={"title": "Generate Kimodo takes", "readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": False, "openWorldHint": False})
def generate(spec_name: str, only: list[str] | None = None, force: bool = False, samples: int | None = None,
             reexport: bool = False) -> dict:
    """Start kimodo_batch.py on specs/<spec_name>.json as a background job and return its job id at once. The model
    loads inside the job (1-3 min), the prompts are encoded (about 10 s, 5.4 GB VRAM peak), then each clip takes
    10-60 s; the process exits at the end, which frees the GPU. Clips whose files exist are skipped unless force=true
    (only = run just these clip names; samples overrides every clip's count). reexport=true rebuilds the Mixamo FBX
    files from the saved .npz without the model or GPU. Refused while another Kimodo process uses the GPU. Follow it
    with job(job_id); write the spec first with write_spec."""
    spec_path = SPECS / f"{spec_name}.json"
    if not spec_path.is_file():
        raise ToolError(f"No spec specs/{spec_name}.json; write it with write_spec.")
    spec = _read_json(spec_path)
    errors, _ = _validate(spec)
    if errors:
        raise ToolError("The spec has errors:\n- " + "\n- ".join(errors))
    names = [c["name"] for c in spec["clips"]]
    if only and set(only) - set(names):
        raise ToolError(f"only: {sorted(set(only) - set(names))} not in the spec (clips: {names})")
    if samples is not None and not 1 <= samples <= 8:
        raise ToolError("samples: 1-8")
    cmd = [str(PY), "-u", str(SCRIPTS / "kimodo_batch.py"), str(spec_path)]
    if only:
        cmd += ["--only", ",".join(only)]
    if force:
        cmd.append("--force")
    if samples:
        cmd += ["--samples", str(samples)]
    LOGS.mkdir(parents=True, exist_ok=True)
    with _Lock("gpu"):
        warning = None
        if reexport:
            cmd.append("--reexport")
            if _running("generate", spec["name"]) or any(k.get("batch") == spec["name"] for k in _scan()[0]):
                raise ToolError(f"A generation of {spec['name']} is running; reexport after it.")
        else:
            warning = _gpu_guard()
        log = LOGS / (f"{spec['name']}_reexport.log" if reexport else f"{spec['name']}.log")
        rec = _start_job("generate" if not reexport else "reexport", spec["name"], cmd, REPO, log,
                         extra={"spec": str(spec_path), "only": only or [], "force": force})
    return {"job_id": rec["id"], "log": rec["log"], "output": str(OUTPUTS / spec["name"]),
            **({"warning": warning} if warning else {}),
            "next": "job(job_id) for progress; preview(batch) when it is done"}


@mcp.tool(annotations={"title": "Render Kimodo previews", "readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": True, "openWorldHint": False})
def preview(batch: str, force: bool = False) -> dict:
    """Start kimodo_preview.py on outputs/<batch> as a background job: Blender renders each take's .npz on Kimodo's
    SOMA body (3/4 front camera following the hips, 0.5 m grid) into <clip>_s<k>.mp4, a <clip>_sheet.png per clip
    (one row of 6 frames per sample) and README.md. About 25 s per 5 s take. Existing previews are kept unless
    force=true. Refused while that batch is still generating. Follow it with job(job_id), then view."""
    d = _batch_dir(batch)
    if not (d / "manifest.json").is_file():
        raise ToolError(f"outputs/{batch} has no manifest.json (not a generated batch; a picks folder already has mp4s).")
    with _Lock("preview"):
        if _running("generate", batch) or any(k.get("batch") == batch for k in _scan()[0]):
            raise ToolError(f"{batch} is still generating; preview it when the job is done.")
        if _running("preview", batch):
            raise ToolError(f"A preview of {batch} is already running: {_running('preview', batch)[0]['id']}")
        cmd = [str(PY), "-u", str(SCRIPTS / "kimodo_preview.py"), str(d)] + (["--force"] if force else [])
        rec = _start_job("preview", batch, cmd, REPO, LOGS / f"{batch}_preview.log")
    return {"job_id": rec["id"], "log": rec["log"], "next": "job(job_id); then view(batch, clip)"}


def _progress(rec, lines):
    kind = rec["kind"]
    p = {}
    if kind in ("generate", "reexport"):
        done = [l for l in lines if l.startswith(("done ", "re-exported "))]
        skipped = [l for l in lines if l.startswith("skip ")]
        m = next((re.search(r"(\d+) clip\(s\)", l) for l in lines if l.startswith("model ")), None)
        total = int(m.group(1)) if m else len(rec.get("only") or []) or None
        if kind == "generate":
            enc = next((l for l in lines if l.startswith("encoded ")), None)
            p["phase"] = ("finished" if any(l.startswith("outputs: ") for l in lines) else
                          f"generating clip {len(done) + 1} of {total}" if enc else
                          "encoding prompts" if m else "starting, then loading the model (1-3 min)")
            if enc:
                p["encoded"] = enc
        p.update(clips_total=total, clips_done=done, clips_skipped=skipped)
    elif kind == "preview":
        p["previews_done"] = [l[8:] for l in lines if l.startswith("preview ")]
        try:
            man = _read_json(OUTPUTS / rec["name"] / "manifest.json")
            p["previews_total"] = sum(len(c["files"]) for c in man["clips"].values())
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    elif kind == "retarget":
        p["clips_started"] = [l[4:].rsplit(" ", 1)[0] for l in lines if l.startswith("=== ") and re.search(r"\d\d:\d\d:\d\d$", l)
                              and not l.startswith(("=== strip", "=== rig", "=== tube", "=== finished"))]
        p["clip_results"] = [l.strip() for l in lines if l.startswith("  ")]
        p["clips_total"] = rec.get("clips_total")
        report = Path(rec["cwd"]) / "anim" / rec["name"] / "report.md"
        if any(l.startswith("=== finished") for l in lines) and report.is_file():
            p["report"] = str(report)
            p["report_head"] = "\n".join(report.read_text(encoding="utf-8", errors="replace").splitlines()[:40])
    return p


@mcp.tool(annotations={"title": "Kimodo job state", "readOnlyHint": True, "openWorldHint": False})
def job(job_id: str | None = None, tail: int = 12) -> dict:
    """State and progress of a background job (generate, preview, retarget): running / done / failed / cancelled /
    stopped, the clips finished so far, the last log lines (tail), and the error lines when it failed. With no job_id,
    lists the latest jobs of every session. A job_id may also be a log name from a CLI run
    (outputs/_logs/<name>.log, e.g. demo_showcase), read as a generation log."""
    if not job_id:
        return {"jobs": [{"id": r["id"], "kind": r["kind"], "name": r["name"], "state": _job_state(r),
                          "started": r["started"], "ended": r.get("ended")} for r in _records()[:20]]}
    rec = _record(job_id)
    if rec is None:
        log = LOGS / f"{job_id}.log"
        if not log.is_file() or not re.match(r"^[\w.-]+$", job_id):
            raise ToolError(f"No job {job_id} and no log outputs/_logs/{job_id}.log. job() lists the jobs.")
        lines = _log_lines(log)
        kimodo = _scan()[0]
        running = any(k.get("batch") == job_id for k in kimodo)
        rec = {"id": job_id, "kind": "generate", "name": job_id, "log": str(log)}
        state = "running" if running else ("done" if any(l.startswith("outputs: ") for l in lines) else "not running")
        return {"id": job_id, "source": "CLI log", "state": state, **_progress(rec, lines),
                "log_tail": lines[-tail:], "log_modified": datetime.fromtimestamp(log.stat().st_mtime).isoformat(timespec="seconds")}
    state = _job_state(rec)
    lines = _log_lines(rec["log"])
    out = {"id": rec["id"], "kind": rec["kind"], "name": rec["name"], "state": state, "started": rec["started"],
           "ended": rec.get("ended"), "exit_code": rec.get("exit_code"), "log": rec["log"], **_progress(rec, lines),
           "log_tail": lines[-tail:]}
    if state == "failed":
        i = next((k for k, l in enumerate(lines) if l.startswith("Traceback")), None)
        out["error"] = lines[i:][-30:] if i is not None else lines[-15:]
    if state in ("done", "failed", "cancelled", "stopped") and rec["kind"] in ("generate", "reexport", "preview"):
        out["gpu_after"] = _gpu()
    return out


@mcp.tool(annotations={"title": "Cancel a Kimodo job", "readOnlyHint": False, "destructiveHint": True,
                       "idempotentHint": True, "openWorldHint": False})
def cancel(job_id: str) -> dict:
    """Stop a running background job that this MCP server started (its whole process tree: Python, Blender, bash).
    Finished clips stay on disk; generate resumes from them when started again. Processes not started through this
    server (CLI runs, the web demo, other people's work) are never touched."""
    rec = _record(job_id)
    if rec is None:
        raise ToolError(f"No MCP job {job_id}. Only jobs started through this server can be cancelled.")
    p = _proc(rec["pid"], rec["create_time"])
    if not p:
        return {"id": job_id, "state": _job_state(rec), "note": "not running"}
    procs = p.children(recursive=True) + [p]
    for q in procs:
        try:
            q.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=10)
    rec = _record(job_id)
    rec.update(state="cancelled", ended=_now())
    _write_json(JOBS_DIR / f"{job_id}.json", rec)
    return {"id": job_id, "state": "cancelled", "killed": len(procs), "gpu_after": _gpu()}


@mcp.tool(annotations={"title": "List Kimodo outputs", "readOnlyHint": True, "openWorldHint": False})
def list_outputs(batch: str | None = None) -> dict:
    """With no batch: every folder in the install's outputs/ (generated batches with their clip and preview
    counts, picks folders with their rows), newest first. With a batch: its clips from manifest.json (prompt, seconds,
    note, path/poses/hands/ground, pin_cm) and which take files exist (npz, fbx, mp4, sheet); for a picks folder, its
    picks.json rows and files."""
    if batch is None:
        rows = []
        for d in sorted((p for p in OUTPUTS.iterdir() if p.is_dir() and not p.name.startswith("_")),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            row = {"name": d.name, "modified": datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="minutes")}
            if (d / "manifest.json").is_file():
                man = _read_json(d / "manifest.json")
                files = [f for c in man.get("clips", {}).values() for f in c.get("files", [])]
                row.update(kind="batch", clips=len(man.get("clips", {})), takes=len(files),
                           previews=sum((d / f).with_suffix(".mp4").is_file() for f in files))
            elif (d / "picks.json").is_file():
                row.update(kind="picks", clips=len(_read_json(d / "picks.json").get("picks", [])))
            else:
                row.update(kind="other", npz=len(list(d.glob("*.npz"))))
            rows.append(row)
        return {"outputs": str(OUTPUTS), "folders": rows}
    d = _batch_dir(batch)
    if (d / "manifest.json").is_file():
        man = _read_json(d / "manifest.json")
        clips = {}
        for name, c in man.get("clips", {}).items():
            files = [d / f for f in c.get("files", [])]
            clips[name] = {k: c[k] for k in ("prompt", "seconds", "frames", "note", "seed", "path", "poses", "hands",
                                               "ground", "pin_cm", "ground_fit") if k in c}
            clips[name]["takes"] = [{"take": f"{batch}/{f.relative_to(d).with_suffix('').as_posix()}",
                                     "npz": f.with_suffix(".npz").is_file(), "fbx": f.is_file(),
                                     "mp4": f.with_suffix(".mp4").is_file()} for f in files]
            sheet = d / name / f"{name}_sheet.png"
            clips[name]["sheet"] = str(sheet) if sheet.is_file() else None
        return {"batch": batch, "model": man.get("model"), "spec": man.get("spec"), "clips": clips}
    out = {"folder": batch, "files": sorted(p.name for p in d.iterdir() if p.is_file())}
    if (d / "picks.json").is_file():
        out["picks"] = _read_json(d / "picks.json")
    return out


def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


@mcp.tool(annotations={"title": "View a Kimodo take", "readOnlyHint": True, "openWorldHint": False})
def view(batch: str, clip: str, sample: int | None = None, fps: float = 3.0) -> list:
    """Return a picture to review. With no sample: the clip's contact sheet (<clip>_sheet.png, one row of 6 frames per
    sample). With a sample (1-based, the s<k> of the take): a grid of frames taken from its mp4 at `fps` frames a
    second (default 3), each labelled with its frame number at 30 fps. In a picks folder, give the clip name and
    sample=1 (the folder's <clip>.mp4). Run preview first: both need the mp4/sheet. In the front view the
    character's right is on the image's left."""
    d = _batch_dir(batch)
    if sample is None:
        sheet = d / clip / f"{clip}_sheet.png"
        if not sheet.is_file():
            raise ToolError(f"No sheet {sheet}; run preview('{batch}') first, or pass a sample for a picks folder.")
        img = PILImage.open(sheet).convert("RGB")
        if img.width > 1600:
            img = img.resize((1600, round(img.height * 1600 / img.width)))
        out = Path(tempfile.gettempdir()) / "kimodo_mcp" / f"{batch}_{clip}_sheet.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, quality=88)
        return [f"{batch}/{clip} sheet ({sheet})", Image(path=out)]
    mp4 = d / clip / f"{clip}_s{sample}.mp4"
    if not mp4.is_file() and (d / f"{clip}.mp4").is_file():
        mp4 = d / f"{clip}.mp4"
    if not mp4.is_file():
        raise ToolError(f"No mp4 {mp4}; " + (f"run preview('{batch}') first." if (d / "manifest.json").is_file() else
                                             "this picks folder has no mp4 of that clip: view its source take."))
    if not 0.5 <= fps <= 10:
        raise ToolError("fps: 0.5-10")
    tmp = Path(tempfile.mkdtemp(prefix="kimodo_view_"))
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    r = subprocess.run([ffmpeg, "-loglevel", "error", "-i", str(mp4), "-vf", f"fps={fps},scale=320:-1",
                        str(tmp / "f_%03d.png")], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    frames = sorted(tmp.glob("f_*.png"))
    if r.returncode or not frames:
        raise ToolError(f"ffmpeg failed on {mp4}: {r.stderr[-500:]}")
    frames = frames[:40]
    cols = 5 if len(frames) > 16 else 4
    tw, th = PILImage.open(frames[0]).size
    rows = math.ceil(len(frames) / cols)
    grid = PILImage.new("RGB", (cols * tw, rows * th), "white")
    draw, font = ImageDraw.Draw(grid), _font(16)
    for i, f in enumerate(frames):
        x, y = (i % cols) * tw, (i // cols) * th
        grid.paste(PILImage.open(f).convert("RGB"), (x, y))
        draw.text((x + 6, y + 4), f"f{round(i * 30 / fps)}", fill="black", font=font)
    out = tmp / "grid.jpg"
    grid.save(out, quality=88)
    for f in tmp.glob("f_*.png"):
        f.unlink()
    return [f"{mp4.relative_to(OUTPUTS).as_posix()}: {len(frames)} frames at {fps:g} fps (labels: frame at 30 fps)",
            Image(path=out)]


@mcp.tool(annotations={"title": "Measure a Kimodo take", "readOnlyHint": True, "openWorldHint": False})
def measure(take: str) -> dict:
    """Measure a take's .npz (posed_joints [T,77,3], metres at human scale, Y up, +Z ahead and +X to the character's
    left at frame 0). take = '<batch>/<clip>/<clip>_s<k>' or '<picks folder>/<clip>' (or an absolute .npz path).
    Returns: hips travel and speed (whole clip, first and last second), body turn, trunk lean forward/sideways (Hips to
    Neck1 from vertical, 0 = the rest pose), per hand the rise above its start, highest frame and frames above the head
    (which hand rises), per foot lift, highest height it stands still at (step up), slide while planted, floor
    penetration, airborne spans (both feet off the floor and moving) with
    landing height, hips rise (jump) and drop (crouch). For a creature, scale distances by its hip-height ratio."""
    path = _take_npz(take)
    d = np.load(path)
    P = d["posed_joints"].astype(float)
    T = P.shape[0]
    fps = float(d["fps"]) if "fps" in d.files else 30.0
    if "joint_names" not in d.files:
        raise ToolError(f"{path.name} has no joint_names; run venv311\\Scripts\\python.exe scripts\\npz_add_skeleton.py on it.")
    J = {str(n): i for i, n in enumerate(d["joint_names"])}
    j = lambda n: P[:, J[n]]  # noqa: E731
    r2 = lambda x, k=2: round(float(x), k)  # noqa: E731
    up = np.array([0.0, 1.0, 0.0])

    hips = j("Hips")
    xz = hips[:, [0, 2]]
    step = np.linalg.norm(np.diff(xz, axis=0), axis=1)
    sec = max(1, int(round(fps)))
    left = j("LeftArm") - j("RightArm")
    left[:, 1] = 0
    left /= np.linalg.norm(left, axis=1, keepdims=True) + 1e-9
    fwd = np.cross(left, up)
    heading = np.degrees(np.arctan2(fwd[:, 0], fwd[:, 2]))
    turn = (heading - heading[0] + 180) % 360 - 180
    spine = j("Neck1") - hips
    rest = np.asarray(d["neutral_joints"], float)
    rest_spine = rest[J["Neck1"]] - rest[J["Hips"]]  # rest pose: +X left, +Z ahead; its lean is the zero
    lean_f = np.degrees(np.arctan2((spine * fwd).sum(1), spine[:, 1])) - np.degrees(np.arctan2(rest_spine[2], rest_spine[1]))
    lean_s = np.degrees(np.arctan2((spine * left).sum(1), spine[:, 1])) - np.degrees(np.arctan2(rest_spine[0], rest_spine[1]))

    out = {"take": str(path), "prompt": str(d["prompt"]) if "prompt" in d.files else None, "frames": T, "fps": fps,
           "seconds": r2(T / fps),
           "hips": {"travel_m": r2(step.sum()), "net_m": r2(np.linalg.norm(xz[-1] - xz[0])),
                    "end_xz": [r2(v) for v in xz[-1] - xz[0]],
                    "speed_mean_ms": r2(step.sum() / max(T - 1, 1) * fps),
                    "speed_first_s_ms": r2(np.linalg.norm(xz[min(sec, T - 1)] - xz[0]) * fps / min(sec, T - 1)),
                    "speed_last_s_ms": r2(np.linalg.norm(xz[-1] - xz[max(0, T - 1 - sec)]) * fps / min(sec, T - 1)),
                    "height_start_m": r2(hips[0, 1]), "rise_max_m": r2(hips[:, 1].max() - hips[0, 1]),
                    "rise_frame": int(hips[:, 1].argmax()), "drop_max_m": r2(hips[0, 1] - hips[:, 1].min()),
                    "drop_frame": int(hips[:, 1].argmin())},
           "body_turn_deg": {"end": r2(turn[-1], 0), "max_left": r2(turn.max(), 0), "max_right": r2(-turn.min(), 0)},
           "trunk_lean_deg": {"forward_mean": r2(lean_f.mean(), 1), "forward_max": r2(lean_f.max(), 1),
                              "forward_max_frame": int(lean_f.argmax()), "back_max": r2(-lean_f.min(), 1),
                              "side_max_left": r2(lean_s.max(), 1), "side_max_right": r2(-lean_s.min(), 1)}}

    head_y = j("Head")[:, 1]
    hands = {}
    for side in ("Left", "Right"):
        w = j(f"{side}Hand")
        rise = w[:, 1] - w[0, 1]
        hands[side.lower()] = {"rise_max_m": r2(rise.max()), "max_frame": int(w[:, 1].argmax()),
                               "max_height_m": r2(w[:, 1].max()), "frames_above_head": int((w[:, 1] > head_y).sum()),
                               "reach_ahead_max_m": r2(((w - hips) * fwd).sum(1).max())}
    rl, rr = hands["left"]["rise_max_m"], hands["right"]["rise_max_m"]
    hands["rises"] = ("none" if max(rl, rr) < 0.15 else "both" if min(rl, rr) >= 0.6 * max(rl, rr)
                      else "left" if rl > rr else "right")
    out["hands"] = hands

    contacts = d["foot_contacts"].astype(bool) if "foot_contacts" in d.files else None
    feet = {}
    floor = min(P[:, J[n], 1].min() for n in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"))
    still = []  # per foot: standing still (ankle and toe under 0.2 m/s), also on a platform Kimodo flags no contact on
    for k, side in enumerate(("Left", "Right")):
        ank, toe = j(f"{side}Foot"), j(f"{side}ToeBase")
        low = np.minimum(ank[:, 1], toe[:, 1])
        speed = np.r_[0.0, np.maximum(np.linalg.norm(np.diff(ank, axis=0), axis=1),
                                      np.linalg.norm(np.diff(toe, axis=0), axis=1)) * fps]
        speed[0] = speed[1] if T > 1 else 0.0
        st = speed < 0.2
        still.append(st)
        f = {"lift_max_m": r2(low.max() - low.min()), "lift_frame": int(low.argmax()),
             "standing_height_max_m": r2(low[st].max()) if st.any() else None,
             "standing_height_max_frame": int(np.where(st)[0][low[st].argmax()]) if st.any() else None}
        if contacts is not None:
            planted = contacts[:, 3 * k: 3 * k + 3].any(1)
            v = np.r_[0.0, np.linalg.norm(np.diff(toe[:, [0, 2]], axis=0), axis=1)]
            both = planted & np.r_[False, planted[:-1]]
            f.update(planted_pct=round(100 * planted.mean()), slide_cm=r2(100 * v[both].sum(), 1),
                     slide_mean_cms=r2(100 * v[both].mean() * fps, 1) if both.any() else 0.0)
        feet[side.lower()] = f
    out["feet"] = {**feet, "floor_min_m": r2(floor, 3)}
    lowest = np.min([P[:, J[n], 1] for n in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")], axis=0)
    air = (lowest > floor + 0.04) & ~still[0] & ~still[1]  # both feet off the floor and moving
    if contacts is not None:
        air &= ~contacts.any(1)
    spans, start = [], None
    for t in range(T + 1):
        if t < T and air[t]:
            start = t if start is None else start
        elif start is not None:
            if spans and start - spans[-1][1] <= 3:  # the apex of a jump: feet still for a frame or two
                spans[-1] = (spans[-1][0], t - 1)
            else:
                spans.append((start, t - 1))
            start = None
    spans = [s for s in spans if s[1] - s[0] >= 2]
    out["airborne"] = [{"frames": [a, b], "seconds": r2((b - a + 1) / fps),
                        "hips_rise_m": r2(hips[a:b + 1, 1].max() - hips[0, 1]),
                        "landing_frame": min(b + 1, T - 1), "landing_height_m": r2(lowest[min(b + 1, T - 1)])}
                       for a, b in spans]
    return out


@mcp.tool(annotations={"title": "Finalize Kimodo picks", "readOnlyHint": False, "destructiveHint": False,
                       "idempotentHint": True, "openWorldHint": False})
def finalize(name: str, picks: list[dict[str, str]], note: str = "", overwrite: bool = False) -> dict:
    """Collect the chosen takes into outputs/<name>/: writes picks.json and runs kimodo_finalize.py, which copies
    <clip>.npz/.mp4 (and .fbx when the take has one) per pick and writes README.md (prompt, source, why, what is left). Each pick:
    {name, from, why?, left?, row?}; from = '<batch>/<clip>/<clip>_s<k>' (relative to outputs, no extension; the take
    needs its mp4, so run preview first). A folder whose picks.json lists other picks needs overwrite=true."""
    if not NAME_RE.match(name):
        raise ToolError("name: lowercase snake_case, e.g. walk_cycle_picks")
    errors = []
    for i, p in enumerate(picks):
        if not NAME_RE.match(p.get("name", "")):
            errors.append(f"picks[{i}]: name must be lowercase snake_case")
        src = p.get("from", "")
        if not re.match(r"^[\w.-]+/[\w.-]+/[\w.-]+$", src):
            errors.append(f"picks[{i}]: from must be '<batch>/<clip>/<clip>_s<k>'")
            continue
        missing = [ext for ext in (".mp4", ".npz") if not (OUTPUTS / (src + ext)).is_file()]
        if missing:
            errors.append(f"picks[{i}]: {src} has no {missing}" + (" (run preview)" if ".mp4" in missing else ""))
    if len({p.get("name") for p in picks}) != len(picks):
        errors.append("pick names must be unique")
    if errors:
        raise ToolError("Not finalized:\n- " + "\n- ".join(errors))
    d = OUTPUTS / name
    data = {"name": name, "note": note, "picks": [{k: p[k] for k in ("row", "name", "from", "why", "left") if k in p}
                                                  for p in picks]}
    if (d / "picks.json").is_file() and _read_json(d / "picks.json") != data and not overwrite:
        raise ToolError(f"outputs/{name}/picks.json exists with other picks; use another name or overwrite=true.")
    if (d / "manifest.json").is_file():
        raise ToolError(f"outputs/{name} is a generated batch, not a picks folder; choose another name.")
    d.mkdir(parents=True, exist_ok=True)
    _write_json(d / "picks.json", data)
    r = subprocess.run([str(PY), str(SCRIPTS / "kimodo_finalize.py"), str(d / "picks.json")], cwd=str(REPO),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300, stdin=subprocess.DEVNULL)
    if r.returncode:
        raise ToolError(f"kimodo_finalize.py failed:\n{(r.stdout + r.stderr)[-2000:]}")
    return {"folder": str(d), "readme": str(d / "README.md"), "output": r.stdout.strip(),
            "files": sorted(p.name for p in d.iterdir())}


def retarget(clips_dir: str, run: str, mesh: str, rig_from: str | None = None, clips: list[str] | None = None,
             jobs: int = 3, fast: bool = False, baseline: str | None = None, tube_arms: str | None = None) -> dict:
    """Start the configured retarget script (KIMODO_RETARGET_SCRIPT, run as it is) as a background job: every take
    <clip>.npz in clips_dir (a picks folder under outputs, or an absolute path) is put on a rigged character and checked;
    the result is anim/<run>/report.md and anim/<run>/<clip>/ in the retarget folder. mesh, rig_from and baseline are
    relative to that folder (rig_from: a folder holding a finished rig_ik.blend, reused instead of rigging the mesh
    again; without it the mesh may be rigged from scratch, which uses the GPU). fast=true: numbers and anim.glb only.
    jobs: clips at once (1-4). An existing run resumes and skips finished clips. Follow it with job(job_id)."""
    if not RUN_RE.match(run):
        raise ToolError("run: letters, digits, _ . - only")
    cd = Path(clips_dir)
    cd = cd if cd.is_absolute() else OUTPUTS / clips_dir
    if not cd.is_dir():
        raise ToolError(f"No clips folder {cd}")
    takes = sorted({p.stem for p in cd.iterdir() if p.suffix in (".npz", ".fbx")})
    if clips:
        missing = [c for c in clips if c not in takes]
        if missing:
            raise ToolError(f"{missing} not in {cd} (takes there: {takes})")
    elif not takes:
        raise ToolError(f"No <clip>.npz in {cd}; finalize the picks into it first.")
    if not (RETARGET_CWD / mesh).is_file():
        raise ToolError(f"No mesh {RETARGET_CWD / mesh}")
    env = {"CLIPS_DIR": cd.as_posix(), "JOBS": str(max(1, min(jobs, 4)))}
    if rig_from:
        if not (RETARGET_CWD / rig_from / "rig_ik.blend").is_file():
            raise ToolError(f"No rig_ik.blend in {RETARGET_CWD / rig_from}")
        env["RIG_FROM"] = rig_from
    if fast:
        env["FAST"] = "1"
    if baseline:
        if not (RETARGET_CWD / baseline).is_file():
            raise ToolError(f"No baseline {RETARGET_CWD / baseline}")
        env["BASELINE"] = baseline
    if tube_arms:
        env["TUBE_ARMS"] = tube_arms
    if WINDOWS and not Path(BASH).is_file():
        raise ToolError(f"Git Bash not found at {BASH} (set GIT_BASH)")
    with _Lock("retarget"):
        if any(r.get("run") == run for r in _scan()[2]) or _running("retarget", run):
            raise ToolError(f"A retarget run '{run}' is already running.")
        warning = None if rig_from else _gpu_guard()  # without a rig to reuse, rigging may run on the GPU
        cmd = [BASH, str(Path(RETARGET_SCRIPT).resolve()), mesh, run] + list(clips or [])
        rec = _start_job("retarget", run, cmd, RETARGET_CWD, LOGS / f"{run}_retarget.log", env=env,
                         extra={"clips_total": len(clips or takes)})
    return {"job_id": rec["id"], "log": rec["log"], "report": str(RETARGET_CWD / "anim" / run / "report.md"),
            "clips": clips or takes, "resumes": (RETARGET_CWD / "anim" / run).is_dir(),
            **({"warning": warning} if warning else {}), "next": "job(job_id); report_head appears when it finishes"}


if RETARGET_SCRIPT:  # the tool exists only when a retarget script is configured
    mcp.tool(retarget, annotations={"title": "Retarget takes onto a rigged character", "readOnlyHint": False,
                                    "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})


if __name__ == "__main__":
    mcp.run(show_banner=False)
