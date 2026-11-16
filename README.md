<div align="center">

# kimodo-mcp

**Text-to-motion as tools for Claude.** An MCP server for [NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo) that
writes motion specs, generates takes as background jobs, renders previews, looks at them, measures them and collects
the picks, without ever holding the model on your GPU between jobs.

[![License: MIT](https://img.shields.io/badge/license-MIT-2a78d6.svg)](LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11-2a78d6.svg)
![MCP](https://img.shields.io/badge/MCP-stdio-eb6834.svg)
![Platform](https://img.shields.io/badge/tested%20on-Windows%2011%20·%20RTX%203070%208%20GB-52514e.svg)

<img src="docs/demo.gif" alt="Four Kimodo takes generated through the MCP server: a steady walk, a right-hand wave, jumping in place, looking around" width="800">

<sub>Four takes from <a href="specs/examples/demo_showcase.json"><code>specs/examples/demo_showcase.json</code></a>, generated, previewed and picked through the tools below. Rendered on Kimodo's own SOMA body.</sub>

</div>

---

## What it does

You talk to Claude; Claude drives Kimodo.

> *"Make me a 7-second take: a person walks at 1.4 m/s, slows down and points ahead with the right arm. Two samples, then show me the frames and tell me which one stops cleaner."*

Claude writes the spec, starts the job, follows its progress, renders the previews, looks at frame grids, reads the
numbers from the motion file and hands you the better take, with frame numbers for what is wrong in the other one.

| | |
|---|---|
| **GPU-friendly by design** | The server never imports torch. Each job loads Kimodo in its own process; when the job ends, the process exits and the GPU is free again for Blender, renders or anything else. |
| **Safe on a shared card** | A generation is refused while another Kimodo process (a CLI batch, the web demo, another session) is on the GPU. Low free VRAM gives a warning. |
| **Jobs outlive the chat** | Jobs are detached. Close the session and the batch keeps going; any session can read any job's state and log. |
| **Claude can see the motion** | `view` returns contact sheets and frame grids with frame numbers as images; `measure` turns the take's joints into numbers (speed, lean, which hand rises, foot slide, airborne spans). |
| **Same files as the CLI** | The tools run the same scripts you can run by hand. Specs, outputs, logs and manifests are identical either way. |

## Tools

| tool | what it does | runs as |
|---|---|---|
| `status` | GPU memory, Kimodo processes on the GPU, other GPU apps (Blender, Houdini, Cascadeur), running jobs; optional install check | instant |
| `write_spec` | validates a spec (keys, seconds, files, prompt form) and writes `specs/<name>.json`; refuses to reuse a name with other prompts | instant |
| `generate` | runs `scripts/kimodo_batch.py` on a spec: one model load, all prompts encoded once, every clip and sample; resumes, `only`, `force`, `samples`, `reexport` | **background job** |
| `job` | state and progress of any job (phase, clips done, error lines, GPU after); also reads a CLI run's log | instant |
| `cancel` | stops a job this server started, with its whole process tree; nothing else is ever touched | instant |
| `preview` | runs `scripts/kimodo_preview.py`: Blender renders each take on the SOMA body to mp4, a contact sheet per clip, README | **background job** |
| `list_outputs` | output folders, or one batch's clips with prompts, notes, constraint residuals and which files exist | instant |
| `view` | a clip's contact sheet, or a frame grid of one take with frame numbers, **as an image** | instant |
| `measure` | numbers from a take's `.npz`: travel, speed per second, body turn, trunk lean, hands, feet, step-up height, slide, airborne spans | instant |
| `finalize` | writes `picks.json` and collects the chosen takes (`.npz`, `.mp4`, `.fbx`) into one folder with a README | instant |
| `retarget` | *(only when configured)* runs your retarget script on a picks folder as a job and reads its report | **background job** |

## How it works

```mermaid
flowchart LR
    subgraph host["Claude Code / Claude Desktop"]
        A["Claude session"]
    end
    subgraph server["kimodo MCP server · no torch"]
        T["11 tools"]
        G{"GPU guard<br/>Kimodo already<br/>on the GPU?"}
    end
    subgraph jobs["Detached job · own process"]
        R["runner<br/>(stores exit code)"]
        B["kimodo_batch.py<br/>load → encode → clips"]
        P["kimodo_preview.py<br/>Blender + ffmpeg"]
    end
    GPU[("NVIDIA GPU")]
    FS[("outputs/<br/>takes · mp4 · sheets<br/>_logs/ · _mcp_jobs/")]

    A <-->|stdio| T
    T --> G
    G -->|"free"| R
    G -.->|"busy: refused"| A
    R --> B & P
    B -->|"model loaded per job"| GPU
    B --> FS
    P --> FS
    T -->|"job · view · measure · list"| FS
```

A job is a record in `outputs/_logs/_mcp_jobs/<id>.json` plus a log in `outputs/_logs/`. The runner that starts the
script waits for it and writes the exit code, so the state is right even when the session that started the job is gone.

```mermaid
sequenceDiagram
    autonumber
    participant C as Claude
    participant S as MCP server
    participant J as Job process
    participant G as GPU
    C->>S: generate("demo_showcase")
    S->>S: scan processes: another Kimodo on the GPU?
    S->>J: start detached runner + kimodo_batch.py
    S-->>C: job_id (returns at once)
    J->>G: load Kimodo
    J->>G: encode every prompt once (5.4 GB peak)
    J->>G: move the text encoder off the GPU
    loop each clip
        J->>G: diffusion, all samples (1.1–1.2 GB)
    end
    J-->>G: process exits: 0 GB held
    C->>S: job(job_id)
    S-->>C: done · 5 clips · GPU after: free
    C->>S: preview → view → measure → finalize
```

### What the GPU sees

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/vram_dark.svg">
  <img src="docs/vram_light.svg" alt="VRAM during one generate job: Kimodo peaks at 5.4 GB while encoding prompts, holds 1.1-1.2 GB per clip, and drops to 0 when the process exits; the whole card also carried a Houdini render" width="880">
</picture>

Measured on the demo batch (5 clips × 2 takes) while a Houdini render shared the card. The blue line is Kimodo's own
torch peak per phase, as the batch logs it; the orange line is the whole card from `nvidia-smi`. With the card full,
the 5.4 GB encode spilled into shared memory and took 81 s (about 15 s on a free card); after it, Kimodo stays near
1.2 GB and lets the render have the rest. When the job ends, nothing of Kimodo is left on the GPU.

## The workflow

```mermaid
flowchart LR
    W["write_spec"] --> Gn["generate"] --> Jb["job"] --> Pv["preview"]
    Pv --> V["view<br/>sheet · frame grid"] & M["measure<br/>numbers from .npz"]
    V & M --> D{"follows<br/>the prompt?"}
    D -->|"no: reword, _v2"| W
    D -->|"yes"| F["finalize<br/>picks folder"]
    F -.->|"optional"| Rt["retarget<br/>onto your rig"]
```

### A spec

```json
{
  "name": "demo_showcase",
  "seed": 2026,
  "samples": 2,
  "clips": [
    {"name": "walk_steady", "prompt": "A person walks forward at a steady, relaxed pace.", "seconds": 5,
     "path": {"speed": 1.4}},
    {"name": "wave_right", "prompt": "A person stands still and waves their right hand above their head.", "seconds": 4},
    {"name": "walk_stop_point", "segments": [
        {"prompt": "A person walks forward at a steady, relaxed pace.", "seconds": 3},
        {"prompt": "A person walking forward slows down and comes to a stop.", "seconds": 1.5},
        {"prompt": "A person standing still raises their right arm and points forward.", "seconds": 2.5}],
     "path": {"speed": 1.4, "stop_at": 4.3}}
  ]
}
```

| key (per clip) | what it does |
|---|---|
| `prompt`, `seconds` | one prompt, up to 10 s |
| `segments` | several prompts as **one continuous take**, joined by Kimodo's own transition overlap (`transition`, frames) |
| `path` | where the root travels, so speed is a number and not a prompt word: `{"speed": 1.4}`, `{"speed": 1.4, "stop_at": 4.3}` or `{"points": [[sec, x, z], ...]}` |
| `poses` | a pose taken from another take at a time, optionally mirrored or for one limb only |
| `hands` | a wrist target in space at a time |
| `ground` | fits the finished take to sloping land: `{"rise": 0.12}` or a height profile |
| `constraints` | a Kimodo constraints file as saved by the demo, passed through |
| `seed`, `samples`, `in_place`, `diffusion_steps` | per clip overrides |

Same spec + same seed gives the same motion. Full key reference: the docstrings of
[`scripts/kimodo_batch.py`](scripts/kimodo_batch.py) and [`scripts/kimodo_author.py`](scripts/kimodo_author.py).

### What `view` shows Claude

<img src="docs/sheet_walk_stop_point.jpg" alt="Contact sheet of the walk_stop_point clip: two samples, six frames each" width="880">

<sub>A contact sheet: one row of six frames per sample. `view(batch, clip, sample=k)` gives a denser grid of one take with frame numbers.</sub>

### What `measure` reads

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/measure_dark.svg">
  <img src="docs/measure_light.svg" alt="Hips speed follows the 1.4 m/s path and drops to 0 at the stop; the right wrist rises to 1.53 m when the person points" width="880">
</picture>

The `walk_stop_point` take above, read from its `.npz`: the hips hold the 1.4 m/s path, come to rest at the stop
(0.00 m/s in the last second), and the right wrist rises to 1.53 m for the point. `measure` returns these as JSON,
for example:

```json
"hips":  {"travel_m": 5.81, "speed_first_s_ms": 1.42, "speed_last_s_ms": 0.0, ...},
"hands": {"right": {"rise_max_m": 0.69, "max_frame": 202, "max_height_m": 1.53, "frames_above_head": 0, ...},
          "rises": "right"},
"feet":  {"left": {"lift_max_m": 0.12, "standing_height_max_m": 0.03, "planted_pct": 68, "slide_cm": 10.7, ...}, ...},
"airborne": []
```

Distances are metres at human scale (Y up, +Z ahead, +X to the character's left). In a front view the character's
right is on the image's left; `measure` settles sides from the data.

## Install

**You need:** Windows 10/11, an NVIDIA GPU with 8 GB or more and its driver, Python 3.11 (`py -3.11`), Git,
[Blender](https://www.blender.org/) 4.x/5.x for previews, `ffmpeg` on `PATH`, and [uv](https://docs.astral.sh/uv/).

```bat
git clone https://github.com/alivnal/kimodo-mcp.git
cd kimodo-mcp
install_kimodo.bat
```

`install_kimodo.bat` builds everything next to itself and is safe to run again: a Python 3.11 venv with pinned
packages (torch cu128, the FBX SDK from PyPI), the NF4-quantised text encoder, Kimodo from
[alivnal/kimodo](https://github.com/alivnal/kimodo) (branch `local`: NVIDIA's v1.2 sync, a Mixamo FBX export and an
NF4 path fix, on top of [Aero-Ex/kimodo](https://github.com/Aero-Ex/kimodo)), the viser web client, then an
8-point install check. Kimodo's own model weights download from Hugging Face on first use.

Then the server's own small venv:

```bat
cd mcp
uv sync --python 3.11
```

For the optional FBX export onto a Mixamo character, download **Y Bot** from [mixamo.com](https://www.mixamo.com)
into `assets/characters/Y Bot.fbx` (see [`assets/README.md`](assets/README.md)), or set `target_fbx` in a spec to any
Mixamo-rigged FBX. Without it, takes are still written as `.npz` and previewed.

### Register it

**Claude Code** (all projects):

```bat
claude mcp add -s user kimodo -- "C:/path/to/kimodo-mcp/mcp/.venv/Scripts/python.exe" "C:/path/to/kimodo-mcp/mcp/kimodo_mcp.py"
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "kimodo": {
      "command": "C:/path/to/kimodo-mcp/mcp/.venv/Scripts/python.exe",
      "args": ["C:/path/to/kimodo-mcp/mcp/kimodo_mcp.py"]
    }
  }
}
```

### Configuration

All optional, as environment variables of the server:

| variable | default | |
|---|---|---|
| `KIMODO_ROOT` | the repo folder | the install with `scripts/`, `specs/`, `outputs/` |
| `KIMODO_PYTHON` | `venv311` in the install | the Python that runs Kimodo |
| `BLENDER` | `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe` | Blender for previews |
| `KIMODO_RETARGET_SCRIPT` | unset | a bash script that puts takes on your rigged character; enables the `retarget` tool |
| `KIMODO_RETARGET_CWD` | the script folder's parent | where the retarget script runs and writes `anim/<run>/report.md` |
| `GIT_BASH` | `C:\Program Files\Git\bin\bash.exe` | bash for the retarget script on Windows |

The retarget script is called as `<script> <mesh> <run> [clip ...]` with `CLIPS_DIR`, `RIG_FROM`, `JOBS`, `FAST`,
`BASELINE` and `TUBE_ARMS` in its environment, and is expected to write `anim/<run>/report.md`. That is the interface
of a SkinTokens retarget pipeline this server was built with; any script with the same shape works.

## Without Claude

Every tool is a script you can run yourself, from the `kimodo` folder with the install's Python:

```bat
..\venv311\Scripts\python.exe -u ..\scripts\kimodo_batch.py ..\specs\examples\demo_showcase.json
..\venv311\Scripts\python.exe ..\scripts\kimodo_preview.py ..\outputs\demo_showcase
..\venv311\Scripts\python.exe ..\scripts\kimodo_finalize.py ..\outputs\my_picks\picks.json
..\venv311\Scripts\python.exe ..\scripts\kimodo_check.py
```

`start_kimodo.bat` starts Kimodo's own web demo.

## Numbers

Measured on an RTX 3070 (8 GB), Windows 11, 16 Nov 2026:

| | |
|---|---|
| model load | 1–3 min per job |
| prompt encoding | 11–15 s on a free card, 5.4 GB peak, once per job (81 s with a render sharing the card) |
| one clip, 2 samples | 18–41 s at 1.1–1.2 GB |
| preview | about 25 s per 5 s take (Blender Workbench) |
| path speed | asked 1.2 / 1.6 / 2.0 m/s, got 1.20 / 1.60 / 2.00 m/s |
| hand targets | 0.0 cm from target after Kimodo's post-processing |

## Writing prompts Kimodo follows

- One person, present tense, starting *"A person …"*.
- Describe the body, not the story: *"raises their right arm and points forward"*, not *"points at the summit"*.
- Name the side for one-arm actions; say *"standing still"* when the character must not travel.
- One action per prompt; a sequence goes in `segments`.
- Repetition words work: *"jumps up and down several times"*.
- Speed is a `path`, not an adjective.
- There are no props: anything stepped onto is mid-air.

## Status

Tested end to end on Windows 11 with an RTX 3070; every tool was run for real. The server's process handling has
Linux code paths (no `creationflags`, a new session per job, `venv311/bin/python`), but the installer is Windows-only
and Linux is untested.

## Credits

- [NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo) (Apache-2.0; model weights under the
  [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)),
  with the SOMA body used in the previews.
- [Aero-Ex/kimodo](https://github.com/Aero-Ex/kimodo) and the NF4 text encoder
  [Aero-Ex/KIMODO-Meta3_llm2vec_NF4](https://huggingface.co/Aero-Ex/KIMODO-Meta3_llm2vec_NF4) (its own licence, from
  its base model).
- The Mixamo FBX export in the fork is adapted from
  [stevelittlefish/ComfyUI-Kimodo-Enhanced](https://github.com/stevelittlefish/ComfyUI-Kimodo-Enhanced) (Apache-2.0).
- [FastMCP](https://gofastmcp.com) for the server.

Model weights, the text encoder and Mixamo characters are not part of this repository; they download from their
sources under their own terms.

## License

MIT © 2026 [Ali Unal](https://github.com/alivnal)
