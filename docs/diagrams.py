"""Draws the README's three diagrams as static SVGs, light and dark (GitHub's Mermaid renderer does not load for
every reader). Run from the repo root: python docs/diagrams.py"""
from pathlib import Path

OUT = Path(__file__).resolve().parent
THEMES = {
    "light": dict(surface="#fcfcfb", card="#f3f2ef", text="#0b0b0b", text2="#52514e", muted="#8a8984",
                  line="#c9c8c3", s1="#2a78d6", s1bg="#e6f0fc", s2="#eb6834", s2bg="#fdeee7"),
    "dark": dict(surface="#1a1a19", card="#242423", text="#f0efec", text2="#c3c2b7", muted="#8f8e88",
                 line="#4a4a46", s1="#3987e5", s1bg="#1c2a3d", s2="#d95926", s2bg="#3a2419"),
}
FONT = "font-family=\"-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif\""


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class D:
    def __init__(self, w, h, c):
        self.w, self.h, self.c, self.b = w, h, c, []

    def svg(self):
        c = self.c
        defs = "".join(
            f'<marker id="a{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>' for k, col in (("n", c["text2"]), ("b", c["s1"]), ("o", c["s2"])))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" {FONT}>'
                f'<defs>{defs}</defs><rect width="{self.w}" height="{self.h}" rx="12" fill="{c["surface"]}"/>{"".join(self.b)}</svg>')

    def text(self, x, y, s, size=12.5, fill=None, weight=400, anchor="middle", italic=False):
        self.b.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
                      f'{"font-style=" + chr(34) + "italic" + chr(34) if italic else ""} fill="{fill or self.c["text"]}">{esc(s)}</text>')

    def box(self, x, y, w, h, title, sub=None, kind="node", rx=10):
        c = self.c
        fill, stroke = {"node": (c["surface"], c["line"]), "blue": (c["s1bg"], c["s1"]), "orange": (c["s2bg"], c["s2"]),
                        "group": (c["card"], c["line"])}[kind]
        dash = ' stroke-dasharray="5 4"' if kind == "group" else ""
        self.b.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')
        if kind == "group":
            self.text(x + 14, y + 20, title, 11.5, c["muted"], 600, "start")
            return
        cy = y + h / 2 + (-3 if sub else 4.5)
        self.text(x + w / 2, cy, title, 13, c["text"], 600)
        if sub:
            self.text(x + w / 2, cy + 17, sub, 11.5, c["text2"])

    def diamond(self, cx, cy, w, h, l1, l2):
        c = self.c
        self.b.append(f'<polygon points="{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}" '
                      f'fill="{c["surface"]}" stroke="{c["text2"]}" stroke-width="1.5"/>')
        self.text(cx, cy - 3, l1, 12, c["text"], 600)
        self.text(cx, cy + 13, l2, 11.5, c["text2"])

    def arrow(self, pts, color="n", dashed=False, label=None, lx=None, ly=None, lanchor="middle"):
        col = {"n": self.c["text2"], "b": self.c["s1"], "o": self.c["s2"]}[color]
        d = "M" + " L".join(f"{x},{y}" for x, y in pts)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.b.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.6"{dash} marker-end="url(#a{color})"/>')
        if label:
            self.text(lx, ly, label, 11.5, col if color != "n" else self.c["text2"], 400, lanchor)


def architecture(c):
    d = D(880, 600, c)
    d.box(300, 20, 280, 62, "Claude session", "Claude Code · Claude Desktop")
    d.arrow([(440, 82), (440, 128)], label="stdio", lx=450, ly=110, lanchor="start")
    d.box(110, 130, 660, 140, "kimodo MCP server · no torch, never holds the model", kind="group")
    d.box(140, 172, 210, 64, "10 tools", "(+ retarget when configured)")
    d.diamond(560, 205, 230, 92, "GPU guard", "Kimodo already on the GPU?")
    d.arrow([(350, 204), (444, 204)])
    d.arrow([(675, 205), (820, 205), (820, 51), (582, 51)], "o", dashed=True, label="busy: refused", lx=804, ly=122, lanchor="end")
    d.arrow([(560, 251), (560, 318)], "b", label="free: start a detached job", lx=572, ly=294, lanchor="start")
    d.box(110, 320, 660, 140, "detached job · its own process, outlives the session", kind="group")
    d.box(140, 362, 190, 64, "runner", "stores the exit code")
    d.box(430, 350, 220, 46, "kimodo_batch.py", "load → encode → clips")
    d.box(430, 404, 220, 46, "kimodo_preview.py", "Blender + ffmpeg")
    d.arrow([(330, 384), (380, 384), (380, 373), (428, 373)])
    d.arrow([(380, 384), (380, 427), (428, 427)])
    d.box(140, 510, 360, 64, "workspace outputs/", "takes .npz · .mp4 · sheets · _logs · _mcp_jobs")
    d.box(600, 510, 170, 64, "NVIDIA GPU", "loaded per job, freed at exit", kind="orange")
    d.arrow([(650, 373), (700, 373), (700, 508)], "o")
    d.arrow([(540, 450), (540, 480), (420, 480), (420, 508)])
    d.arrow([(140, 220), (90, 220), (90, 542), (138, 542)])
    d.text(78, 380, "job · view · measure · list", 11.5, c["text2"], 400, "middle")
    d.b[-1] = d.b[-1].replace("<text ", '<text transform="rotate(-90 78 380)" ', 1)
    return d


def sequence(c):
    d = D(880, 660, c)
    lanes = [("Claude", 110), ("MCP server", 330), ("Job process", 560), ("GPU", 780)]
    for name, x in lanes:
        d.box(x - 70, 20, 140, 40, name, kind="orange" if name == "GPU" else "node")
        d.b.append(f'<line x1="{x}" x2="{x}" y1="60" y2="632" stroke="{c["line"]}" stroke-width="1.5" stroke-dasharray="4 4"/>')
    X = dict((n, x) for n, x in lanes)
    y = 100
    steps = [("Claude", "MCP server", 'generate("demo_showcase")', "n", False),
             ("MCP server", "MCP server", "scan processes: another Kimodo on the GPU?", "n", False),
             ("MCP server", "Job process", "start detached runner + kimodo_batch.py", "b", False),
             ("MCP server", "Claude", "job_id, at once", "n", True),
             ("Job process", "GPU", "load Kimodo", "o", False),
             ("Job process", "GPU", "encode every prompt once · 5.4 GB peak", "o", False),
             ("Job process", "GPU", "move the text encoder off the GPU", "o", False),
             ("Job process", "GPU", "diffusion per clip, all samples · 1.1–1.2 GB", "o", False),
             ("Job process", "GPU", "process exits: 0 GB held", "o", True),
             ("Claude", "MCP server", "job(job_id)", "n", False),
             ("MCP server", "Claude", "done · 5 clips · GPU free", "n", True),
             ("Claude", "MCP server", "preview → view → measure → finalize", "b", False)]
    for i, (a, b, label, col, dashed) in enumerate(steps, 1):
        if i == 8:
            d.b.append(f'<rect x="{X["Job process"] - 32}" y="{y - 24}" width="{X["GPU"] - X["Job process"] + 64}" height="38" rx="6" '
                       f'fill="none" stroke="{c["line"]}" stroke-width="1.2"/>')
            d.text(X["Job process"] - 28, y - 30, "loop · each clip", 10.5, c["muted"], 600, "start")
        if a == b:
            x = X[a]
            d.arrow([(x, y - 6), (x + 40, y - 6), (x + 40, y + 6), (x + 2, y + 6)], col)
            d.text(x + 50, y + 4, label, 12, c["text2"], 400, "start")
        else:
            x1, x2 = X[a], X[b]
            s = 1 if x2 > x1 else -1
            d.arrow([(x1 + 4 * s, y), (x2 - 4 * s, y)], col, dashed)
            d.text((x1 + x2) / 2, y - 7, label, 12, c["text2"])
        d.b.append(f'<circle cx="{X[a] - 18 * (1 if X[b] >= X[a] else -1) if a != b else X[a] - 18}" cy="{y}" r="9" fill="{c["text2"]}"/>')
        d.text(X[a] - 18 * (1 if X[b] >= X[a] else -1) if a != b else X[a] - 18, y + 4, str(i), 10.5, c["surface"], 700)
        y += 52 if i in (4, 7) else 42
    return d


def workflow(c):
    d = D(880, 240, c)
    pills = [("write_spec", 20, 62, 104), ("generate", 148, 62, 96), ("job", 268, 62, 64), ("preview", 356, 62, 92)]
    for name, x, y, w in pills:
        d.box(x, y, w, 44, name, kind="blue" if name in ("generate", "preview") else "node", rx=22)
    for (_, x1, _, w1), (_, x2, _, _) in zip(pills, pills[1:]):
        d.arrow([(x1 + w1, 84), (x2 - 2, 84)])
    d.box(482, 26, 128, 46, "view", "sheet · frame grid", rx=22)
    d.box(482, 98, 128, 46, "measure", "numbers from .npz", rx=22)
    d.arrow([(448, 84), (466, 84), (466, 49), (480, 49)])
    d.arrow([(466, 84), (466, 121), (480, 121)])
    d.diamond(690, 84, 118, 80, "follows", "the prompt?")
    d.arrow([(610, 49), (618, 49), (618, 84), (629, 84)])
    d.b.append(f'<path d="M610,121 L618,121 L618,84" fill="none" stroke="{c["text2"]}" stroke-width="1.6"/>')
    d.box(770, 62, 92, 44, "finalize", kind="blue", rx=22)
    d.arrow([(749, 84), (768, 84)], label="yes", lx=758, ly=76)
    d.box(770, 160, 92, 40, "retarget", rx=20)
    d.b[-2] = d.b[-2].replace('stroke-width="1.5"', 'stroke-width="1.5" stroke-dasharray="5 4"')
    d.arrow([(816, 106), (816, 158)], dashed=True, label="optional", lx=826, ly=138, lanchor="start")
    d.arrow([(690, 124), (690, 214), (72, 214), (72, 108)], label="no: reword as <name>_v2", lx=380, ly=206)
    return d


for name, c in THEMES.items():
    for stem, fn in (("architecture", architecture), ("sequence", sequence), ("workflow", workflow)):
        (OUT / f"{stem}_{name}.svg").write_text(fn(c).svg(), encoding="utf-8")
print("diagrams written to", OUT)
