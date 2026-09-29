"""Figures 1-6 as vector PDF and 3000-px PNG, drawn with reportlab from results.json."""
import json, math, os, sys
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, black, white
import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FIGDIR", os.path.join(HERE, "figures")); os.makedirs(OUT, exist_ok=True)
RES = json.load(open(os.environ.get("RESULTS", os.path.join(HERE, "..", "results.json"))))
# Okabe-Ito
C = {"flat": HexColor("#D55E00"), "tree": HexColor("#0072B2"), "recursive": HexColor("#009E73"),
     "warm": HexColor("#E69F00"), "grey": HexColor("#6B6B6B"), "light": HexColor("#DDDDDD"), "ink": HexColor("#1A1A1A"),
     "fill0": HexColor("#EAF2FA"), "fill1": HexColor("#FDF1E6"), "fill2": HexColor("#E8F5EF")}
NAME = {"flat": "Flat (one CMLM)", "tree": "Tree", "recursive": "Cascade"}
MARK = {"flat": "circle", "tree": "square", "recursive": "triangle"}
F, FB = "Helvetica", "Helvetica-Bold"


def fmt(v):
    return f"{v:,.0f}"


class Fig:
    def __init__(self, name, w=504, h=300):
        self.path = os.path.join(OUT, name + ".pdf"); self.png = os.path.join(OUT, name + ".png")
        self.c = canvas.Canvas(self.path, pagesize=(w, h)); self.w, self.h = w, h
        self.c.setTitle(name); self.c.setAuthor("Devieswar Kancheti")

    def text(self, x, y, s, size=8.5, font=F, anchor="l", color=None, angle=0):
        c = self.c; c.saveState(); c.setFillColor(color or C["ink"]); c.setFont(font, size)
        c.translate(x, y); c.rotate(angle)
        {"l": c.drawString, "c": c.drawCentredString, "r": c.drawRightString}[anchor](0, 0, s)
        c.restoreState()

    def marker(self, x, y, kind, color, r=3.2):
        c = self.c; c.setFillColor(color); c.setStrokeColor(white); c.setLineWidth(0.6)
        if kind == "circle": c.circle(x, y, r, stroke=1, fill=1)
        elif kind == "square": c.rect(x - r, y - r, 2 * r, 2 * r, stroke=1, fill=1)
        else:
            p = c.beginPath(); p.moveTo(x, y + r * 1.2); p.lineTo(x - r * 1.1, y - r * 0.8); p.lineTo(x + r * 1.1, y - r * 0.8); p.close()
            c.drawPath(p, stroke=1, fill=1)

    def save(self):
        self.c.showPage(); self.c.save()
        d = fitz.open(self.path); pg = d[0]
        zoom = 3000 / pg.rect.width
        pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False).save(self.png)
        print("wrote", self.path, self.png)


class Axes:
    def __init__(self, fig, x0, y0, x1, y1, xr, yr, logy=False):
        self.f, self.x0, self.y0, self.x1, self.y1, self.xr, self.yr, self.logy = fig, x0, y0, x1, y1, xr, yr, logy

    def X(self, v):
        return self.x0 + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * (self.x1 - self.x0)

    def Y(self, v):
        if self.logy:
            a, b = math.log10(self.yr[0]), math.log10(self.yr[1]); t = (math.log10(v) - a) / (b - a)
        else:
            t = (v - self.yr[0]) / (self.yr[1] - self.yr[0])
        return self.y0 + t * (self.y1 - self.y0)

    def frame(self, xticks, yticks, xlabel, ylabel, xfmt=str, yfmt=str, grid=True):
        c = self.f.c; c.setLineWidth(0.4)
        for v in yticks:
            y = self.Y(v)
            if grid:
                c.setStrokeColor(C["light"]); c.line(self.x0, y, self.x1, y)
            c.setStrokeColor(C["ink"]); c.line(self.x0 - 3, y, self.x0, y)
            self.f.text(self.x0 - 5, y - 3, yfmt(v), 8, anchor="r")
        for v in xticks:
            x = self.X(v) if not isinstance(v, tuple) else v[0]
            lab = xfmt(v) if not isinstance(v, tuple) else v[1]
            c.setStrokeColor(C["ink"]); c.line(x, self.y0, x, self.y0 - 3)
            self.f.text(x, self.y0 - 12, lab, 8, anchor="c")
        c.setStrokeColor(C["ink"]); c.setLineWidth(0.7)
        c.line(self.x0, self.y0, self.x1, self.y0); c.line(self.x0, self.y0, self.x0, self.y1)
        self.f.text((self.x0 + self.x1) / 2, self.y0 - 26, xlabel, 8.5, anchor="c")
        self.f.text(self.x0 - 42, (self.y0 + self.y1) / 2, ylabel, 8.5, anchor="c", angle=90)

    def line(self, xs, ys, color, width=1.4, dash=None, mark=None):
        c = self.f.c; c.setStrokeColor(color); c.setLineWidth(width)
        c.setDash(*(dash or ([], 0)) if isinstance(dash, tuple) else (dash or []))
        p = c.beginPath(); p.moveTo(self.X(xs[0]), self.Y(ys[0]))
        for x, y in zip(xs[1:], ys[1:]):
            p.lineTo(self.X(x), self.Y(y))
        c.drawPath(p, stroke=1, fill=0); c.setDash([])
        if mark:
            for x, y in zip(xs, ys):
                self.f.marker(self.X(x), self.Y(y), mark, color)


def legend(fig, x, y, items, dy=13):
    c = fig.c
    for i, (label, color, mark, dash) in enumerate(items):
        yy = y - i * dy
        c.setStrokeColor(color); c.setLineWidth(1.4); c.setDash(dash or [])
        c.line(x, yy + 3, x + 18, yy + 3); c.setDash([])
        if mark: fig.marker(x + 9, yy + 3, mark, color)
        fig.text(x + 24, yy, label, 8.0)


# ------------------------------------------------------------------ Figure 1
def fig1():
    f = Fig("Figure 1", 504, 268); c = f.c

    def box(x, y, w, h, title, sub, fill, stroke=C["ink"], dash=None):
        c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(0.9); c.setDash(dash or [])
        c.roundRect(x, y, w, h, 4, stroke=1, fill=1); c.setDash([])
        f.text(x + w / 2, y + h - 13, title, 8.6, FB, "c")
        for i, s in enumerate(sub):
            f.text(x + w / 2, y + h - 25 - i * 10, s, 7.3, F, "c", C["grey"])

    def arrow(x1, y1, x2, y2, color=C["ink"], dash=None, width=0.9):
        c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(width); c.setDash(dash or [])
        c.line(x1, y1, x2, y2); c.setDash([])
        a = math.atan2(y2 - y1, x2 - x1); L = 5.5
        p = c.beginPath(); p.moveTo(x2, y2)
        p.lineTo(x2 - L * math.cos(a - 0.4), y2 - L * math.sin(a - 0.4)); p.lineTo(x2 - L * math.cos(a + 0.4), y2 - L * math.sin(a + 0.4)); p.close()
        c.drawPath(p, stroke=0, fill=1)

    # columns
    box(8, 196, 70, 46, "Person", ["asks in words"], white)
    box(116, 180, 108, 78, "Frontier model", ["rules only,", "no transcript,", "tools: recall, teach"], C["fill0"])
    box(278, 196, 100, 46, "CMLM level 0", ["at most 30 facts,", "trains every turn"], C["fill2"])
    box(278, 110, 100, 46, "CMLM level 1", ["at most 30 facts"], C["fill2"])
    box(278, 24, 100, 46, "CMLM level n", ["the oldest facts"], C["fill2"])
    for y in (196, 110, 24):
        box(416, y + 6, 80, 34, "Life", ["exact key table"], C["fill1"], dash=[2, 2])
        arrow(378, y + 27, 414, y + 27, C["warm"], dash=[2, 2])
        arrow(414, y + 17, 378, y + 17, C["warm"], dash=[2, 2])
    box(116, 60, 108, 52, "Tool model", ["learns the frontier's", "calls; gated"], white, dash=[4, 2])
    # the turn
    arrow(78, 226, 114, 226); f.text(96, 231, "question", 6.8, F, "c")
    arrow(114, 206, 78, 206); f.text(96, 196, "answer", 6.8, F, "c")
    arrow(224, 236, 276, 236); f.text(250, 244, "recall(q, k, key)", 6.6, F, "c")
    arrow(276, 220, 224, 220); f.text(250, 210, "k facts as text", 6.6, F, "c")
    arrow(224, 200, 276, 204); f.text(250, 188, "teach(cue, fact, key)", 6.6, F, "c")
    # hand-down as vectors
    arrow(318, 194, 318, 158, C["recursive"], width=1.2); arrow(318, 108, 318, 72, C["recursive"], width=1.2)
    arrow(338, 158, 338, 194, C["recursive"], width=1.2, dash=[3, 2]); arrow(338, 72, 338, 108, C["recursive"], width=1.2, dash=[3, 2])
    f.text(262, 176, "cue vectors,", 6.8, F, "r", C["recursive"]); f.text(262, 167, "0 tokens", 6.8, F, "r", C["recursive"])
    f.text(346, 176, "ranked facts", 6.8, F, "l", C["recursive"])
    c.setFillColor(C["recursive"])
    for yy in (84, 90, 96): c.circle(328, yy, 1.1, stroke=0, fill=1)
    # tool model
    arrow(170, 112, 170, 178, C["grey"], dash=[4, 2]); f.text(176, 140, "recovered calls", 6.8, F, "l", C["grey"])
    # legend
    f.text(8, 150, "Words", 7.4, FB); f.text(8, 140, "person to frontier,", 7.0, F, "l", C["grey"])
    f.text(8, 131, "and the readout into an", 7.0, F, "l", C["grey"]); f.text(8, 122, "API-served frontier", 7.0, F, "l", C["grey"])
    f.text(8, 104, "Vectors", 7.4, FB, "l", C["recursive"]); f.text(8, 94, "between CMLM levels", 7.0, F, "l", C["grey"])
    f.text(8, 76, "Keys", 7.4, FB, "l", C["warm"]); f.text(8, 66, "CMLM to its Life", 7.0, F, "l", C["grey"])
    f.save()


# ------------------------------------------------------------------ Figure 2
def fig2():
    t5 = RES["t5"]
    groups = [("60 turns, k = 6", 60, 6), ("90 turns, k = 9", 90, 9), ("120 turns, k = 12", 120, 12)]
    f = Fig("Figure 2", 504, 290); ax = Axes(f, 70, 50, 366, 270, (0, 3), (0, 0.8))
    ax.frame([(ax.X(i + 0.5), g[0]) for i, g in enumerate(groups)], [0, 0.2, 0.4, 0.6, 0.8],
             "Session length and readout size", "Recall@k at the last turn (fraction)", yfmt=lambda v: f"{v:.1f}")
    c = f.c; bw = 26
    for i, (lab, turns, k) in enumerate(groups):
        for j, shape in enumerate(("flat", "tree", "recursive")):
            key = f"{turns} turns, k={k}, {shape}"
            if key not in t5: continue
            st = t5[key]["per_seed"]["final"]; m = st["mean"]; lo, hi = st["ci95"]
            x = ax.X(i + 0.5) + (j - 1) * (bw + 4)
            c.setFillColor(C[shape]); c.setStrokeColor(C["ink"]); c.setLineWidth(0.4)
            c.rect(x - bw / 2, ax.Y(0), bw, ax.Y(m) - ax.Y(0), stroke=1, fill=1)
            if shape == "tree":
                c.saveState(); p = c.beginPath(); p.rect(x - bw / 2, ax.Y(0), bw, ax.Y(m) - ax.Y(0)); c.clipPath(p, stroke=0)
                c.setStrokeColor(white); c.setLineWidth(0.7)
                for d in range(-60, 200, 6): c.line(x - bw / 2 + d, ax.Y(0), x - bw / 2 + d + 60, ax.Y(0) + 60 * 3)
                c.restoreState()
            if shape == "recursive":
                c.saveState(); p = c.beginPath(); p.rect(x - bw / 2, ax.Y(0), bw, ax.Y(m) - ax.Y(0)); c.clipPath(p, stroke=0)
                c.setFillColor(white)
                for yy in range(int(ax.Y(0)) + 3, int(ax.Y(m)), 6):
                    for xx in range(int(x - bw / 2) + 3, int(x + bw / 2), 6): c.circle(xx, yy, 0.8, stroke=0, fill=1)
                c.restoreState()
            c.setStrokeColor(C["ink"]); c.setLineWidth(0.8)
            c.line(x, ax.Y(lo), x, ax.Y(hi)); c.line(x - 4, ax.Y(lo), x + 4, ax.Y(lo)); c.line(x - 4, ax.Y(hi), x + 4, ax.Y(hi))
            f.text(x, ax.Y(hi) + 4, f"{m:.3f}", 7, anchor="c")
    lx = 384
    for i, shape in enumerate(("flat", "tree", "recursive")):
        y = 250 - i * 16
        c.setFillColor(C[shape]); c.setStrokeColor(C["ink"]); c.setLineWidth(0.4); c.rect(lx, y, 12, 10, stroke=1, fill=1)
        f.text(lx + 18, y + 2, NAME[shape], 8.2)
    f.text(lx, 190, "Bars: mean of 20 seeds.", 7.4, F, "l", C["grey"])
    f.text(lx, 180, "Whiskers: 95% confidence", 7.4, F, "l", C["grey"]); f.text(lx, 170, "interval over seeds.", 7.4, F, "l", C["grey"])
    f.text(lx, 154, "k equals facts per topic,", 7.4, F, "l", C["grey"]); f.text(lx, 144, "so the ceiling is 1.", 7.4, F, "l", C["grey"])
    f.save()


# ------------------------------------------------------------------ Figure 3
def fig3():
    t10 = RES["t10"]
    f = Fig("Figure 3", 504, 280); ax = Axes(f, 70, 50, 366, 260, (0, 310), (0, 1))
    ax.frame([50, 100, 150, 200, 250, 300], [0, 0.2, 0.4, 0.6, 0.8, 1.0], "Session turn (count)",
             "Recall as a share of its ceiling (fraction)", yfmt=lambda v: f"{v:.1f}")
    dash = {"flat": [], "tree": [4, 2], "recursive": [1.5, 2]}
    for shape in ("flat", "tree", "recursive"):
        rows = t10[shape]["seed1_checkpoints"]
        ax.line([r["turn"] for r in rows], [r["share"] for r in rows], C[shape], dash=dash[shape], mark=MARK[shape])
    legend(f, 384, 250, [(NAME[s], C[s], MARK[s], dash[s]) for s in ("flat", "tree", "recursive")])
    f.text(384, 196, "Seed 1 of the run with a", 7.4, F, "l", C["grey"]); f.text(384, 186, "3,300-token document per", 7.4, F, "l", C["grey"])
    f.text(384, 176, "turn; k = 12. The ceiling", 7.4, F, "l", C["grey"]); f.text(384, 166, "falls as facts per topic", 7.4, F, "l", C["grey"])
    f.text(384, 156, "pass k.", 7.4, F, "l", C["grey"])
    f.save()


# ------------------------------------------------------------------ Figure 4
def fig4():
    rows = RES["t10"]["recursive"]["seed1_per_turn"]
    f = Fig("Figure 4", 504, 280); ax = Axes(f, 76, 50, 366, 260, (0, 310), (100, 3e6), logy=True)
    ax.frame([50, 100, 150, 200, 250, 300], [100, 1000, 10000, 100000, 1000000], "Session turn (count)",
             "Input tokens per question (log scale)", yfmt=lambda v: {100: "100", 1000: "1,000", 10000: "10,000", 100000: "100,000", 1000000: "1,000,000"}[v])
    c = f.c; c.setStrokeColor(C["grey"]); c.setLineWidth(0.7); c.setDash([3, 2]); c.line(ax.x0, ax.Y(1e6), ax.x1, ax.Y(1e6)); c.setDash([])
    f.text(ax.x0 + 4, ax.Y(1e6) + 4, "a one-million-token context window", 7.2, F, "l", C["grey"])
    xs = [r["t"] for r in rows]
    ax.line(xs, [max(r["transcript"], 100) for r in rows], C["flat"], width=1.6)
    ax.line(xs, [r["sent"] for r in rows], C["tree"], width=1.6, dash=[4, 2])
    legend(f, 384, 250, [("Transcript path", C["flat"], None, []), ("CMLM path, sent", C["tree"], None, [4, 2])])
    last = rows[-1]
    f.text(384, 212, f"At turn {last['t']}:", 7.4, F, "l", C["grey"])
    f.text(384, 202, f"transcript {fmt(last['transcript'])}", 7.4, F, "l", C["grey"])
    f.text(384, 192, f"sent {fmt(last['sent'])}", 7.4, F, "l", C["grey"])
    f.text(384, 176, "Cascade, seed 1; token", 7.4, F, "l", C["grey"]); f.text(384, 166, "counts are characters / 4.", 7.4, F, "l", C["grey"])
    f.save()


# ------------------------------------------------------------------ Figure 5
def fig5():
    kh = RES["t7"]["key head alone"]
    pts = sorted({r["examples"] for r in kh["rows"]})
    f = Fig("Figure 5", 504, 280); ax = Axes(f, 70, 50, 366, 260, (0, 135), (0, 1))
    ax.frame(pts, [0, 0.2, 0.4, 0.6, 0.8, 1.0], "Keyed examples taught (count)", "Share of 30 unseen facts (fraction)", yfmt=lambda v: f"{v:.1f}")
    c = f.c; c.setStrokeColor(C["grey"]); c.setLineWidth(0.8); c.setDash([3, 2]); c.line(ax.x0, ax.Y(0.85), ax.x1, ax.Y(0.85)); c.setDash([])
    f.text(ax.x0 + 4, ax.Y(0.85) + 4, "prequential gate, 0.85", 7.2, F, "l", C["grey"])
    acc = [sum(r["correct"] for r in kh["rows"] if r["examples"] == n) / sum(r["of"] for r in kh["rows"] if r["examples"] == n) for n in pts]
    com = [sum(r["committed"] for r in kh["rows"] if r["examples"] == n) / sum(r["of"] for r in kh["rows"] if r["examples"] == n) for n in pts]
    for r in kh["rows"]:
        f.marker(ax.X(r["examples"]) + (r["seed"] - 2) * 3, ax.Y(r["correct"] / r["of"]), "circle", HexColor("#9DC3E6"), r=2.0)
    ax.line(pts, acc, C["tree"], mark="square")
    ax.line(pts, com, C["flat"], dash=[4, 2], mark="triangle")
    legend(f, 384, 250, [("Correct key", C["tree"], "square", []), ("Committed", C["flat"], "triangle", [4, 2])])
    f.text(384, 212, "Lines: mean of 3 seeds;", 7.4, F, "l", C["grey"]); f.text(384, 202, "dots: single seeds.", 7.4, F, "l", C["grey"])
    f.text(384, 186, "Committed: softmax share", 7.4, F, "l", C["grey"]); f.text(384, 176, "of at least 0.6.", 7.4, F, "l", C["grey"])
    f.save()


# ------------------------------------------------------------------ Figure 6
def fig6():
    rs = RES["real_session"]["per_turn"]
    cum = {"cold": [], "warm": [], "cmlm": []}; s = {"cold": 0, "warm": 0, "cmlm": 0}
    for r in rs:
        for k in s: s[k] += r[k]; cum[k].append(max(s[k], 1))
    f = Fig("Figure 6", 504, 280); ax = Axes(f, 76, 50, 366, 260, (0, 35), (1, 2e7), logy=True)
    ax.frame([1, 5, 10, 15, 20, 25, 30, 34], [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000], "Turn of the real session (count)",
             "Cumulative input tokens (log scale)", yfmt=lambda v: f"{v:,.0f}")
    xs = [r["turn"] for r in rs]
    ax.line(xs, cum["cold"][:], C["flat"], mark="circle")
    ax.line(xs, cum["warm"], C["warm"], dash=[4, 2], mark="triangle")
    ax.line(xs, cum["cmlm"], C["tree"], dash=[1.5, 2], mark="square")
    legend(f, 384, 250, [("Transcript, no cache", C["flat"], "circle", []), ("Transcript, warm cache", C["warm"], "triangle", [4, 2]),
                         ("CMLM path", C["tree"], "square", [1.5, 2])])
    f.text(384, 196, "Totals after 34 turns:", 7.4, F, "l", C["grey"])
    f.text(384, 186, f"{fmt(s['cold'])} no cache", 7.4, F, "l", C["grey"])
    f.text(384, 176, f"{fmt(s['warm'])} warm (equivalents)", 7.4, F, "l", C["grey"])
    f.text(384, 166, f"{fmt(s['cmlm'])} CMLM", 7.4, F, "l", C["grey"])
    f.text(384, 150, "Token counts are", 7.4, F, "l", C["grey"]); f.text(384, 140, "characters / 4.", 7.4, F, "l", C["grey"])
    f.save()


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    for w in which:
        globals()["fig" + w]()
