"""Render results/qual_examples.json as a transcript-style artifact with the injected concept
vocabulary highlighted -> paper/figures/fig_steer_example.pdf. Real generations, no fabrication."""
import json, textwrap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.size": 8.5, "savefig.bbox": "tight", "figure.dpi": 220})
RED, BLUE, INK, GREY = "#B23A5E", "#2C6E9E", "#242424", "#8A8A8A"

d = json.load(open("results/qual_examples.json"))
def get(c, coef):
    return next(e["text"] for e in d["examples"] if e["concept"] == c and abs(e["coef"] - coef) < 1e-6)
def clip(t, n): return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " ..."

PARIS = {"paris", "parisian", "eiffel", "tower", "france", "french"}
SEA = {"ocean", "waves", "wave", "salty", "sea", "coastal", "marine", "water"}
over = "Parisian Parisian Parisian Parisian ... Paris France Paris France Paris France Paris France ..."

blocks = [  # (label, label-color, text, highlight-set, highlight-color)
 ("prompt →", GREY, d["prompt"], set(), GREY),
 ("no steering →", INK, clip(d["baseline"], 190), set(), INK),
 ("+ Paris direction  (c=0.2) →", RED, clip(get("Paris", 0.2), 210), PARIS, RED),
 ("+ Dolphin direction  (c=0.2) →", BLUE, clip(get("Dolphin", 0.2), 210), SEA, BLUE),
 ("+ Paris, over-steered  (c=0.35) →", RED, over, PARIS, RED),
]

fig, ax = plt.subplots(figsize=(7.0, 4.7)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.add_patch(FancyBboxPatch((0.004, 0.01), 0.99, 0.985, boxstyle="round,pad=0.008",
             fc="#FBFAF8", ec="#DADADA", lw=1.0, zorder=0))
fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = ax.transAxes.inverted()
probe = ax.text(0, 0, "M" * 20, family="monospace", fontsize=8.5, transform=ax.transAxes)
e = probe.get_window_extent(r); charw = (inv.transform((e.x1, 0))[0] - inv.transform((e.x0, 0))[0]) / 20
probe.remove()
x0, y, lh = 0.028, 0.945, 0.049
NC = int(0.93 / charw)
for label, lc, text, hl, hc in blocks:
    ax.text(x0 - 0.004, y, label, transform=ax.transAxes, fontsize=8.3, fontweight="bold", color=lc, va="top")
    y -= lh * 1.02
    for line in (textwrap.wrap(text, width=NC) or [""]):
        col = 0
        for w in line.split(" "):
            clean = w.strip(".,;:!?\"'’").lower()
            hit = clean in hl
            ax.text(x0 + col * charw, y, w, transform=ax.transAxes, fontsize=8.5, family="monospace",
                    color=(hc if hit else "#2A2A2A"), fontweight=("bold" if hit else "normal"), va="top")
            col += len(w) + 1
        y -= lh
    y -= lh * 0.45
fig.savefig("paper/figures/fig_steer_example.pdf"); fig.savefig("paper/figures/fig_steer_example.png")
print("wrote fig_steer_example")
