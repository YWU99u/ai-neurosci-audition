"""Real graphical abstract -> paper/figures/fig_graphical.pdf. Left: the same steering effect
looks like emergence in raw units but is flat once made comparable (real data). Right: the four
audit verdicts."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.size": 8.5, "savefig.bbox": "tight", "figure.dpi": 220,
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "axes.spines.top": False, "axes.spines.right": False})
RED, QW, GREEN, ORANGE, INK = "#B23A5E", "#B23A5E", "#2F8F63", "#D98A3D", "#242424"

nc = json.load(open("results/naive_concept.json"))
sc = json.load(open("results/steer_cv2.json"))
raw = sorted([(v["size"], v["naive_gain"]) for v in nc.values()])
hel = sorted([(v["size"], v["heldout_eff"]) for k, v in sc.items()
              if k.startswith("Qwen3") and "bf16" not in k and v.get("size", 99) <= 14 and "heldout_eff" in v])

fig = plt.figure(figsize=(7.0, 2.75))
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], height_ratios=[1, 1], hspace=0.72, wspace=0.32)

axa = fig.add_subplot(gs[0, 0])
axa.plot([p[0] for p in raw], [p[1] for p in raw], "-o", ms=3.2, color=RED, mfc="white", mew=1.2)
axa.set_xscale("log"); axa.set_title("raw units: looks like emergence", fontsize=7.6, color=RED, loc="left", pad=1)
axa.set_xticks([p[0] for p in raw]); axa.set_xticklabels([]); axa.tick_params(labelsize=6.5)
axa.set_ylabel("gain", fontsize=7)

axb = fig.add_subplot(gs[1, 0])
axb.plot([p[0] for p in hel], [p[1] for p in hel], "-o", ms=3.2, color=QW)
axb.axhline(np.mean([p[1] for p in hel]), ls=(0, (4, 3)), color="#AAA", lw=0.9)
axb.set_xscale("log"); axb.set_title("comparable units + held-out: no trend", fontsize=7.4, color=INK, loc="left", pad=1)
axb.set_xticks([p[0] for p in hel]); axb.set_xticklabels([f"{p[0]:g}" for p in hel], fontsize=6.3)
axb.set_xlabel("model size (B)", fontsize=7); axb.set_ylabel("effect", fontsize=7); axb.set_ylim(0, None)

# right: verdict chips
axr = fig.add_subplot(gs[:, 1]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
axr.text(0.0, 0.99, "Four neuroscience parallels, audited to 72B", fontsize=8.4, fontweight="bold", va="top")
rows = [("Concept steering  (recording)", "ARTIFACT", RED, "units + operating point"),
        ("Number magnitude  (tuning)", "MIXED", ORANGE, "‘never bell’ was selection bias"),
        ("Language localization  (lesion)", "NOT ROBUST", RED, "attribution-dependent"),
        ("World map  (decoding)", "ROBUST", GREEN, "survives every control")]
y = 0.80
for name, verd, col, why in rows:
    axr.text(0.0, y, name, fontsize=7.6, va="center")
    axr.add_patch(FancyBboxPatch((0.63, y - 0.055), 0.36, 0.11, boxstyle="round,pad=0.008",
                  fc=col, ec="none", alpha=0.92, transform=axr.transAxes))
    axr.text(0.81, y, verd, fontsize=6.9, fontweight="bold", color="white", ha="center", va="center")
    axr.text(0.0, y - 0.085, why, fontsize=6.3, color="#666", va="center", style="italic")
    y -= 0.205
axr.text(0.0, -0.02, "Controls, not new phenomena, decide which parallels are real.",
         fontsize=7.4, fontweight="bold", va="top", color=INK)

fig.savefig("paper/figures/fig_graphical.pdf"); fig.savefig("paper/figures/fig_graphical.png")
print("wrote fig_graphical")
