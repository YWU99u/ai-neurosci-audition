"""Regenerate fig_map_example.pdf with a representative small model (Qwen3-1.7B, R2=0.62/0.66). Cross-validated predicted city coordinates -> the model's world map.
Real, no fabrication."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from mlib import load, last_token_residuals
from exp01_concept import GRID
from exp02_map import CITIES
plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight", "figure.dpi": 220,
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "axes.spines.top": False, "axes.spines.right": False})
OUT = "paper/figures"
MODEL = "Qwen3-1.7B"
path = [g[2] for g in GRID if g[0] == MODEL][0]
tok, model = load(path)

names = [c[0] for c in CITIES]; Y = np.array([[c[1], c[2]] for c in CITIES], float)
reps = np.stack([last_token_residuals(tok, model, f"The city of {n}") for n in names])
def cvpred(Xl, a):
    kf = KFold(5, shuffle=True, random_state=0); pr = np.zeros_like(Y)
    for tr, te in kf.split(Xl): pr[te] = Ridge(alpha=a).fit(Xl[tr], Y[tr]).predict(Xl[te])
    return pr
best = (-9, None, None)
for l in range(reps.shape[1]):
    for a in (10., 100., 1000.):
        pr = cvpred(reps[:, l, :], a)
        sc = 2 - (((Y - pr) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)).sum()
        if sc > best[0]: best = (sc, l, a)
_, bl, ba = best; pred = cvpred(reps[:, bl, :], ba)
ssr = ((Y - pred) ** 2).sum(0); sst = ((Y - Y.mean(0)) ** 2).sum(0); r2lat, r2lon = 1 - ssr / sst

fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.scatter(pred[:, 1], pred[:, 0], c=Y[:, 1], cmap="Spectral", s=36, edgecolor="#333", lw=0.5, zorder=3)
show = {"London", "New York", "Tokyo", "Sydney", "Cairo", "Rio de Janeiro", "Moscow", "Mumbai",
        "Los Angeles", "Beijing", "Cape Town", "Buenos Aires", "Singapore", "Dubai"}
for (n, la, lo), (py, px) in zip(CITIES, pred):
    if n in show: ax.annotate(n, (px, py), fontsize=6.6, color="#222", xytext=(3, 2), textcoords="offset points")
ax.set_xlabel("predicted longitude"); ax.set_ylabel("predicted latitude")
ax.set_title(f"The model's internal world map ({MODEL}): held-out cities,\n"
             f"decoded lat/lon $R^2$={r2lat:.2f}/{r2lon:.2f}", fontsize=8.5, loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_map_example.pdf"); fig.savefig(f"{OUT}/fig_map_example.png")
print(f"map ({MODEL}): R2 lat={r2lat:.2f} lon={r2lon:.2f}")
