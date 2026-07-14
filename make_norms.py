"""#4 Real norm decomposition for the units argument (Fig 2a data).
Across the Qwen3 ladder, at the operating layer (~0.8 depth), measure:
  (a) residual-stream norm ||h||,
  (b) the RAW (unnormalized) mean-difference direction norm ||d||,
  (c) the injection/residual ratio ||d||/||h||.
The naive pipeline injects the raw direction, so if ||d||/||h|| grows with scale the raw effect
grows spuriously even though steerability does not. Writes results/norms.json + paper/figures/fig_norms.pdf."""
import json, gc, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlib
from mlib import load, layer_residuals, residual_norm, n_layers
from exp01_concept import NEUTRAL, PROBES, GRID
from make_steer_cv2 import S, CONC

LADDER = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]
path = {n: p for n, s, p in GRID}; size = {n: s for n, s, p in GRID}
import os
out = {}
for name in LADDER:
    if not os.path.exists(path[name]): print("skip", name); continue
    tok, model = load(path[name])
    Lc = round(0.8 * n_layers(model))
    nmean = np.stack([layer_residuals(tok, model, t, Lc) for t in NEUTRAL]).mean(0)
    rn = residual_norm(tok, model, PROBES[:3], Lc)                       # ||h||
    dn = [float(np.linalg.norm(np.stack([layer_residuals(tok, model, t, Lc) for t in S[c][1]]).mean(0) - nmean))
          for c in CONC]
    dm = float(np.mean(dn))                                              # mean ||d||
    out[name] = {"size": size[name], "resid_norm": rn, "dir_norm": dm, "ratio": dm / rn}
    print(f"{name}: ||h||={rn:.1f}  ||d||={dm:.1f}  ratio={dm/rn:.3f}", flush=True)
    del model, tok; mlib._CACHE.clear(); gc.collect(); torch.cuda.empty_cache()
json.dump(out, open("results/norms.json", "w"), indent=2)
print("DONE; wrote results/norms.json")
