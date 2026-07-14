"""2×2 factor decomposition (W4.1 reviewer response).
Isolates the contribution of (a) unit normalization and (b) operating-point selection
to the apparent scaling law, using the SAME metric and SAME 24 concepts for all 4 cells.

Factor 1 — Units:
  raw:  inject d_raw (unnormalized mean-diff) at scalar alpha
  norm: inject d/||d|| at alpha = coef × residual_norm

Factor 2 — Operating point:
  fixed:   layer 2/3, single coef (raw=1.0; norm=1.0)
  heldout: grid search (3 layers × 5 coefs), 4-fold concept CV

Metric for ALL cells: specificity contrast = log P(target_words) − log P(control_words),
via full-sequence log-prob (mlib.target_score). Same probes, same concepts, same controls.

Output: results/factor_2x2.json + paper/figures/fig_factor.pdf
"""
import os, json, gc, contextlib, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlib
from mlib import load, layer_residuals, residual_norm, steer_residual, target_score, unit, n_layers
from exp01_concept import NEUTRAL, PROBES as ALLPROBES, GRID, available
from make_steer_cv2 import S, CONC

PROBES = ALLPROBES[:3]
LFRAC = [0.4, 0.6, 0.8]; COEFS = [0.25, 0.5, 1.0, 2.0, 4.0]
MODELS = [("Qwen3-0.6B",0.6),("Qwen3-1.7B",1.7),("Qwen3-4B",4.0),("Qwen3-8B",8.0),("Qwen3-14B",14.0)]

def contrast(tok, model, Lc, d, alpha, tw, cw):
    """Specificity contrast under steering with direction d at strength alpha."""
    v = []
    for p in PROBES:
        ctx = steer_residual(model, Lc, d, alpha) if alpha else contextlib.nullcontext()
        with ctx: v.append(target_score(tok, model, p, tw) - target_score(tok, model, p, cw))
    return float(np.nanmean(v))

def run_cell(tok, model, Lc, dirs, rn, unit_type, coef_val):
    """Run all 24 concepts at a single (layer, coef) with given unit type.
    Returns {concept: effect}."""
    out = {}
    for c in CONC:
        if unit_type == "raw":
            alpha = coef_val                            # raw scalar
            d = dirs[c]["raw"]                           # unnormalized
        else:
            alpha = coef_val * rn                        # fraction of residual norm
            d = dirs[c]["norm"]                           # unit-normalized
        tw, cw = S[c][2], S[S[c][3]][2]
        et = contrast(tok, model, Lc, d, alpha, tw, cw)
        out[c] = et
    return out

def cv_heldout(cells):
    """4-fold concept CV: pick best (layer,coef) on select folds, record held-out effect."""
    rng = np.random.default_rng(0); idx = rng.permutation(len(CONC)); folds = np.array_split(idx, 4)
    ho_eff = {}
    for f in range(4):
        ev = [CONC[i] for i in folds[f]]; sel = [CONC[i] for i in idx if CONC[i] not in ev]
        best = max(cells, key=lambda k: np.mean([cells[k][c] for c in sel]))
        for c in ev:
            ho_eff[c] = cells[best][c]
    return ho_eff

def trend_slope(sizes, effects):
    """Bootstrap slope of effect vs log2(size)."""
    x = np.log2(sizes); y = np.array(effects)
    slope = np.polyfit(x, y, 1)[0]
    rng = np.random.default_rng(42); boots = []
    for _ in range(2000):
        idx = rng.choice(len(x), len(x))
        boots.append(np.polyfit(x[idx], y[idx], 1)[0])
    ci = np.percentile(boots, [2.5, 97.5])
    return float(slope), [float(ci[0]), float(ci[1])]

out = {}
for name, sz in MODELS:
    path = [g[2] for g in GRID if g[0] == name]
    if not path or not available(path[0]):
        print(f"skip {name}"); continue
    print(f"\n==== {name} ====", flush=True)
    tok, model = load(path[0])
    L = n_layers(model)

    # build directions (both raw and normalized) at each layer fraction
    layer_dirs = {}
    for lf in LFRAC + [2/3]:
        Lc = max(1, min(L - 1, int(round(L * lf))))
        nmean = np.stack([layer_residuals(tok, model, t, Lc) for t in NEUTRAL]).mean(0)
        rn = residual_norm(tok, model, PROBES, Lc)
        d = {}
        for c in CONC:
            raw = np.stack([layer_residuals(tok, model, t, Lc) for t in S[c][1]]).mean(0) - nmean
            d[c] = {"raw": raw, "norm": unit(raw)}
        layer_dirs[lf] = {"dirs": d, "rn": rn, "Lc": Lc}

    # --- Cell A: raw + fixed (L·2/3, coef=1.0) ---
    ld = layer_dirs[2/3]
    cell_a = run_cell(tok, model, ld["Lc"], ld["dirs"], ld["rn"], "raw", 1.0)
    print(f"  A (raw+fixed):  mean={np.mean(list(cell_a.values())):.2f}", flush=True)

    # --- Cell C: norm + fixed (L·2/3, coef=1.0) ---
    cell_c = run_cell(tok, model, ld["Lc"], ld["dirs"], ld["rn"], "norm", 1.0)
    print(f"  C (norm+fixed): mean={np.mean(list(cell_c.values())):.2f}", flush=True)

    # --- Cell B: raw + held-out ---
    cells_raw = {}
    for lf in LFRAC:
        ld2 = layer_dirs[lf]
        for coef in COEFS:
            key = f"{lf}|{coef}"
            cells_raw[key] = run_cell(tok, model, ld2["Lc"], ld2["dirs"], ld2["rn"], "raw", coef)
    ho_raw = cv_heldout(cells_raw)
    print(f"  B (raw+HO):     mean={np.mean(list(ho_raw.values())):.2f}", flush=True)

    # --- Cell D: norm + held-out ---
    cells_norm = {}
    for lf in LFRAC:
        ld2 = layer_dirs[lf]
        for coef in COEFS:
            key = f"{lf}|{coef}"
            cells_norm[key] = run_cell(tok, model, ld2["Lc"], ld2["dirs"], ld2["rn"], "norm", coef)
    ho_norm = cv_heldout(cells_norm)
    print(f"  D (norm+HO):    mean={np.mean(list(ho_norm.values())):.2f}", flush=True)

    out[name] = {
        "size": sz,
        "cell_A_raw_fixed": float(np.mean(list(cell_a.values()))),
        "cell_B_raw_heldout": float(np.mean(list(ho_raw.values()))),
        "cell_C_norm_fixed": float(np.mean(list(cell_c.values()))),
        "cell_D_norm_heldout": float(np.mean(list(ho_norm.values()))),
    }
    del model, tok; mlib._CACHE.clear(); gc.collect(); torch.cuda.empty_cache()

json.dump(out, open("results/factor_2x2.json", "w"), indent=2)

# --- trend slopes ---
sizes = np.array([out[n]["size"] for n in sorted(out, key=lambda k: out[k]["size"])])
for cell in ["cell_A_raw_fixed", "cell_B_raw_heldout", "cell_C_norm_fixed", "cell_D_norm_heldout"]:
    effs = np.array([out[n][cell] for n in sorted(out, key=lambda k: out[k]["size"])])
    sl, ci = trend_slope(sizes, effs)
    print(f"  {cell}: slope={sl:+.3f} CI=[{ci[0]:+.3f},{ci[1]:+.3f}]")

# --- figure ---
FAM = {"Qwen": "#B23A5E"}
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw={"wspace": 0.36})
labels = {"cell_A_raw_fixed": "raw units", "cell_C_norm_fixed": "normalized"}
for i, (cell, lab) in enumerate(labels.items()):
    effs = [out[n][cell] for n in sorted(out, key=lambda k: out[k]["size"])]
    ax[0].plot(sizes, effs, "-o", ms=5, label=lab)
ax[0].set_xscale("log"); ax[0].set_xlabel("model size (B)"); ax[0].set_ylabel("specificity contrast")
ax[0].set_title("Fixed operating point (L·2/3)", loc="left", fontsize=9); ax[0].legend(frameon=False, fontsize=7.5)

labels2 = {"cell_B_raw_heldout": "raw units", "cell_D_norm_heldout": "normalized"}
for i, (cell, lab) in enumerate(labels2.items()):
    effs = [out[n][cell] for n in sorted(out, key=lambda k: out[k]["size"])]
    ax[1].plot(sizes, effs, "-o", ms=5, label=lab)
ax[1].set_xscale("log"); ax[1].set_xlabel("model size (B)"); ax[1].set_ylabel("specificity contrast")
ax[1].set_title("Held-out operating point", loc="left", fontsize=9); ax[1].legend(frameon=False, fontsize=7.5)

plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight", "figure.dpi": 220,
                     "axes.spines.top": False, "axes.spines.right": False})
fig.savefig("paper/figures/fig_factor.pdf"); fig.savefig("paper/figures/fig_factor.png")
print("\nDONE; wrote results/factor_2x2.json + paper/figures/fig_factor.pdf")
