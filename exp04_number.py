"""EXP 04 (canonical) — Number-magnitude tuning across the model grid.

Question: do single neurons tune to number magnitude, and is the code MONOTONIC
(ramp, temperature-gauge style) or BELL-SHAPED (a preferred number, like biological
'number neurons', Nieder)? Monotonic-everywhere would be a real AI-vs-brain divergence.

Rigor (post-audit):
  - take the activation at the DIGIT token (reduce='last' on a prompt ending in the
    number), NOT a mean over the template.                                   [B2]
  - neuron chosen on TRAIN numbers, correlation reported on held-out TEST.    [R1]
  - shuffle-label null; 5 seeds -> mean +/- std.                              [D1/D2]
  - monotonic vs bell: compare linear vs quadratic fit R^2 for top neurons.
"""
import os, json, gc, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import mlib
from mlib import load, record_batch, n_layers
from exp01_concept import GRID, available, FAMILY

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); RES = os.path.join(HERE, "results")
NUMS = list(range(1, 41))

def run(tok, model):
    # digit-token activation: prompt ends in the number, take last token
    A = record_batch(tok, model, [f"The number is {n}" for n in NUMS], reduce="last")
    N, L, U = A.shape; Af = A.reshape(N, L * U); X = np.array(NUMS, float)
    best_rs, nulls = [], []
    for seed in range(5):
        rng = np.random.default_rng(seed); idx = rng.permutation(N); tr, te = idx[:N // 2], idx[N // 2:]
        xtr = X[tr] - X[tr].mean(); atr = Af[tr] - Af[tr].mean(0)
        r = (atr * xtr[:, None]).sum(0) / (np.sqrt((atr ** 2).sum(0) * (xtr ** 2).sum()) + 1e-8)
        fb = int(np.argmax(np.abs(r)))
        best_rs.append(abs(float(pearsonr(Af[te, fb], X[te])[0])))
        # shuffle null (best |r| under permuted labels)
        Xs = X[rng.permutation(N)]; xt = Xs[tr] - Xs[tr].mean()
        rr = (atr * xt[:, None]).sum(0) / (np.sqrt((atr ** 2).sum(0) * (xt ** 2).sum()) + 1e-8)
        b = int(np.argmax(np.abs(rr))); nulls.append(abs(float(pearsonr(Af[te, b], Xs[te])[0])))
    # monotonic vs bell for the overall top-8 |corr| neurons (full data)
    xc = X - X.mean(); ac = Af - Af.mean(0)
    rfull = (ac * xc[:, None]).sum(0) / (np.sqrt((ac ** 2).sum(0) * (xc ** 2).sum()) + 1e-8)
    Xd = np.stack([np.ones(N), X, X ** 2], 1)
    def r2(y, cols):
        b, *_ = np.linalg.lstsq(Xd[:, cols], y, rcond=None); pred = Xd[:, cols] @ b
        return 1 - ((y - pred) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-9)
    bell = 0
    for f in np.argsort(-np.abs(rfull))[:8]:
        y = Af[:, f]; lin = r2(y, [0, 1]); quad = r2(y, [0, 1, 2])
        if quad - lin > 0.15:  # quadratic adds a lot -> bell-ish
            bell += 1
    return {"best_r_mean": float(np.mean(best_rs)), "best_r_std": float(np.std(best_rs)),
            "null_mean": float(np.mean(nulls)), "bell_fraction": bell / 8.0}

def main():
    runnable = [(n, s, p) for n, s, p in GRID if available(p)]
    print("[number] models:", [n for n, _, _ in runnable] or "NONE")
    res = {}
    for name, size, path in runnable:
        try:
            tok, model = load(path)
        except Exception as e:
            print(f"{name}: load failed {e}"); continue
        r = run(tok, model); r["size"] = size; r["family"] = FAMILY(name); res[name] = r
        print(f"{name:14s} sz={size:<4} best_r={r['best_r_mean']:.2f}±{r['best_r_std']:.2f} "
              f"(null {r['null_mean']:.2f})  bell={r['bell_fraction']:.2f}")
        mlib._CACHE.clear(); del model, tok; gc.collect(); torch.cuda.empty_cache()
    if res:
        json.dump(res, open(os.path.join(RES, "number_tuning.json"), "w"), ensure_ascii=False, indent=2)
        print("[number] saved results/number_tuning.json")

if __name__ == "__main__":
    main()
