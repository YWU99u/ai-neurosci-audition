"""Magnitude control: remove the selection bias in exp04.

exp04 picks neurons by |linear correlation with number|, which by construction favours
MONOTONIC neurons, so "bell fraction = 0" is nearly circular. Here we select neurons by a
SHAPE-AGNOSTIC tuning measure -- held-out R^2 of a *quadratic* fit (a bell/parabola fits a
quadratic as well as a line does) -- and only THEN classify each selected neuron as monotonic
vs bell. We report both selections side by side: if the shape-agnostic selection also yields
~0 bell-shaped neurons, the monotonic-code conclusion is real, not a selection artifact.
"""
import os, json, gc, numpy as np, torch, mlib
from mlib import load, record_batch
from exp01_concept import GRID, available, FAMILY

NUMS = list(range(1, 41)); X = np.array(NUMS, float); N = len(NUMS)
K = 30                                                  # neurons to select
TR = np.arange(0, N, 2); TE = np.arange(1, N, 2)        # even/odd held-out split

def _r2(Xtr, Xte, Ytr, Yte):                            # vectorised held-out R^2 for all neurons
    B = np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]
    sse = ((Yte - Xte @ B) ** 2).sum(0)
    sst = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-9
    return 1 - sse / sst

def analyze(A):                                         # A: [40, U]
    lin = lambda x: np.stack([np.ones_like(x), x], 1)
    quad = lambda x: np.stack([np.ones_like(x), x, x ** 2], 1)
    qr = _r2(quad(X[TR]), quad(X[TE]), A[TR], A[TE])     # SHAPE-AGNOSTIC tuning strength
    lr = _r2(lin(X[TR]),  lin(X[TE]),  A[TR], A[TE])     # linear tuning strength
    Xc = X - X.mean(); Ac = A - A.mean(0)
    rlin = (Ac * Xc[:, None]).sum(0) / (np.sqrt((Ac ** 2).sum(0) * (Xc ** 2).sum()) + 1e-9)  # |linear r|
    # full-data quad fit for shape classification
    Bq = np.linalg.lstsq(quad(X), A, rcond=None)[0]      # [3,U] -> a,b,c
    b, c = Bq[1], Bq[2]
    vertex = np.where(np.abs(c) > 1e-9, -b / (2 * c + 1e-12), 1e9)
    qfull = _r2(quad(X), quad(X), A, A); lfull = _r2(lin(X), lin(X), A, A)
    is_bell = (qfull - lfull > 0.15) & (vertex >= 5) & (vertex <= 36) & (qfull > 0.5) & (Bq[2] < 0)  # interior PEAK only (negative quadratic coef)
    old_sel = np.argsort(-np.abs(rlin))[:K]             # exp04's biased selection
    new_sel = np.argsort(-qr)[:K]                       # shape-agnostic selection
    overlap = len(set(old_sel) & set(new_sel)) / K
    return {"old_bell_frac": float(is_bell[old_sel].mean()),
            "new_bell_frac": float(is_bell[new_sel].mean()),
            "new_sel_qr_mean": float(qr[new_sel].mean()),
            "n_bell_anywhere": int(is_bell[qr > 0.7].sum()),   # bell neurons among ALL well-tuned
            "n_welltuned": int((qr > 0.7).sum()), "overlap_old_new": float(overlap)}

def main():
    out = json.load(open("results/number_control.json")) if os.path.exists("results/number_control.json") else {}
    for name, size, path in [g for g in GRID if available(g[2])]:
        print(f"==== {name} ====", flush=True)
        try:
            tok, model = load(path)
            A = record_batch(tok, model, [f"The number is {n}" for n in NUMS], reduce="last")
            A = A.reshape(N, -1).astype(np.float64)
            r = analyze(A); r["size"] = size; r["family"] = FAMILY(name); out[name] = r
            json.dump(out, open("results/number_control.json", "w"), indent=2)
            print(f"  old_bell={r['old_bell_frac']:.2f}  NEW_bell={r['new_bell_frac']:.2f}  "
                  f"bell among all well-tuned: {r['n_bell_anywhere']}/{r['n_welltuned']}", flush=True)
            del model, tok
        except Exception as e:
            import traceback; print("FAIL", e); traceback.print_exc()
        mlib._CACHE.clear(); gc.collect(); torch.cuda.empty_cache()
    print("DONE")

if __name__ == "__main__":
    main()
