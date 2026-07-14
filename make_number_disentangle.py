"""Decisive control for magnitude SHAPE (= reviewers' #1 ask = Direction-B Step 1).

The bell-shaped 'number neurons' we found could be genuine numerosity tuning OR just digit-token
identity. Gold-standard test (as in neuroscience): CROSS-FORMAT CONSISTENCY. Present each number
as a DIGIT ('17') and SPELLED-OUT WORD ('seventeen'); a genuine number neuron keeps the same
tuning curve across formats, a digit-token detector does not.

For neurons classed bell in the digit format, we report the correlation of their digit-format
tuning curve with their word-format tuning curve. High -> genuine numerosity. Low -> token artifact.
"""
import os, json, gc, numpy as np, torch, mlib
from mlib import load, record_batch
from exp01_concept import GRID, available, FAMILY

NUMS = list(range(1, 41)); X = np.array(NUMS, float); N = len(NUMS)
TR = np.arange(0, N, 2); TE = np.arange(1, N, 2)
_ONES = ["", "one","two","three","four","five","six","seven","eight","nine","ten","eleven",
         "twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
_TENS = {20: "twenty", 30: "thirty", 40: "forty"}
def n2w(n):
    if n < 20: return _ONES[n]
    t, o = (n // 10) * 10, n % 10
    return _TENS[t] if o == 0 else f"{_TENS[t]}-{_ONES[o]}"

def _r2(Xtr, Xte, Ytr, Yte):
    B = np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]
    sse = ((Yte - Xte @ B) ** 2).sum(0); sst = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-9
    return 1 - sse / sst

def bell_mask(A):                                    # shape-agnostic well-tuned bell (interior peak/trough)
    quad = lambda x: np.stack([np.ones_like(x), x, x ** 2], 1); lin = lambda x: np.stack([np.ones_like(x), x], 1)
    qr = _r2(quad(X[TR]), quad(X[TE]), A[TR], A[TE])
    B = np.linalg.lstsq(quad(X), A, rcond=None)[0]; b, c = B[1], B[2]
    vertex = np.where(np.abs(c) > 1e-9, -b / (2 * c + 1e-12), 1e9)
    qf = _r2(quad(X), quad(X), A, A); lf = _r2(lin(X), lin(X), A, A)
    return (qf - lf > 0.15) & (vertex >= 5) & (vertex <= 36) & (qf > 0.5), qr

def consistency(Ad, Aw):                             # per-neuron Pearson r between digit & word tuning curves
    a = Ad - Ad.mean(0); b = Aw - Aw.mean(0)
    return (a * b).sum(0) / (np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + 1e-9)

def main():
    out = json.load(open("results/number_disentangle.json")) if os.path.exists("results/number_disentangle.json") else {}
    for name, size, path in [g for g in GRID if available(g[2])]:
        print(f"==== {name} ====", flush=True)
        try:
            tok, model = load(path)
            Ad = record_batch(tok, model, [f"The number is {n}" for n in NUMS], reduce="last").reshape(N, -1).astype(np.float64)
            Aw = record_batch(tok, model, [f"The number is {n2w(n)}" for n in NUMS], reduce="last").reshape(N, -1).astype(np.float64)
            bell_d, qr_d = bell_mask(Ad)              # bell neurons in DIGIT format
            cons = consistency(Ad, Aw)                # cross-format consistency
            top_bell = np.argsort(-np.where(bell_d, qr_d, -1))[:30]     # 30 strongest bell (digit)
            allwt = qr_d > 0.7
            r = {"size": size, "family": FAMILY(name),
                 "bell_consistency_mean": float(np.mean(cons[top_bell])),
                 "bell_consistency_frac_high": float(np.mean(cons[top_bell] > 0.5)),   # kept across format
                 "welltuned_consistency_mean": float(np.mean(cons[allwt])) if allwt.any() else 0.0,
                 "random_consistency_mean": float(np.mean(np.random.default_rng(0).choice(cons, 500)))}
            out[name] = r; json.dump(out, open("results/number_disentangle.json", "w"), indent=2)
            print(f"  bell cross-format r={r['bell_consistency_mean']:+.2f}  frac(r>0.5)={r['bell_consistency_frac_high']:.2f}  "
                  f"(well-tuned {r['welltuned_consistency_mean']:+.2f}, random {r['random_consistency_mean']:+.2f})", flush=True)
            del model, tok
        except Exception as e:
            import traceback; print("FAIL", e); traceback.print_exc()
        mlib._CACHE.clear(); gc.collect(); torch.cuda.empty_cache()
    print("DONE")

if __name__ == "__main__":
    main()
