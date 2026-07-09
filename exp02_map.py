"""EXP 02 (canonical) — Cognitive map (geographic space) across the model grid.

Question: is a metric, linear world-map universal, or does its fidelity depend on
scale / family?

Rigor (post-audit):
  - residual taken at the CITY-NAME token (prompt ends with the city), not a period. [B1]
  - Ridge alpha chosen by cross-validation per layer; best layer per model.          [A3]
  - held-out KFold R^2; shuffled-label null; RSA of activation vs geographic distance.
"""
import os, json, gc, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
import mlib
from mlib import load, last_token_residuals
from exp01_concept import GRID, available, FAMILY

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); RES = os.path.join(HERE, "results")
CITIES = [
 ("London",51,0),("Paris",49,2),("Berlin",52,13),("Madrid",40,-4),("Rome",42,12),("Moscow",56,38),
 ("Istanbul",41,29),("Cairo",30,31),("Lagos",6,3),("Nairobi",-1,37),("Cape Town",-34,18),("Dubai",25,55),
 ("Mumbai",19,73),("Delhi",29,77),("Bangkok",14,100),("Beijing",40,116),("Shanghai",31,121),("Tokyo",36,140),
 ("Seoul",37,127),("Jakarta",-6,107),("Sydney",-34,151),("Melbourne",-38,145),("Auckland",-37,175),
 ("Singapore",1,104),("Manila",15,121),("New York",41,-74),("Los Angeles",34,-118),("Chicago",42,-88),
 ("Toronto",44,-79),("Mexico City",19,-99),("Bogota",5,-74),("Lima",-12,-77),("Santiago",-33,-71),
 ("Buenos Aires",-35,-58),("Rio de Janeiro",-23,-43),("Vancouver",49,-123),("Stockholm",59,18),("Athens",38,24),
 ("Lisbon",39,-9),("Dublin",53,-6),("Warsaw",52,21),("Tehran",36,51),("Karachi",25,67),("Hong Kong",22,114),
 ("Johannesburg",-26,28),
]

def run(tok, model):
    names = [c[0] for c in CITIES]; Y = np.array([[c[1], c[2]] for c in CITIES], float)
    reps = np.stack([last_token_residuals(tok, model, f"The city of {n}") for n in names])  # [N, L+1, H]
    Ln = reps.shape[1]
    def cv_r2(X, Yt, alpha):
        kf = KFold(5, shuffle=True, random_state=0); pr = np.zeros_like(Yt)
        for tr, te in kf.split(X):
            pr[te] = Ridge(alpha=alpha).fit(X[tr], Yt[tr]).predict(X[te])
        ssr = ((Yt - pr) ** 2).sum(0); sst = ((Yt - Yt.mean(0)) ** 2).sum(0)
        return 1 - ssr / sst
    best = (-9, None, None)  # (score, layer, alpha)
    for l in range(Ln):
        for a in (10.0, 100.0, 1000.0):
            r2 = cv_r2(reps[:, l, :], Y, a)
            if r2.sum() > best[0]:
                best = (r2.sum(), l, a)
    _, bl, ba = best; r2b = cv_r2(reps[:, bl, :], Y, ba)
    # shuffle null
    rng = np.random.default_rng(0); nulls = []
    for _ in range(100):
        nulls.append(cv_r2(reps[:, bl, :], Y[rng.permutation(len(Y))], ba).mean())
    null_p95 = float(np.percentile(nulls, 95))
    # RSA: activation cosine distance vs great-circle-ish geo distance
    ad = pdist(reps[:, bl, :], "cosine"); gd = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            gd.append(np.hypot(Y[i,0]-Y[j,0], min(abs(Y[i,1]-Y[j,1]), 360-abs(Y[i,1]-Y[j,1]))))
    rsa = float(pearsonr(gd, ad)[0])
    return {"best_layer": bl, "alpha": ba, "r2_lat": float(r2b[0]), "r2_lon": float(r2b[1]),
            "null_p95": null_p95, "rsa_r": rsa}

def main():
    runnable = [(n, s, p) for n, s, p in GRID if available(p)]
    print("[map] models:", [n for n, _, _ in runnable] or "NONE")
    res = {}
    for name, size, path in runnable:
        try:
            tok, model = load(path)
        except Exception as e:
            print(f"{name}: load failed {e}"); continue
        r = run(tok, model); r["size"] = size; r["family"] = FAMILY(name); res[name] = r
        print(f"{name:14s} sz={size:<4} R2 lat={r['r2_lat']:.2f} lon={r['r2_lon']:.2f} "
              f"(null {r['null_p95']:+.2f})  RSA={r['rsa_r']:.2f}  @L{r['best_layer']}")
        mlib._CACHE.clear(); del model, tok; gc.collect(); torch.cuda.empty_cache()
    if res:
        json.dump(res, open(os.path.join(RES, "cognitive_map.json"), "w"), ensure_ascii=False, indent=2)
        print("[map] saved results/cognitive_map.json")

if __name__ == "__main__":
    main()
