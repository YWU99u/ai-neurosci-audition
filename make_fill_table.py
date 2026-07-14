"""Fill the per-model table: run held-out steering (fine grid) on the six non-Qwen <=14B models
and lesion on the two tiny Llamas that were never run, merge into results/. Real data."""
import json, numpy as np, gc, torch, mlib
from mlib import load
import make_steer_cv2 as scv
import make_lesion_control as lc
from exp01_concept import GRID, available, FAMILY

STEER = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B", "Phi-3.5-mini", "phi-4", "Ministral-8B"]
LESION = ["Llama-3.2-1B", "Llama-3.2-3B"]
path = {n: p for n, s, p in GRID}; size = {n: s for n, s, p in GRID}

def bootci(x, n=5000):
    rng = np.random.default_rng(0)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

def merge(fn, name, rec):
    d = json.load(open(f"results/{fn}.json")); d[name] = rec
    json.dump(d, open(f"results/{fn}.json", "w"), ensure_ascii=False, indent=2)

for name in STEER:
    if not available(path[name]): print("skip (missing)", name, flush=True); continue
    print("==== " + name + " ====", flush=True)
    tok, model = load(path[name])
    try:
        cells = scv.cells_for_model(tok, model); he, hp = scv.cv_heldout(cells)
        eff = np.array([he[c] for c in scv.CONC]); pas = np.array([hp[c] for c in scv.CONC])
        merge("steer_cv2", name, {"size": size[name], "family": FAMILY(name),
              "heldout_eff": float(eff.mean()), "heldout_pass": float(pas.mean()),
              "eff_ci": bootci(eff), "pass_ci": bootci(pas),
              "per_concept_eff": {c: he[c] for c in scv.CONC}})
        print(f"  steer eff={eff.mean():+.2f} pass={pas.mean():.2f}", flush=True)
    except Exception as e:
        import traceback; print("  STEER FAIL", e); traceback.print_exc()
    if name in LESION:
        try:
            r = lc.run(tok, model); r["size"] = size[name]; r["family"] = FAMILY(name)
            merge("lesion_control", name, r)
            print(f"  lesion zh={r['zh_sel']:+.2f}({r['zh_sig']}) en={r['en_sel']:+.2f}({r['en_sig']})", flush=True)
        except Exception as e:
            import traceback; print("  LESION FAIL", e); traceback.print_exc()
    del model, tok; mlib._CACHE.clear(); gc.collect(); torch.cuda.empty_cache()
print("DONE")
