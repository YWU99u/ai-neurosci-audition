"""Robustness of the lesion result to the attribution method (reviewer #6).

exp03 selects language neurons by gradient x activation. Here we select by a genuinely different
criterion --- ACTIVATION MAGNITUDE difference (which neurons respond more to one language) ---
and re-run the same ablation + bootstrap. If Chinese stays significantly localizable and English
stays mostly non-significant, the asymmetry is not an artifact of one attribution method.
"""
import os, json, gc, numpy as np, torch, mlib
from mlib import load, record_batch
import make_lesion_control as lcm
from exp01_concept import FAMILY

def act_attr(tok, model, texts):                     # mean over texts of max-token activation, per neuron
    A = record_batch(tok, model, texts, reduce="max")     # [N, L, U]
    return A.mean(0)                                       # [L, U]

def run_alt(tok, model):
    az = act_attr(tok, model, lcm.ZH_TR); ae = act_attr(tok, model, lcm.EN_TR)
    L, U = az.shape; K = max(50, int(lcm.KFRAC * L * U)); diff = (az - ae).reshape(-1)
    zh = [divmod(int(i), U) for i in np.argsort(diff)[::-1][:K]]
    en = [divmod(int(i), U) for i in np.argsort(diff)[:K]]
    dzh_aZH = lcm.per_sent_dnll(tok, model, lcm.ZH_TE, zh); den_aZH = lcm.per_sent_dnll(tok, model, lcm.EN_TE, zh)
    den_aEN = lcm.per_sent_dnll(tok, model, lcm.EN_TE, en); dzh_aEN = lcm.per_sent_dnll(tok, model, lcm.ZH_TE, en)
    rng = np.random.default_rng(0)
    def boot(a, b):
        v = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(3000)]
        return float(np.mean(v)), [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    zs, zci = boot(dzh_aZH, den_aZH); es, eci = boot(den_aEN, dzh_aEN)
    return {"zh_sel": zs, "zh_ci": zci, "en_sel": es, "en_ci": eci,
            "zh_sig": bool(zci[0] > 0), "en_sig": bool(eci[0] > 0)}

def main():
    out = json.load(open("results/lesion_altattr.json")) if os.path.exists("results/lesion_altattr.json") else {}
    for name, size, path in lcm.MODELS:
        if not os.path.exists(path): print("skip", name); continue
        print(f"==== {name} ====", flush=True)
        try:
            tok, model = load(path); r = run_alt(tok, model); r["size"] = size; r["family"] = FAMILY(name)
            out[name] = r; json.dump(out, open("results/lesion_altattr.json", "w"), indent=2)
            print(f"  zh_sel={r['zh_sel']:+.2f} sig={r['zh_sig']} | en_sel={r['en_sel']:+.2f} sig={r['en_sig']}", flush=True)
            del model, tok
        except Exception as e:
            import traceback; print("FAIL", e); traceback.print_exc()
        mlib._CACHE.clear(); gc.collect(); torch.cuda.empty_cache()
    print("DONE")

if __name__ == "__main__":
    main()
