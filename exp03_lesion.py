"""EXP 03 (canonical) — Lesion & language double-dissociation across the model grid.

Question: is language processing spatially specialized, and is the "dominant language
is harder to lesion" asymmetry a general (cross-family) property?

Rigor (post-audit):
  - neurons selected by CAUSAL IMPORTANCE (gradient x activation attribution of the
    per-language LM loss), NOT mean-activation difference.                    [C3]
  - K as a FRACTION of neurons (comparable across models).                    [A3]
  - ablate -> measure held-out NLL change; random-ablation baseline.
  - report both directions (double dissociation) + a mean-diff selection for contrast.
"""
import os, json, gc, numpy as np, torch, torch.nn.functional as F
import mlib
from mlib import load, text_nll, edited_neurons, record_batch
from exp01_concept import GRID, available, FAMILY

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ZH_TR = ["今天天气很好,我们去公园散步吧。", "他昨天买了一本新书。", "这道菜的味道非常鲜美。",
         "北京是中国的首都,历史悠久。", "孩子们在操场上快乐地玩耍。", "请把窗户关上,外面有点冷。",
         "这部电影讲述了一个感人的故事。", "我每天早上都会喝一杯茶。"]
ZH_TE = ["她正在认真地准备明天的考试。", "这座桥是三年前建成的。", "我们计划暑假去南方旅行。",
         "厨房里飘来了饭菜的香味。", "老师耐心地讲解了这道题。"]
EN_TR = ["The weather is nice today, let's walk in the park.", "He bought a new book yesterday.",
         "This dish tastes absolutely delicious.", "Beijing is the capital of China.",
         "The children played happily on the playground.", "Please close the window, it's cold outside.",
         "The film tells a very moving story.", "I drink a cup of tea every morning."]
EN_TE = ["She is carefully preparing for tomorrow's exam.", "This bridge was built three years ago.",
         "We plan to travel south this summer.", "The smell of cooking drifted from the kitchen.",
         "The teacher patiently explained the problem."]

def attribution(model, tok, texts):
    """Causal importance per neuron = mean_t |act * dNLL/dact| (grad x activation).
    Memory-efficient: freeze all weights so backward allocates NO weight-gradient buffers
    (~= model size saved -> lets 24-72B fit); we only need gradients w.r.t. the activations,
    obtained by marking each captured activation requires_grad. Values are identical to the
    naive version (d NLL / d act does not depend on whether weights require grad)."""
    for p in model.parameters(): p.requires_grad_(False)          # <- no weight-grad buffers
    layers = mlib._layers(model); L = len(layers); total = None
    for t in texts:
        caps = {}
        def mk(i):
            def pre(mod, args):
                a = args[0]; a.requires_grad_(True); a.retain_grad(); caps[i] = a
            return pre
        hs = [layers[i].mlp.down_proj.register_forward_pre_hook(mk(i)) for i in range(L)]
        ids = tok(t, return_tensors="pt").to(model.device)["input_ids"]
        logits = model(ids).logits[0]
        logp = F.log_softmax(logits.float(), -1)
        nll = -logp[torch.arange(ids.shape[1] - 1), ids[0, 1:]].mean()
        model.zero_grad(set_to_none=True); nll.backward()
        for h in hs: h.remove()
        attr = np.stack([(caps[i][0].detach().float() * caps[i].grad[0].float()).abs().mean(0).cpu().numpy()
                         if caps[i].grad is not None else np.zeros(caps[i].shape[-1]) for i in range(L)])
        total = attr if total is None else total + attr
    return total / len(texts)

def mean_nll(tok, model, texts, abl=None):
    if abl is None:
        return float(np.mean([text_nll(tok, model, t) for t in texts]))
    with edited_neurons(model, ablate=abl):
        return float(np.mean([text_nll(tok, model, t) for t in texts]))

def run(tok, model):
    az = attribution(model, tok, ZH_TR); ae = attribution(model, tok, EN_TR)
    L, U = az.shape; K = max(50, int(0.002 * L * U))
    diff = (az - ae).reshape(-1)
    zh = [divmod(int(i), U) for i in np.argsort(diff)[::-1][:K]]     # causally Chinese
    en = [divmod(int(i), U) for i in np.argsort(diff)[:K]]           # causally English
    rng = np.random.default_rng(0)
    rnd = [divmod(int(i), U) for i in rng.choice(L * U, K, replace=False)]
    bz, be = mean_nll(tok, model, ZH_TE), mean_nll(tok, model, EN_TE)
    def sel(abl):
        return (mean_nll(tok, model, ZH_TE, abl) - bz), (mean_nll(tok, model, EN_TE, abl) - be)
    zz, ze = sel(zh); ez, ee = sel(en); rz, re = sel(rnd)
    return {"K": K, "K_frac": K / (L * U),
            "ablate_zh": {"dnll_zh": zz, "dnll_en": ze},
            "ablate_en": {"dnll_zh": ez, "dnll_en": ee},
            "ablate_random": {"dnll_zh": rz, "dnll_en": re},
            "zh_selectivity": zz - ze, "en_selectivity": ee - ez,
            "double_dissociation": bool(zz - ze > 0.1 and ee - ez > 0.1)}

def main():
    runnable = [(n, s, p) for n, s, p in GRID if available(p)]
    print("[lesion] models:", [n for n, _, _ in runnable] or "NONE")
    res = {}
    for name, size, path in runnable:
        try:
            tok, model = load(path)
        except Exception as e:
            print(f"{name}: load failed {e}"); continue
        try:
            r = run(tok, model)
        except Exception as e:
            import traceback; print(f"{name}: run failed {e}"); traceback.print_exc()
            mlib._CACHE.clear(); del model, tok; gc.collect(); torch.cuda.empty_cache(); continue
        r["size"] = size; r["family"] = FAMILY(name); res[name] = r
        print(f"{name:14s} sz={size:<4} zh_sel={r['zh_selectivity']:+.2f} en_sel={r['en_selectivity']:+.2f} "
              f"rand(zh/en)={r['ablate_random']['dnll_zh']:+.2f}/{r['ablate_random']['dnll_en']:+.2f} "
              f"dbl={r['double_dissociation']}")
        mlib._CACHE.clear(); del model, tok; gc.collect(); torch.cuda.empty_cache()
    if res:
        json.dump(res, open(os.path.join(RES, "lesion.json"), "w"), ensure_ascii=False, indent=2)
        print("[lesion] saved results/lesion.json")

if __name__ == "__main__":
    main()
