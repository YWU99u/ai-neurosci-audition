"""EXP 01 (canonical) — Concept steering across a model grid, rigorously.

Question: does DIRECTION-SPECIFIC causal steerability of concepts emerge with scale,
and is it scale- or family-dependent?

Rigor (post-audit):
  - steering direction UNIT-normalized; strength = fraction of each model's residual
    norm at layer 2/3 (comparable across models).                      [A1]
  - target = sequence log-prob of fixed English word sets (tokenizer-agnostic). [A2]
  - per-concept SPECIFICITY contrast: logP(concept) - logP(paired control).
  - three controls: target direction (want +), CONTROL-concept direction (want -,
    symmetry), RANDOM direction (want ~0).
  - 8 concepts; bootstrap CI over concepts on pass-rate and mean effect.  [D1]
Outputs results/concept_steering.json and figures/concept_steering.png.
"""
import os, json, gc, numpy as np, torch, contextlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlib
from mlib import (load, layer_residuals, residual_norm, steer_residual,
                  target_score, unit, n_layers)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); RES = os.path.join(HERE, "results")
DL = os.path.join(HERE, "models_dl")

# (name, size_B, path). Skipped automatically if weights are absent.
GRID = [
    ("Qwen3-0.6B", 0.6, f"{DL}/Qwen3-0.6B"),
    ("Llama-3.2-1B", 1.0, "/home/user/Public/yide/models/Llama3.2-1B-Instruct"),
    ("Qwen3-1.7B", 1.7, f"{DL}/Qwen3-1.7B"),
    ("Llama-3.2-3B", 3.0, f"{DL}/Llama-3.2-3B-Instruct"),
    ("Phi-3.5-mini", 3.8, f"{DL}/Phi-3.5-mini-instruct"),
    ("Qwen3-4B", 4.0, f"{DL}/Qwen3-4B"),
    ("Qwen3.5-4B", 4.0, f"{DL}/Qwen3.5-4B"),
    ("Mistral-7B", 7.0, f"{DL}/Mistral-7B-Instruct-v0.3"),
    ("Qwen3-8B", 8.0, f"{DL}/Qwen3-8B"),
    ("Ministral-8B", 8.0, f"{DL}/Ministral-8B-Instruct-2410"),
    ("Llama-3.1-8B", 8.0, f"{DL}/Llama-3.1-8B-Instruct"),
    ("phi-4", 14.0, f"{DL}/phi-4"),
    ("Qwen3-14B", 14.0, f"{DL}/Qwen3-14B"),
]
FAMILY = lambda n: ("Qwen" if n.startswith("Qwen") else "Llama" if "Llama" in n
                    else "Phi" if "hi" in n else "Mistral" if "istral" in n
                    else "Gemma" if "emma" in n else "?")

# concept -> sentences to build the direction (name + description, multi-lingual)
CONCEPTS = {
 "Paris": ["Paris is the capital of France.", "巴黎是法国的首都。", "We flew into Paris last summer.",
           "The city where the Eiffel Tower stands.", "The French capital is famous for its cafés."],
 "Tokyo": ["Tokyo is the capital of Japan.", "东京是日本最大的城市。", "We took the bullet train to Tokyo.",
           "The city with Shibuya crossing.", "The Japanese metropolis near Mount Fuji."],
 "Einstein": ["Albert Einstein developed relativity.", "爱因斯坦提出了相对论。",
              "The physicist who wrote E=mc².", "Einstein won the Nobel Prize in physics."],
 "Shakespeare": ["Shakespeare wrote Hamlet and Macbeth.", "莎士比亚写了《哈姆雷特》。",
                 "The Bard of Avon wrote many tragedies.", "The Elizabethan playwright penned the sonnets."],
 "Moon": ["The Moon orbits the Earth.", "今晚的月亮又大又圆。", "Astronauts walked on it in 1969.",
          "The pale disc that causes the ocean tides."],
 "Ocean": ["The ocean covers most of the Earth.", "海洋覆盖了地球大部分表面。",
           "Salt water full of fish and coral.", "Waves crashed on the deep blue ocean."],
 "Piano": ["She played a melody on the piano.", "他在钢琴上弹了一首曲子。",
           "An instrument with 88 black and white keys.", "The pianist played a beautiful sonata."],
 "Dinosaur": ["The Tyrannosaurus was a fearsome dinosaur.", "恐龙在白垩纪灭绝了。",
              "Giant reptiles from millions of years ago.", "Fossils show dinosaurs were enormous."],
}
WORDS = {
 "Paris": ["Paris", "France", "Eiffel", "French"], "Tokyo": ["Tokyo", "Japan", "Shibuya", "Japanese"],
 "Einstein": ["Einstein", "relativity", "physicist", "physics"],
 "Shakespeare": ["Shakespeare", "Hamlet", "playwright", "sonnet"],
 "Moon": ["Moon", "lunar", "orbit", "crater"], "Ocean": ["ocean", "sea", "waves", "marine"],
 "Piano": ["piano", "keys", "melody", "keyboard"], "Dinosaur": ["dinosaur", "fossil", "extinct", "reptile"],
}
CONTROL = {"Paris": "Tokyo", "Tokyo": "Paris", "Einstein": "Shakespeare", "Shakespeare": "Einstein",
           "Moon": "Ocean", "Ocean": "Moon", "Piano": "Dinosaur", "Dinosaur": "Piano"}
NEUTRAL = ["I need to buy groceries tonight.", "The quarterly report is due Friday.", "她昨天去公园散步。",
           "Please format the data into a spreadsheet.", "The match ended in a draw.", "他每天早上跑步。",
           "The printer is out of toner.", "我们下周开会讨论预算。", "Remember to water the plants.",
           "The train was delayed twenty minutes."]
PROBES = ["My favorite thing is", "The best one in the world is", "I keep thinking about",
          "The most remarkable is", "Let me tell you about", "The one I dream of is"]
FRAC = 1.0; N_RAND = 5

def available(p):
    return os.path.isdir(p) and any(f.endswith(".safetensors") for f in os.listdir(p))

def contrast(tok, model, Lc, d, alpha, tw, cw):
    vals = []
    for p in PROBES:
        ctx = steer_residual(model, Lc, d, alpha) if alpha else contextlib.nullcontext()
        with ctx:
            vals.append(target_score(tok, model, p, tw) - target_score(tok, model, p, cw))
    return float(np.nanmean(vals))

def run(tok, model):
    L = n_layers(model); Lc = int(round(L * 2 / 3))
    nmean = np.stack([layer_residuals(tok, model, t, Lc) for t in NEUTRAL]).mean(0)
    H = nmean.shape[0]; rn = residual_norm(tok, model, PROBES, Lc); alpha = FRAC * rn
    rng = np.random.default_rng(0)
    dirs = {c: unit(np.stack([layer_residuals(tok, model, t, Lc) for t in CONCEPTS[c]]).mean(0) - nmean)
            for c in CONCEPTS}
    per = {}
    for c in CONCEPTS:
        tw, cw = WORDS[c], WORDS[CONTROL[c]]
        base = contrast(tok, model, Lc, dirs[c], 0.0, tw, cw)
        e_t = contrast(tok, model, Lc, dirs[c], alpha, tw, cw) - base            # want +
        e_s = contrast(tok, model, Lc, dirs[CONTROL[c]], alpha, tw, cw) - base    # want - (symmetry)
        e_r = np.mean([contrast(tok, model, Lc, unit(rng.standard_normal(H)), alpha, tw, cw) - base
                       for _ in range(N_RAND)])
        per[c] = {"target": e_t, "symmetry": float(e_s), "random": float(e_r),
                  "pass": bool(e_t > 0.5 and e_t > 2 * abs(e_r))}
    # bootstrap CI over the 8 concepts
    cs = list(per); eff = np.array([per[c]["target"] for c in cs]); pas = np.array([per[c]["pass"] for c in cs])
    bt = rng.integers(0, len(cs), size=(2000, len(cs)))
    pr = pas[bt].mean(1); me = eff[bt].mean(1)
    return {"Lc": Lc, "pass_rate": float(pas.mean()), "pass_ci": [float(np.percentile(pr, 2.5)), float(np.percentile(pr, 97.5))],
            "mean_effect": float(eff.mean()), "eff_ci": [float(np.percentile(me, 2.5)), float(np.percentile(me, 97.5))],
            "per_concept": per}

def main():
    runnable = [(n, s, p) for n, s, p in GRID if available(p)]
    print("[concept] models available:", [n for n, _, _ in runnable] or "NONE — waiting for downloads")
    res = {}
    for name, size, path in runnable:
        try:
            tok, model = load(path)
        except Exception as e:
            print(f"{name}: load failed {e}"); continue
        r = run(tok, model); r["size"] = size; r["family"] = FAMILY(name); res[name] = r
        print(f"{name:14s} sz={size:<4} pass={r['pass_rate']:.2f} {r['pass_ci']} "
              f"eff={r['mean_effect']:+.2f} {r['eff_ci']}")
        mlib._CACHE.clear(); del model, tok; gc.collect(); torch.cuda.empty_cache()
    if not res:
        print("no models yet; download first."); return
    json.dump(res, open(os.path.join(RES, "concept_steering.json"), "w"), ensure_ascii=False, indent=2)
    # figure: pass-rate vs size, colored by family, with CI
    col = {"Qwen": "#c0392b", "Llama": "#2980b9", "Phi": "#27ae60", "Mistral": "#e67e22", "?": "#888"}
    plt.figure(figsize=(8, 5))
    for fam in set(r["family"] for r in res.values()):
        pts = sorted([(r["size"], r["pass_rate"], r["pass_ci"]) for r in res.values() if r["family"] == fam])
        if not pts: continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        lo = [p[1] - p[2][0] for p in pts]; hi = [p[2][1] - p[1] for p in pts]
        plt.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, color=col[fam], label=fam)
    plt.xscale("log"); plt.xlabel("model size (B params)"); plt.ylim(-0.05, 1.05)
    plt.ylabel("fraction of 8 concepts specifically steerable")
    plt.title("Concept steerability vs scale & family (95% bootstrap CI)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "concept_steering.png"), dpi=130); plt.close()
    print("[concept] saved results/concept_steering.json, figures/concept_steering.png")

if __name__ == "__main__":
    main()
