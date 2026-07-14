"""Real qualitative steering examples for the appendix: generate the SAME neutral prompt with
steering off vs. on, using the exact direction/normalization from the main experiment.
Writes results/qual_examples.json. No fabrication --- these are actual model generations."""
import json, numpy as np, mlib
from mlib import load, layer_residuals, residual_norm, steer_residual, generate, unit, n_layers
from exp01_concept import NEUTRAL, PROBES
from make_steer_cv2 import S

MODEL = ("Llama-3.1-8B", "")   # clean prose (no <think>); path resolved from GRID below
DEMO = ["Paris", "Volcano", "Dolphin"]
PROMPT = "Tell me a short, everyday story in two sentences."
COEFS = [0.1, 0.2, 0.35]

def main():
    import os
    # resolve model path from the grid if the hardcoded path is absent
    from exp01_concept import GRID
    cand = [g[2] for g in GRID if g[0] == MODEL[0]]
    path = cand[0] if cand else MODEL[1]
    tok, model = load(path)
    Lc = round(0.8 * n_layers(model))
    nmean = np.stack([layer_residuals(tok, model, t, Lc) for t in NEUTRAL]).mean(0)
    rn = residual_norm(tok, model, PROBES[:3], Lc)
    out = {"model": MODEL[0], "layer": Lc, "prompt": PROMPT, "resid_norm": rn, "examples": []}
    base = generate(tok, model, PROMPT, max_new_tokens=48)
    out["baseline"] = base.strip()
    print("BASELINE:", base.strip(), flush=True)
    for c in DEMO:
        d = unit(np.stack([layer_residuals(tok, model, t, Lc) for t in S[c][1]]).mean(0) - nmean)
        for coef in COEFS:
            with steer_residual(model, Lc, d, alpha=coef * rn):
                g = generate(tok, model, PROMPT, max_new_tokens=48)
            rec = {"concept": c, "category": S[c][0], "coef": coef, "text": g.strip()}
            out["examples"].append(rec)
            json.dump(out, open("results/qual_examples.json", "w"), ensure_ascii=False, indent=2)
            print(f"[{c} c={coef}] {g.strip()}", flush=True)
    print("DONE")

if __name__ == "__main__":
    main()
