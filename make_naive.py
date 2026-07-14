"""Reconstruct the NAIVE (un-normalized) concept-steering metric on the Qwen3 ladder,
to pair with the rigorous results (results/concept_steering.json) for the artifact figure.

Naive = raw mean-difference direction (NOT unit), fixed raw coefficient, first-token-only
target logprob. This is the pre-audit method; its 'effect' is incomparable across models
(residual norms differ), producing a spurious impression of scale-dependence.
"""
import os, json, gc, numpy as np, torch, torch.nn.functional as F, contextlib
import mlib
from mlib import load, layer_residuals, steer_residual, n_layers
from exp01_concept import CONCEPTS, NEUTRAL, PROBES

LADDER = [("Qwen3-0.6B",0.6,"models_dl/Qwen3-0.6B"),("Qwen3-1.7B",1.7,"models_dl/Qwen3-1.7B"),
          ("Qwen3-4B",4.0,"models_dl/Qwen3-4B"),("Qwen3-8B",8.0,"models_dl/Qwen3-8B"),
          ("Qwen3-14B",14.0,"models_dl/Qwen3-14B")]
TW = ["Paris"," Paris","France"," France","Eiffel"," Eiffel"]

def naive_gain(tok, model):
    L=n_layers(model); Lc=int(round(L*2/3))
    draw=(np.stack([layer_residuals(tok,model,t,Lc) for t in CONCEPTS["Paris"]]).mean(0)
          -np.stack([layer_residuals(tok,model,t,Lc) for t in NEUTRAL]).mean(0))  # RAW, not unit
    tgt=sorted({tok(w,add_special_tokens=False)["input_ids"][0] for w in TW
                if tok(w,add_special_tokens=False)["input_ids"]})
    def score(coef):
        vals=[]
        for p in PROBES:
            ids=tok(p,return_tensors="pt").to(model.device)
            ctx=steer_residual(model,Lc,draw,coef) if coef else contextlib.nullcontext()
            with ctx:
                with torch.no_grad():
                    lp=F.log_softmax(model(**ids).logits[0,-1].float(),-1)
            vals.append(float(torch.logsumexp(lp[tgt],0)))
        return np.mean(vals)
    base=score(0.0); peak=max(score(c) for c in [1,2,4])
    return float(peak-base)

def main():
    out={}
    for name,size,path in LADDER:
        try: tok,model=load(path)
        except Exception as e: print(f"{name} load fail {e}"); continue
        g=naive_gain(tok,model); out[name]={"size":size,"naive_gain":g}
        print(f"{name:12s} naive_gain={g:+.2f}")
        mlib._CACHE.clear(); del model,tok; gc.collect(); torch.cuda.empty_cache()
    json.dump(out,open("results/naive_concept.json","w"),indent=2)
    print("saved results/naive_concept.json")

if __name__=="__main__":
    main()
