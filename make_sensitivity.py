"""Depth experiments for the main phenomenon (concept steering) + supporting sensitivity
curves, so the paper has a thick, defensible main experiment rather than a shallow audit.

Produces results/sensitivity.json with:
  steering_dose[model]      : normalized specificity effect vs steering strength (coef x ||h||)
  steering_dose_naive[model]: raw (un-normalized) target log-prob gain vs raw coef
  layer_sweep[model]        : specificity effect vs intervention-layer fraction
  lesion_frac[model]        : zh/en lesion selectivity vs ablated fraction
"""
import os, json, gc, numpy as np, torch, torch.nn.functional as F, contextlib
import mlib
from mlib import (load, layer_residuals, residual_norm, steer_residual, target_score,
                  unit, n_layers, record_batch, text_nll, edited_neurons)
from exp01_concept import CONCEPTS, WORDS, CONTROL, NEUTRAL, PROBES
from exp03_lesion import ZH_TR, ZH_TE, EN_TR, EN_TE, attribution, mean_nll

# subset for tractable sweeps
CONC = ["Paris","Tokyo","Einstein","Shakespeare","Moon","Ocean"]
PR = PROBES[:4]; NRAND = 3
# <=8B only: 14B models OOM on the shared GPU; the "no clean scaling" point is already
# covered by the full-grid results/concept_steering.json.
STEER_MODELS = [("Qwen3-0.6B",0.6,"models_dl/Qwen3-0.6B"),("Qwen3-1.7B",1.7,"models_dl/Qwen3-1.7B"),
   ("Qwen3-4B",4.0,"models_dl/Qwen3-4B"),("Qwen3-8B",8.0,"models_dl/Qwen3-8B"),
   ("Llama-3.1-8B",8.0,"models_dl/Llama-3.1-8B-Instruct"),
   ("Ministral-8B",8.0,"models_dl/Ministral-8B-Instruct-2410")]
LAYER_MODELS = [("Qwen3-1.7B","models_dl/Qwen3-1.7B"),("Qwen3-8B","models_dl/Qwen3-8B"),
                ("Llama-3.1-8B","models_dl/Llama-3.1-8B-Instruct")]
LESION_MODELS = [("Qwen3-0.6B","models_dl/Qwen3-0.6B"),("Qwen3-1.7B","models_dl/Qwen3-1.7B"),
                 ("Qwen3-4B","models_dl/Qwen3-4B"),("Qwen3-8B","models_dl/Qwen3-8B")]
COEFS=[0,0.25,0.5,1,2,4,8]; RAWCOEFS=[0,0.5,1,2,4]; LFRACS=[0.6,0.4,0.5,0.67,0.8,0.9]
KFRACS=[0.0005,0.001,0.002,0.005,0.01]
TWraw=["Paris"," Paris","France"," France","Eiffel"," Eiffel"]

def contrast(tok,model,Lc,d,alpha,tw,cw):
    v=[]
    for p in PR:
        ctx=steer_residual(model,Lc,d,alpha) if alpha else contextlib.nullcontext()
        with ctx: v.append(target_score(tok,model,p,tw)-target_score(tok,model,p,cw))
    return float(np.nanmean(v))

def steering_dose(tok,model,Lc=None):
    L=n_layers(model); Lc=Lc if Lc is not None else int(round(L*2/3))
    nmean=np.stack([layer_residuals(tok,model,t,Lc) for t in NEUTRAL]).mean(0)
    rn=residual_norm(tok,model,PR,Lc); rng=np.random.default_rng(0)
    dirs={c:unit(np.stack([layer_residuals(tok,model,t,Lc) for t in CONCEPTS[c]]).mean(0)-nmean) for c in CONC}
    out=[]
    for coef in COEFS:
        eff=[]
        for c in CONC:
            tw,cw=WORDS[c],WORDS[CONTROL[c]]
            base=contrast(tok,model,Lc,dirs[c],0.0,tw,cw)
            eff.append(contrast(tok,model,Lc,dirs[c],coef*rn,tw,cw)-base)
        out.append(float(np.mean(eff)))
    return out

def steering_dose_naive(tok,model):
    L=n_layers(model);Lc=int(round(L*2/3))
    draw=(np.stack([layer_residuals(tok,model,t,Lc) for t in CONCEPTS["Paris"]]).mean(0)
          -np.stack([layer_residuals(tok,model,t,Lc) for t in NEUTRAL]).mean(0))
    tgt=sorted({tok(w,add_special_tokens=False)["input_ids"][0] for w in TWraw if tok(w,add_special_tokens=False)["input_ids"]})
    def sc(coef):
        v=[]
        for p in PR:
            ids=tok(p,return_tensors="pt").to(model.device)
            ctx=steer_residual(model,Lc,draw,coef) if coef else contextlib.nullcontext()
            with ctx:
                with torch.no_grad(): lp=F.log_softmax(model(**ids).logits[0,-1].float(),-1)
            v.append(float(torch.logsumexp(lp[tgt],0)))
        return np.mean(v)
    b=sc(0.0); return [float(sc(c)-b) for c in RAWCOEFS]

def layer_sweep(tok,model):
    L=n_layers(model); res=[]   # specificity effect at coef=1.0 across intervention layers
    for fr in sorted(LFRACS):
        Lc=max(1,min(L-1,int(round(L*fr))))
        nmean=np.stack([layer_residuals(tok,model,t,Lc) for t in NEUTRAL]).mean(0)
        rn=residual_norm(tok,model,PR,Lc)
        eff=[]
        for c in CONC:
            d=unit(np.stack([layer_residuals(tok,model,t,Lc) for t in CONCEPTS[c]]).mean(0)-nmean)
            tw,cw=WORDS[c],WORDS[CONTROL[c]]
            base=contrast(tok,model,Lc,d,0.0,tw,cw)
            eff.append(contrast(tok,model,Lc,d,1.0*rn,tw,cw)-base)
        res.append((fr,float(np.mean(eff))))
    return res

def lesion_frac(tok,model):
    az=attribution(model,tok,ZH_TR); ae=attribution(model,tok,EN_TR)
    L,U=az.shape; diff=(az-ae).reshape(-1)
    bz,be=mean_nll(tok,model,ZH_TE),mean_nll(tok,model,EN_TE); rng=np.random.default_rng(0)
    out=[]
    for kf in KFRACS:
        K=max(20,int(kf*L*U))
        zh=[divmod(int(i),U) for i in np.argsort(diff)[::-1][:K]]
        en=[divmod(int(i),U) for i in np.argsort(diff)[:K]]
        zz=mean_nll(tok,model,ZH_TE,zh)-bz; ze=mean_nll(tok,model,EN_TE,zh)-be
        ez=mean_nll(tok,model,ZH_TE,en)-bz; ee=mean_nll(tok,model,EN_TE,en)-be
        out.append({"kfrac":kf,"K":K,"zh_sel":zz-ze,"en_sel":ee-ez})
    return out

def save(R): json.dump(R,open("results/sensitivity.json","w"),indent=2)

def main():
    R={"steering_dose":{},"steering_dose_naive":{},"layer_sweep":{},"lesion_frac":{},"coefs":COEFS,"rawcoefs":RAWCOEFS}
    for name,size,path in STEER_MODELS:
        try:
            tok,model=load(path)
            R["steering_dose"][name]={"size":size,"eff":steering_dose(tok,model)}
            R["steering_dose_naive"][name]={"size":size,"eff":steering_dose_naive(tok,model)}
            if any(name==m[0] for m in LAYER_MODELS): R["layer_sweep"][name]=layer_sweep(tok,model)
            print(f"[steer] {name} done",flush=True); save(R)
            mlib._CACHE.clear(); del model,tok; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            import traceback; print(f"[steer] {name} FAIL {e}"); traceback.print_exc()
            mlib._CACHE.clear(); gc.collect(); torch.cuda.empty_cache()
    for name,path in LESION_MODELS:
        try:
            tok,model=load(path)
            R["lesion_frac"][name]=lesion_frac(tok,model); print(f"[lesion] {name} done",flush=True); save(R)
            mlib._CACHE.clear(); del model,tok; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            import traceback; print(f"[lesion] {name} FAIL {e}"); traceback.print_exc()
            mlib._CACHE.clear(); gc.collect(); torch.cuda.empty_cache()
    save(R); print("saved results/sensitivity.json")

if __name__=="__main__": main()
