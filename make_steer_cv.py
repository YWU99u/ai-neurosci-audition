"""Principled steering: select the operating point (layer, strength) the way exp02 (map)
selects its layer/alpha -- by held-out performance -- instead of the arbitrary fixed (2/3, 1.0).

For each model we compute the per-concept specificity effect at every (layer, strength) cell,
then do 2-fold CV over the 8 concepts: pick the best cell on the SELECT half, score on the
disjoint EVAL half. The reported number is thus on concepts that did not choose the operating
point -> no arbitrary layer, no circularity. We also record the old fixed-point pass-rate.
"""
import os, json, gc, numpy as np, torch, contextlib
import mlib
from mlib import load, layer_residuals, residual_norm, steer_residual, target_score, unit, n_layers
import exp01_concept as e1

LFRAC=[0.4,0.55,0.67,0.8]; COEFS=[0.25,0.5,1.0,2.0,4.0]; NRAND=2
PROBES=e1.PROBES[:4]; DL="models_dl"
MODELS=[("Qwen3-0.6B",0.6),("Qwen3-1.7B",1.7),("Qwen3-4B",4.0),("Qwen3-8B",8.0),("Qwen3-14B",14.0)]
CONC=list(e1.CONCEPTS)                                   # 8 concepts

def contrast(tok,model,Lc,d,alpha,tw,cw):
    v=[]
    for p in PROBES:
        ctx=steer_residual(model,Lc,d,alpha) if alpha else contextlib.nullcontext()
        with ctx: v.append(target_score(tok,model,p,tw)-target_score(tok,model,p,cw))
    return float(np.nanmean(v))

def cells_for_model(tok,model):
    L=n_layers(model); rng=np.random.default_rng(0); cells={}
    for lf in LFRAC:
        Lc=max(1,min(L-1,int(round(L*lf))))
        nmean=np.stack([layer_residuals(tok,model,t,Lc) for t in e1.NEUTRAL]).mean(0); H=nmean.shape[0]
        rn=residual_norm(tok,model,PROBES,Lc)
        dirs={c:unit(np.stack([layer_residuals(tok,model,t,Lc) for t in e1.CONCEPTS[c]]).mean(0)-nmean) for c in CONC}
        rand=[unit(rng.standard_normal(H)) for _ in range(NRAND)]
        base={c:contrast(tok,model,Lc,dirs[c],0.0,e1.WORDS[c],e1.WORDS[e1.CONTROL[c]]) for c in CONC}
        for coef in COEFS:
            cell={}
            for c in CONC:
                tw,cw=e1.WORDS[c],e1.WORDS[e1.CONTROL[c]]
                et=contrast(tok,model,Lc,dirs[c],coef*rn,tw,cw)-base[c]
                er=float(np.mean([contrast(tok,model,Lc,rd,coef*rn,tw,cw)-base[c] for rd in rand]))
                cell[c]={"et":float(et),"er":er}
            cells[f"{lf}|{coef}"]=cell
    return cells

def passed(cell,c): return cell[c]["et"]>0.5 and cell[c]["et"]>2*abs(cell[c]["er"])

def cv_summary(cells):
    folds=[(CONC[:4],CONC[4:]),(CONC[4:],CONC[:4])]; hp=[]; he=[]; picks=[]
    for sel,ev in folds:
        best=max(cells,key=lambda k:np.mean([cells[k][c]["et"] for c in sel]))   # pick on SELECT
        hp.append(float(np.mean([passed(cells[best],c) for c in ev])))            # score on EVAL
        he.append(float(np.mean([cells[best][c]["et"] for c in ev]))); picks.append(best)
    fx=cells["0.67|1.0"]
    return {"heldout_pass":float(np.mean(hp)),"heldout_eff":float(np.mean(he)),
            "fixed_pass":float(np.mean([passed(fx,c) for c in CONC])),"picks":picks}

def main():
    out=json.load(open("results/steer_cv.json")) if os.path.exists("results/steer_cv.json") else {}
    for name,size in MODELS:
        path=f"{DL}/{name}"
        if not os.path.exists(path): print("skip",name); continue
        print(f"==== {name} ====",flush=True)
        try:
            tok,model=load(path); cells=cells_for_model(tok,model); s=cv_summary(cells)
            s["size"]=size; out[name]=s; json.dump(out,open("results/steer_cv.json","w"),indent=2)
            print(f"  held-out pass={s['heldout_pass']:.2f}  (fixed 2/3,1.0 = {s['fixed_pass']:.2f})  picks={s['picks']}",flush=True)
            del model,tok
        except Exception as e:
            import traceback; print("FAIL",e); traceback.print_exc()
        mlib._CACHE.clear(); gc.collect(); torch.cuda.empty_cache()
    print("DONE")

if __name__=="__main__": main()
