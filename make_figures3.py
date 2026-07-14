"""Premium publication figures (v3). Refined palette, conceptual schematic, clean sans
typography, minimal chartjunk. Reads results/*.json -> paper/figures/*.pdf."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({
 "font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],
 "font.size":9,"axes.titlesize":9.5,"axes.titleweight":"bold","axes.labelsize":9,
 "legend.fontsize":7.5,"xtick.labelsize":8,"ytick.labelsize":8,"axes.linewidth":0.9,
 "xtick.major.width":0.9,"ytick.major.width":0.9,"lines.linewidth":1.8,"lines.markersize":5,
 "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":220,"savefig.bbox":"tight",
 "axes.grid":True,"grid.color":"#EEEEEE","grid.linewidth":0.8,"axes.axisbelow":True})
FAM={"Qwen":"#B23A5E","Llama":"#2C6E9E","Phi":"#2F8F63","Mistral":"#D98A3D","Gemma":"#6C4FB0"}
INK="#22252A"
SEQ=LinearSegmentedColormap.from_list("seq",["#F3D9B1","#E08E45","#B23A5E","#5E2750"])
def fam(n): return "Qwen" if n.startswith("Qwen") else "Llama" if "Llama" in n else "Phi" if "hi" in n else "Gemma" if "emma" in n else "Mistral"
def L(f): return json.load(open(f"results/{f}.json"))
def keep(d): return {k:v for k,v in d.items() if "8bit" not in k}   # drop the 8-bit quantization-control entry
def panel(ax,letter):
    ax.text(-0.16,1.06,letter,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top",color=INK)
OUT="paper/figures"; sen=L("sensitivity"); rig=L("concept_steering")

# ===================== FIG MAIN =====================
fig=plt.figure(figsize=(7.4,5.4)); gs=fig.add_gridspec(2,2,hspace=0.46,wspace=0.42)
# (a) units confound: raw injection fraction is uncontrolled (real data)
axa=fig.add_subplot(gs[0,0]); panel(axa,"a"); axa.set_title("Raw injection: uncontrolled fraction",loc="left")
nrm=L("norms"); Gn=sorted(nrm.values(),key=lambda v:v["size"])
axa.plot([v["size"] for v in Gn],[v["ratio"] for v in Gn],"-o",color=FAM["Qwen"],mfc="white",mew=1.6,ms=6,zorder=3)
axa.set_xscale("log"); axa.set_ylim(0,None); axa.set_xlabel("model size (B)")
axa.set_ylabel(r"$\Vert d\Vert/\Vert h\Vert$ (raw)")
axa.text(0.04,0.94,"a fixed raw coefficient displaces an\nunpredictable fraction of the residual;\nfix: normalize $\\alpha=c\\,\\Vert h\\Vert$",
         transform=axa.transAxes,fontsize=6.4,color="#555",va="top")
# (b) naive apparent scaling
axb=fig.add_subplot(gs[0,1]); panel(axb,"b"); axb.set_title("Naive: apparent scaling (Qwen3)",loc="left")
nc=L("naive_concept"); pa=sorted([(v["size"],v["naive_gain"]) for v in nc.values()])
axb.plot([p[0] for p in pa],[p[1] for p in pa],"-o",color=FAM["Qwen"],mfc="white",mew=1.6,zorder=3)
axb.set_xscale("log"); axb.set_xticks([p[0] for p in pa]); axb.set_xticklabels([f"{p[0]}" for p in pa])
axb.set_xlabel("model size (B)"); axb.set_ylabel("naive raw-unit gain")
axb.annotate("looks like\nemergence",xy=(pa[-1][0],pa[-1][1]),xytext=(2.2,3.35),fontsize=7.6,ha="center",
             color="#555",arrowprops=dict(arrowstyle="->",color="#999",lw=0.9))
# (c) normalized dose-response
axc=fig.add_subplot(gs[1,0]); panel(axc,"c"); axc.set_title("Normalized dose–response",loc="left")
sd=sen["steering_dose"]; co=sen["coefs"]; sizes=sorted({v["size"] for v in sd.values()})
snorm=plt.Normalize(np.log10(min(sizes)),np.log10(max(sizes)))
for m,v in sorted(sd.items(),key=lambda kv:kv[1]["size"]):
    axc.plot(co,v["eff"],"-o",ms=3.5,color=SEQ(snorm(np.log10(v["size"]))),mfc="white",mew=1)
axc.axhline(0,color="#BBB",lw=0.8); axc.set_xlabel(r"steering strength ($\times\,\Vert h\Vert$)")
axc.set_ylabel("specificity effect")
from matplotlib.lines import Line2D
ssz=sorted({v["size"] for v in sd.values()}); pick=[ssz[0],ssz[len(ssz)//2],ssz[-1]]
axc.legend(handles=[Line2D([],[],marker="o",ls="-",mfc="white",color=SEQ(snorm(np.log10(s))),
           label=f"{s:g}B") for s in pick],frameon=False,fontsize=6.8,title="model size",
           title_fontsize=6.8,loc="lower right",handlelength=1.3,labelspacing=0.25)
axc.annotate("inverted-U",xy=(1,9),xytext=(3.4,7.6),fontsize=7.6,color="#555",
             arrowprops=dict(arrowstyle="->",color="#999",lw=0.9))
# (d) fixed operating point is erratic; per-model-tuned is flat (Qwen3 ladder scan)
axd=fig.add_subplot(gs[1,1]); panel(axd,"d"); axd.set_title("Fixed-recipe scaling is an artifact",loc="left")
scf=L("steer_cv"); sc2=L("steer_cv2")
qf=sorted([(v["size"],v["fixed_pass"]) for v in scf.values()])
axd.plot([p[0] for p in qf],[p[1] for p in qf],"-o",color="#B8BCC2",ms=4,mfc="white",mew=1.2,zorder=2,label="fixed layer 2/3")
q2=sorted([(v["size"],v["heldout_pass"],v["pass_ci"]) for k,v in sc2.items() if k.startswith("Qwen3") and "pass_ci" in v])
xs=[p[0] for p in q2]; ys=[p[1] for p in q2]
lo=[max(0,p[1]-p[2][0]) for p in q2]; hi=[max(0,p[2][1]-p[1]) for p in q2]
axd.errorbar(xs,ys,yerr=[lo,hi],fmt="-o",color=FAM["Qwen"],ms=5,mfc=FAM["Qwen"],capsize=2,elinewidth=1.1,zorder=4,label="held-out, 24 concepts")
axd.annotate("trend slope n.s.\n(neither emerges\nnor collapses)",xy=(7,0.80),xytext=(1.05,0.22),fontsize=6.6,color="#555",
             arrowprops=dict(arrowstyle="->",color="#999",lw=0.9))
axd.set_xscale("log"); axd.set_ylim(-0.05,1.14); axd.set_xlabel("model size (B)")
axd.set_ylabel("fraction steerable"); axd.legend(frameon=False,loc="lower left",fontsize=6.6)
fig.savefig(f"{OUT}/fig_main.pdf"); fig.savefig(f"{OUT}/fig_main.png"); plt.close()

# ===================== FIG LAYER (appendix) =====================
fig,ax=plt.subplots(figsize=(3.5,2.7))
shades={"Qwen3-1.7B":"#D98A3D","Qwen3-8B":"#B23A5E","Qwen3-14B":"#5E2750","Llama-3.1-8B":"#2C6E9E"}
for m,pts in sen["layer_sweep"].items():
    ax.plot([p[0] for p in pts],[p[1] for p in pts],"-o",ms=3.5,color=shades.get(m,"#888"),
            mfc="white",mew=1,label=m)
ax.axhline(0,color="#BBB",lw=0.8); ax.set_xlabel("intervention layer (fraction of depth)")
ax.set_ylabel("specificity effect"); ax.set_title("Layer sensitivity",loc="left")
ax.legend(frameon=False)
fig.savefig(f"{OUT}/fig_layer.pdf"); fig.savefig(f"{OUT}/fig_layer.png"); plt.close()

# ===================== FIG MAGNITUDE =====================
nt=keep(L("number_tuning")); nc=L("number_control")
fig,ax=plt.subplots(1,2,figsize=(7.0,2.7),gridspec_kw={"wspace":0.34})
for m,r in nt.items():
    ax[0].scatter(r["size"],r["best_r_mean"],color=FAM[fam(m)],s=34,zorder=3,edgecolor="white",lw=.6)
ax[0].axhline(np.mean([r["null_mean"] for r in nt.values()]),ls=(0,(4,3)),color="#999",lw=1)
ax[0].set_xscale("log"); ax[0].set_ylim(0,1.02); ax[0].set_xlabel("model size (B)")
ax[0].set_ylabel(r"held-out tuning $r$"); ax[0].set_title("Magnitude tuning: strong (solid)",loc="left")
ax[0].text(0.6,0.14,"null",fontsize=7,color="#999")
for m,r in nc.items():
    ax[1].scatter(r["size"],r["new_bell_frac"],color=FAM[fam(m)],s=34,zorder=3,edgecolor="white",lw=.6)
ax[1].axhline(0,color="#BBB",lw=1,ls=(0,(4,3)))
ax[1].set_xscale("log"); ax[1].set_ylim(-0.03,0.6); ax[1].set_xlabel("model size (B)")
ax[1].set_ylabel("bell fraction (top units)"); ax[1].set_title("Bell neurons exist; selection hid them",loc="left")
ax[1].legend(handles=[plt.Line2D([],[],marker="o",ls="",color=c,label=f) for f,c in FAM.items()],
             frameon=False,ncol=2,loc="upper right",handletextpad=0.3,columnspacing=0.8,fontsize=6.6)
fig.savefig(f"{OUT}/fig_magnitude.pdf"); fig.savefig(f"{OUT}/fig_magnitude.png"); plt.close()

# ===================== FIG LESION =====================
lc=L("lesion_control"); lf=sen["lesion_frac"]
fig,ax=plt.subplots(1,2,figsize=(7.2,3.1),gridspec_kw={"wspace":0.34})
items=sorted(lc.items(),key=lambda kv:kv[1]["size"])
for i,(m,r) in enumerate(items):
    lo=r["en_sel"]-r["en_ci"][0]; hi=r["en_ci"][1]-r["en_sel"]; sig=(r["en_ci"][0]>0 or r["en_ci"][1]<0)
    ax[0].errorbar(i,r["en_sel"],yerr=[[max(0,lo)],[max(0,hi)]],fmt="o",ms=5,color=FAM[fam(m)],
                   ecolor=FAM[fam(m)],capsize=2,elinewidth=1.1,mfc=FAM[fam(m)] if sig else "white",mew=1.3,zorder=3)
ax[0].axhline(0,color="#999",lw=1)
ax[0].set_xticks(range(len(items))); ax[0].set_xticklabels([m for m,_ in items],rotation=45,ha="right",fontsize=6)
ax[0].set_ylabel("English lesion selectivity"); ax[0].set_title("English lesioning: mostly n.s. (95% CI)",loc="left")
ax[0].text(0.02,0.96,"filled = significant.  Chinese sel. > 0 in ALL 17 models.",transform=ax[0].transAxes,fontsize=6.2,color="#2F8F63",va="top")
shades={"Qwen3-0.6B":"#F3C48B","Qwen3-1.7B":"#D98A3D","Qwen3-4B":"#B23A5E","Qwen3-8B":"#5E2750"}
for m,rows in lf.items():
    ax[1].plot([r["kfrac"]*100 for r in rows],[r["zh_sel"] for r in rows],"-o",ms=3.5,
               color=shades.get(m,"#888"),mfc="white",mew=1,label=m)
ax[1].axhline(0,color="#BBB",lw=.8); ax[1].set_xscale("log")
ax[1].set_xlabel("ablated fraction (% of neurons)"); ax[1].set_ylabel("Chinese selectivity")
ax[1].set_title("Stable across ablation size",loc="left"); ax[1].legend(frameon=False)
fig.savefig(f"{OUT}/fig_lesion.pdf"); fig.savefig(f"{OUT}/fig_lesion.png"); plt.close()

# ===================== FIG MAP =====================
cm2=keep(L("cognitive_map"))
fig,ax=plt.subplots(figsize=(3.4,2.7))
for m,r in cm2.items():
    ax.scatter(r["size"],(r["r2_lat"]+r["r2_lon"])/2,color=FAM[fam(m)],s=34,zorder=3,edgecolor="white",lw=.6)
ax.axhline(0,ls=(0,(4,3)),color="#999",lw=1)
ax.set_xscale("log"); ax.set_ylim(-0.1,1); ax.set_xlabel("model size (B)")
ax.set_ylabel(r"held-out geographic $R^2$"); ax.set_title("World map: universal",loc="left")
ax.legend(handles=[plt.Line2D([],[],marker="o",ls="",color=c,label=f) for f,c in FAM.items()],
          frameon=False,fontsize=6.5,loc="lower right")
fig.savefig(f"{OUT}/fig_map.pdf"); fig.savefig(f"{OUT}/fig_map.png"); plt.close()

# ===================== FIG STEER DIST (appendix; continuous effect size, not just pass-rate) =====================
sc2=L("steer_cv2")
q3=sorted([(v["size"],list(v["per_concept_eff"].values())) for k,v in sc2.items()
           if k.startswith("Qwen3") and "bf16" not in k and "per_concept_eff" in v and v.get("size",0)<=14])
if q3:
    fig,ax=plt.subplots(figsize=(4.3,2.8)); rng=np.random.default_rng(0)
    for size,effs in q3:
        e=np.array(effs); xs=size*np.exp(0.05*rng.standard_normal(len(e)))
        ax.scatter(xs,e,s=11,color=FAM["Qwen"],alpha=0.45,edgecolor="none",zorder=2)
        ax.plot([size*0.86,size*1.16],[e.mean()]*2,color=INK,lw=2.2,zorder=3)
    ax.axhline(0,color="#BBB",lw=.8); ax.set_xscale("log")
    ax.set_xticks([p[0] for p in q3]); ax.set_xticklabels([f"{p[0]:g}" for p in q3])
    ax.set_xlabel("model size (B)"); ax.set_ylabel("per-concept held-out effect")
    ax.set_title("Steering effect-size distribution (24 concepts)",loc="left")
    fig.savefig(f"{OUT}/fig_steer_dist.pdf"); fig.savefig(f"{OUT}/fig_steer_dist.png"); plt.close()

# ===================== FIG SCALE (cross-family audit to 72B; uses the cloud large-model results) =====================
scA=L("steer_cv2"); ndA=L("number_disentangle"); leA=L("lesion_control")
figS=plt.figure(figsize=(7.4,5.2)); gsS=figS.add_gridspec(2,2,hspace=0.5,wspace=0.42)
# (a) steering effect vs size, all scales
ax=figS.add_subplot(gsS[0,0]); panel(ax,"a"); ax.set_title("Steering: significant to 72B",loc="left")
for k,v in scA.items():
    if k=="_trend" or "bf16" in k or "heldout_eff" not in v: continue
    coarse=(v.get("size",0)>=24); c=FAM.get(v.get("family",fam(k)),INK)
    ci=v.get("eff_ci")
    if ci: ax.errorbar(v["size"],v["heldout_eff"],yerr=[[v["heldout_eff"]-ci[0]],[ci[1]-v["heldout_eff"]]],
                       fmt="none",ecolor=c,elinewidth=1,capsize=2,zorder=2)
    ax.scatter(v["size"],v["heldout_eff"],marker="s" if coarse else "o",s=34,
               facecolor="white" if coarse else c,edgecolor=c,linewidth=1.5,zorder=3)
ax.axhline(0,color="#BBB",lw=.8); ax.set_xscale("log"); ax.set_ylim(0,None)
ax.set_xlabel("model size (B)"); ax.set_ylabel("held-out effect")
ax.scatter([],[],marker="o",facecolor=INK,edgecolor=INK,label="fine grid ($\\leq$14B)")
ax.scatter([],[],marker="s",facecolor="white",edgecolor=INK,label="coarse ($\\geq$24B)")
ax.legend(frameon=False,fontsize=6.6,loc="lower center",ncol=2)
# (b) quantization control at 32B
ax=figS.add_subplot(gsS[0,1]); panel(ax,"b"); ax.set_title("Quant control (32B): 8-bit $\\approx$ bf16",loc="left")
q8,qb=scA["Qwen3-32B"],scA["Qwen3-32B-bf16"]
ax.bar([0,0.9],[q8["heldout_eff"],qb["heldout_eff"]],width=0.7,color=[FAM["Qwen"],"#C98"],zorder=3)
for x,val in [(0,q8),(0.9,qb)]: ax.text(x,val["heldout_eff"]+0.15,f"eff {val['heldout_eff']:+.1f}\npass {val['heldout_pass']:.2f}",ha="center",fontsize=7)
ax.set_xticks([0,0.9]); ax.set_xticklabels(["8-bit","bf16"]); ax.set_ylabel("held-out effect"); ax.set_ylim(0,q8["heldout_eff"]*1.5)
# (c) number cross-format consistency
ax=figS.add_subplot(gsS[1,0]); panel(ax,"c"); ax.set_title("Number: cross-format tuning",loc="left")
ax.axhspan(0.5,1.0,color="#EAF3EA",zorder=0); ax.axhspan(0.30,0.51,color="#EEEEEE",zorder=0)
for k,v in keep(ndA).items():
    ax.scatter(v["size"],v["bell_consistency_mean"],s=32,color=FAM.get(v.get("family",fam(k)),INK),zorder=3)
ax.text(0.55,0.80,"partly format-invariant",fontsize=6.0,color="#3a7a4a"); ax.text(0.55,0.40,"random-neuron baseline",fontsize=5.8,color="#999")
ax.set_xscale("log"); ax.set_ylim(0,0.85); ax.set_xlabel("model size (B)"); ax.set_ylabel("digit vs. word $r$")
from matplotlib.lines import Line2D as _L2fam
ax.legend(handles=[_L2fam([],[],marker='o',ls='',color=FAM[f],label=f,ms=4.5) for f in ["Qwen","Llama","Phi","Mistral","Gemma"]],
          frameon=False,fontsize=5.5,loc="lower left",ncol=2,handletextpad=0.15,columnspacing=0.6,labelspacing=0.25,borderpad=0.1)
# (d) lesion selectivity ZH/EN
ax=figS.add_subplot(gsS[1,1]); panel(ax,"d"); ax.set_title("Lesion: ZH robust, EN mixed",loc="left")
for k,v in keep(leA).items():
    if "zh_sel" not in v: continue
    s=v["size"]
    ax.scatter(s,v["zh_sel"],marker="^",s=34,facecolor=FAM["Qwen"] if v["zh_sig"] else "white",edgecolor=FAM["Qwen"],linewidth=1.4,zorder=3)
    ax.scatter(s,v["en_sel"],marker="v",s=34,facecolor=FAM["Llama"] if v["en_sig"] else "white",edgecolor=FAM["Llama"],linewidth=1.4,zorder=3)
ax.axhline(0,color="#BBB",lw=.8); ax.set_xscale("log"); ax.set_xlabel("model size (B)"); ax.set_ylabel("selectivity")
ax.scatter([],[],marker="^",facecolor=FAM["Qwen"],edgecolor=FAM["Qwen"],label="Chinese")
ax.scatter([],[],marker="v",facecolor=FAM["Llama"],edgecolor=FAM["Llama"],label="English")
ax.legend(frameon=False,fontsize=6.6,loc="upper right",title="filled = sig.",title_fontsize=6.2)
figS.savefig(f"{OUT}/fig_scale.pdf"); figS.savefig(f"{OUT}/fig_scale.png"); plt.close()
print("premium figures written")

# ===================== FIG NORMS (units argument, real data) =====================
try:
    nm=L("norms"); G=sorted(nm.values(),key=lambda v:v["size"]); xs=[v["size"] for v in G]
    fig,ax=plt.subplots(1,2,figsize=(6.6,2.6),gridspec_kw={"wspace":0.36})
    ax[0].plot(xs,[v["resid_norm"] for v in G],"-o",ms=4,color="#9AA0A6",label=r"residual $\Vert h\Vert$")
    ax[0].plot(xs,[v["dir_norm"] for v in G],"-o",ms=4,color=FAM["Qwen"],label=r"steering direction $\Vert d\Vert$")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("model size (B)")
    ax[0].set_ylabel("norm"); ax[0].legend(frameon=False,fontsize=7.5)
    ax[0].set_title("Raw norms do not track scale",loc="left",fontsize=9)
    ax[1].plot(xs,[v["ratio"] for v in G],"-o",ms=4,color="#5E2750")
    ax[1].set_xscale("log"); ax[1].set_xlabel("model size (B)"); ax[1].set_ylabel(r"$\Vert d\Vert / \Vert h\Vert$")
    ax[1].set_ylim(0,None); ax[1].set_title("Injection fraction is uncontrolled",loc="left",fontsize=9)
    fig.savefig(f"{OUT}/fig_norms.pdf"); fig.savefig(f"{OUT}/fig_norms.png"); plt.close()
except Exception as e:
    print("fig_norms skipped:",e)

# ===================== FIG FACTOR (2x2 decomposition) =====================
try:
    fc=L("factor_2x2"); Gf=sorted(fc.values(),key=lambda v:v["size"]); xs=[v["size"] for v in Gf]
    fig,ax=plt.subplots(1,2,figsize=(6.6,2.8),gridspec_kw={"wspace":0.36})
    for a,fix_label,raw_key,norm_key in [
        (ax[0],"Fixed operating point (layer 2/3)","cell_A_raw_fixed","cell_C_norm_fixed"),
        (ax[1],"Held-out operating point","cell_B_raw_heldout","cell_D_norm_heldout")]:
        a.plot(xs,[v[raw_key] for v in Gf],"-o",ms=5,color="#2C6E9E",label="raw units")
        a.plot(xs,[v[norm_key] for v in Gf],"-o",ms=5,color="#D98A3D",label="normalized")
        a.set_xscale("log"); a.set_xlabel("model size (B)"); a.set_ylabel("specificity contrast")
        a.set_title(fix_label,loc="left",fontsize=9); a.legend(frameon=False,fontsize=7.5)
        a.set_ylim(0,None)
    fig.savefig(f"{OUT}/fig_factor.pdf"); fig.savefig(f"{OUT}/fig_factor.png"); plt.close()
except Exception as e:
    print("fig_factor skipped:",e)
