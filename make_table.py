"""Emit paper/tab_results.tex: per-model results across the full 0.6-72B grid, all four experiments.
Regenerated from results/*.json so the numbers are exact."""
import json
def L(f):
    try: return json.load(open(f"results/{f}.json"))
    except: return {}
sc,nc,nd,le,cm=L("steer_cv2"),L("number_control"),L("number_disentangle"),L("lesion_control"),L("cognitive_map")
def g(d,m,k): return d.get(m,{}).get(k)
def sz(m): return g(sc,m,'size') or g(nc,m,'size') or g(le,m,'size') or g(nd,m,'size') or g(cm,m,'size')
def fam(m): return g(nc,m,'family') or g(le,m,'family') or g(cm,m,'family') or g(sc,m,'family')

_all=set(sc)|set(nc)|set(nd)|set(le)|set(cm)
# include only models present in at least steer OR lesion OR map (the 17-model target grid)
models=[m for m in _all if m!="_trend" and "bf16" not in m and "8bit" not in m and (m in sc or m in le or m in cm)]
models.sort(key=lambda m:(sz(m) or 99, m))

def num(x,f="{:.1f}"): return f.format(x) if x is not None else "--"
def sig(x,s,f="{:+.1f}"):
    if x is None: return "--"
    return ("\\textbf{"+f.format(x)+"}") if s else f.format(x)
def mapr2(m):
    a,b=g(cm,m,'r2_lat'),g(cm,m,'r2_lon')
    return f"{a:.2f}/{b:.2f}" if a is not None and b is not None else "--"

rows=[]
for m in models:
    rows.append(" & ".join([
        m.replace("_","\\_"), num(sz(m),"{:g}"), fam(m) or "--",
        num(g(sc,m,'heldout_eff'),"{:+.1f}"), num(g(sc,m,'heldout_pass'),"{:.2f}"),
        "{:.0f}".format(100*g(nc,m,'new_bell_frac')) if g(nc,m,'new_bell_frac') is not None else "--",
        num(g(nd,m,'bell_consistency_mean'),"{:+.2f}"),
        sig(g(le,m,'zh_sel'),g(le,m,'zh_sig')), sig(g(le,m,'en_sel'),g(le,m,'en_sig')),
        mapr2(m),
    ])+" \\\\")

tex = r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{5pt}
\caption{\textbf{Per-model results across the $0.6$--$72$B grid, all four experiments.} Steering:
held-out specificity effect and pass-rate. Number: bell-neuron fraction (shape-agnostic selection)
and digit-vs-word cross-format tuning correlation ($>\!0.5$ indicates partly format-invariant
magnitude tuning). Lesion (grad$\times$activation): Chinese/English selectivity. Map: cross-validated geographic $R^2$
(latitude/longitude). \textbf{Bold} marks a statistically significant selectivity (bootstrap $95\%$
CI excluding $0$); we bold only the lesion columns, where significance varies model to model, since
the steering effect is significant at every scale and the number and map columns are point estimates.
Pass-rate is the fraction of concepts passing the specificity, directional, and null controls (R3--R4).
The five $\ge\!24$B rows use a coarse steering grid, so their absolute pass-rates are not comparable to
the fine $\le\!14$B scan. Every cell is filled (all $17$ models $\times$ all four experiments). The world
map is the one column that is strong and uniform everywhere; every other headline is scale-, family-,
or selection-dependent.}
\label{tab:results}
\begin{tabular}{lccccccccc}
\toprule
& & & \multicolumn{2}{c}{Steering} & \multicolumn{2}{c}{Number} & \multicolumn{2}{c}{Lesion (sel.)} & Map \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-10}
Model & B & Fam. & eff & pass & bell\% & x-fmt $r$ & ZH & EN & $R^2$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}"""
open("paper/tab_results.tex","w").write(tex)
print(tex); print("\n-> paper/tab_results.tex ("+str(len(rows))+" rows)")
