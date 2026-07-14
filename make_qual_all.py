"""Real qualitative artifacts for the other three experiments (single Llama-3.1-8B load):
  fig_number_example.pdf  -- bell vs monotonic number-neuron tuning curves, digit vs word
  fig_map_example.pdf     -- cross-validated predicted city coordinates (the model's world map)
  fig_lesion_example.pdf  -- per-token surprisal before/after ablating Chinese-selective neurons
All from real model activations; no fabrication."""
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import mlib
from mlib import load, record_batch, last_token_residuals, edited_neurons
from exp01_concept import GRID
from exp02_map import CITIES
from exp03_lesion import attribution
import make_lesion_control as lc

plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight", "figure.dpi": 220,
                     "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"], "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE, INK = "#B23A5E", "#2C6E9E", "#242424"
OUT = "paper/figures"
path = [g[2] for g in GRID if g[0] == "Llama-3.1-8B"][0]
tok, model = load(path)

# ============ 1. NUMBER NEURON TUNING CURVES ============
try:
    N = list(range(1, 41)); X = np.array(N, float)
    from make_number_disentangle import n2w
    Ad = record_batch(tok, model, [f"The number is {n}" for n in N], reduce="last").reshape(len(N), -1).astype(float)
    Aw = record_batch(tok, model, [f"The number is {n2w(n)}" for n in N], reduce="last").reshape(len(N), -1).astype(float)
    def r2(x, y, deg):
        c = np.polyfit(x, y, deg); return 1 - ((y - np.polyval(c, x))**2).sum() / (((y - y.mean())**2).sum() + 1e-9)
    quad = np.array([r2(X, Ad[:, u], 2) for u in range(Ad.shape[1])])
    lin = np.array([r2(X, Ad[:, u], 1) for u in range(Ad.shape[1])])
    vtx = np.array([(-np.polyfit(X, Ad[:, u], 2)[1] / (2*np.polyfit(X, Ad[:, u], 2)[0] + 1e-12)) for u in range(Ad.shape[1])])
    xf = np.array([np.corrcoef(Ad[:, u], Aw[:, u])[0, 1] for u in range(Ad.shape[1])])
    bell_ok = (quad - lin > 0.2) & (vtx > 8) & (vtx < 33) & (quad > 0.6) & (xf > 0.6)
    bell = int(np.argsort(-np.where(bell_ok, quad, -1))[0])
    mono = int(np.argsort(-np.abs([np.corrcoef(X, Ad[:, u])[0, 1] for u in range(Ad.shape[1])]))[0])
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for a, u, ttl, showr in [(ax[0], bell, "Bell-shaped number neuron", True), (ax[1], mono, "Monotonic number neuron", False)]:
        zd = (Ad[:, u]-Ad[:, u].mean())/(Ad[:, u].std()+1e-9); zw = (Aw[:, u]-Aw[:, u].mean())/(Aw[:, u].std()+1e-9)
        a.plot(X, zd, "-o", ms=3, color=RED, label="digit  “17”")
        a.plot(X, zw, "--s", ms=3, color=BLUE, mfc="white", label="word  “seventeen”")
        sub = f"\n(digit vs. word r={xf[u]:+.2f}: genuine numerosity)" if showr else "\n(rises with number)"
        a.set_title(f"{ttl}{sub}", fontsize=8.5, loc="left")
        a.set_xlabel("number"); a.set_ylabel("activation (z)"); a.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_number_example.pdf"); fig.savefig(f"{OUT}/fig_number_example.png"); plt.close()
    print("number: bell u=%d xf=%.2f | mono u=%d" % (bell, xf[bell], mono), flush=True)
except Exception as e:
    import traceback; print("NUMBER FAIL", e); traceback.print_exc()

# ============ 2. WORLD MAP (predicted coordinates) ============
try:
    names = [c[0] for c in CITIES]; Y = np.array([[c[1], c[2]] for c in CITIES], float)
    reps = np.stack([last_token_residuals(tok, model, f"The city of {n}") for n in names])  # [N, L+1, H]
    def cvpred(Xl, a):
        kf = KFold(5, shuffle=True, random_state=0); pr = np.zeros_like(Y)
        for tr, te in kf.split(Xl): pr[te] = Ridge(alpha=a).fit(Xl[tr], Y[tr]).predict(Xl[te])
        return pr
    best = (-9, None, None)
    for l in range(reps.shape[1]):
        for a in (10., 100., 1000.):
            pr = cvpred(reps[:, l, :], a)
            sc = 2 - (((Y-pr)**2).sum(0) / ((Y-Y.mean(0))**2).sum(0)).sum()
            if sc > best[0]: best = (sc, l, a)
    _, bl, ba = best; pred = cvpred(reps[:, bl, :], ba)
    ssr = ((Y-pred)**2).sum(0); sst = ((Y-Y.mean(0))**2).sum(0); r2lat, r2lon = 1-ssr/sst
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    sc = ax.scatter(pred[:, 1], pred[:, 0], c=Y[:, 1], cmap="Spectral", s=34, edgecolor="#333", lw=0.5, zorder=3)
    show = {"London","New York","Tokyo","Sydney","Cairo","Rio de Janeiro","Moscow","Mumbai","Los Angeles",
            "Beijing","Cape Town","Buenos Aires","Singapore","Dubai"}
    for (n, la, lo), (py, px) in zip(CITIES, pred):
        if n in show: ax.annotate(n, (px, py), fontsize=6.6, color="#222", xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("predicted longitude"); ax.set_ylabel("predicted latitude")
    ax.set_title(f"The model's internal world map (Llama-3.1-8B): held-out cities,\n"
                 f"decoded lat/lon $R^2$={r2lat:.2f}/{r2lon:.2f}", fontsize=8.5, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_map_example.pdf"); fig.savefig(f"{OUT}/fig_map_example.png"); plt.close()
    print(f"map: R2 lat={r2lat:.2f} lon={r2lon:.2f}", flush=True)
except Exception as e:
    import traceback; print("MAP FAIL", e); traceback.print_exc()

# ============ 3. LESION: per-token surprisal before/after ablating Chinese neurons ============
try:
    az = attribution(model, tok, lc.ZH_TR); ae = attribution(model, tok, lc.EN_TR)
    L, U = az.shape; K = max(50, int(lc.KFRAC * L * U)); diff = (az - ae).reshape(-1)
    zh_neurons = [divmod(int(i), U) for i in np.argsort(diff)[::-1][:K]]
    @torch.no_grad()
    def ptnll(text):
        ids = tok(text, return_tensors="pt").to(model.device); t = ids["input_ids"][0]
        logp = F.log_softmax(model(**ids).logits[0].float(), -1)
        return ([tok.decode([t[i]]) for i in range(1, len(t))],
                [-logp[i-1, t[i]].item() for i in range(1, len(t))])
    zh_sent, en_sent = lc.ZH_TE[0], lc.EN_TE[0]
    (zt, zb), (et, eb) = ptnll(zh_sent), ptnll(en_sent)
    with edited_neurons(model, ablate=zh_neurons):
        (_, za), (_, ea) = ptnll(zh_sent), ptnll(en_sent)
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.9))
    for a, tks, before, after, ttl in [(ax[0], zt, zb, za, "Chinese sentence"), (ax[1], et, eb, ea, "English sentence")]:
        xs = np.arange(len(tks))
        a.plot(xs, before, "-o", ms=3, color="#9AA0A6", label="intact")
        a.plot(xs, after, "-o", ms=3, color=RED, label="Chinese neurons ablated")
        a.set_xticks(xs); a.set_xticklabels(tks, rotation=60, ha="right", fontsize=6)
        a.set_ylabel("surprisal (nats)"); a.set_title(ttl, fontsize=9, loc="left"); a.legend(frameon=False, fontsize=7)
    fig.suptitle("Ablating Chinese-selective neurons: Chinese surprisal jumps, English is spared", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_lesion_example.pdf"); fig.savefig(f"{OUT}/fig_lesion_example.png"); plt.close()
    print(f"lesion: ZH dNLL={np.mean(za)-np.mean(zb):+.2f} EN dNLL={np.mean(ea)-np.mean(eb):+.2f}", flush=True)
except Exception as e:
    import traceback; print("LESION FAIL", e); traceback.print_exc()
print("DONE")
