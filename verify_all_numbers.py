"""Regenerate every number that appears in the paper (table, figures, in-text stats)
from results/*.json only, for human proofreading.

Usage: python verify_all_numbers.py
Input:  results/*.json (16 files)
Output: printed audit with exact JSON values, rounded paper values, and pass/fail checks.
"""
import json, numpy as np
from collections import OrderedDict

def L(f):
    try: return json.load(open(f"results/{f}.json"))
    except FileNotFoundError: return {}

# ── Load all result files ──
sc2  = L("steer_cv2")        # held-out steering (main)
scv  = L("steer_cv")         # fixed-point steering
nc   = L("number_control")   # bell fractions (shape-agnostic)
nd   = L("number_disentangle")  # cross-format tuning
nt   = L("number_tuning")    # magnitude tuning r
lc   = L("lesion_control")   # lesion selectivity (grad×act)
la   = L("lesion_altattr")   # lesion alt attribution
cm   = L("cognitive_map")    # world map R²
f2   = L("factor_2x2")       # 2×2 decomposition
nm   = L("norms")            # residual norms
nv   = L("naive_concept")    # naive scaling artifact
sn   = L("sensitivity")      # dose-response, layer sweep, lesion frac
cs   = L("concept_steering") # full concept-steering results
qe   = L("qual_examples")    # qualitative examples

# ── Helpers ──
def rd(x, d=1):
    """Round to d decimal places, return string."""
    if x is None: return "None"
    return f"{x:.{d}f}"

def check(name, json_val, paper_val, tol=0.051):
    """Check if json_val rounds to paper_val."""
    if json_val is None:
        return f"  {name}: JSON=None  paper={paper_val}  ⚠️ MISSING"
    diff = abs(json_val - paper_val)
    ok = "✓" if diff < tol else "✗ MISMATCH"
    return f"  {name}: JSON={json_val:.6f}  →round→ {rd(json_val,len(str(paper_val).split('.')[-1]) if '.' in str(paper_val) else 0)}  paper={paper_val}  {ok}"

SEP = "=" * 80

# ────────────────────────────────────────────────────────────────────────
print(SEP)
print("1. TABLE 1 — Per-model results (17 models × 9 data columns)")
print(SEP)

def sz(d, m):
    for src in [sc2, nc, lc, nd, cm]:
        v = src.get(m, {}).get('size')
        if v is not None: return v
    return None

def fam(d, m):
    for src in [nc, lc, cm, sc2, nd]:
        v = src.get(m, {}).get('family')
        if v is not None: return v
    return None

# Build model list matching make_table.py logic
_all = set(sc2) | set(nc) | set(nd) | set(lc) | set(cm)
models = [m for m in _all if m != "_trend" and "bf16" not in m and "8bit" not in m
          and (m in sc2 or m in lc or m in cm)]
models.sort(key=lambda m: (sz(None, m) or 99, m))

print(f"\nModel count: {len(models)}  (paper says 17)")
print()

header = f"{'Model':<22} {'B':>4} {'Fam':<7} {'eff':>6} {'pass':>5} {'bell%':>5} {'x-fmt':>6} {'ZH':>6} {'EN':>6} {'R²_lat':>6} {'R²_lon':>6}"
print(header)
print("-" * len(header))

for m in models:
    s = sz(None, m)
    f = fam(None, m) or "--"
    eff = sc2.get(m, {}).get('heldout_eff')
    pas = sc2.get(m, {}).get('heldout_pass')
    bf = nc.get(m, {}).get('new_bell_frac')
    xf = nd.get(m, {}).get('bell_consistency_mean')
    zh = lc.get(m, {}).get('zh_sel')
    en = lc.get(m, {}).get('en_sel')
    zh_sig = lc.get(m, {}).get('zh_sig')
    en_sig = lc.get(m, {}).get('en_sig')
    r2lat = cm.get(m, {}).get('r2_lat')
    r2lon = cm.get(m, {}).get('r2_lon')

    def fmt(v, f=".1f"):
        return f"{v:{f}}" if v is not None else "--"
    def fmtp(v):
        return f"{v:.2f}" if v is not None else "--"
    def fmtb(v):
        return f"{100*v:.0f}" if v is not None else "--"
    def fmts(v, sig):
        if v is None: return "--"
        s = f"{v:+.1f}"
        return f"**{s}**" if sig else s

    print(f"{m:<22} {fmt(s,'4g'):>4} {f:<7} {fmt(eff,'+.1f'):>6} {fmtp(pas):>5} "
          f"{fmtb(bf):>5} {fmt(xf,'+.2f'):>6} {fmts(zh, zh_sig):>6} {fmts(en, en_sig):>6} "
          f"{fmt(r2lat,'.2f'):>6}/{fmt(r2lon,'.2f')}")

# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("2. IN-TEXT STEERING NUMBERS (§4.1)")
print(SEP)

# Norm ratios
print("\n── Norm ratios ‖d‖/‖h‖ (paper: 0.12–0.27) ──")
ratios = sorted([(v['size'], v['ratio']) for v in nm.values()], key=lambda x: x[0])
for s, r in ratios:
    print(f"  {s:5.1f}B: ratio = {r:.4f}")
rvals = [r for _, r in ratios]
print(f"  Range: {min(rvals):.2f}–{max(rvals):.2f}  (paper: 0.12–0.27)")

# Fixed-point pass rates
print("\n── Fixed-point pass rates (paper: 0.50,0.88,0.25,0.50,0.25) ──")
qwen_fixed = sorted([(v['size'], v['fixed_pass']) for k, v in scv.items()], key=lambda x: x[0])
for s, p in qwen_fixed:
    print(f"  {s:5.1f}B: {p:.2f}")

# Held-out pass rates (Qwen3 ladder, ≤14B)
print("\n── Held-out pass rates, Qwen3 ≤14B (paper: 0.71,1.00,0.79,0.79,0.75) ──")
qwen_ho = sorted([(v['size'], v['heldout_pass'], v.get('pass_ci'))
                   for k, v in sc2.items() if k.startswith("Qwen3") and v.get('size', 99) <= 14],
                  key=lambda x: x[0])
for s, p, ci in qwen_ho:
    ci_s = f"  CI={ci}" if ci else ""
    print(f"  {s:5.1f}B: {p:.2f}{ci_s}")

# Trend slope
print("\n── Trend slope (paper: +0.31, CI [-0.11, +0.73]) ──")
tr = sc2.get("_trend", {})
print(f"  slope = {tr.get('slope_eff_vs_log2size')}")
print(f"  CI = {tr.get('slope_ci')}")

# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("3. LARGE MODEL EFFECTS AND CIs (§4.2)")
print(SEP)

large = ["Gemma-2-27B", "Mistral-Small-24B", "Qwen3-32B", "Qwen2.5-72B", "Llama-3.1-70B"]
print("\nPaper reports these five with CIs:")
for m in large:
    v = sc2.get(m, {})
    eff = v.get('heldout_eff')
    ci = v.get('eff_ci')
    pas = v.get('heldout_pass')
    print(f"  {m}: eff={eff}  CI={ci}  pass={pas}")

# Quantization control
print("\n── Quantization control (paper: +4.9 vs +4.3) ──")
q8 = sc2.get("Qwen3-32B", {})
qbf = sc2.get("Qwen3-32B-bf16", {})
print(f"  8-bit:  eff={q8.get('heldout_eff')}  pass={q8.get('heldout_pass')}")
print(f"  bf16:   eff={qbf.get('heldout_eff')}  pass={qbf.get('heldout_pass')}")

# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("4. FACTOR DECOMPOSITION (§4.1 / App C)")
print(SEP)

print("\n── 2×2 cell means per model ──")
for m in sorted(f2, key=lambda k: f2[k]['size']):
    v = f2[m]
    print(f"  {m} ({v['size']}B): A_raw_fixed={v['cell_A_raw_fixed']:.2f}  "
          f"B_raw_ho={v['cell_B_raw_heldout']:.2f}  "
          f"C_norm_fixed={v['cell_C_norm_fixed']:.2f}  "
          f"D_norm_ho={v['cell_D_norm_heldout']:.2f}")

print("\n── Trend slopes (paper: raw+fixed +0.22 CI [-0.02,+0.62]; norm+HO +0.31 CI [-2.87,+4.05]) ──")
sizes = np.array([f2[n]["size"] for n in sorted(f2, key=lambda k: f2[k]["size"])])
for cell in ["cell_A_raw_fixed", "cell_B_raw_heldout", "cell_C_norm_fixed", "cell_D_norm_heldout"]:
    effs = np.array([f2[n][cell] for n in sorted(f2, key=lambda k: f2[k]["size"])])
    x = np.log2(sizes); y = effs
    slope = np.polyfit(x, y, 1)[0]
    rng = np.random.default_rng(42); boots = []
    for _ in range(2000):
        idx = rng.choice(len(x), len(x))
        boots.append(np.polyfit(x[idx], y[idx], 1)[0])
    ci = np.percentile(boots, [2.5, 97.5])
    print(f"  {cell}: slope={slope:+.2f}  CI=[{ci[0]:+.2f}, {ci[1]:+.2f}]")

# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("5. MAGNITUDE CODING (§4.3)")
print(SEP)

# Best tuning r
print("\n── Best held-out tuning r (paper: 0.68–0.98) ──")
for m in sorted(nt, key=lambda k: nt[k].get('size', 0)):
    if "8bit" in m: continue
    v = nt[m]
    print(f"  {m} ({v.get('size')}B): best_r={v['best_r_mean']:.4f}  null={v['null_mean']:.4f}")
rvals = [v['best_r_mean'] for k, v in nt.items() if "8bit" not in k]
print(f"  Range: {min(rvals):.2f}–{max(rvals):.2f}")

# Bell fractions
print("\n── Bell fractions (paper: 0.00–0.43; L/P/M: 13–43%) ──")
for m in sorted(nc, key=lambda k: nc[k].get('size', 0)):
    v = nc[m]
    print(f"  {m} ({v.get('size')}B, {v.get('family','')}): new_bell_frac={v['new_bell_frac']:.4f} = {100*v['new_bell_frac']:.0f}%")
bvals = [v['new_bell_frac'] for v in nc.values()]
print(f"  Range: {min(bvals):.2f}–{max(bvals):.2f}")
# L/P/M subset
lpm = [v['new_bell_frac'] for k, v in nc.items()
       if v.get('family') in ('Llama', 'Phi', 'Mistral')]
if lpm:
    print(f"  Llama/Phi/Mistral range: {min(lpm)*100:.0f}–{max(lpm)*100:.0f}%")

# Cross-format r
print("\n── Cross-format tuning r (paper: 0.59–0.70 for L/P/M; 0.01–0.64 for Qwen) ──")
for m in sorted(nd, key=lambda k: nd[k].get('size', 0)):
    v = nd[m]
    print(f"  {m} ({v.get('size')}B, {v.get('family','')}): bell_consistency={v['bell_consistency_mean']:.4f}")
lpm_xf = [v['bell_consistency_mean'] for k, v in nd.items()
          if v.get('family') in ('Llama', 'Phi', 'Mistral')]
qwen_xf = [v['bell_consistency_mean'] for k, v in nd.items()
           if v.get('family') == 'Qwen']
if lpm_xf:
    print(f"  L/P/M range: {min(lpm_xf):.2f}–{max(lpm_xf):.2f}")
if qwen_xf:
    print(f"  Qwen range: {min(qwen_xf):.2f}–{max(qwen_xf):.2f}")

# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("6. LESION RESULTS (§4.4)")
print(SEP)

print("\n── Per-model selectivity, significance, and interaction ──")
zh_sig_count = 0; en_sig_count = 0; dd_count = 0; inter_sig_count = 0
en_sig_models = []
for m in sorted(lc, key=lambda k: lc[k].get('size', 0)):
    v = lc[m]
    zh_s = "SIG" if v.get('zh_sig') else "n.s."
    en_s = "SIG" if v.get('en_sig') else "n.s."
    dd = "DD" if v.get('zh_sig') and v.get('en_sig') else ""
    inter = v.get('interaction')
    inter_ci = v.get('inter_ci')
    inter_s = "SIG" if v.get('inter_sig') else "n.s."
    print(f"  {m:<22} ZH={v.get('zh_sel',0):+.1f} [{zh_s}]  EN={v.get('en_sel',0):+.1f} [{en_s}]  "
          f"inter={inter:+.1f} CI={inter_ci} [{inter_s}]  {dd}")
    if v.get('zh_sig'): zh_sig_count += 1
    if v.get('en_sig'):
        en_sig_count += 1
        en_sig_models.append(m)
    if v.get('zh_sig') and v.get('en_sig'): dd_count += 1
    if v.get('inter_sig'): inter_sig_count += 1

print(f"\n  Summary:")
print(f"    ZH significant: {zh_sig_count}/17  (paper: 17/17)")
print(f"    EN significant: {en_sig_count}/17  (paper: 5/17)")
print(f"    EN sig models: {en_sig_models}")
print(f"    Double dissoc:  {dd_count}/17  (paper: 5/17)")
print(f"    Interaction sig: {inter_sig_count}/17 (paper: 17/17)")

# Alt attribution
if la:
    print("\n── Alternative attribution (activation magnitude) ──")
    la_zh_sig = sum(1 for v in la.values() if v.get('zh_sig'))
    la_en_sig = sum(1 for v in la.values() if v.get('en_sig'))
    n_la = len(la)
    print(f"  Models tested: {n_la}  (paper: 9)")
    print(f"  ZH sig: {la_zh_sig}/{n_la}  (paper: 7/9)")
    print(f"  EN sig: {la_en_sig}/{n_la}  (paper: 7/9)")
    for m in sorted(la, key=lambda k: la[k].get('size', 0)):
        v = la[m]
        print(f"    {m}: ZH={v.get('zh_sel',0):+.1f} [{'SIG' if v.get('zh_sig') else 'n.s.'}]  "
              f"EN={v.get('en_sel',0):+.1f} [{'SIG' if v.get('en_sig') else 'n.s.'}]")

# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("7. WORLD MAP (§4.5)")
print(SEP)

print("\n── R² per model (paper: 0.43–0.68) ──")
for m in sorted(cm, key=lambda k: cm[k].get('size', 0)):
    if "8bit" in m: continue
    v = cm[m]
    avg = (v['r2_lat'] + v['r2_lon']) / 2
    print(f"  {m} ({v.get('size')}B): lat={v['r2_lat']:.4f}  lon={v['r2_lon']:.4f}  avg={avg:.4f}")
r2_all = [v for k, v in cm.items() if "8bit" not in k]
all_r2 = [v['r2_lat'] for v in r2_all] + [v['r2_lon'] for v in r2_all]
print(f"  Range (all lat+lon): {min(all_r2):.2f}–{max(all_r2):.2f}")

# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("8. CATEGORY MEANS (App C, paper: obj +20.4, per +17.4, city +14.5, ani +8.1, abs +7.5)")
print(SEP)

# Use Qwen3-8B per_concept_eff
q8_data = sc2.get("Qwen3-8B", {}).get("per_concept_eff", {})
cat_data = sc2.get("Qwen3-8B", {}).get("cat_eff", {})
if cat_data:
    print("\n── From cat_eff (Qwen3-8B) ──")
    for cat in ["object", "person", "city", "animal", "abstract"]:
        v = cat_data.get(cat)
        print(f"  {cat}: {v}")
if q8_data:
    print("\n── From per_concept_eff (Qwen3-8B), manual grouping ──")
    # Typical category assignments
    cats = {
        "city": ["Paris", "Tokyo", "London", "Sydney", "NewYork", "Cairo"],
        "person": ["Einstein", "Shakespeare", "Mozart", "Cleopatra", "DaVinci", "Napoleon"],
        "object": ["Piano", "Telescope", "Compass", "Diamond", "Sword", "Lighthouse"],
        "animal": ["Eagle", "Dolphin", "Wolf", "Elephant", "Tiger", "Butterfly"],
        "abstract": ["Freedom", "Justice", "Gravity", "Time", "Chaos", "Harmony"],
    }
    for cat, concepts in cats.items():
        vals = [q8_data[c] for c in concepts if c in q8_data]
        if vals:
            print(f"  {cat}: concepts={[c for c in concepts if c in q8_data]}  "
                  f"vals={[round(v,2) for v in vals]}  mean={np.mean(vals):.3f}")

# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("9. DESIGN PARAMETERS (methods / stimuli)")
print(SEP)

print("""
  Paper claims to verify:
  - 17 models, 5 families, 0.6–72B range
  - 24 concepts, 5 categories, 12 matched pairs
  - 4-fold CV for operating-point selection
  - Layer grid: {0.4, 0.6, 0.8} × depth
  - Strength grid: {0.25, 0.5, 1, 2, 4}
  - Dose-response: {0, 0.25, 0.5, 1, 2, 4, 8}
  - Layer sweep: {0.4, 0.5, 0.6, 0.67, 0.8, 0.9} × depth
  - 6 neutral readout prompts
  - 10 neutral baseline sentences
  - 5 random directions
  - Numbers 1–40
  - 45 world cities
  - Ridge α by inner CV, 5-fold held-out R²
  - 200 label shuffles
  - 44 sentences/language (train 32, test 12)
  - 3000 bootstrap resamples (lesion)
  - Lesion fractions: {0.05, 0.1, 0.2, 0.5, 1}%
""")

# Count models in each source
print("  Model counts per result file:")
for name, d in [("steer_cv2", sc2), ("number_control", nc), ("number_disentangle", nd),
                ("lesion_control", lc), ("cognitive_map", cm), ("number_tuning", nt),
                ("factor_2x2", f2), ("norms", nm), ("steer_cv", scv),
                ("naive_concept", nv), ("lesion_altattr", la)]:
    keys = [k for k in d if k != "_trend" and "bf16" not in k and "8bit" not in k]
    print(f"    {name}: {len(keys)} models")

# Count families
all_fams = set()
for src in [nc, lc, cm, sc2, nd]:
    for v in src.values():
        if isinstance(v, dict) and 'family' in v:
            all_fams.add(v['family'])
print(f"  Families: {sorted(all_fams)}  count={len(all_fams)}")

# Size range
all_sizes = set()
for src in [sc2, nc, lc, cm, nd]:
    for k, v in src.items():
        if isinstance(v, dict) and 'size' in v and "bf16" not in k and "8bit" not in k:
            all_sizes.add(v['size'])
print(f"  Size range: {min(all_sizes)}–{max(all_sizes)}B")

# Concepts count
concepts_sc2 = set()
for k, v in sc2.items():
    if isinstance(v, dict) and 'per_concept_eff' in v:
        concepts_sc2.update(v['per_concept_eff'].keys())
        break
print(f"  Concepts in steer_cv2: {len(concepts_sc2)}  names: {sorted(concepts_sc2)}")

# Dose-response coefs
print(f"  Dose-response coefs: {sn.get('coefs')}")
print(f"  Raw coefs: {sn.get('rawcoefs')}")

# Layer sweep points
lsw = sn.get("layer_sweep", {})
if lsw:
    sample = list(lsw.values())[0]
    print(f"  Layer sweep fractions: {[p[0] for p in sample]}")

# Lesion fractions
lfrac = sn.get("lesion_frac", {})
if lfrac:
    sample = list(lfrac.values())[0]
    print(f"  Lesion fractions: {[r['kfrac']*100 for r in sample]}%")

# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("10. FIGURE DATA POINTS")
print(SEP)

# Fig main (a): norms
print("\n── Fig main (a): ‖d‖/‖h‖ ratio vs size ──")
for s, r in ratios:
    print(f"  {s}B → {r:.4f}")

# Fig main (b): naive gain
print("\n── Fig main (b): naive gain vs size ──")
naive_pts = sorted([(v["size"], v["naive_gain"]) for v in nv.values()])
for s, g in naive_pts:
    print(f"  {s}B → {g:.4f}")

# Fig main (c): dose-response
print("\n── Fig main (c): dose-response (steering_dose) ──")
sd = sn.get("steering_dose", {})
coefs = sn.get("coefs", [])
print(f"  Coefs: {coefs}")
for m in sorted(sd, key=lambda k: sd[k]['size']):
    v = sd[m]
    print(f"  {m} ({v['size']}B): {v['eff']}")

# Fig main (d): fixed vs held-out pass rates
print("\n── Fig main (d): fixed vs held-out (Qwen3) ──")
print("  Fixed:")
for s, p in qwen_fixed:
    print(f"    {s}B → {p:.2f}")
print("  Held-out:")
for s, p, ci in qwen_ho:
    print(f"    {s}B → {p:.2f}  CI={ci}")

# Fig magnitude (a): tuning r
print("\n── Fig magnitude (a): held-out tuning r vs size ──")
for m in sorted(nt, key=lambda k: nt[k].get('size', 0)):
    if "8bit" in m: continue
    v = nt[m]
    print(f"  {m} ({v.get('size')}B): r={v['best_r_mean']:.4f}  null={v['null_mean']:.4f}")

# Fig magnitude (b): bell fractions
print("\n── Fig magnitude (b): bell fraction vs size ──")
for m in sorted(nc, key=lambda k: nc[k].get('size', 0)):
    v = nc[m]
    print(f"  {m} ({v.get('size')}B): bell_frac={v['new_bell_frac']:.4f}")

# Fig lesion (a): EN selectivity with CI
print("\n── Fig lesion (a): EN selectivity with CI ──")
for m in sorted(lc, key=lambda k: lc[k].get('size', 0)):
    v = lc[m]
    sig = v.get('en_ci', [0, 0])[0] > 0 or v.get('en_ci', [0, 0])[1] < 0
    print(f"  {m}: en_sel={v.get('en_sel',0):+.2f}  CI={v.get('en_ci')}  sig={sig}")

# Fig lesion (b): lesion fraction sweep
print("\n── Fig lesion (b): ZH selectivity vs ablation fraction ──")
for m in sorted(lfrac):
    rows = lfrac[m]
    print(f"  {m}:")
    for r in rows:
        print(f"    kfrac={r['kfrac']*100:.2f}%  zh_sel={r['zh_sel']:.2f}")

# Fig map: average R²
print("\n── Fig map: average R² vs size ──")
for m in sorted(cm, key=lambda k: cm[k].get('size', 0)):
    if "8bit" in m: continue
    v = cm[m]
    print(f"  {m} ({v.get('size')}B): avg_R²={(v['r2_lat']+v['r2_lon'])/2:.4f}")

# Fig scale (a): steering effect all models
print("\n── Fig scale (a): steering held-out effect, all models ──")
for m in sorted(sc2, key=lambda k: sc2[k].get('size', 0) if isinstance(sc2[k], dict) else 0):
    if m == "_trend" or "bf16" in m: continue
    v = sc2[m]
    if not isinstance(v, dict) or 'heldout_eff' not in v: continue
    ci = v.get('eff_ci')
    coarse = v.get('size', 0) >= 24
    print(f"  {m} ({v.get('size')}B, {'coarse' if coarse else 'fine'}): "
          f"eff={v['heldout_eff']:+.2f}  CI={ci}")

# Fig scale (b): quant control bar chart
print("\n── Fig scale (b): quant control Qwen3-32B ──")
for label, key in [("8-bit", "Qwen3-32B"), ("bf16", "Qwen3-32B-bf16")]:
    v = sc2.get(key, {})
    print(f"  {label}: eff={v.get('heldout_eff')}  pass={v.get('heldout_pass')}")

# Fig scale (c): cross-format consistency
print("\n── Fig scale (c): digit-vs-word cross-format r ──")
for m in sorted(nd, key=lambda k: nd[k].get('size', 0)):
    if "8bit" in m: continue
    v = nd[m]
    print(f"  {m} ({v.get('size')}B, {v.get('family','')}): r={v['bell_consistency_mean']:.4f}")

# Fig scale (d): ZH/EN selectivity
print("\n── Fig scale (d): ZH and EN selectivity ──")
for m in sorted(lc, key=lambda k: lc[k].get('size', 0)):
    if "8bit" in m: continue
    v = lc[m]
    print(f"  {m} ({v.get('size')}B): zh={v.get('zh_sel',0):+.2f} [{'SIG' if v.get('zh_sig') else 'n.s.'}]  "
          f"en={v.get('en_sel',0):+.2f} [{'SIG' if v.get('en_sig') else 'n.s.'}]")

# Fig factor: 2×2 decomposition
print("\n── Fig factor: 2×2 cell values ──")
for m in sorted(f2, key=lambda k: f2[k]['size']):
    v = f2[m]
    print(f"  {m} ({v['size']}B): A={v['cell_A_raw_fixed']:.4f}  B={v['cell_B_raw_heldout']:.4f}  "
          f"C={v['cell_C_norm_fixed']:.4f}  D={v['cell_D_norm_heldout']:.4f}")

# Fig norms
print("\n── Fig norms: raw norms ──")
for s, _ in ratios:
    v = [x for x in nm.values() if x['size'] == s][0]
    print(f"  {s}B: resid_norm={v['resid_norm']:.2f}  dir_norm={v['dir_norm']:.2f}  ratio={v['ratio']:.4f}")

# Fig steer_dist: per-concept effects
print("\n── Fig steer_dist: per-concept held-out effects (Qwen3 ≤14B) ──")
q3_models = sorted([(k, v) for k, v in sc2.items()
                     if k.startswith("Qwen3") and "bf16" not in k
                     and isinstance(v, dict) and v.get('size', 99) <= 14
                     and 'per_concept_eff' in v],
                    key=lambda x: x[1]['size'])
for m, v in q3_models:
    effs = list(v['per_concept_eff'].values())
    print(f"  {m} ({v['size']}B): mean={np.mean(effs):.2f}  "
          f"min={min(effs):.2f}  max={max(effs):.2f}  n={len(effs)}")

# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("11. QUALITATIVE EXAMPLES (App D)")
print(SEP)

if qe:
    print(f"  Model: {qe.get('model')}")
    print(f"  Layer: {qe.get('layer')}")
    print(f"  Resid norm: {qe.get('resid_norm')}")
    n_ex = len(qe.get('examples', []))
    print(f"  Number of examples: {n_ex}")
    for ex in qe.get('examples', [])[:3]:
        print(f"    concept={ex.get('concept')}  cat={ex.get('category')}  coef={ex.get('coef')}")
else:
    print("  qual_examples.json not found or empty")

# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("12. SPECIFIC IN-TEXT NUMBER CHECKS")
print(SEP)

print("\n── Checking specific numbers mentioned in paper text ──")

# §4.3: "r=0.68–0.98"
rvals_nt = [v['best_r_mean'] for k, v in nt.items() if "8bit" not in k]
print(f"  Tuning r range: {min(rvals_nt):.2f}–{max(rvals_nt):.2f}  (paper: 0.68–0.98)")

# §4.3: "cross-format 0.59–0.70" for L/P/M
print(f"  Cross-fmt L/P/M: {min(lpm_xf):.2f}–{max(lpm_xf):.2f}  (paper: 0.59–0.70)")

# §4.3: "random baseline 0.30–0.51"
rand_baselines = [v.get('random_consistency_mean') for v in nd.values()
                  if v.get('family') in ('Llama', 'Phi', 'Mistral') and v.get('random_consistency_mean') is not None]
if rand_baselines:
    print(f"  Random baseline L/P/M: {min(rand_baselines):.2f}–{max(rand_baselines):.2f}  (paper: 0.30–0.51)")

# §4.3: "73–83% of bell units > 0.5"
frac_high = [v.get('bell_consistency_frac_high') for v in nd.values()
             if v.get('family') in ('Llama', 'Phi', 'Mistral') and v.get('bell_consistency_frac_high') is not None]
if frac_high:
    print(f"  Bell >0.5 fraction L/P/M: {min(frac_high)*100:.0f}–{max(frac_high)*100:.0f}%  (paper: 73–83%)")

# §4.4: "44 sentences per language" — design parameter, can't verify from JSON
# §4.4: "+10.7 nats ... +0.5 nats" — qualitative example
print(f"\n  Lesion example: +10.7 / +0.5 nats — from qualitative, not in these JSONs")

# §4.5: "R² ≈ 0.43–0.68"
print(f"  Map R² range: {min(all_r2):.2f}–{max(all_r2):.2f}  (paper: 0.43–0.68)")

# Map example: Qwen3-1.7B R²=0.62/0.66
q17 = cm.get("Qwen3-1.7B", {})
print(f"  Qwen3-1.7B map: lat={q17.get('r2_lat')}  lon={q17.get('r2_lon')}  (paper: 0.62/0.66)")

# Number example: cross-format r=0.96
print(f"  Number example r=0.96 — from qualitative, not in these aggregated JSONs")

# §4.1: "d' ≈ 6 in-sample, ≈ 2 out-of-sample"
print(f"  d' values — from concept_steering.json per_concept data, not directly stored")

# Large model specific: Qwen3-32B en_sel = +3.7, Qwen2.5-72B en_sel = +0.5
q32_en = lc.get("Qwen3-32B", {}).get('en_sel')
q72_en = lc.get("Qwen2.5-72B", {}).get('en_sel')
print(f"  Qwen3-32B en_sel: {q32_en}  (paper: +3.7)")
print(f"  Qwen2.5-72B en_sel: {q72_en}  (paper: +0.5)")

# Llama-70B x-fmt: paper says ≈0.7
l70_xf = nd.get("Llama-3.1-70B", {}).get('bell_consistency_mean')
print(f"  Llama-70B cross-fmt r: {l70_xf}  (paper: ≈0.7)")

# Mistral-24B x-fmt: paper says ≈0.7
m24_xf = nd.get("Mistral-Small-24B", {}).get('bell_consistency_mean')
print(f"  Mistral-24B cross-fmt r: {m24_xf}  (paper: ≈0.7)")

print(f"\n{SEP}")
print("DONE. Review above for any ✗ MISMATCH or unexpected values.")
print(SEP)
