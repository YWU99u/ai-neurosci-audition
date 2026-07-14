"""Load-bearing control for the paper's positive claim ('steerability is flat, not scaling').

Upgrades the 8-concept, near-binary, no-CI, layer-2/3 scan to a defensible test:
  - 24 concepts across 5 categories (cities/persons/objects/animals/abstract);
  - operating point (layer x strength) chosen by HELD-OUT concepts (4-fold CV);
  - a CONTINUOUS effect size per concept, not just a binary pass;
  - bootstrap 95% CIs over concepts for held-out pass-rate and effect;
  - per-category effects; and a trend test (slope of effect vs log-size) so 'flat' is earned.
Qwen3 0.6-14B (local). Saves results/steer_cv2.json.
"""
import os, json, gc, numpy as np, torch, contextlib
import mlib
from mlib import load, layer_residuals, residual_norm, steer_residual, target_score, unit, n_layers
from exp01_concept import NEUTRAL, PROBES as ALLPROBES

PROBES = ALLPROBES[:3]
LFRAC = [0.4, 0.6, 0.8]; COEFS = [0.25, 0.5, 1.0, 2.0, 4.0]; NRAND = 1
DL = "models_dl"
MODELS = [("Qwen3-0.6B",0.6),("Qwen3-1.7B",1.7),("Qwen3-4B",4.0),("Qwen3-8B",8.0),("Qwen3-14B",14.0)]

# 24 concepts, 12 same-category control pairs. sents build the direction; words score the effect.
S = {  # concept: (category, [sentences], [target words], control-concept)
 "Paris":("city",["Paris is the capital of France.","巴黎有埃菲尔铁塔。","We flew into Paris in spring."],["Paris","France","Eiffel","French"],"Tokyo"),
 "Tokyo":("city",["Tokyo is the capital of Japan.","东京是日本的大城市。","We took the train across Tokyo."],["Tokyo","Japan","Shibuya","Japanese"],"Paris"),
 "London":("city",["London sits on the river Thames.","伦敦是英国的首都。","Big Ben chimes over London."],["London","England","Thames","British"],"Beijing"),
 "Beijing":("city",["Beijing is the capital of China.","北京有故宫和长城。","We walked through old Beijing."],["Beijing","China","Forbidden","Chinese"],"London"),
 "Cairo":("city",["Cairo lies beside the Nile.","开罗靠近金字塔。","The desert stretches past Cairo."],["Cairo","Egypt","Nile","Egyptian"],"Sydney"),
 "Sydney":("city",["Sydney has a famous opera house.","悉尼在澳大利亚。","Boats fill Sydney harbour."],["Sydney","Australia","harbour","Australian"],"Cairo"),
 "Einstein":("person",["Einstein developed relativity.","爱因斯坦研究物理。","Einstein wrote E=mc^2."],["Einstein","relativity","physicist","physics"],"Shakespeare"),
 "Shakespeare":("person",["Shakespeare wrote Hamlet.","莎士比亚是剧作家。","Shakespeare penned many sonnets."],["Shakespeare","Hamlet","playwright","sonnet"],"Einstein"),
 "Mozart":("person",["Mozart composed symphonies.","莫扎特是音乐家。","Mozart wrote operas as a child."],["Mozart","symphony","composer","opera"],"Newton"),
 "Newton":("person",["Newton described gravity.","牛顿研究力学。","Newton invented calculus."],["Newton","gravity","calculus","motion"],"Mozart"),
 "Piano":("object",["She played a tune on the piano.","他在弹钢琴。","The piano has 88 keys."],["piano","keys","melody","keyboard"],"Dinosaur"),
 "Dinosaur":("object",["The dinosaur was a giant reptile.","恐龙早已灭绝。","Fossils reveal huge dinosaurs."],["dinosaur","fossil","extinct","reptile"],"Piano"),
 "Telescope":("object",["The telescope reveals distant stars.","望远镜观测星空。","Astronomers peer through a telescope."],["telescope","lens","stars","astronomy"],"Volcano"),
 "Volcano":("object",["The volcano erupted with lava.","火山喷出岩浆。","Ash rose above the volcano."],["volcano","lava","eruption","magma"],"Telescope"),
 "Elephant":("animal",["The elephant has a long trunk.","大象是陆地上最大的动物。","Elephants roam the savanna."],["elephant","trunk","tusk","savanna"],"Dolphin"),
 "Dolphin":("animal",["The dolphin leaps from the sea.","海豚很聪明。","Dolphins swim in pods."],["dolphin","ocean","fin","marine"],"Elephant"),
 "Eagle":("animal",["The eagle soars over the cliffs.","老鹰在天空盘旋。","An eagle has sharp talons."],["eagle","wings","talon","soar"],"Spider"),
 "Spider":("animal",["The spider spun a web.","蜘蛛结了一张网。","Spiders have eight legs."],["spider","web","legs","silk"],"Eagle"),
 "Justice":("abstract",["Justice demands fair treatment.","正义需要公平。","The court seeks justice."],["justice","fairness","court","law"],"Freedom"),
 "Freedom":("abstract",["Freedom lets people choose.","自由是一种权利。","They fought for freedom."],["freedom","liberty","rights","free"],"Justice"),
 "Sadness":("abstract",["Sadness weighed on her heart.","悲伤让人流泪。","A wave of sadness passed."],["sadness","grief","sorrow","tears"],"Courage"),
 "Courage":("abstract",["Courage means facing fear.","勇气战胜恐惧。","She showed great courage."],["courage","bravery","brave","valor"],"Sadness"),
 "Music":("abstract",["Music filled the concert hall.","音乐让人放松。","He studies classical music."],["music","melody","song","rhythm"],"Medicine"),
 "Medicine":("abstract",["Medicine heals the sick.","医学能治病。","She practises modern medicine."],["medicine","doctor","cure","health"],"Music"),
}
CONC = list(S)

def contrast(tok, model, Lc, d, alpha, tw, cw):
    v = []
    for p in PROBES:
        ctx = steer_residual(model, Lc, d, alpha) if alpha else contextlib.nullcontext()
        with ctx: v.append(target_score(tok, model, p, tw) - target_score(tok, model, p, cw))
    return float(np.nanmean(v))

def cells_for_model(tok, model):
    L = n_layers(model); rng = np.random.default_rng(0); cells = {}
    for lf in LFRAC:
        Lc = max(1, min(L - 1, int(round(L * lf))))
        nmean = np.stack([layer_residuals(tok, model, t, Lc) for t in NEUTRAL]).mean(0); H = nmean.shape[0]
        rn = residual_norm(tok, model, PROBES, Lc)
        dirs = {c: unit(np.stack([layer_residuals(tok, model, t, Lc) for t in S[c][1]]).mean(0) - nmean) for c in CONC}
        rand = [unit(rng.standard_normal(H)) for _ in range(NRAND)]
        base = {c: contrast(tok, model, Lc, dirs[c], 0.0, S[c][2], S[S[c][3]][2]) for c in CONC}
        for coef in COEFS:
            cell = {}
            for c in CONC:
                tw, cw = S[c][2], S[S[c][3]][2]
                et = contrast(tok, model, Lc, dirs[c], coef * rn, tw, cw) - base[c]
                er = float(np.mean([contrast(tok, model, Lc, rd, coef * rn, tw, cw) - base[c] for rd in rand]))
                cell[c] = {"et": float(et), "er": er}
            cells[f"{lf}|{coef}"] = cell
    return cells

def cv_heldout(cells):
    """4-fold CV over concepts: pick cell on select folds, record each concept's effect held-out."""
    rng = np.random.default_rng(0); idx = rng.permutation(len(CONC)); folds = np.array_split(idx, 4)
    ho_eff = {}; ho_pass = {}
    for f in range(4):
        ev = [CONC[i] for i in folds[f]]; sel = [CONC[i] for i in idx if CONC[i] not in ev]
        best = max(cells, key=lambda k: np.mean([cells[k][c]["et"] for c in sel]))
        for c in ev:
            ho_eff[c] = cells[best][c]["et"]
            ho_pass[c] = float(cells[best][c]["et"] > 0.5 and cells[best][c]["et"] > 2 * abs(cells[best][c]["er"]))
    return ho_eff, ho_pass

def main():
    out = json.load(open("results/steer_cv2.json")) if os.path.exists("results/steer_cv2.json") else {}
    for name, size in MODELS:
        path = f"{DL}/{name}"
        if not os.path.exists(path): print("skip", name); continue
        print(f"==== {name} ====", flush=True)
        try:
            tok, model = load(path); cells = cells_for_model(tok, model)
            he, hp = cv_heldout(cells)
            eff = np.array([he[c] for c in CONC]); pas = np.array([hp[c] for c in CONC])
            rng = np.random.default_rng(1); bt = rng.integers(0, len(CONC), size=(4000, len(CONC)))
            eff_ci = [float(np.percentile(eff[bt].mean(1), 2.5)), float(np.percentile(eff[bt].mean(1), 97.5))]
            pass_ci = [float(np.percentile(pas[bt].mean(1), 2.5)), float(np.percentile(pas[bt].mean(1), 97.5))]
            cats = sorted(set(S[c][0] for c in CONC))
            cat_eff = {ct: float(np.mean([he[c] for c in CONC if S[c][0] == ct])) for ct in cats}
            out[name] = {"size": size, "heldout_eff": float(eff.mean()), "eff_ci": eff_ci,
                         "heldout_pass": float(pas.mean()), "pass_ci": pass_ci,
                         "per_concept_eff": {c: he[c] for c in CONC}, "cat_eff": cat_eff}
            json.dump(out, open("results/steer_cv2.json", "w"), indent=2)
            print(f"  eff={eff.mean():+.2f} CI{[round(x,2) for x in eff_ci]}  pass={pas.mean():.2f} CI{[round(x,2) for x in pass_ci]}", flush=True)
            del model, tok
        except Exception as e:
            import traceback; print("FAIL", e); traceback.print_exc()
        mlib._CACHE.clear(); gc.collect(); torch.cuda.empty_cache()
    # trend test: slope of held-out effect vs log2(size), bootstrap over concepts
    if len(out) >= 3:
        names = [m for m, _ in MODELS if m in out]; sizes = np.log2([out[m]["size"] for m in names])
        rng = np.random.default_rng(2); slopes = []
        for _ in range(4000):
            bi = rng.integers(0, len(CONC), len(CONC))
            ys = [np.mean([out[m]["per_concept_eff"][CONC[i]] for i in bi]) for m in names]
            slopes.append(np.polyfit(sizes, ys, 1)[0])
        out["_trend"] = {"slope_eff_vs_log2size": float(np.mean(slopes)),
                         "slope_ci": [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]}
        json.dump(out, open("results/steer_cv2.json", "w"), indent=2)
        print(f"\nTREND slope(eff vs log2 size) = {out['_trend']['slope_eff_vs_log2size']:+.3f} "
              f"CI{[round(x,3) for x in out['_trend']['slope_ci']]}  (flat if CI includes 0)")
    print("DONE")

if __name__ == "__main__":
    main()
