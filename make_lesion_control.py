"""Lesion control: is 'English lesioning is fragile / flips by model' real or noise?

exp03 used only 8 sentences per language and reported a point estimate. Here we (1) expand to
44 sentences per language and (2) bootstrap the test sentences to put a 95% CI on the Chinese-
and English-selectivity. If a selectivity's CI excludes 0, the (non-)localization is real for
that model; if CIs straddle 0, the 'fragility' is just noise.
"""
import os, json, gc, numpy as np, torch, mlib
from mlib import load, text_nll, edited_neurons
from exp03_lesion import attribution
from exp01_concept import FAMILY

ZH = ["今天天气很好,我们去公园散步吧。","他昨天买了一本新书。","这道菜的味道非常鲜美。",
 "北京是中国的首都,历史悠久。","孩子们在操场上快乐地玩耍。","请把窗户关上,外面有点冷。",
 "这部电影讲述了一个感人的故事。","我每天早上都会喝一杯茶。","她正在认真地准备明天的考试。",
 "这座桥是三年前建成的。","我们计划暑假去南方旅行。","厨房里飘来了饭菜的香味。",
 "老师耐心地讲解了这道难题。","春天到了,花园里开满了鲜花。","他喜欢在周末去图书馆看书。",
 "火车马上就要进站了,请大家做好准备。","这家餐厅的服务态度很好。","我的手机快没电了,需要充电。",
 "医生建议他多运动、少熬夜。","昨晚下了一场很大的雨。","公司下周要开一个重要的会议。",
 "小猫蜷缩在沙发上睡着了。","这条街上新开了一家咖啡馆。","爷爷每天早上都去河边锻炼。",
 "考试成绩公布后,大家都松了一口气。","这本小说的结局让人意想不到。","超市里的水果又新鲜又便宜。",
 "他花了整个下午修理那辆自行车。","远处的山峰被白雪覆盖着。","妹妹正在学习弹钢琴。",
 "这次旅行让我们留下了美好的回忆。","邻居家的狗特别喜欢叫。","会议一直持续到很晚才结束。",
 "他把房间打扫得干干净净。","秋天的落叶铺满了整条小路。","我们约好周五一起去看电影。",
 "这台电脑的运行速度很快。","母亲做的饺子是我最爱吃的。","天空中飘着几朵白云。",
 "他终于通过了驾驶考试。","这幅画挂在客厅的墙上。","孩子把积木堆得很高。",
 "早市上人来人往非常热闹。","雨过天晴后出现了一道彩虹。"]
EN = ["The weather is lovely today, let's take a walk in the park.","He bought a brand new book yesterday.",
 "This dish tastes absolutely delicious.","Beijing is the capital of China, with a long history.",
 "The children are playing happily on the playground.","Please close the window, it is a bit cold outside.",
 "The film tells a very moving story.","I drink a cup of tea every single morning.",
 "She is carefully preparing for tomorrow's exam.","This bridge was built three years ago.",
 "We plan to travel to the south this summer.","The smell of cooking drifted from the kitchen.",
 "The teacher patiently explained the difficult problem.","Spring has come and the garden is full of flowers.",
 "He likes going to the library to read on weekends.","The train is about to arrive, please get ready.",
 "The service at this restaurant is excellent.","My phone is almost dead and needs charging.",
 "The doctor advised him to exercise more and sleep earlier.","It rained very heavily last night.",
 "The company will hold an important meeting next week.","The kitten curled up on the sofa and fell asleep.",
 "A new coffee shop just opened on this street.","Grandpa exercises by the river every morning.",
 "Everyone felt relieved after the exam results came out.","The ending of this novel was completely unexpected.",
 "The fruit at the supermarket is fresh and cheap.","He spent the whole afternoon fixing that bicycle.",
 "The distant mountain peaks are covered with white snow.","My younger sister is learning to play the piano.",
 "This trip left us with wonderful memories.","The neighbour's dog barks an awful lot.",
 "The meeting went on until very late at night.","He cleaned the room until it was spotless.",
 "Autumn leaves covered the entire little path.","We agreed to go see a movie together on Friday.",
 "This computer runs remarkably fast.","The dumplings my mother makes are my favourite.",
 "A few white clouds are drifting across the sky.","He finally passed his driving test.",
 "This painting hangs on the living-room wall.","The child stacked the wooden blocks very high.",
 "The morning market was bustling with people.","A rainbow appeared after the rain cleared up."]
ZH_TR, ZH_TE = ZH[:32], ZH[32:]; EN_TR, EN_TE = EN[:32], EN[32:]
KFRAC = 0.002
MODELS = [("Qwen3-0.6B",0.6,"models_dl/Qwen3-0.6B"),("Qwen3-1.7B",1.7,"models_dl/Qwen3-1.7B"),
 ("Qwen3-4B",4.0,"models_dl/Qwen3-4B"),("Qwen3-8B",8.0,"models_dl/Qwen3-8B"),
 ("Qwen3-14B",14.0,"models_dl/Qwen3-14B"),("Llama-3.1-8B",8.0,"models_dl/Llama-3.1-8B-Instruct"),
 ("Ministral-8B",8.0,"models_dl/Ministral-8B-Instruct-2410"),
 ("Phi-3.5-mini",3.8,"models_dl/Phi-3.5-mini-instruct"),("phi-4",14.0,"models_dl/phi-4")]

def per_sent_dnll(tok, model, texts, abl):
    base = np.array([text_nll(tok, model, t) for t in texts])
    with edited_neurons(model, ablate=abl):
        ab = np.array([text_nll(tok, model, t) for t in texts])
    return ab - base

def run(tok, model):
    az = attribution(model, tok, ZH_TR); ae = attribution(model, tok, EN_TR)
    L, U = az.shape; K = max(50, int(KFRAC * L * U)); diff = (az - ae).reshape(-1)
    zh = [divmod(int(i), U) for i in np.argsort(diff)[::-1][:K]]   # Chinese neurons
    en = [divmod(int(i), U) for i in np.argsort(diff)[:K]]         # English neurons
    dzh_aZH = per_sent_dnll(tok, model, ZH_TE, zh); den_aZH = per_sent_dnll(tok, model, EN_TE, zh)
    den_aEN = per_sent_dnll(tok, model, EN_TE, en); dzh_aEN = per_sent_dnll(tok, model, ZH_TE, en)
    rng = np.random.default_rng(0)
    def boot(a, b):
        v = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
             for _ in range(3000)]
        return float(np.mean(v)), [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    zs, zci = boot(dzh_aZH, den_aZH)     # Chinese selectivity
    es, eci = boot(den_aEN, dzh_aEN)     # English selectivity
    return {"zh_sel": zs, "zh_ci": zci, "en_sel": es, "en_ci": eci,
            "zh_sig": bool(zci[0] > 0), "en_sig": bool(eci[0] > 0), "K": K, "n_test": len(ZH_TE)}

def main():
    out = json.load(open("results/lesion_control.json")) if os.path.exists("results/lesion_control.json") else {}
    for name, size, path in MODELS:
        if not os.path.exists(path): print("skip", name); continue
        print(f"==== {name} ====", flush=True)
        try:
            tok, model = load(path); r = run(tok, model); r["size"] = size; r["family"] = FAMILY(name)
            out[name] = r; json.dump(out, open("results/lesion_control.json", "w"), indent=2)
            print(f"  zh_sel={r['zh_sel']:+.2f} CI{[round(x,2) for x in r['zh_ci']]} sig={r['zh_sig']} | "
                  f"en_sel={r['en_sel']:+.2f} CI{[round(x,2) for x in r['en_ci']]} sig={r['en_sig']}", flush=True)
            del model, tok
        except Exception as e:
            import traceback; print("FAIL", e); traceback.print_exc()
        mlib._CACHE.clear(); gc.collect(); torch.cuda.empty_cache()
    print("DONE")

if __name__ == "__main__":
    main()
