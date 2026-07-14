# When Is a Steerable Concept Representation Real? — Code & Data

Audit protocol, stimuli, per-model results, and figure/table code for the paper
*"When Is a Steerable Concept Representation Real? Measurement Confounds in a Cross-Family Audit of
Neuroscience Parallels in LLMs."*

Everything here is **real and reproducible**: every number, figure, and table in the paper is
generated from the JSON files in `results/` by the scripts listed below. No values are hand-entered
into the paper except through these files.

## Setup
```bash
pip install -r requirements.txt          # torch, transformers, scikit-learn, numpy, matplotlib
# models are open-weight; set a local path or let transformers download from Hugging Face.
# edit the GRID paths in exp01_concept.py to point at your checkpoints.
```
Interventions use forward hooks on the residual stream and MLP layers (`mlib.py`). Checkpoints
≤14B run in bf16 on one GPU; 24–72B use 4/8-bit quantization.

## Reproduce the results
```bash
# experiments (require GPU + model checkpoints):
python make_steer_cv2.py        # held-out steering (Qwen3 ladder)      -> results/steer_cv2.json
python make_fill_table.py       # steering+lesion on remaining models   -> merges into results/
python make_number_control.py   # shape-agnostic bell-neuron selection  -> results/number_control.json
python make_number_disentangle.py  # digit-vs-word cross-format test     -> results/number_disentangle.json
python make_lesion_control.py   # language lesion, bootstrap CIs         -> results/lesion_control.json
python make_lesion_altattr.py   # lesion under activation-magnitude attr -> results/lesion_altattr.json
python exp02_map.py             # world-map ridge probes                 -> results/cognitive_map.json
python make_qual_all.py         # qualitative artifacts (number/map/lesion)
python make_qual_fig.py         # steering transcript artifact
# figures + table (no GPU needed, reads results/*.json):
python make_figures3.py         # -> paper/figures/*.pdf
python make_graphical.py        # -> paper/figures/fig_graphical.pdf
python make_table.py            # -> paper/tab_results.tex
# verification (no GPU needed):
python verify_all_numbers.py    # prints every paper number with JSON source for proofreading
```

## Data provenance — every paper artifact traces to a file

| Paper artifact | Script | Data file |
|---|---|---|
| Fig. 1a–c (units, dose–response, layer) | `make_figures3.py` | `sensitivity.json`, `naive_concept.json`, `concept_steering.json` |
| Fig. 1d (fixed vs held-out operating point) | `make_figures3.py` | `steer_cv.json`, `steer_cv2.json` |
| Table 2 (per-model, all four experiments) | `make_table.py` | `steer_cv2`, `number_control`, `number_disentangle`, `lesion_control`, `cognitive_map` |
| Steering trend `slope +0.31 [−0.11,+0.73]` | `make_steer_cv2.py` | `steer_cv2.json["_trend"]` |
| Large-model steering + CIs; 32B 8-bit vs bf16 | `make_fill_table.py` | `steer_cv2.json` |
| Magnitude tuning `r=0.68–0.98` | `exp04_number.py` | `number_tuning.json` |
| Bell fraction `13–43%` (L/P/M); `0%` under linear sel. | `make_number_control.py` | `number_control.json` |
| Cross-format `r 0.59–0.70` vs random `0.30–0.51` | `make_number_disentangle.py` | `number_disentangle.json` |
| Lesion: Chinese 17/17, English 5/17 sig, dbl 5/17 | `make_lesion_control.py` | `lesion_control.json` |
| Lesion attribution flip (7/9, 7/9) | `make_lesion_altattr.py` | `lesion_altattr.json` |
| World map `R² 0.43–0.68` | `exp02_map.py` | `cognitive_map.json` |
| Fig. 2 graphical abstract | `make_graphical.py` | `naive_concept.json`, `steer_cv2.json` |
| Fig. 6 steering transcript | `make_qual_fig.py` | `qual_examples.json` |
| Figs. 7–9 (number curves, lesion, map) | `make_qual_all.py` | live model activations |

## Models (17 open checkpoints, 0.6–72B, 5 families)
Qwen3 {0.6, 1.7, 4, 8, 14, 32}B and Qwen2.5-72B; Llama-3.2 {1, 3}B, Llama-3.1-8B, and
Llama-3.1-70B; Phi-3.5-mini and Phi-4; Ministral-8B and Mistral-Small-24B; Gemma-2 {9, 27}B.
All are open-weight (Hugging Face).

## Notes
- Only open-weight models and curated/synthetic text stimuli are used; no human-subject or sensitive
  data. Chinese stimuli live in `make_lesion_control.py`.
