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
Interventions use forward hooks on the residual stream and MLP layers (`mlib.py`). 



## Models (14 open checkpoints, 0.6–72B)
Qwen3 {0.6, 1.7, 4, 8, 14, 32}B and Qwen2.5-72B; Llama-3.2 {1, 3}B and Llama-3.1-8B; Phi-3.5-mini
and Phi-4; Ministral-8B and Mistral-Small-24B. All are open-weight (Hugging Face).

## Citation
```

```
