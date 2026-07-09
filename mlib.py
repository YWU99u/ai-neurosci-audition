"""Shared utilities for neuron-level LLM-neuroscience experiments."""
import os, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

_CACHE = {}

def freest_gpu():
    best, bestfree = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > bestfree:
            best, bestfree = i, free
    return f"cuda:{best}"

def load(model_id):
    """Load a model. Cloud/large-model options via env vars:
       MULTI_GPU=1  -> device_map='auto' (shard across all GPUs; needs `accelerate`)
       LOAD_8BIT=1  -> 8-bit quantization + device_map='auto' (needs `bitsandbytes`; ~70GB for
                       a 70B model — best fidelity that fits a single 96GB GH200)
       LOAD_4BIT=1  -> 4-bit NF4 quantization + device_map='auto' (~40GB for a 70B model,
                       fits a single 80GB GPU; noisier for steering)
       DEVICE=cuda:k -> pin to one GPU (default: freest single GPU)."""
    if model_id in _CACHE:
        return _CACHE[model_id]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kw = dict(dtype=torch.bfloat16)
    if os.environ.get("LOAD_4BIT") == "1":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        kw["device_map"] = "auto"
    elif os.environ.get("LOAD_8BIT") == "1":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kw["device_map"] = "auto"
    elif os.environ.get("MULTI_GPU") == "1":
        kw["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except (ValueError, KeyError) as e:
        msg = str(e).lower()
        if any(k in msg for k in ("memory", "cpu or the disk", "offload", "quantiz")):
            raise
        if "trust_remote_code" in msg or "custom code" in msg:
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, **kw)
        else:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    if "device_map" not in kw:                     # single-GPU: place manually
        dev = os.environ.get("DEVICE") or freest_gpu()
        model = model.to(dev)
    model.eval()
    print(f"[loaded {model_id} | device_map={kw.get('device_map','single')} ]")
    _CACHE[model_id] = (tok, model)
    return tok, model

# ---- single-neuron recording (concept-cell experiments) ----

def _layers(model):
    """Decoder layer list, across standard CausalLM and vision-language models."""
    m = model.model if hasattr(model, "model") else model
    if hasattr(m, "layers"):
        return m.layers
    if hasattr(m, "language_model"):
        return m.language_model.layers
    raise AttributeError("cannot locate decoder layers")

def _down_proj(layer):
    if hasattr(layer, "mlp"):
        return layer.mlp.down_proj
    if hasattr(layer, "feed_forward"):
        return layer.feed_forward.w2
    raise AttributeError(f"no MLP down-projection found in {type(layer)}")

@torch.no_grad()
def record_mlp_neurons(tok, model, text, reduce="max"):
    """Return array [n_layers, intermediate] of each MLP neuron's activation on
    `text`, reduced over token positions ('max' = peak 'firing rate', or 'mean').
    Neuron = a unit of the MLP intermediate layer (input to down_proj)."""
    layers = _layers(model)
    caps = {}
    def mk(i):
        def pre_hook(mod, args):
            caps[i] = args[0].detach()[0]        # [seq, intermediate]
        return pre_hook
    handles = [_down_proj(layers[i]).register_forward_pre_hook(mk(i))
               for i in range(len(layers))]
    ids = tok(text, return_tensors="pt").to(model.device)
    model(**ids)
    for h in handles:
        h.remove()
    out = []
    for i in range(len(layers)):
        a = caps[i].float()
        if reduce == "max":
            v = a.max(0).values
        elif reduce == "last":
            v = a[-1]
        else:
            v = a.mean(0)
        out.append(v.cpu().numpy())
    return np.stack(out)                          # [n_layers, intermediate]

def record_batch(tok, model, texts, reduce="max"):
    return np.stack([record_mlp_neurons(tok, model, t, reduce) for t in texts])

def dprime(pos, neg):
    """Sensitivity index d' between two activation samples (per neuron)."""
    mp, mn = pos.mean(0), neg.mean(0)
    vp, vn = pos.var(0), neg.var(0)
    return (mp - mn) / np.sqrt(0.5 * (vp + vn) + 1e-8)

@torch.no_grad()
def neuron_token_trace(tok, model, text, layer, unit):
    """Per-token activation of ONE neuron on `text` -> (tokens, activations)."""
    cap = {}
    def pre_hook(mod, args):
        cap["a"] = args[0].detach()[0, :, unit].float().cpu().numpy()
    h = _down_proj(_layers(model)[layer]).register_forward_pre_hook(pre_hook)
    ids = tok(text, return_tensors="pt").to(model.device)
    model(**ids)
    h.remove()
    toks = [tok.decode([t]) for t in ids["input_ids"][0].tolist()]
    return toks, cap["a"]

def top_activating_tokens(tok, model, texts, layer, unit, topn=5):
    """Across `texts`, the tokens where this neuron fires hardest (concept-cell
    'preferred stimulus'). Returns list of (activation, token, context)."""
    hits = []
    for t in texts:
        toks, acts = neuron_token_trace(tok, model, t, layer, unit)
        j = int(acts.argmax())
        ctx = "".join(toks[max(0, j-3):j]) + "⟦" + toks[j] + "⟧" + "".join(toks[j+1:j+3])
        hits.append((float(acts[j]), toks[j].strip(), ctx.replace("\n", " ")))
    return sorted(hits, reverse=True)[:topn]

# ---- extra tools: residuals, perplexity, lesion/microstimulation, generation ----
import contextlib
import torch.nn.functional as F
from collections import defaultdict

@torch.no_grad()
def last_token_residuals(tok, model, text):
    """Residual-stream vector at the LAST token, each layer -> [n_layers+1, hidden]."""
    ids = tok(text, return_tensors="pt").to(model.device)
    hs = model(**ids, output_hidden_states=True).hidden_states
    return np.stack([h[0, -1].float().cpu().numpy() for h in hs])

@torch.no_grad()
def text_nll(tok, model, text):
    """Mean next-token negative log-likelihood (nats/token) = language-modeling loss."""
    ids = tok(text, return_tensors="pt").to(model.device)["input_ids"]
    logp = F.log_softmax(model(ids).logits[0].float(), dim=-1)
    tgt = ids[0, 1:]
    return float(-logp[torch.arange(tgt.shape[0]), tgt].mean().item())

@contextlib.contextmanager
def edited_neurons(model, clamp=None, ablate=None):
    """Temporarily clamp MLP neurons to a value (microstimulation) or zero them
    (lesion). clamp={(layer,unit):value}; ablate=[(layer,unit),...]."""
    per_layer = defaultdict(dict)
    for (l, u), v in (clamp or {}).items():
        per_layer[l][u] = float(v)
    for (l, u) in (ablate or []):
        per_layer[l][u] = 0.0
    handles = []
    def mk(units):
        def pre(mod, args):
            x = args[0].clone()
            for u, v in units.items():
                x[:, :, u] = v
            return (x,) + tuple(args[1:])
        return pre
    for l, units in per_layer.items():
        handles.append(_down_proj(_layers(model)[l]).register_forward_pre_hook(mk(units)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()

@torch.no_grad()
def generate(tok, model, prompt, max_new_tokens=40, chat=True):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True) if chat else prompt
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

def n_layers(model):
    return len(_layers(model))

def intermediate_size(model):
    return _down_proj(_layers(model)[0]).weight.shape[1]

@contextlib.contextmanager
def steer_residual(model, layer, vec, alpha):
    """Add alpha*vec to the residual stream at `layer` output (population
    microstimulation). vec: 1-D array [hidden]."""
    vec = np.asarray(vec)
    def hook(mod, inp, out):
        h0 = out[0] if isinstance(out, tuple) else out          # device-safe under multi-GPU
        v = torch.as_tensor(vec, dtype=h0.dtype, device=h0.device)
        h0 = h0 + alpha * v
        return (h0,) + tuple(out[1:]) if isinstance(out, tuple) else h0
    h = _layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()

@torch.no_grad()
def layer_residuals(tok, model, text, layer):
    """Mean-pooled residual-stream vector at a given layer -> [hidden]."""
    ids = tok(text, return_tensors="pt").to(model.device)
    hs = model(**ids, output_hidden_states=True).hidden_states[layer]
    return hs[0].float().mean(0).cpu().numpy()

# ---- audit fixes: comparable steering & targets, token-position helpers ----

@torch.no_grad()
def residual_norm(tok, model, texts, layer):
    """Mean L2 norm of the residual stream at `layer` over texts. Steering
    strength is expressed as a fraction of this, so it is comparable ACROSS
    models/layers (as in the emotions paper)."""
    ns = []
    for t in texts:
        ids = tok(t, return_tensors="pt").to(model.device)
        h = model(**ids, output_hidden_states=True).hidden_states[layer][0]
        ns.append(h.float().norm(dim=-1).mean().item())
    return float(np.mean(ns))

@torch.no_grad()
def word_logprob(tok, model, prompt, word):
    """TOTAL log-prob of ' word' (its full token sequence) as a continuation of
    prompt. Tokenizer-agnostic in meaning -> comparable across models (fixes the
    'first-token only' bug)."""
    p = tok(prompt, return_tensors="pt").to(model.device)["input_ids"]
    w = tok(" " + word, add_special_tokens=False, return_tensors="pt").to(model.device)["input_ids"]
    if w.shape[1] == 0:
        return float("nan")
    full = torch.cat([p, w], dim=1)
    logp = F.log_softmax(model(full).logits[0].float(), dim=-1)
    tot = 0.0
    for k in range(w.shape[1]):
        pos = p.shape[1] + k - 1
        tot += logp[pos, full[0, p.shape[1] + k]].item()
    return tot

def target_score(tok, model, prompt, words):
    """log P(any target word) via logsumexp of per-word total log-probs."""
    lps = [word_logprob(tok, model, prompt, w) for w in words]
    lps = [x for x in lps if x == x]  # drop nan
    if not lps:
        return float("nan")
    m = max(lps)
    return m + float(np.log(sum(np.exp(x - m) for x in lps)))

def unit(vec):
    v = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
