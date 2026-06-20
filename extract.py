"""
extract.py  --  model loading and activation extraction.

Architecture-agnostic: we run a forward pass with output_hidden_states=True and
read the residual-stream hidden state at a chosen layer and the LAST token. This
works for Llama / Qwen / Mistral / Gemma / OLMoE without arch-specific hook paths.

If the model is a mixture-of-experts and exposes router logits
(output_router_logits=True), we also record, per probe, the top-1 routed expert at
the final layer, enabling the optional MoE sub-basis check (see run_drift_probe.py).
"""

from __future__ import annotations

import numpy as np
import torch


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, device: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # right padding so the last real token sits at index (mask.sum - 1)
    tok.padding_side = "right"
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.to(device).eval()
    is_moe = bool(getattr(model.config, "num_local_experts", 0)) or \
        bool(getattr(model.config, "num_experts", 0))
    return model, tok, device, is_moe


def _format(tok, context_user_msgs: list[str], probe_user_msg: str) -> str:
    """Render running dialogue context + a final probe user turn into a prompt."""
    messages = []
    for i, m in enumerate(context_user_msgs):
        messages.append({"role": "user", "content": m})
        # a minimal neutral assistant turn keeps the chat structure well-formed
        messages.append({"role": "assistant", "content": "Sure, happy to help with that."})
    messages.append({"role": "user", "content": probe_user_msg})
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # fallback for base models without a chat template
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


@torch.no_grad()
def last_token_activations(
    model, tok, device, context_user_msgs, probe_msgs, layer, batch_size=8,
    want_router=False,
):
    """
    Return (acts, experts) where acts is (n, d) last-token residual activations at
    `layer` for each probe appended to the running context, and experts is (n,)
    top-1 routed expert ids at the final layer (or None if not MoE / unavailable).
    """
    prompts = [_format(tok, context_user_msgs, p) for p in probe_msgs]
    acts, experts = [], []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(device)
        out = model(**enc, output_hidden_states=True,
                    output_router_logits=want_router, use_cache=False)
        hs = out.hidden_states[layer]                      # (b, seq, d)
        # last non-pad token index per row
        lengths = enc["attention_mask"].sum(dim=1) - 1     # (b,)
        idx = lengths.clamp(min=0)
        rows = hs[torch.arange(hs.shape[0]), idx]          # (b, d)
        acts.append(rows.float().cpu().numpy())
        if want_router and getattr(out, "router_logits", None):
            # router_logits: tuple over layers of (b*seq, n_experts); take last layer
            rl = out.router_logits[-1]
            n_exp = rl.shape[-1]
            rl = rl.view(hs.shape[0], -1, n_exp)           # (b, seq, n_exp)
            last = rl[torch.arange(hs.shape[0]), idx]      # (b, n_exp)
            experts.append(last.argmax(dim=-1).cpu().numpy())
    acts = np.concatenate(acts, axis=0)
    experts = np.concatenate(experts, axis=0) if experts else None
    return acts, experts
