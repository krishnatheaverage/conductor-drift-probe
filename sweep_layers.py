"""
sweep_layers.py  --  is the hallucination-subspace drift a single-layer fluke or
robust across depth? One forward pass yields all layers, so this is nearly free.

Reports, per layer, the rank-1 consecutive-turn drift of the hallucination /
uncertainty subspace on a chosen dialogue, against the within-turn noise floor.
A premise that only "passes" at one isolated layer out of many is consistent with
chance (multiple comparisons), not a robust effect.

Usage:
    python3 sweep_layers.py --model Qwen/Qwen2.5-0.5B-Instruct --dialogue benign_escalation
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

import probes as P
from extract import load_model, _format
from drift_metrics import subspace_from_contrast, subspace_drift, noise_corrected_drift


@torch.no_grad()
def all_layer_acts(model, tok, device, context, msgs, batch_size=8):
    prompts = [_format(tok, context, m) for m in msgs]
    outs = []
    for s in range(0, len(prompts), batch_size):
        enc = tok(prompts[s:s + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(device)
        o = model(**enc, output_hidden_states=True, use_cache=False)
        idx = enc["attention_mask"].sum(1) - 1
        hs = torch.stack(o.hidden_states, 0)                 # (L+1, b, seq, d)
        rows = hs[:, torch.arange(hs.shape[1]), idx]         # (L+1, b, d)
        outs.append(rows.float().cpu().numpy())
    return np.concatenate(outs, axis=1)                      # (L+1, n, d)


def floor95(pos, neg, rank, n_boot=150, seed=0):
    rng = np.random.default_rng(seed)
    n = pos.shape[0]
    out = np.empty(n_boot)
    for b in range(n_boot):
        i1 = rng.integers(0, n, n); i2 = rng.integers(0, n, n)
        out[b] = subspace_drift(subspace_from_contrast(pos[i1], neg[i1], rank),
                                subspace_from_contrast(pos[i2], neg[i2], rank))
    return float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dialogue", default="benign_escalation", choices=list(P.DIALOGUES))
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--max-probes", type=int, default=32)
    ap.add_argument("--stride", type=int, default=1, help="layer stride")
    args = ap.parse_args()

    model, tok, device, _ = load_model(args.model)
    pairs = P.UNCERTAINTY_PAIRS_BIG[: args.max_probes]
    script = P.DIALOGUES[args.dialogue]
    pos = [all_layer_acts(model, tok, device, script[:t], [a for a, _ in pairs]) for t in range(len(script))]
    neg = [all_layer_acts(model, tok, device, script[:t], [b for _, b in pairs]) for t in range(len(script))]
    L = pos[0].shape[0]

    print(f"\nmodel={args.model}  dialogue={args.dialogue}  rank={args.rank}  "
          f"probes={len(pairs)}  layers={L}")
    print(f"{'layer':>5} {'drift_med':>9} {'floor95':>8} {'corrected':>9} {'significant':>11}")
    n_sig = 0
    for l in range(0, L, args.stride):
        Us = [subspace_from_contrast(pos[t][l], neg[t][l], args.rank) for t in range(len(script))]
        drift = [subspace_drift(Us[t - 1], Us[t]) for t in range(1, len(Us))]
        med = float(np.median(drift)); fh = floor95(pos[0][l], neg[0][l], args.rank)
        sig = med > fh; n_sig += int(sig)
        print(f"{l:>5} {med:>9.3f} {fh:>8.3f} {noise_corrected_drift(med, fh):>9.3f} {str(sig):>11}")
    n_checked = len(range(0, L, args.stride))
    print(f"\n{n_sig}/{n_checked} layers significant. A robust effect shows a CONTIGUOUS band "
          f"of significant layers; isolated single-layer hits are consistent with chance.")


if __name__ == "__main__":
    main()
