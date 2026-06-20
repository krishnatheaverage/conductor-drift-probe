"""
placebo_control.py  --  the decisive control the adversarial review demanded.

The powered run found that the hallucination subspace differs across 3 dialogues by
far more than a SAME-CONTEXT bootstrap floor. But that floor holds context (hence
token length, position, topic) constant, so "spread > floor" is near-tautological
for any context-dependent readout: it does not isolate dependence on dialogue
CONTENT from dependence on token length / final-token position / surface topic.

This script supplies the correct null: a length- and topic-matched CONTENT PLACEBO.
For each real dialogue we build a scrambled twin by shuffling the word order within
every user turn (deterministic). Scrambling preserves the exact bag of tokens (so
token length, sequence position, and topical lexicon are held fixed) but destroys
coherent dialogue meaning / epistemic structure. We then recompute the
between-dialogue spread on the scrambled twins and compare:

  * if real_spread is statistically indistinguishable from scrambled_spread, the
    between-dialogue movement is explained by length / position / lexical topic, NOT
    by coherent dialogue content -> the content-conditional claim is NOT supported;
  * if real_spread clearly exceeds scrambled_spread (and scrambled sits near the
    bootstrap floor), coherent content genuinely moves the subspace.

Reuses the real activations from the cache; only the scrambled twins are collected.

Usage:
    python3 placebo_control.py --model Qwen/Qwen2.5-0.5B-Instruct --cache results/acts_cache.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import probes as P
from extract import load_model
from run_strong import all_layer_acts, load_cache
from drift_metrics import subspace_from_contrast, subspace_drift


def scramble(msg: str, seed: int) -> str:
    words = msg.split()
    rng = np.random.default_rng(seed)
    return " ".join(words[i] for i in rng.permutation(len(words)))


def between_dialogue_spread(subspaces_by_dialogue, layer):
    """median pairwise chordal distance across dialogues at the last turn, one layer."""
    keys = list(subspaces_by_dialogue)
    last = len(subspaces_by_dialogue[keys[0]]) - 1
    vals = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            vals.append(subspace_drift(subspaces_by_dialogue[keys[i]][last][layer],
                                       subspaces_by_dialogue[keys[j]][last][layer]))
    return float(np.median(vals))


def subspaces_per_layer(acts_pos, acts_neg, rank, L):
    return [subspace_from_contrast(acts_pos[l].astype(np.float32),
                                   acts_neg[l].astype(np.float32), rank) for l in range(L)]


def run(args):
    # real spreads from the saved run
    real = json.load(open(os.path.join(args.outdir, "strong_results.json")))
    real_spread = {s["layer"]: s["between_dialogue_spread"] for s in real["per_layer_H"]}
    real_floor = {s["layer"]: s["floor_ci_hi"] for s in real["per_layer_H"]}

    model, tok, device, _ = load_model(args.model, args.device)
    uncert = P.UNCERTAINTY_PAIRS_BIG[: args.max_probes]
    hp = [a for a, _ in uncert]; hn = [b for _, b in uncert]
    dialogues = {k: P.DIALOGUES[k] for k in (args.dialogues or P.DIALOGUES)}

    # build scrambled twins (preserve tokens -> length+topic; destroy coherence)
    scr = {name: [scramble(m, 1000 * di + ti) for ti, m in enumerate(script)]
           for di, (name, script) in enumerate(dialogues.items())}

    # report the token-length match (the confound we are controlling)
    def toklen(msgs):
        from extract import _format
        return len(tok(_format(tok, msgs, hp[0]))["input_ids"])
    print("token-length check at last turn (real vs scrambled):")
    for name, script in dialogues.items():
        print(f"  {name:18s} real={toklen(script):4d}  scrambled={toklen(scr[name]):4d}")

    # collect scrambled subspaces per (dialogue, turn) at all layers
    subs = {}
    L = None
    for name, script in scr.items():
        subs[name] = []
        for t in range(len(script)):
            ctx = script[:t]
            ap = all_layer_acts(model, tok, device, ctx, hp)
            an = all_layer_acts(model, tok, device, ctx, hn)
            L = ap.shape[0]
            subs[name].append(subspaces_per_layer(ap, an, args.rank, L))
        print(f"  collected scrambled {name}", flush=True)

    rows = []
    for l in range(L):
        scr_spread = between_dialogue_spread(subs, l)
        rows.append({"layer": l, "real_spread": real_spread.get(l, float("nan")),
                     "scrambled_spread": scr_spread, "floor95": real_floor.get(l, float("nan"))})

    # verdict at the real best layer and across the band
    bestL = real["hallucination"]["best_layer"]
    rb = next(r for r in rows if r["layer"] == bestL)
    band = real["hallucination"]["content_band_layers"]
    band_rows = [r for r in rows if r["layer"] in band]
    # content survives only where real clearly exceeds the (length/topic-matched) placebo
    survive = [r["layer"] for r in band_rows if r["real_spread"] > r["scrambled_spread"] + 0.05]
    content_supported = len(survive) >= 3

    out = {
        "model": args.model,
        "interpretation": ("scrambled twins hold token length + position + topic fixed and "
                           "destroy coherent content; real_spread must exceed scrambled_spread "
                           "for the movement to be attributable to dialogue content"),
        "best_layer": bestL,
        "best_layer_real_spread": rb["real_spread"],
        "best_layer_scrambled_spread": rb["scrambled_spread"],
        "best_layer_floor95": rb["floor95"],
        "band_layers_where_real_beats_placebo_by_0.05": survive,
        "content_conditional_supported_by_placebo": bool(content_supported),
        "per_layer": rows,
    }
    with open(os.path.join(args.outdir, "placebo_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 64)
    print("PLACEBO CONTROL (length/topic-matched, content destroyed)")
    print("=" * 64)
    print(f"{'layer':>5} {'real':>7} {'scrambled':>10} {'floor95':>8}  real>placebo+.05?")
    for r in rows:
        mark = "Y" if r["real_spread"] > r["scrambled_spread"] + 0.05 else "."
        print(f"{r['layer']:>5} {r['real_spread']:>7.3f} {r['scrambled_spread']:>10.3f} "
              f"{r['floor95']:>8.3f}      {mark}")
    print(f"\nbest layer {bestL}: real={rb['real_spread']:.3f} scrambled={rb['scrambled_spread']:.3f} "
          f"floor95={rb['floor95']:.3f}")
    print(f"layers where real beats placebo by >0.05: {survive}")
    print(f"CONTENT-CONDITIONAL SUPPORTED BY PLACEBO: {content_supported}")
    print("  (False => the between-dialogue spread is explained by length/position/topic, "
          "not coherent dialogue content; downgrade to INCONCLUSIVE.)")
    print(f"\nresults -> {os.path.join(args.outdir, 'placebo_results.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--max-probes", type=int, default=96)
    ap.add_argument("--dialogues", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cache", default="results/acts_cache.npz")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
