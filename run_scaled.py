"""
run_scaled.py  --  harden the content-conditionality result against the n=3 critique.

The placebo control established that, at middle-to-late layers, the hallucination
subspace differs across 3 dialogues by more than a length/topic-matched scrambled
twin. The main remaining weakness: 3 dialogues, a median of 3 pairwise distances, no
confidence interval, partly driven by one (chemistry) dialogue.

This scales to 12 topically diverse dialogues and puts a DIALOGUE-LEVEL bootstrap CI
on the between-dialogue spread, for both the real dialogues and their scrambled twins
(length/topic-matched, coherence destroyed). Content-conditionality is declared at a
layer only if the real spread's lower CI exceeds the scrambled spread's upper CI, so
the claim no longer rests on any single dialogue. Only the final (full-context) turn
is collected, so this is cheap.

Usage:
    python3 run_scaled.py --model Qwen/Qwen2.5-0.5B-Instruct --max-probes 64
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import probes as P
from extract import load_model, _format
from run_strong import all_layer_acts
from placebo_control import scramble
from drift_metrics import subspace_from_contrast, subspace_drift


def final_turn_subspaces(model, tok, device, dialogues, hp, hn, rank):
    """Return {name: [subspace per layer]} using each dialogue's full (final-turn) context."""
    out, L = {}, None
    for name, script in dialogues.items():
        ctx = script[:len(script)]                       # full context = final turn
        ap = all_layer_acts(model, tok, device, ctx, hp)
        an = all_layer_acts(model, tok, device, ctx, hn)
        L = ap.shape[0]
        out[name] = [subspace_from_contrast(ap[l].astype(np.float32), an[l].astype(np.float32), rank)
                     for l in range(L)]
        print(f"  collected {name}", flush=True)
    return out, L


def pairwise_spread(subs_at_layer, idx):
    """median pairwise chordal distance among the dialogues in idx (skip same-dialogue pairs)."""
    vals = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if idx[a] != idx[b]:
                vals.append(subspace_drift(subs_at_layer[idx[a]], subs_at_layer[idx[b]]))
    return float(np.median(vals)) if vals else 0.0


def spread_ci(subs_by_dialogue, layer, n_boot=600, seed=0):
    """Dialogue-level bootstrap CI of the median pairwise spread at one layer."""
    names = list(subs_by_dialogue)
    subs = [subs_by_dialogue[n][layer] for n in names]
    n = len(subs)
    point = pairwise_spread(subs, list(range(n)))
    rng = np.random.default_rng(seed)
    boots = np.array([pairwise_spread(subs, rng.integers(0, n, n)) for _ in range(n_boot)])
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run(args):
    model, tok, device, _ = load_model(args.model, args.device)
    uncert = P.UNCERTAINTY_PAIRS_BIG[: args.max_probes]
    hp = [a for a, _ in uncert]; hn = [b for _, b in uncert]
    dialogues = dict(list(P.DIALOGUES_SCALED.items())[: args.max_dialogues]) \
        if args.max_dialogues else P.DIALOGUES_SCALED
    print(f"{len(dialogues)} dialogues, {len(uncert)} probes, model {args.model}")

    scr = {name: [scramble(m, 1000 * di + ti) for ti, m in enumerate(script)]
           for di, (name, script) in enumerate(dialogues.items())}

    print("collecting real ...")
    real, L = final_turn_subspaces(model, tok, device, dialogues, hp, hn, args.rank)
    print("collecting scrambled twins ...")
    scramb, _ = final_turn_subspaces(model, tok, device, scr, hp, hn, args.rank)

    rows, band = [], []
    for l in range(L):
        rp, rlo, rhi = spread_ci(real, l, seed=l)
        sp, slo, shi = spread_ci(scramb, l, seed=1000 + l)
        content = rlo > shi                              # real CI strictly above placebo CI
        if content:
            band.append(l)
        rows.append({"layer": l, "real": rp, "real_ci": [rlo, rhi],
                     "scrambled": sp, "scrambled_ci": [slo, shi], "content_significant": content})

    # longest contiguous band of dialogue-level-significant layers
    best = cur = 0; end = -1
    for i, r in enumerate(rows):
        cur = cur + 1 if r["content_significant"] else 0
        if cur > best:
            best, end = cur, i
    band_layers = list(range(end - best + 1, end + 1)) if best else []

    result = {
        "model": args.model, "n_dialogues": len(dialogues), "n_probes": len(uncert),
        "rank": args.rank,
        "content_conditional_band_len": best, "content_conditional_band_layers": band_layers,
        "verdict": ("CONTENT_CONDITIONAL_MID_LATE" if best >= 3 else "NOT_CONTENT_CONDITIONAL"),
        "per_layer": rows,
        "method": ("12 diverse dialogues; final-turn H subspace; dialogue-level bootstrap CI "
                   "(real vs length/topic-matched scrambled twins); significant iff real CI_lo "
                   "> scrambled CI_hi, so no single dialogue can carry the result."),
    }
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "scaled_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 72)
    print(f"VERDICT: {result['verdict']}  (dialogue-level CIs, n={len(dialogues)} dialogues)")
    print("=" * 72)
    print(f"{'layer':>5} {'real[ci]':>20} {'scrambled[ci]':>20}  sig?")
    for r in rows:
        print(f"{r['layer']:>5}  {r['real']:.3f}[{r['real_ci'][0]:.2f},{r['real_ci'][1]:.2f}]"
              f"   {r['scrambled']:.3f}[{r['scrambled_ci'][0]:.2f},{r['scrambled_ci'][1]:.2f}]"
              f"   {'Y' if r['content_significant'] else '.'}")
    print(f"\ncontent-conditional contiguous band (real CI_lo > scrambled CI_hi): {band_layers}")
    try:
        _plot(rows, result, args.outdir)
        print(f"plot -> {os.path.join(args.outdir, 'scaled_compare.png')}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    print(f"results -> {os.path.join(args.outdir, 'scaled_results.json')}")


def _plot(rows, result, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = [r["layer"] for r in rows if r["layer"] > 0]
    rows = [r for r in rows if r["layer"] > 0]
    real = np.array([r["real"] for r in rows]); rlo = np.array([r["real_ci"][0] for r in rows])
    rhi = np.array([r["real_ci"][1] for r in rows])
    scr = np.array([r["scrambled"] for r in rows]); slo = np.array([r["scrambled_ci"][0] for r in rows])
    shi = np.array([r["scrambled_ci"][1] for r in rows])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(L, real, c="tab:blue", marker="o", label="real dialogues (content)")
    ax.fill_between(L, rlo, rhi, color="tab:blue", alpha=0.2)
    ax.plot(L, scr, c="tab:orange", marker="s", label="scrambled twins (length+topic)")
    ax.fill_between(L, slo, shi, color="tab:orange", alpha=0.2)
    for l in result["content_conditional_band_layers"]:
        ax.axvline(l, color="green", alpha=0.06)
    ax.set_xlabel("layer"); ax.set_ylabel("between-dialogue spread [0,1]")
    ax.set_title(f"Dialogue-level CIs, n={result['n_dialogues']} dialogues ({result['model'].split('/')[-1]})")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scaled_compare.png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--max-probes", type=int, default=64)
    ap.add_argument("--max-dialogues", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default="results")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
