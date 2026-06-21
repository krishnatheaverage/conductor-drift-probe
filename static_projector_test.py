"""
static_projector_test.py  --  does a STATIC disentangling projector actually fail across
turns, for CONTENT reasons? This turns the paper's recommendation ("conditional-alignment
methods should show a failing static baseline") into a demonstration.

Construction. Conditional-disentanglement methods replace a single static projection with a
per-context one. The static baseline builds, at turn 1, the projector that nulls the
hallucination subspace H_1 (M = I - U_{H_1} U_{H_1}^T). Its residual "re-entanglement leak"
at turn t is exactly the fraction of the turn-t hallucination subspace H_t that the frozen
projector fails to remove, which equals the normalized chordal distance d(H_1, H_t) in
[0,1] (0 = H_t still fully removed; 1 = H_t orthogonal to H_1, nothing removed). A
conditional projector rebuilt at turn t has leak 0 by construction (modulo the estimation
floor). So the static-minus-conditional gap is d(H_1, H_t).

The honest test (same confound control as the rest of the harness). A static projector
will "leak" simply because prompts get longer; that is not a reason to condition on
*content*. We therefore compare the real-dialogue leak to the scrambled-twin leak
(identical token length / topic, coherent content destroyed), with a dialogue-level
bootstrap CI. The static baseline genuinely fails for CONTENT reasons only where real leak
clears scrambled leak. If real <= scrambled, the static projector's degradation is a
length/topic artifact and a content-conditional projector would not fix it: the conditional
machinery is unjustified for the behaviors and models studied.

Raw-activation backend (runs locally). For the SAE-feature-space version, swap the two
all_layer_acts calls for the SAE encoder used in run_scaled_sae.py.

Usage:
    python3 static_projector_test.py --model Qwen/Qwen2.5-0.5B-Instruct --max-probes 48
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import probes as P
from extract import load_model
from run_strong import all_layer_acts
from placebo_control import scramble
from drift_metrics import subspace_from_contrast, subspace_drift


def collect(model, tok, device, dialogues, hp, hn, rank):
    """{dialogue: [ [subspace per layer] per turn ]} for the hallucination subspace."""
    out, L = {}, None
    for name, script in dialogues.items():
        turns = []
        for t in range(len(script)):
            ctx = script[:t]
            ap = all_layer_acts(model, tok, device, ctx, hp)
            an = all_layer_acts(model, tok, device, ctx, hn)
            L = ap.shape[0]
            turns.append([subspace_from_contrast(ap[l].astype(np.float32),
                                                 an[l].astype(np.float32), rank) for l in range(L)])
        out[name] = turns
        print(f"  collected {name}", flush=True)
    return out, L


def leak_ci(subs, layer, base, t, n_boot=600, seed=0):
    """Median over dialogues of d(H_base, H_t) at one layer, with a dialogue-level CI.
    This is the static projector's residual re-entanglement leak at turn t."""
    names = list(subs)
    vals = np.array([subspace_drift(subs[d][base][layer], subs[d][t][layer]) for d in names])
    rng = np.random.default_rng(seed)
    boots = np.array([np.median(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(n_boot)])
    return float(np.median(vals)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run(args):
    model, tok, device, _ = load_model(args.model, args.device)
    uncert = P.UNCERTAINTY_PAIRS_BIG[: args.max_probes]
    hp = [a for a, _ in uncert]; hn = [b for _, b in uncert]
    dialogues = dict(list(P.DIALOGUES_SCALED.items())[: args.max_dialogues]) \
        if args.max_dialogues else P.DIALOGUES_SCALED
    scr = {name: [scramble(m, 1000 * di + ti) for ti, m in enumerate(s)]
           for di, (name, s) in enumerate(dialogues.items())}
    print(f"{len(dialogues)} dialogues, {len(uncert)} probes, model {args.model}")

    print("collecting real ...")
    real, L = collect(model, tok, device, dialogues, hp, hn, args.rank)
    print("collecting scrambled twins ...")
    scrb, _ = collect(model, tok, device, scr, hp, hn, args.rank)

    base = 1                                    # static projector fit at the first contextful turn
    last = len(next(iter(dialogues.values()))) - 1
    rows, genuine = [], []
    for l in range(L):
        rp, rlo, rhi = leak_ci(real, l, base, last, seed=l)
        sp, slo, shi = leak_ci(scrb, l, base, last, seed=1000 + l)
        g = rlo > shi                           # static fails MORE for real than placebo
        if g:
            genuine.append(l)
        rows.append({"layer": l, "static_leak_real": rp, "real_ci": [rlo, rhi],
                     "static_leak_scrambled": sp, "scrambled_ci": [slo, shi],
                     "genuine_content_reentanglement": g})

    best = cur = 0
    for r in rows:
        cur = cur + 1 if r["genuine_content_reentanglement"] else 0
        best = max(best, cur)
    result = {
        "model": args.model, "rank": args.rank, "n_dialogues": len(dialogues),
        "n_probes": len(uncert), "base_turn": base, "eval_turn": last,
        "static_projector_leak": "normalized chordal d(H_base, H_eval) in [0,1]",
        "genuine_content_reentanglement_layers": genuine,
        "longest_contiguous_band": best,
        "verdict": ("STATIC_GENUINELY_FAILS" if best >= 3 else "STATIC_FAILURE_IS_CONFOUND"),
        "per_layer": rows,
    }
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "static_projector_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 72)
    print(f"STATIC-PROJECTOR VERDICT: {result['verdict']}  (model {args.model})")
    print("=" * 72)
    print(f"static projector built at turn {base}, evaluated at turn {last}. Leak = residual")
    print("hallucination subspace the frozen projector fails to remove (0=ok, 1=fully re-entangled).")
    print(f"{'layer':>5} {'real_leak[ci]':>22} {'scrambled_leak[ci]':>22}  content-fail?")
    for r in rows:
        print(f"{r['layer']:>5}  {r['static_leak_real']:.3f}[{r['real_ci'][0]:.2f},{r['real_ci'][1]:.2f}]"
              f"   {r['static_leak_scrambled']:.3f}[{r['scrambled_ci'][0]:.2f},{r['scrambled_ci'][1]:.2f}]"
              f"   {'Y' if r['genuine_content_reentanglement'] else '.'}")
    print(f"\nlayers where the static projector fails for CONTENT reasons "
          f"(real CI_lo > scrambled CI_hi): {genuine}")
    print("INTERPRET: STATIC_FAILURE_IS_CONFOUND => the static projector's re-entanglement is "
          "explained by length/topic, not content; a content-conditional projector would not "
          "fix it, so the conditional machinery is unjustified here. STATIC_GENUINELY_FAILS => "
          "content drives re-entanglement; the conditional approach is motivated.")
    print(f"results -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--max-probes", type=int, default=48)
    ap.add_argument("--max-dialogues", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default="results")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
