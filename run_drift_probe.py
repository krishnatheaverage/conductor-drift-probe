"""
run_drift_probe.py  --  the real-model experiment.

Tests the CONDUCTOR load-bearing premise on an actual LLM: as a multi-turn
dialogue accumulates, do the safety subspace S_t and the hallucination/uncertainty
subspace H_t rotate (consecutive-turn drift above the estimation-noise floor), and
does their cross-overlap omega(S_t, H_t) move across turns?

Verdict is mechanical via the pre-registered rule in drift_metrics.decide, which
the synthetic self-test validates against analytic ground truth.

Usage:
    python3 run_drift_probe.py --model Qwen/Qwen2.5-0.5B-Instruct
    python3 run_drift_probe.py --model <hf-id> --layer 12 --rank 3 --max-probes 16

If no model can be loaded (offline / no weights), this prints a clear message and
exits; the synthetic test is the offline validation path.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import probes as P
from drift_metrics import (
    subspace_from_contrast, subspace_drift, overlap_omega,
    noise_corrected_drift, _summ, decide,
)


def turn_subspaces(model, tok, device, context, pairs, layer, rank, want_router):
    from extract import last_token_activations
    pos_msgs = [a for a, _ in pairs]
    neg_msgs = [b for _, b in pairs]
    pos, exp_pos = last_token_activations(model, tok, device, context, pos_msgs, layer,
                                          want_router=want_router)
    neg, _ = last_token_activations(model, tok, device, context, neg_msgs, layer,
                                    want_router=want_router)
    U = subspace_from_contrast(pos, neg, rank)
    return U, pos, neg, exp_pos


def run(args):
    from extract import load_model
    print(f"Loading {args.model} ...")
    model, tok, device, is_moe = load_model(args.model, args.device)
    n_layers = model.config.num_hidden_layers
    layer = args.layer if args.layer is not None else n_layers // 2
    print(f"device={device}  layers={n_layers}  using layer={layer}  MoE={is_moe}  rank={args.rank}")

    bank_S = P.SAFETY_PAIRS_BIG if args.big_banks else P.SAFETY_PAIRS
    bank_H = P.UNCERTAINTY_PAIRS_BIG if args.big_banks else P.UNCERTAINTY_PAIRS
    safety = bank_S[: args.max_probes]
    uncert = bank_H[: args.max_probes]
    dialogues = {k: P.DIALOGUES[k] for k in (args.dialogues or P.DIALOGUES)}

    per_dialogue = {}
    pooled_H_drift, pooled_S_drift = [], []
    floor_H_draws, floor_S_draws, omega_floor_draws = [], [], []

    for name, script in dialogues.items():
        print(f"\n=== dialogue: {name} ({len(script)} turns) ===")
        US, UH, omega = [], [], []
        cache_S, cache_H = [], []
        for t in range(len(script)):
            context = script[:t]                       # t=0 -> single-turn (no context)
            U_s, ps, ns, _ = turn_subspaces(model, tok, device, context, safety, layer,
                                            args.rank, want_router=False)
            U_h, ph, nh, _ = turn_subspaces(model, tok, device, context, uncert, layer,
                                            args.rank, want_router=is_moe)
            US.append(U_s); UH.append(U_h)
            cache_S.append((ps, ns)); cache_H.append((ph, nh))
            omega.append(overlap_omega(U_s, U_h))
            print(f"  turn {t}: omega(S,H)={omega[-1]:.3f}")

        S_drift = [subspace_drift(US[t - 1], US[t]) for t in range(1, len(US))]
        H_drift = [subspace_drift(UH[t - 1], UH[t]) for t in range(1, len(UH))]
        pooled_S_drift += S_drift
        pooled_H_drift += H_drift

        # noise floors estimated at every turn, pooled into one null distribution
        for (ps, ns) in cache_S:
            floor_S_draws.append(_raw_boot(ps, ns, args.rank, 80))
        for (ph, nh) in cache_H:
            floor_H_draws.append(_raw_boot(ph, nh, args.rank, 80))
        for (ps, ns), (ph, nh) in zip(cache_S, cache_H):
            omega_floor_draws.append(_omega_boot_array(ps, ns, ph, nh, args.rank, 80))

        per_dialogue[name] = {
            "omega_series": omega,
            "omega_range": float(max(omega) - min(omega)),
            "S_drift": S_drift,
            "H_drift": H_drift,
        }

    floor_S = _summ(np.concatenate(floor_S_draws))
    floor_H = _summ(np.concatenate(floor_H_draws))
    omega_floor = _summ(np.concatenate(omega_floor_draws))

    # headline: hallucination subspace drift + omega movement (pooled across dialogues
    # for drift power; omega movement judged per dialogue then aggregated)
    omega_concat = [v for d in per_dialogue.values() for v in d["omega_series"]]
    headline = decide(pooled_H_drift, floor_H, omega_concat, omega_floor,
                      min_effect=args.min_effect)
    # omega movement is a within-dialogue quantity; aggregate per-dialogue ranges
    omega_band = omega_floor["ci_hi"] - omega_floor["ci_lo"]
    omega_moves_per_dialogue = {k: d["omega_range"] > omega_band for k, d in per_dialogue.items()}
    headline["omega_moves_any_dialogue"] = bool(any(omega_moves_per_dialogue.values()))

    S_summary = _drift_summary(pooled_S_drift, floor_S, args.min_effect)
    H_summary = _drift_summary(pooled_H_drift, floor_H, args.min_effect)

    # Final verdict: the hallucination subspace (the one the projector removes) must
    # drift significantly above the no-change null AND its overlap with the safety
    # subspace must move WITHIN at least one dialogue (a temporal, not between-dialogue,
    # quantity). This supersedes decide()'s pooled-omega heuristic for the real run.
    drift_significant = H_summary["significant"] and H_summary["effect_exceeds_min"]
    final_supported = bool(drift_significant and headline["omega_moves_any_dialogue"])
    headline["verdict"] = "PREMISE_SUPPORTED" if final_supported else "PREMISE_FALSIFIED"

    result = {
        "model": args.model, "layer": int(layer), "rank": int(args.rank),
        "n_probes_per_side": len(safety), "is_moe": bool(is_moe),
        "headline_verdict": headline["verdict"],
        "headline": headline,
        "safety_subspace_drift": S_summary,
        "hallucination_subspace_drift": H_summary,
        "omega_noise_band_width": float(omega_band),
        "omega_moves_per_dialogue": omega_moves_per_dialogue,
        "per_dialogue": per_dialogue,
        "caveats": [
            "Small probe banks: estimation-noise floor is high; treat a FALSIFIED "
            "verdict as possibly underpowered (see README Statistical power).",
            "Subspaces are recovered in raw activation space, not SAE feature space "
            "(no SAE library installed); SAE features are the interpretable lift.",
            "Last-token activation at one layer; the geometry may differ by layer.",
        ],
    }

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "drift_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    _print_report(result)
    try:
        _plot(result, per_dialogue, args.outdir)
        print(f"\nplot -> {os.path.join(args.outdir, 'drift_trajectories.png')}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    print(f"results -> {out}")
    return result


def _raw_boot(pos, neg, rank, n_boot):
    rng = np.random.default_rng(0)
    n = pos.shape[0]
    out = np.empty(n_boot)
    for b in range(n_boot):
        i1 = rng.integers(0, n, size=n); i2 = rng.integers(0, n, size=n)
        U1 = subspace_from_contrast(pos[i1], neg[i1], rank)
        U2 = subspace_from_contrast(pos[i2], neg[i2], rank)
        out[b] = subspace_drift(U1, U2)
    return out


def _omega_boot_array(ps, ns, ph, nh, rank, n_boot):
    rng = np.random.default_rng(1)
    n = ps.shape[0]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        U_s = subspace_from_contrast(ps[idx], ns[idx], rank)
        U_h = subspace_from_contrast(ph[idx], nh[idx], rank)
        out[b] = overlap_omega(U_s, U_h)
    return out


def _drift_summary(drift, floor, min_effect):
    med = float(np.median(drift))
    corrected = noise_corrected_drift(med, floor["median"])
    return {
        "median": med,
        "noise_floor_median": floor["median"],
        "noise_floor_ci_hi": floor["ci_hi"],
        "noise_corrected": corrected,
        "significant": bool(med > floor["ci_hi"]),
        "effect_exceeds_min": bool(corrected >= min_effect),
        "values": [float(x) for x in drift],
    }


def _print_report(r):
    print("\n" + "=" * 64)
    print(f"VERDICT: {r['headline_verdict']}")
    print("=" * 64)
    h = r["hallucination_subspace_drift"]; s = r["safety_subspace_drift"]
    print(f"hallucination subspace: drift median={h['median']:.3f} "
          f"floor_ci_hi={h['noise_floor_ci_hi']:.3f} corrected={h['noise_corrected']:.3f} "
          f"significant={h['significant']} effect={h['effect_exceeds_min']}")
    print(f"safety subspace:        drift median={s['median']:.3f} "
          f"floor_ci_hi={s['noise_floor_ci_hi']:.3f} corrected={s['noise_corrected']:.3f} "
          f"significant={s['significant']} effect={s['effect_exceeds_min']}")
    print(f"omega moves per dialogue: {r['omega_moves_per_dialogue']} "
          f"(band width {r['omega_noise_band_width']:.3f})")
    print(f"note: {r['headline']['note']}")


def _plot(result, per_dialogue, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, d in per_dialogue.items():
        ax1.plot(range(1, len(d["H_drift"]) + 1), d["H_drift"], marker="o", label=name)
        ax2.plot(range(len(d["omega_series"])), d["omega_series"], marker="o", label=name)
    fh = result["hallucination_subspace_drift"]["noise_floor_ci_hi"]
    ax1.axhline(fh, ls="--", c="k", alpha=0.6, label="noise floor (95%)")
    ax1.set_title("Hallucination subspace: consecutive-turn drift")
    ax1.set_xlabel("turn"); ax1.set_ylabel("normalized chordal drift [0,1]"); ax1.legend(fontsize=8)
    ax2.set_title("Cross-overlap omega(S,H) across turns")
    ax2.set_xlabel("turn"); ax2.set_ylabel("normalized overlap [0,1]"); ax2.legend(fontsize=8)
    fig.suptitle(f"{result['model']}  layer {result['layer']}  ->  {result['headline_verdict']}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "drift_trajectories.png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer", type=int, default=None, help="default: middle layer")
    ap.add_argument("--rank", type=int, default=1,
                    help="subspace rank; rank>1 needs many probes (see power curve)")
    ap.add_argument("--max-probes", type=int, default=32)
    ap.add_argument("--big-banks", action="store_true", default=True,
                    help="use the templated expanded probe banks (default on)")
    ap.add_argument("--small-banks", dest="big_banks", action="store_false")
    ap.add_argument("--min-effect", type=float, default=0.05)
    ap.add_argument("--dialogues", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    try:
        run(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\nCould not complete the real-model run (offline, missing weights, or "
              "memory). The offline validation path is:  python3 test_synthetic.py")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
