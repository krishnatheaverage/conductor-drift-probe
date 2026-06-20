"""
run_strong.py  --  powered version of the drift premise test.

Improvements over run_drift_probe.py, all aimed at the two weaknesses of the first
run (low power, single cherry-picked layer):

  * hundreds of probe pairs (probes.*_BIG) -> lower estimation-noise floor;
  * ALL layers analyzed from one forward pass (no layer cherry-picking); a real
    effect must show a CONTIGUOUS band of significant layers, not an isolated hit;
  * cumulative displacement d(U_0, U_t) as the primary statistic (more sensitive to
    slow accumulation than consecutive-turn drift);
  * a power curve (noise floor vs n_probes) emitted as an artifact;
  * between-dialogue spread as an extra reference.

Verdict logic is built from the same validated primitives in drift_metrics.

Usage:
    python3 run_strong.py --model Qwen/Qwen2.5-1.5B-Instruct
    python3 run_strong.py --model Qwen/Qwen2.5-0.5B-Instruct --max-probes 100
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import probes as P
from extract import load_model, _format
from drift_metrics import (
    subspace_from_contrast, subspace_drift, cumulative_displacement, overlap_omega,
    noise_corrected_drift,
)


@torch.no_grad()
def all_layer_acts(model, tok, device, context, msgs, batch_size=16):
    prompts = [_format(tok, context, m) for m in msgs]
    outs = []
    for s in range(0, len(prompts), batch_size):
        enc = tok(prompts[s:s + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(device)
        o = model(**enc, output_hidden_states=True, use_cache=False)
        b = enc["attention_mask"].shape[0]
        idx = enc["attention_mask"].sum(1) - 1               # (b,) last real token
        ar = torch.arange(b, device=device)
        # index the last token PER LAYER before stacking, so we never materialize the
        # full (L+1, b, seq, d) tensor (that was ~700MB/batch and thrashed MPS).
        rows = [hs[ar, idx].float().cpu().numpy().astype(np.float16) for hs in o.hidden_states]
        outs.append(np.stack(rows, axis=0))                  # (L+1, b, d)
        del o
    return np.concatenate(outs, axis=1)                      # (L+1, n, d)


def collect(model, tok, device, dialogues, safety, uncert):
    data = {}
    sp = [a for a, _ in safety]; sn = [b for _, b in safety]
    hp = [a for a, _ in uncert]; hn = [b for _, b in uncert]
    for name, script in dialogues.items():
        data[name] = []
        for t in range(len(script)):
            ctx = script[:t]
            data[name].append({
                "S_pos": all_layer_acts(model, tok, device, ctx, sp),
                "S_neg": all_layer_acts(model, tok, device, ctx, sn),
                "H_pos": all_layer_acts(model, tok, device, ctx, hp),
                "H_neg": all_layer_acts(model, tok, device, ctx, hn),
            })
            print(f"  collected {name} turn {t}", flush=True)
    return data


def save_cache(data, path, model_name=""):
    flat, meta = {}, {}
    for dia, turns in data.items():
        meta[dia] = len(turns)
        for t, d in enumerate(turns):
            for k, arr in d.items():
                flat[f"{dia}|{t}|{k}"] = arr
    np.savez_compressed(path, __meta__=json.dumps(meta), __model__=np.array(model_name), **flat)


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    data = {dia: [dict() for _ in range(n)] for dia, n in meta.items()}
    for key in z.files:
        if key in ("__meta__", "__model__"):
            continue
        dia, t, k = key.split("|")
        data[dia][int(t)][k] = z[key]
    return data


def cache_model(path):
    z = np.load(path, allow_pickle=False)
    return str(z["__model__"]) if "__model__" in z.files else None


def floor_at(pos, neg, rank, n_boot=160, seed=0):
    """Two-sample bootstrap no-change null at one turn/layer (pos,neg are (n,d))."""
    rng = np.random.default_rng(seed)
    n = pos.shape[0]
    out = np.empty(n_boot)
    for b in range(n_boot):
        i1 = rng.integers(0, n, n); i2 = rng.integers(0, n, n)
        out[b] = subspace_drift(subspace_from_contrast(pos[i1], neg[i1], rank),
                                subspace_from_contrast(pos[i2], neg[i2], rank))
    return float(np.median(out)), float(np.percentile(out, 97.5))


def _slope(ys):
    """Least-squares slope of ys against index 0..len-1 (drift growth per turn)."""
    x = np.arange(len(ys), dtype=np.float64)
    if len(ys) < 2:
        return 0.0
    return float(np.polyfit(x, np.asarray(ys, float), 1)[0])


def analyze(data, key, rank, min_effect):
    """
    Per-layer analysis of subspace `key` ('S' or 'H'). The premise test baselines at
    turn 1 (steady state), so the generic no-context->context jump (turn0->turn1,
    reported separately as `onset`) cannot masquerade as multi-turn accumulation.
    """
    dialogues = list(data)
    L = data[dialogues[0]][0][f"{key}_pos"].shape[0]
    T = len(data[dialogues[0]])
    base = 1 if T >= 3 else 0
    stats = []
    for l in range(L):
        steady_last, onsets, slopes = [], [], []
        Us = {}
        for d in dialogues:
            turns = data[d]
            U = [subspace_from_contrast(turns[t][f"{key}_pos"][l].astype(np.float32),
                                        turns[t][f"{key}_neg"][l].astype(np.float32), rank)
                 for t in range(T)]
            Us[d] = U
            disp = cumulative_displacement(U, base=base)        # d(U_base, U_t), t>base
            steady_last.append(disp[-1])
            slopes.append(_slope([0.0] + disp))
            onsets.append(subspace_drift(U[0], U[1]) if T >= 2 else 0.0)
        # no-change floor estimated at the BASELINE (contextful) turn, pooled
        fl_med, fl_hi = [], []
        for d in dialogues:
            m, hi = floor_at(data[d][base][f"{key}_pos"][l].astype(np.float32),
                             data[d][base][f"{key}_neg"][l].astype(np.float32), rank)
            fl_med.append(m); fl_hi.append(hi)
        floor_med = float(np.median(fl_med)); floor_hi = float(np.max(fl_hi))
        # between-dialogue spread at the last turn (content sensitivity reference)
        last = T - 1
        spread = [subspace_drift(Us[dialogues[i]][last], Us[dialogues[j]][last])
                  for i in range(len(dialogues)) for j in range(i + 1, len(dialogues))]
        steady_med = float(np.median(steady_last))
        spread_med = float(np.median(spread)) if spread else 0.0
        stats.append({
            "layer": l,
            # PRIMARY, length-controlled: different dialogue content, same #turns. If this
            # exceeds the noise floor, the subspace is genuinely CONDITIONAL on context
            # content (CONDUCTOR's core premise), not merely a function of prompt length.
            "between_dialogue_spread": spread_med,
            "content_corrected": noise_corrected_drift(spread_med, floor_med),
            "content_significant": bool(spread_med > floor_hi and
                                        noise_corrected_drift(spread_med, floor_med) >= min_effect),
            # SECONDARY, NOT length-controlled: within-dialogue drift confounds content
            # with growing prompt length, so it is reported but not used for the verdict.
            "steady_disp_last_median": steady_med,
            "onset_0to1_median": float(np.median(onsets)),
            "slope_per_turn_median": float(np.median(slopes)),
            "disp_corrected": noise_corrected_drift(steady_med, floor_med),
            "floor_median": floor_med,
            "floor_ci_hi": floor_hi,
        })
    return stats


def longest_band(stats, field="content_significant"):
    best = cur = 0; end = -1
    for i, s in enumerate(stats):
        cur = cur + 1 if s[field] else 0
        if cur > best:
            best, end = cur, i
    layers = list(range(end - best + 1, end + 1)) if best else []
    return best, layers


def omega_by_layer(data, rank):
    dialogues = list(data)
    L = data[dialogues[0]][0]["H_pos"].shape[0]
    res = []
    for l in range(L):
        ranges = []
        for d in dialogues:
            turns = data[d]
            om = []
            for t in range(len(turns)):
                U_s = subspace_from_contrast(turns[t]["S_pos"][l].astype(np.float32),
                                             turns[t]["S_neg"][l].astype(np.float32), rank)
                U_h = subspace_from_contrast(turns[t]["H_pos"][l].astype(np.float32),
                                             turns[t]["H_neg"][l].astype(np.float32), rank)
                om.append(overlap_omega(U_s, U_h))
            ranges.append(max(om) - min(om))
        res.append({"layer": l, "omega_range_median": float(np.median(ranges))})
    return res


def power_curve(data, key, layer, rank):
    """Noise floor (median, ci_hi) vs n_probes at a fixed layer, on one contextful turn.
    Uses turn 1 (not turn 0, where the empty context is identical across dialogues and
    would triplicate the same probes and understate the floor)."""
    d0 = list(data)[0]
    t = 1 if len(data[d0]) > 1 else 0
    pos = data[d0][t][f"{key}_pos"][layer].astype(np.float32)
    neg = data[d0][t][f"{key}_neg"][layer].astype(np.float32)
    n_tot = pos.shape[0]
    rng = np.random.default_rng(3)
    curve = []
    ns = sorted({n for n in [16, 32, 64, 96, 128, 256] if n <= n_tot} | {n_tot})
    for n in ns:
        sub = rng.choice(n_tot, n, replace=False)
        m, hi = floor_at(pos[sub], neg[sub], rank, n_boot=160, seed=int(n))
        curve.append({"n_probes": int(n), "floor_median": m, "floor_ci_hi": hi})
    return curve


def run(args):
    is_moe = False
    model_label = args.model
    if args.from_cache and args.cache and os.path.exists(args.cache):
        print(f"Loading activations from cache {args.cache} (no model run)")
        data = load_cache(args.cache)
        model_label = cache_model(args.cache) or args.model  # avoid the CLI-default mislabel
    else:
        print(f"Loading {args.model} ...")
        model, tok, device, is_moe = load_model(args.model, args.device)
        print(f"device={device}  layers={model.config.num_hidden_layers}  MoE={is_moe}")
        safety = P.SAFETY_PAIRS_BIG[: args.max_probes] if args.max_probes else P.SAFETY_PAIRS_BIG
        uncert = P.UNCERTAINTY_PAIRS_BIG[: args.max_probes] if args.max_probes else P.UNCERTAINTY_PAIRS_BIG
        dialogues = {k: P.DIALOGUES[k] for k in (args.dialogues or P.DIALOGUES)}
        print(f"probes: safety={len(safety)} uncert={len(uncert)}  dialogues={list(dialogues)}")
        print("Collecting all-layer activations ...")
        data = collect(model, tok, device, dialogues, safety, uncert)
        if args.cache:
            save_cache(data, args.cache, args.model)
            print(f"cached activations -> {args.cache}")

    n_uncert = data[list(data)[0]][0]["H_pos"].shape[1]
    n_safety = data[list(data)[0]][0]["S_pos"].shape[1]
    print("Analyzing ...")
    S = analyze(data, "S", args.rank, args.min_effect)
    H = analyze(data, "H", args.rank, args.min_effect)
    om = omega_by_layer(data, args.rank)
    # PRIMARY premise test: content-conditional band (between-dialogue spread > floor)
    H_band, H_layers = longest_band(H, "content_significant")
    S_band, S_layers = longest_band(S, "content_significant")
    # pick the best layer among RECOVERABLE ones (floor not near-degenerate); a layer
    # where two estimates of the same subspace are already near-orthogonal (floor ~1)
    # carries no signal, so excluding it avoids reporting noise as the "best" layer.
    def best_layer(stats):
        rec = [s for s in stats[1:] if s["floor_ci_hi"] < 0.5]
        return max(rec or stats[1:], key=lambda s: s["content_corrected"])
    best_H = best_layer(H)
    best_S = best_layer(S)
    pc = power_curve(data, "H", best_H["layer"], args.rank)
    # report omega only where the safety subspace is actually recoverable (low floor);
    # at early layers S is ~orthogonal to itself under resampling, so omega is noise.
    om_rec = [x for x in om if S[x["layer"]]["floor_ci_hi"] < 0.5]
    om_best = max(om_rec or om, key=lambda x: x["omega_range_median"])

    # spread > floor shows the subspace is PROMPT-DEPENDENT, but the same-context floor
    # carries no length/topic variance, so this does NOT by itself isolate dialogue
    # CONTENT. The content verdict requires the length/topic-matched placebo control
    # (placebo_control.py), which on this model confirms content-conditionality only at
    # MIDDLE-TO-LATE layers (early-layer spread is a length/lexical artifact).
    H_prompt_dependent = H_band >= 3
    verdict = "GEOMETRY_PROMPT_DEPENDENT" if H_prompt_dependent else "GEOMETRY_STATIC"

    result = {
        "model": model_label, "rank": args.rank, "is_moe": bool(is_moe),
        "n_probes_safety": int(n_safety), "n_probes_uncert": int(n_uncert),
        "verdict": verdict,
        "premise_tested": ("hallucination subspace varies across dialogues beyond a "
                           "SAME-CONTEXT noise floor (PROMPT-DEPENDENT). spread>floor does "
                           "NOT isolate content from token length/position/topic; run "
                           "placebo_control.py for the length/topic-matched content verdict."),
        "content_verdict_source": "placebo_results.json (real vs length/topic-matched scrambled)",
        "hallucination": {
            "content_band": H_band, "content_band_layers": H_layers,
            "best_layer": best_H["layer"], "best_layer_stats": best_H,
        },
        "safety": {
            "content_band": S_band, "content_band_layers": S_layers,
            "best_layer": best_S["layer"], "best_layer_stats": best_S,
        },
        "omega_best_layer": om_best,
        "power_curve_H": pc,
        "per_layer_H": H, "per_layer_S": S,
        "caveats": [
            "PRIMARY test is between-dialogue spread (different content, matched #turns) "
            "vs noise floor: length-controlled evidence the subspace is context-conditional.",
            "Within-dialogue displacement is reported but NOT used for the verdict: it "
            "confounds content with growing prompt length.",
            "Raw activation space (no SAE); last-token; rank-1 by default.",
            "Safety/refusal direction is poorly recoverable below mid layers (high floor); "
            "the cross-subspace entanglement omega is only meaningful where S is stable.",
            "Verdict is for this model/probe-bank, not LLMs in general.",
        ],
    }
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "strong_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    _report(result)
    try:
        _plot(result, data, args)
        print(f"plot -> {os.path.join(args.outdir, 'strong_trajectories.png')}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    print(f"results -> {os.path.join(args.outdir, 'strong_results.json')}")
    return result


def _report(r):
    print("\n" + "=" * 72)
    print(f"VERDICT: {r['verdict']}")
    print(f"premise: {r['premise_tested']}")
    print("=" * 72)
    for key, lab in (("hallucination", "hallucination"), ("safety", "safety     ")):
        b = r[key]; s = b["best_layer_stats"]
        print(f"{lab}: content-conditional band = {b['content_band']} layers {b['content_band_layers']}")
        print(f"             best L{s['layer']}: between-dialogue spread={s['between_dialogue_spread']:.3f} "
              f"floor95={s['floor_ci_hi']:.3f} content-corrected={s['content_corrected']:.3f}  "
              f"(PRIMARY, length-controlled)")
        print(f"             secondary [length-confounded]: within-dialogue disp(turn1->last)"
              f"={s['steady_disp_last_median']:.3f} slope/turn={s['slope_per_turn_median']:.3f} "
              f"onset(0->1)={s['onset_0to1_median']:.3f}")
    print(f"omega best layer {r['omega_best_layer']['layer']}: "
          f"range={r['omega_best_layer']['omega_range_median']:.3f}")
    print("power curve (H, best layer) floor95 vs n_probes:")
    for p in r["power_curve_H"]:
        print(f"    n={p['n_probes']:>4}  floor95={p['floor_ci_hi']:.3f}")


def _plot(result, data, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    # (1) cumulative displacement vs layer, with floor band, for H and S
    for key, st, c in (("H", result["per_layer_H"], "tab:blue"),
                       ("S", result["per_layer_S"], "tab:red")):
        ls = [s["layer"] for s in st]
        ax[0].plot(ls, [s["between_dialogue_spread"] for s in st], c=c, marker=".",
                   label=f"{key} content spread")
        ax[0].plot(ls, [s["floor_ci_hi"] for s in st], c=c, ls="--", alpha=0.5, label=f"{key} floor95")
    ax[0].set_title("Content spread (length-controlled) vs floor, by layer")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("normalized chordal [0,1]"); ax[0].legend(fontsize=7)
    # (2) displacement from turn 1 vs turn at best H layer, per dialogue (steady-state)
    l = result["hallucination"]["best_layer"]
    for d in data:
        turns = data[d]
        U = [subspace_from_contrast(turns[t]["H_pos"][l].astype(np.float32),
                                    turns[t]["H_neg"][l].astype(np.float32), args.rank)
             for t in range(len(turns))]
        cum = [0.0] + cumulative_displacement(U, base=1)
        ax[1].plot(range(1, len(cum) + 1), cum, marker="o", label=d)
    ax[1].axhline(result["hallucination"]["best_layer_stats"]["floor_ci_hi"], ls="--", c="k", alpha=0.6)
    ax[1].set_title(f"H displacement from turn 1 (layer {l})")
    ax[1].set_xlabel("turn"); ax[1].set_ylabel("d(U1, Ut)"); ax[1].legend(fontsize=7)
    # (3) power curve
    pc = result["power_curve_H"]
    ax[2].plot([p["n_probes"] for p in pc], [p["floor_ci_hi"] for p in pc], marker="o")
    ax[2].set_title("Noise floor (95%) vs n_probes (H, best layer)")
    ax[2].set_xlabel("n_probes"); ax[2].set_ylabel("floor95 chordal")
    fig.suptitle(f"{result['model']}  ->  {result['verdict']}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "strong_trajectories.png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--max-probes", type=int, default=None)
    ap.add_argument("--min-effect", type=float, default=0.05)
    ap.add_argument("--dialogues", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cache", default="results/acts_cache.npz",
                    help="path to save/load collected activations (skip the model run)")
    ap.add_argument("--from-cache", action="store_true",
                    help="load activations from --cache instead of running the model")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
