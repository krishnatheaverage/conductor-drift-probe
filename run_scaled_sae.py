"""
run_scaled_sae.py  --  the SAE-feature-space version of the content-conditionality test.

Same confound-controlled protocol as run_scaled.py (between-dialogue spread vs scrambled
twins, dialogue-level bootstrap CIs), but subspaces are recovered in SAE FEATURE space,
the space conditional-disentanglement methods actually operate in.

Tests TWO subspaces:
  * hallucination/epistemic  (always)  -- answerable vs unanswerable-fictional entity probes.
  * safety/refusal           (chat backends, or --safety) -- harmful vs harmless requests.
    GPT-2 is not a chat model, so its safety side is meaningless and skipped by default.

Backends:
  --backend gpt2   ungated, no HF token, free Colab T4. Hallucination subspace only.
  --backend gemma  gemma-2-2b-it + Gemma Scope. GATED: accept the Gemma + Gemma Scope
                   licenses and export HF_TOKEN. Tests BOTH subspaces. NOTE: the public
                   Gemma Scope SAEs were trained on the BASE (pt) model; applying them to
                   the instruct model is a mild distribution mismatch (results indicative).
                   Use --gemma-model google/gemma-2-2b for an in-distribution (base) run of
                   the hallucination side.

Install:  pip install sae-lens transformer-lens
Run:      python run_scaled_sae.py --backend gpt2  --max-probes 40
          HF_TOKEN=... python run_scaled_sae.py --backend gemma --max-probes 48
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import probes as P
from drift_metrics import subspace_from_contrast, subspace_drift

BACKENDS = {
    "gpt2": dict(model="gpt2", release="gpt2-small-res-jb",
                 sae_id=lambda l: f"blocks.{l}.hook_resid_pre",
                 layers=list(range(12)), chat=False),
    "gemma": dict(model="google/gemma-2-2b-it", release="gemma-scope-2b-pt-res-canonical",
                  sae_id=lambda l: f"layer_{l}/width_16k/canonical",
                  layers=[3, 6, 9, 12, 15, 18, 21], chat=True),
}


def scramble(msg, seed):
    w = msg.split()
    rng = np.random.default_rng(seed)
    return " ".join(w[i] for i in rng.permutation(len(w)))


def fmt(context, probe):
    lines = []
    for m in context:
        lines += ["User: " + m, "Assistant: Sure, happy to help with that."]
    lines += ["User: " + probe, "Assistant:"]
    return "\n".join(lines)


def load_sae(release, sae_id, device):
    from sae_lens import SAE
    res = SAE.from_pretrained(release, sae_id, device=device)
    sae = res[0] if isinstance(res, (tuple, list)) else res
    return sae.to(device)


def sae_hook_name(sae):
    c = sae.cfg
    if hasattr(c, "hook_name"):
        return c.hook_name
    md = getattr(c, "metadata", None)
    if md is not None and getattr(md, "hook_name", None):
        return md.hook_name
    if getattr(sae, "hook_name", None):
        return sae.hook_name
    raise AttributeError("could not locate hook_name on the SAE (cfg/metadata/sae)")


@torch.no_grad()
def sae_features(model, saes, hooks, prompts, device):
    out = {l: [] for l in saes}
    names = set(hooks.values())
    for p in prompts:
        toks = model.to_tokens(p)
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n in names)
        for l, sae in saes.items():
            acts = cache[hooks[l]][:, -1, :]
            dtype = sae.W_enc.dtype if hasattr(sae, "W_enc") else next(sae.parameters()).dtype
            z = sae.encode(acts.to(device=device, dtype=dtype))
            out[l].append(z.float().cpu().numpy())
    return {l: np.concatenate(v, 0) for l, v in out.items()}


def pairwise_spread(subs, idx):
    vals = [subspace_drift(subs[idx[a]], subs[idx[b]])
            for a in range(len(idx)) for b in range(a + 1, len(idx)) if idx[a] != idx[b]]
    return float(np.median(vals)) if vals else 0.0


def spread_ci(subs_by_dialogue, layer, n_boot=600, seed=0):
    names = list(subs_by_dialogue)
    subs = [subs_by_dialogue[n][layer] for n in names]
    n = len(subs)
    rng = np.random.default_rng(seed)
    boots = np.array([pairwise_spread(subs, rng.integers(0, n, n)) for _ in range(n_boot)])
    return (pairwise_spread(subs, list(range(n))),
            float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def collect(model, saes, hooks, dialogues, pos, neg, rank, device):
    out = {}
    for name, script in dialogues.items():
        fp = sae_features(model, saes, hooks, [fmt(script, q) for q in pos], device)
        fn = sae_features(model, saes, hooks, [fmt(script, q) for q in neg], device)
        out[name] = {l: subspace_from_contrast(fp[l], fn[l], rank) for l in saes}
        print(f"    {name}", flush=True)
    return out


def content_test(model, saes, hooks, dialogues, scr, pos, neg, rank, device, layers, label):
    print(f"  [{label}] real ...")
    real = collect(model, saes, hooks, dialogues, pos, neg, rank, device)
    print(f"  [{label}] scrambled twins ...")
    scramb = collect(model, saes, hooks, scr, pos, neg, rank, device)
    rows, band = [], []
    for l in layers:
        rp, rlo, rhi = spread_ci(real, l, seed=l)
        sp, slo, shi = spread_ci(scramb, l, seed=1000 + l)
        sig = rlo > shi
        if sig:
            band.append(l)
        rows.append({"layer": l, "real": rp, "real_ci": [rlo, rhi],
                     "scrambled": sp, "scrambled_ci": [slo, shi], "content_significant": sig})
    best = cur = 0
    for r in rows:
        cur = cur + 1 if r["content_significant"] else 0
        best = max(best, cur)
    return {"label": label, "content_conditional_layers": band, "longest_contiguous_band": best,
            "verdict": "CONTENT_CONDITIONAL" if best >= 3 else "NOT_CONTENT_CONDITIONAL",
            "per_layer": rows}


def run(args):
    cfg = BACKENDS[args.backend]
    model_name = args.gemma_model if (args.backend == "gemma" and args.gemma_model) else cfg["model"]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"backend={args.backend} model={model_name} device={device}")
    if args.backend == "gemma":
        print("NOTE: Gemma Scope SAEs are base-model-trained; on the -it model this is a mild "
              "distribution mismatch (results indicative). See --gemma-model.")
    from transformer_lens import HookedTransformer
    print("loading model (this downloads weights on first run) ...", flush=True)
    model = HookedTransformer.from_pretrained(model_name, device=device)
    layers = cfg["layers"]
    print(f"loading {len(layers)} SAEs (each ~150-300MB on first run; this phase can be "
          f"quiet for a minute, do NOT interrupt) ...", flush=True)
    saes = {}
    for l in layers:
        saes[l] = load_sae(cfg["release"], cfg["sae_id"](l), device)
        print(f"  SAE layer {l} ready", flush=True)
    hooks = {l: sae_hook_name(saes[l]) for l in layers}
    print(f"loaded {len(saes)} SAEs at layers {layers}; starting collection", flush=True)

    unc = P.UNCERTAINTY_PAIRS_BIG[: args.max_probes]
    saf = P.SAFETY_PAIRS_BIG[: args.max_probes]
    dialogues = dict(list(P.DIALOGUES_SCALED.items())[: args.max_dialogues]) \
        if args.max_dialogues else P.DIALOGUES_SCALED
    scr = {name: [scramble(m, 1000 * di + ti) for ti, m in enumerate(s)]
           for di, (name, s) in enumerate(dialogues.items())}

    tests = [("hallucination", [a for a, _ in unc], [b for _, b in unc])]
    if cfg["chat"] or args.safety:
        tests.append(("safety", [a for a, _ in saf], [b for _, b in saf]))

    results = {}
    for label, pos, neg in tests:
        results[label] = content_test(model, saes, hooks, dialogues, scr, pos, neg,
                                       args.rank, device, layers, label)

    out_obj = {"backend": args.backend, "model": model_name, "space": "SAE features",
               "n_dialogues": len(dialogues), "n_probes": len(unc), "rank": args.rank,
               "layers": layers, "subspaces": results}
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"sae_results_{args.backend}.json")
    with open(out, "w") as f:
        json.dump(out_obj, f, indent=2)

    print("\n" + "=" * 72)
    for label, res in results.items():
        print(f"SAE-SPACE VERDICT [{label}] ({model_name}): {res['verdict']}")
        print(f"{'layer':>5} {'real[ci]':>22} {'scrambled[ci]':>22}  sig?")
        for r in res["per_layer"]:
            print(f"{r['layer']:>5}  {r['real']:.3f}[{r['real_ci'][0]:.2f},{r['real_ci'][1]:.2f}]"
                  f"   {r['scrambled']:.3f}[{r['scrambled_ci'][0]:.2f},{r['scrambled_ci'][1]:.2f}]"
                  f"   {'Y' if r['content_significant'] else '.'}")
        print(f"  content-conditional layers: {res['content_conditional_layers']}\n")
    print("INTERPRET: CONTENT_CONDITIONAL => premise holds where methods operate (revives the "
          "framework). NOT_CONTENT_CONDITIONAL => the raw-space null replicates in feature space.")
    print(f"results -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(BACKENDS), default="gpt2")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--max-probes", type=int, default=40)
    ap.add_argument("--max-dialogues", type=int, default=None)
    ap.add_argument("--safety", action="store_true", help="also test the safety subspace")
    ap.add_argument("--gemma-model", default=None,
                    help="override the gemma model (e.g. google/gemma-2-2b for a base, "
                         "in-distribution hallucination run)")
    ap.add_argument("--outdir", default="results")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
