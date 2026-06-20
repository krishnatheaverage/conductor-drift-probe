# CONDUCTOR drift probe: the cheapest falsification first

The CONDUCTOR framework (conditional, per-expert operator-coupling disentanglement
of hallucination vs safety features, with an anytime-valid multi-turn certificate)
rests on one empirical premise:

> In multi-turn dialogue, the activation-space subspace encoding **safety/refusal**
> (`S_t`) and the subspace encoding **hallucination/epistemic-uncertainty** (`H_t`)
> **rotate as turns accumulate**, so a single static disentangling projection
> re-entangles or over-projects later in a conversation.

If that premise is false (the subspaces do not move beyond estimation noise), a
static projection suffices and the entire conditional/transport machinery is
unmotivated. This repo tests the premise before any of the framework is built. It
is deliberately minimal and runs on a laptop.

## What it measures

At each dialogue turn `t`, the running conversation is used as context and a bank
of contrastive probes is appended as the next user turn:

- **Safety subspace `S_t`**: directions separating a harmful request from a
  surface-matched harmless one (refusal-trigger geometry; Arditi et al. 2024).
- **Hallucination subspace `H_t`**: directions separating an answerable, real-entity
  question from a surface-matched unanswerable / fictional-entity one
  (known-vs-unknown geometry; Ferrando et al. 2025).

Because the probes are fixed and only the prepended context changes, any movement
in the recovered subspace is caused by accumulating dialogue, which is exactly the
premise under test.

Statistics (all in `drift_metrics.py`, pure numpy):

- **Subspace drift**: normalized chordal distance in `[0,1]` between consecutive-turn
  subspaces, from principal angles. **Gauge-invariant**, so it measures the subspace
  moving, never the arbitrary basis spinning. `0` = identical, `1` = orthogonal.
- **Cross-overlap `omega(S_t, H_t)`** in `[0,1]`: the entanglement a static
  orthogonalizer drives to 0 once; the premise is that it **moves** across turns.
- **Noise floor (the null)**: drift between two independent bootstrap estimates of
  the *same, unchanged* subspace. This is the right null for consecutive-turn drift.
- **Noise-corrected drift**: `sqrt(max(0, d_meas^2 - floor^2))`, de-biasing the
  upward inflation from estimation noise.
- **Verdict (pre-registered, mechanical)**: `PREMISE_SUPPORTED` iff consecutive
  drift is significantly above the no-change null (disjoint 95% bands) **and** the
  noise-corrected effect exceeds `min_effect` **and** `omega` moves beyond its noise
  band. Otherwise `PREMISE_FALSIFIED`.

## Statistical power (read this before trusting a verdict)

Subspace estimation in high-dimensional activation space is noisy, and the noise
floor scales like `1/sqrt(n_probes)`. Measured on `Qwen2.5-0.5B-Instruct`, layer 12,
from the 16 hand-written entity probes:

| rank | n_probes | noise floor (median) | noise floor (95%) |
|-----:|---------:|---------------------:|------------------:|
| 1 | 16 | 0.243 | 0.361 |
| 1 |  8 | 0.324 | 0.503 |
| 2 | 16 | 0.683 | 0.733 |
| 3 | 16 | 0.663 | 0.800 |

Lessons baked into the defaults:

- **Use rank 1** (the difference-of-means direction) unless you have hundreds of
  probes. Rank 2-3 from a few dozen probes is pure noise (floor ~0.7).
- **Use many probes.** To detect, say, a 6 degree per-turn rotation (chordal ~0.1)
  with rank 1, you need roughly 90+ probe pairs. The default banks are expanded with
  surface-matched templates to make a real run adequately powered; for a publishable
  result, expand them much further (AdvBench-scale).
- **A `PREMISE_FALSIFIED` verdict with a high noise floor is underpowered, not
  evidence of no drift.** The runner prints the floor so you can tell the difference.

## Run

```bash
pip install -r requirements.txt

# 1) Offline validation of the math, null model, and decision rule (no model, no net).
#    Validates the estimator against analytic ground truth (known rotation angles).
python3 test_synthetic.py

# 2) Real model. Defaults: Qwen2.5-0.5B-Instruct, middle layer, rank 1, expanded banks.
python3 run_drift_probe.py

# options
python3 run_drift_probe.py --model <hf-id> --layer 12 --rank 1 --max-probes 48
python3 run_drift_probe.py --dialogues crescendo benign_escalation   # subset
python3 run_drift_probe.py --model allenai/OLMoE-1B-7B-0924-Instruct  # MoE (needs RAM)
```

Outputs `results/drift_results.json` and `results/drift_trajectories.png`.

## Results: an honest arc on Qwen2.5-0.5B-Instruct

The premise was tested in escalating rigor. Each step caught a confound in the
previous one; that is the point of the harness, and why the headline moved.

1. **v1 (`run_drift_probe.py`, 32 probes, one layer, consecutive drift).**
   Underpowered: noise floor ~0.27, only 1 of 8 layers marginally significant
   (chance). Verdict FALSIFIED-but-underpowered.

2. **v2 naive (`run_strong.py`, 96 probes, all layers, cumulative displacement from
   turn 0).** Flipped to SUPPORTED, but the tell was that between-dialogue spread
   (~0.37) was far below the turn-0 displacement (~0.88): the motion was the generic
   no-context -> context jump, not multi-turn accumulation. Confound caught; baseline
   moved to turn 1.

3. **v2 turn-1 (between-dialogue spread vs same-context bootstrap floor).** SUPPORTED
   again (H spread 0.51 >> floor 0.22 across 21 layers). But an adversarial review
   (4 independent skeptics) showed the floor holds context length constant while the
   spread carries it, so "spread > floor" is near-tautological for ANY context-
   dependent readout, and the spread tracks the token-length gap monotonically
   (0.37 -> 0.51 as the gap grows 3 -> 20 tokens across turns). Downgraded: the
   geometry is PROMPT-DEPENDENT, but not yet shown to be content-specific.

4. **Placebo control (`placebo_control.py`), n=3 dialogues.** Scrambled twins
   (word-shuffled dialogues: identical tokens, so identical length + topic, coherent
   meaning destroyed) supply the length/topic-matched null. At n=3 this looked like a
   real, depth-localized effect: early layers (1-10) scrambled >= real (length/lexical
   artifact), but mid-late layers (11-21) real > scrambled by 0.05-0.20 (L12 real 0.51
   vs scrambled 0.39). Suggestive, but a point estimate with no confidence interval,
   and the review had warned the n=3 median rests on the one topically-divergent
   (chemistry) dialogue.

5. **Scaled control (`run_scaled.py`), n=12 diverse dialogues + dialogue-level
   bootstrap CIs.** The decisive test. Twelve topically diverse 5-turn dialogues, real
   vs scrambled twins, with a bootstrap CI over DIALOGUES so no single script can carry
   the result. Outcome: **the mid-late effect vanishes.** At EVERY layer the real-spread
   CI overlaps the scrambled-spread CI (e.g. L12 real 0.347 [0.27,0.47] vs scrambled
   0.322 [0.29,0.37]; L15 0.323 [0.24,0.41] vs 0.272 [0.24,0.32]). There is a tiny,
   consistent, NON-significant trend (real slightly > scrambled at mid-late layers), but
   no layer clears `real CI_lo > scrambled CI_hi`. The n=3 "content" signal was a
   small-sample artifact.

**Honest final conclusion: the premise is NOT supported on this model.** Once you
control for length/topic with scrambled twins AND use enough dialogues to get a
confidence interval, the hallucination subspace does NOT measurably rotate with
coherent dialogue content beyond what prompt length and topic already explain. The
apparent multi-turn drift is dominated by prompt length, and the apparent
content-conditionality is within noise. CONDUCTOR's core premise (dialogue content
rotates the disentanglement geometry, so a static projection is mis-specified) is not
established here; the evidence leans against it at this scale. The cheap experiment did
its job: **do not build CONDUCTOR on the strength of this premise without first**
finding the effect on (a) a larger model in SAE feature space, (b) a real MoE for the
per-expert claim, and (c) a directly-fit static projector shown to re-entangle. On the
current evidence, the conditional machinery is not yet motivated.

(This arc is the deliverable: each control overturned the previous headline. A result
that only appears with n=3 and a same-context null is not a result.)

## Powered v2 (`run_strong.py`)

`run_strong.py` is the stronger test, addressing the two weaknesses of the first run
(low power, single cherry-picked layer):

- **Hundreds of probes** (`probes.*_BIG`, ~136 safety / ~106 uncertainty pairs via
  surface-matched templates and a deterministic fake-entity generator), which lowers
  the noise floor.
- **All layers from one forward pass.** A real effect must show a **contiguous band of
  >= 2 significant layers**, not an isolated hit (which is consistent with chance).
- **Cumulative displacement baselined at turn 1.** The premise statistic is
  `d(U_1, U_t)`, the steady-state drift once context exists. Critically it does NOT use
  turn 0: `d(U_0, U_t)` is dominated by the generic "no-context -> has-context" jump,
  which every dialogue shares and which is not the multi-turn accumulation the
  framework is about. That onset is reported separately as `onset(turn0->1)`. (In a
  first cut, baselining at turn 0 produced a spurious PREMISE_SUPPORTED; the tell was
  that between-dialogue spread was far smaller than the turn-0 displacement, i.e. the
  movement was context-onset, not content. Hence the turn-1 baseline.)
- **Accumulation slope** `d(U_1, U_t)` vs `t`, and **between-dialogue spread** (content
  sensitivity), reported per layer.
- **Power curve** (noise floor vs n_probes) emitted as a figure.

```bash
python3 run_strong.py --model Qwen/Qwen2.5-1.5B-Instruct      # powered default
python3 run_strong.py --model Qwen/Qwen2.5-0.5B-Instruct --max-probes 100
```

Outputs `results/strong_results.json` and `results/strong_trajectories.png`
(displacement vs layer with floor band; displacement-from-turn-1 vs turn; power curve).

## What the verdict means for the framework

- `PREMISE_SUPPORTED`: the subspaces measurably rotate and `omega` moves across
  turns. A static projection is mis-specified; the conditional CONDUCTOR machinery
  is motivated. Next step: test whether a *conditional* projector actually beats a
  per-turn re-solve (the next falsification gate in the paper's design).
- `PREMISE_FALSIFIED` (and adequately powered): a static projection suffices. Do not
  build the conditional framework. This is the cheap, decisive negative the paper
  should want to know first.

## Honest limitations

- **Raw activation space, not SAE features.** No SAE library is required; subspaces
  are found by contrast directly in the residual stream (exactly how refusal/truth
  directions are found). SAE features are the interpretable lift and would change the
  *coordinates*, not the *geometry* claim. Add `sae_lens` to test in feature space.
- **One layer, last token.** The geometry can differ by layer and token position;
  sweep `--layer` before concluding.
- **The MoE sub-basis claim is only partly testable here.** `extract.py` records the
  routed expert when the model exposes router logits, but the dense default model
  does not exercise it. A real MoE run is needed for the per-expert part of the
  premise.
- **The verdict is about this model, layer, and probe bank**, not LLMs in general.
- **Templated probes** trade some quality for power; inspect them in `probes.py`.

## Files

- `drift_metrics.py`  geometry + null models + decision rule (pure numpy, model-free)
- `test_synthetic.py` analytic-ground-truth validation of the above (runs anywhere)
- `probes.py`         contrastive probe banks (+templated expansion) + dialogue scripts
- `extract.py`        architecture-agnostic activation extraction (+ optional router)
- `run_drift_probe.py` v1 experiment (consecutive drift, single layer)
- `run_strong.py`     v2 powered experiment (all layers, content-spread, activation cache)
- `sweep_layers.py`   per-layer drift sweep from one forward pass
- `placebo_control.py` length/topic-matched scrambled control (the decisive content test)
- `run_scaled.py`      12 diverse dialogues + dialogue-level bootstrap CIs (hardens n=3)

Verdict precedence: `run_strong.py` reports whether the geometry is prompt-dependent;
`placebo_control.py` is what actually adjudicates content-conditionality. Trust the
placebo over the raw spread-vs-floor verdict.
