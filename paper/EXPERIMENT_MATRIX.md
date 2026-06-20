# Path to a TMLR submission: the evidence matrix

TMLR accepts on two criteria: (1) claims are correct and supported by the evidence,
(2) some of the audience would find it useful. Novelty/SOTA are explicitly not
required. This paper's claim is a *scoped negative*: "the obvious measurements of
context-conditional safety/hallucination geometry are confounded, and the controlled
measurement returns a null on the models tested." The acceptance risk is NOT novelty;
it is a reviewer saying **"you didn't look hard enough"** (wrong model, wrong space,
wrong rank). The whole job before submission is to close that objection with breadth.

Each row below is a reviewer objection mapped to the exact command that answers it.
The harness already produces the statistic; these just fill cells.

## Status

| cell | objection it closes | status | command |
|---|---|---|---|
| Qwen2.5-0.5B, raw acts, rank-1 | baseline | DONE | `python3 run_scaled.py --model Qwen/Qwen2.5-0.5B-Instruct` |
| Qwen2.5-1.5B, raw acts | "only the smallest model" | RUNNING (laptop, slow) | `python3 run_scaled.py --model Qwen/Qwen2.5-1.5B-Instruct` |
| Qwen2.5-3B / 7B-Instruct | "too small to have the geometry" | TODO (GPU) | `python3 run_scaled.py --model Qwen/Qwen2.5-7B-Instruct` |
| a second model family | "Qwen-specific" | TODO | e.g. `--model microsoft/Phi-3.5-mini-instruct` (ungated) |
| rank-2 and rank-3 | "rank-1 is unstable / too crude" | TODO | add `--rank 3` (needs >=200 probes; see power note) |
| **SAE feature space** | "raw activations aren't where methods operate" | TODO (needs `sae_lens` + gated Gemma) | see below |
| MoE model | per-expert variant of the premise | TODO (GPU) | `--model allenai/OLMoE-1B-7B-0924-Instruct` |
| within-topic content control | "you conflated topic with content" | TODO | add same-topic / vary-epistemic-state dialogues to `probes.DIALOGUES_SCALED` |
| static-projector mis-specification | "you never tested the actual claim" | TODO | fit one static disentangling projector on turn 1, measure re-entanglement at turn 5 |

## The SAE-feature-space cell (the most important one)

This is the space conditional-disentanglement methods actually use, so a reviewer will
want it. Two practical blockers on a laptop, both resolved with the user's resources:

1. **`sae_lens` install.** `pip install sae-lens` pulls `transformer_lens` and may pin
   `torch`/`transformers`; install in a fresh venv to avoid disturbing the current env.
2. **Gemma Scope is for Gemma-2, which is GATED on Hugging Face.** You must accept the
   Gemma license on HF and set `HF_TOKEN`. Then `Gemma-2-2B` + the matching
   `gemma-scope-2b-pt-res` SAEs load via `sae_lens`.

Port plan (one new backend, ~30 lines): in `extract.py`, after reading the residual
activation at a layer, pass it through the frozen SAE encoder to get the sparse feature
vector `z = E(x)`, and run the identical contrast/drift pipeline on `z` instead of `x`.
Everything downstream (`drift_metrics.py`, `run_scaled.py`, `placebo_control.py`) is
unchanged because it is feature-agnostic. Ungated fallback for a first pass:
`gpt2-small` or `pythia-*` SAEs (well supported in `sae_lens`) for the *hallucination*
subspace only (these are not chat models, so the refusal side is not meaningful).

## Decision: the result is no-lose for publication

- If the controlled null **holds** across the matrix (raw + SAE, several models): the
  paper is the cautionary/measurement contribution as written. Scope the title to the
  tested family and the claim is airtight.
- If the effect **appears** robustly in SAE space at scale (survives scrambled twins +
  dialogue-level CIs): that is a *positive* finding that genuinely motivates conditional
  disentanglement, and you upgrade to a measurement+method paper.

Either branch is a defensible TMLR submission. Run the SAE-space row next; the headline
depends only on it.

## Writing guardrails (these are what win/lose TMLR review)

- Scope every claim to the tested models/spaces ("on the instruction-tuned models we
  evaluate", not "in LLMs"). Over-generalization is the #1 reject reason for negatives.
- Lead with the protocol (gauge-invariant drift + scrambled-twin null + dialogue-level
  CI) as a reusable contribution, so the paper is useful even to readers who don't care
  about the specific null.
- Keep the rigor cascade (Table 1): reviewers reward demonstrated falsification of one's
  own intermediate positives.
- Pre-register the decision rule (done: real CI_lo > scrambled CI_hi over >=3 contiguous
  layers) and report the power/noise floor so a null is never an under-powered run.
