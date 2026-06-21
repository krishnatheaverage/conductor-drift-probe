# Running the SAE-feature-space test on Colab

This is the decisive experiment. It runs the same confound-controlled protocol
(`run_scaled_sae.py`) in SAE feature space. Two paths.

## Path A (do this first): GPT-2 SAEs, ungated, free Colab T4

No HF token, no Gemma license. Tests the hallucination subspace in real SAE space.

1. https://colab.research.google.com -> New notebook -> Runtime -> Change runtime type
   -> **T4 GPU** -> Save.
2. Paste and run this one cell (the repo is private, so the clone uses a token; see note):

```python
# --- clone the repo ---
# PRIVATE repo: paste a GitHub token with repo read scope (github.com/settings/tokens).
GH_TOKEN = "PASTE_GITHUB_TOKEN"   # or make the repo public and use a plain https clone
!git clone https://{GH_TOKEN}@github.com/krishnatheaverage/conductor-drift-probe.git
%cd conductor-drift-probe

# --- install and run ---
!pip -q install sae-lens transformer-lens
!python run_scaled_sae.py --backend gpt2 --max-probes 40
```

3. Read the verdict it prints (`SAE-SPACE VERDICT (gpt2): ...`). Download
   `results/sae_results_gpt2.json`.

## Path B (step 2: the instruct + SAFETY version): Gemma-2 + Gemma Scope, GATED

This run tests BOTH subspaces: hallucination AND safety/refusal (the side GPT-2 could
not do). It prints two verdicts.

1. Accept the license: log into HF, open https://huggingface.co/google/gemma-2-2b-it
   AND https://huggingface.co/google/gemma-scope , click "Acknowledge license" (instant).
2. Get a **Read** token: https://huggingface.co/settings/tokens .
3. In Colab (A100 recommended; T4 is tight but usually works for 2B):

```python
import os
os.environ["HF_TOKEN"] = "PASTE_HF_TOKEN"
!pip -q install sae-lens transformer-lens
!git -C conductor-drift-probe pull 2>/dev/null || git clone https://github.com/krishnatheaverage/conductor-drift-probe.git
%cd conductor-drift-probe
!python run_scaled_sae.py --backend gemma --max-probes 48
```

Caveat printed by the script: Gemma Scope SAEs were trained on the BASE model, so on the
-it model there is a mild distribution mismatch (indicative, fine for a first pass). For an
in-distribution hallucination-only run, add `--gemma-model google/gemma-2-2b`.

## Step 3: larger model (raw space) + rank-3 for breadth

No SAEs needed; these run `run_scaled.py` (raw activations) on the GPU. Each is one line.

```python
%cd /content/conductor-drift-probe
!pip -q install transformers accelerate
# a bigger instruct model, raw space:
!python run_scaled.py --model Qwen/Qwen2.5-7B-Instruct --max-probes 64
# a different family:
!python run_scaled.py --model microsoft/Phi-3.5-mini-instruct --max-probes 64
# rank-3 subspaces (higher variance; uses all probes):
!python run_scaled.py --model Qwen/Qwen2.5-1.5B-Instruct --rank 3
```

Download each `results/*.json` (or just paste the verdict + per-layer table back) and I
fold them into the paper.

## How to read the result (this is the whole paper)

- **NOT_CONTENT_CONDITIONAL** (real and scrambled CIs overlap, as in raw space):
  the null replicates in feature space -> the cautionary/measurement paper is
  airtight. Add this as the SAE row and submit Path 1.
- **CONTENT_CONDITIONAL** (real CI clears scrambled CI over >=3 contiguous layers):
  the premise holds where methods operate -> a positive finding that revives the
  CONDUCTOR framework. Escalate.

## Notes / first-run gotchas

- If `SAE.from_pretrained` errors on the return signature, the API drifted: it is
  isolated in `load_sae()` (run_scaled_sae.py); adjust there and re-run.
- GPT-2 is not instruct-tuned, so only the hallucination/uncertainty subspace is
  meaningful on Path A; the safety subspace needs Path B.
- To avoid the token-in-clone, you can make the GitHub repo public (it is a research
  harness with no secrets) and use a plain `git clone https://github.com/...`.
