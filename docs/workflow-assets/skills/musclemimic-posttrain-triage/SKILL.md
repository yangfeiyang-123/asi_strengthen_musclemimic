---
name: musclemimic-posttrain-triage
description: Triage MuscleMimic post-training experiments by comparing candidate arms against baseline, reading validation metrics/videos, checking WandB or local outputs, and recommending the next experiment. Use when the user asks about E1/E2/E3 arms, baseline vs posttrain quality, latest validation results, bad reward/error trends, whether a run is working, or what to try next for ForehandNetLift, ForehandClear, grip-hold, or static-hit posttrain runs.
---

# Musclemimic Posttrain Triage

## Overview

Use this workflow to turn scattered training logs, validation videos, config diffs, and checkpoint paths into a concrete run verdict. The output should say whether the candidate is better than baseline, what evidence supports that, what failure mode is most likely, and the smallest next experiment to run.

## Workflow

1. Anchor the comparison.
   - Identify the action, arm name, run name, checkpoint, config, and baseline checkpoint.
   - Prefer explicit spec files under `experiments/posttrain/` and configs under `fullbody/config_specific_task/posttrain/`.
   - If the user only names an arm such as `E2c`, locate the matching config and output directory before interpreting metrics.

2. Reconstruct what changed.
   - Diff candidate config against baseline and the previous arm.
   - Summarize only behaviorally relevant changes: reward weights, PPO/KL/anchor terms, init policy, reset behavior, curriculum, action scale, env count, validation cadence.
   - State whether the change should affect early learning, late stability, or only validation.

3. Gather evidence from three surfaces.
   - Training curves: reward components, errors, termination length, entropy/KL if present, and wall-clock progress.
   - Deterministic evaluation: baseline and candidate on the same motion paths, seeds, and metrics flags.
   - Visual validation: latest videos or screenshots for pose collapse, foot sliding, grip loss, shuttle/racket contact, or orientation errors.

4. Compare fairly.
   - Do not compare different validation splits, motion paths, seeds, or checkpoint ages without calling that out.
   - For "better than baseline", require at least one shared deterministic metric comparison plus a visual sanity check.
   - Treat reward improvements without error or video improvement as weak evidence.

5. Diagnose before proposing.
   - If candidate and baseline trends are nearly identical, check whether the new config is actually loaded and whether reward terms are nonzero.
   - If validation gets worse while training reward rises, suspect reward hacking, invalid reference weighting, or evaluation mismatch.
   - If early metrics do not move, check whether the new term is gated by phase/contact/threshold and has not activated.
   - If videos fail while scalar metrics look good, prioritize visual failure mode over aggregate reward.

6. Recommend the smallest next run.
   - Give one primary next experiment and one fallback.
   - Include exact config changes and the run command when possible.
   - Say what observation would falsify the recommendation and when to stop the run.

## Useful Commands

Use these as starting points, adjusting paths and arm names to the current task:

```bash
git diff -- experiments/posttrain fullbody/config_specific_task/posttrain
```

```bash
MM_CUDA_VISIBLE_DEVICES=0 scripts/run_with_cuda_compat.sh \
  .venv/bin/python musclemimic/badminton/scripts/evaluate_posttrain_protocol.py \
  --spec experiments/posttrain/<spec>.yaml \
  --arm <arm_name> \
  --run-name <run_name> \
  --splits train,validation,stress_test \
  --execute
```

## Output Format

Return:

- Verdict: `improving`, `neutral`, `regressed`, or `inconclusive`.
- Evidence: metric comparison, visual evidence, and config/run evidence.
- Failure mode: one or two likely causes, with confidence.
- Next action: exact run/config change and stop condition.
- Gaps: missing metrics, videos, checkpoint paths, or WandB access.
