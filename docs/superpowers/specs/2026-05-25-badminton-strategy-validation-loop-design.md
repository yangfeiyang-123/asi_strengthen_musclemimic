# Badminton Strategy Validation Loop Design

## Context

The current badminton action-stage tooling can reproducibly separate motions into `base`, `posttrain`, `repair`, and `exclude` buckets, then generate stage manifests for training. The relevant implementation is:

- `musclemimic/utils/action_stage.py`
- `musclemimic/badminton/scripts/recommend_action_stages.py`
- `musclemimic/badminton/scripts/build_stage_manifests.py`
- `manifests/generated/`
- `doc/PostTrain_Advice.md`

That tooling is useful, but it does not by itself prove the paper-level method claim. It proves that the current heuristic is executable and repeatable. It does not prove that the thresholds are optimal, that the stage assignments improve learning, or that the method generalizes across badminton actions.

The latest generated recommendations show exactly why this needs a stricter validation loop. There are borderline examples where a tiny threshold change could alter the training stage:

- `ForehandNetLift/best/video02_best_stage7_smpl`: `base/net_frontcourt`, root displacement `0.336`, peak speed `1.184`, just below the `1.2` posttrain speed cutoff.
- `ForehandNetLift/best/video10_best_stage6_smpl`: `repair/net_frontcourt`, root displacement `0.238`, peak speed `1.370`, showing that speed alone does not resolve questionable root motion.
- `Smash/best/video10_best_smpl`: `base/smash`, yaw `0.790`, just below the `0.8` rotation cutoff.
- `ForehandClear/raw/video1_raw_wham`: `posttrain/general`, peak speed `3.564`, suggesting large-motion detection can catch raw movement-heavy clips.
- `Backhand/best/video2_best_smpl`: `base/rotation`, root displacement `0.591`, near the `0.60` large-displacement cutoff.

These examples are not failures by themselves. They are evidence that the method needs explicit confidence scores, diagnostics, ablations, and claim-to-evidence gates before being treated as paper-ready.

## Confidence Position

There is no honest way to claim literal 100% confidence in this strategy before running validation experiments. The achievable target is practical confidence:

> After repeated adversarial audits, data diagnostics, ablations, held-out evaluations, and claim-to-evidence checks, there are no known critical loopholes left, and all remaining limitations are explicitly scoped.

This design therefore treats "100% confidence" as a stopping rule for a validation loop, not as a statement that the heuristic is mathematically guaranteed.

## Goal

Build a method-level validation loop for the badminton training strategy so that every paper claim is backed by diagnostics and experiments.

The loop should answer four questions:

1. Is each motion clip credible enough to train from?
2. Is the assigned stage/family defensible?
3. Does staged training improve learning compared with simpler alternatives?
4. Which claims are supported, weakened, or rejected by evidence?

## Non-Goals

- Do not claim fine hand, finger, racket-face, or shuttle-contact control when the current capture pipeline cannot observe those details reliably.
- Do not replace the PPO algorithm in this design. The first target is validating the data/stage strategy around the existing training stack.
- Do not require a full large-scale training sweep before the validation machinery exists.
- Do not treat current thresholds as final universal constants.
- Do not use a successful single-action posttrain run as evidence for generality.

## Main Vulnerabilities

### 1. Threshold Brittleness

Current cutoffs such as root displacement `0.25/0.60`, peak speed `1.2`, and yaw `0.8` are reasonable first-pass heuristics, but several clips are close to those boundaries. A threshold-only decision can silently flip a motion between `base` and `posttrain`.

Mitigation: add confidence bands around thresholds and mark borderline clips as `review_required`. Stage manifests should preserve both the hard assignment and the reason it may be unstable.

### 2. Observability Gaps

The current method cannot reliably inspect detailed hand shape, racket face, shuttle contact, or subtle finger-driven net actions. The code already has `fine_hand_dominant` and `contact_unreliable` hints, but the strategy still needs a stronger rule: unobservable fine details cannot become main claimed capabilities.

Mitigation: restrict claims to body-scale badminton imitation, footwork, trunk rotation, reaching, lunging, clears, smashes, lifts, and larger net-frontcourt movements. Fine net-shot hand actions should be excluded, repaired, or discussed as limitations unless extra sensors/features are added.

### 3. Retarget/Data Failure Modes

The stage classifier receives qpos/qvel/site signals after the capture and retargeting pipeline. If SMPL/WHAM/root reconstruction is wrong, the downstream stage assignment can look mathematically consistent while being physically wrong.

Likely failure modes include camera scale errors, drifting root, foot sliding, ground penetration, qpos/qvel spikes, discontinuities, crop loss, and missing endpoint reliability.

Mitigation: add a data credibility gate before training. A clip with unreliable kinematics should be sent to `repair` or `manual_review`, even if its action name would normally be trainable.

### 4. Stage/Family Assignment Issues

The current classifier uses root motion, speed, yaw, and hints. It requires `right_hand_world_path_length`, but that metric is not yet used in stage selection. That leaves a gap between the stated badminton task structure and the actual rule.

Mitigation: keep root/trunk/footwork as the first-order signal, but add hand-root consistency diagnostics. The strategy should not say it models fine hand control; it should say hand-path diagnostics help reject or review clips whose arm motion is inconsistent with the action family.

### 5. Experimental Evidence Gaps

The current tooling can generate manifests, but paper credibility needs comparison. Without baselines and ablations, the method can only claim "we organized data this way", not "this improves general badminton imitation".

Mitigation: require baseline-vs-method experiments and ablations before strong claims. At minimum, compare all-mix training, action-name grouping, current metric-gated staging, and metric-gated staging with key gates removed.

### 6. Reproducibility and Operations

Generated manifests are reproducible, but the larger training/evaluation loop still needs explicit records: which checkpoint was used, which manifest generated it, which thresholds were active, which clips were rejected, and which metrics passed.

Mitigation: every run should emit a small machine-readable report containing manifest hash, stage counts, thresholds, excluded clips, checkpoint path, evaluation metrics, and claim-gate status.

## Selected Approach

Three approaches were considered:

### A. Static Audit Only

Manually inspect the recommendations and update the documentation.

This is fast but insufficient. It can find obvious mistakes but cannot prove that staged training improves learning.

### B. Training-Only Validation

Run more PPO jobs and judge whether the final motions look better.

This may catch gross failures, but it is hard to debug. If a result improves or fails, we will not know whether the cause is data quality, stage assignment, reward balance, checkpoint selection, or action family grouping.

### C. Adversarial Validation Loop

Add diagnostics, confidence labels, experiment gates, ablations, and claim-to-evidence checks. Repeat until no known critical issue remains.

This is the recommended approach because it converts the strategy from a heuristic into an auditable method. It also gives the paper a defensible story: the method is general where the observable motion features support it, and it explicitly rejects or weakens claims where the data cannot support them.

## Validation Loop

The loop has eight steps:

1. Generate motion diagnostics for every candidate clip.
2. Assign stage/family plus confidence and failure modes.
3. Review low-confidence and borderline samples.
4. Build training and evaluation manifests.
5. Run baseline, method, and ablation experiments.
6. Analyze metrics into a claim-to-evidence table.
7. Update thresholds, hints, repair rules, or claims.
8. Repeat until the stop criteria pass.

The loop should be run at two levels:

- Dataset level: before training, to decide what can be used.
- Result level: after training, to decide what the paper is allowed to claim.

## Data Credibility Gate

Each motion should receive:

- `stage`: `base`, `posttrain`, `repair`, or `exclude`.
- `family`: `general`, `rotation`, `net_frontcourt`, `smash`, or a later family if added.
- `confidence`: `high`, `medium`, or `low`.
- `failure_modes`: a list of concrete issues.
- `review_required`: boolean.
- `required_action`: `train`, `posttrain`, `repair_first`, `exclude`, or `manual_review`.

Recommended diagnostics:

- root displacement, root path length, and root displacement ratio.
- peak root speed and average root speed.
- yaw range and trunk/root orientation change.
- right-hand path length and right-hand/root consistency.
- qpos/qvel spike detection.
- site position discontinuity.
- foot height, ground penetration proxy, and foot sliding proxy when foot sites are available.
- frame count and effective FPS.
- missing or unreliable endpoint/contact flags.

Confidence rules:

- High confidence: all required metrics exist, no major data-quality failures, far from decision thresholds.
- Medium confidence: metrics exist, but the sample is near a threshold or has minor quality concerns.
- Low confidence: missing key metrics, contradictory signals, suspected retarget failure, or unobservable fine-hand-dominant behavior.

## Training Effectiveness Gate

The method should be compared against simpler alternatives:

- `all_mix`: train all usable clips together.
- `action_name_grouping`: group only by action label.
- `metric_gated_staging`: current proposed method.
- `no_repair_gate`: remove repair/manual-review filtering.
- `no_rotation_speed_gate`: remove yaw/speed-based posttrain routing.
- `no_posttrain_root_focus`: use staged manifests but remove the posttrain root-motion emphasis.

Recommended evaluation metrics:

- root position RMSE.
- root displacement ratio error.
- root velocity or peak-speed error.
- heading/yaw error.
- right-hand position error where reliable.
- relative body pose error.
- early termination rate.
- foot slip/penetration proxy.
- control/action-rate cost.
- held-out action-family performance.

The evaluation set should include held-out clips and at least one held-out action family or source condition when possible. A method that only improves the exact posttrain clip does not support a generality claim.

## Claim Gate

Every paper claim should be mapped to required evidence.

### Claim 1: Staging improves training stability

Required evidence: lower early termination, fewer catastrophic failures, or better reward curves than `all_mix` and `action_name_grouping`.

If not supported: weaken to "staging organizes data for controlled training" rather than "improves stability".

### Claim 2: Posttraining helps movement-heavy badminton actions

Required evidence: improved root/yaw/path metrics on held-out or repeated movement-heavy actions such as clears, smashes, footwork, lunges, or larger lifts.

If only ForehandNetLift improves: claim action-specific fine-tuning, not general posttraining.

### Claim 3: Repair/exclusion prevents corrupted references from hurting training

Required evidence: `no_repair_gate` performs worse or produces visibly/quantitatively worse tracking on clips flagged as unreliable.

If not supported: repair gate remains a data hygiene practice, not a performance claim.

### Claim 4: Metric-gated staging is better than action-name grouping

Required evidence: the metric-gated method outperforms action-name grouping on at least one aggregate metric and does not regress badly elsewhere.

If not supported: the method should be reframed as a diagnostic tool rather than a superior training strategy.

## Stop Criteria

The strategy can be treated as practically credible only when all of these hold:

- No unresolved critical data or training failure remains.
- High-risk clips are either repaired, excluded, or manually reviewed.
- Borderline assignments are marked and do not silently drive strong claims.
- Each major paper claim has a corresponding experiment and metric table.
- At least one simple baseline and at least two ablations have been run.
- Held-out evaluation does not contradict the generality claim.
- Known unsupported regions, especially fine hand/racket/shuttle-contact details, are listed as limitations.

If any item fails, the loop continues by either fixing the method, changing the thresholds, rerunning an experiment, or weakening the claim.

## Error Handling and Fallbacks

- Missing required metrics: assign `low` confidence and route to `manual_review` or `repair_first`.
- Diagnostics disagree with visual inspection: visual/manual review wins, and the failed diagnostic becomes a tracked issue.
- Training improves one metric but degrades physical plausibility: do not pass the claim gate until the degradation is explained or fixed.
- GPU unavailable: generate manifests and experiment commands, but do not claim evidence until the jobs finish.
- Ablation results contradict the hypothesis: update the claim, not just the threshold.

## Testing and Verification

Implementation should add focused tests for:

- confidence-band behavior around thresholds.
- missing metric handling.
- failure-mode serialization.
- deterministic manifest generation with confidence fields.
- claim-table parsing or report generation.
- backwards compatibility with existing stage manifests.

Verification should include:

- unit tests for new diagnostics.
- a smoke run over the existing badminton data.
- a diff check confirming generated manifests are reproducible.
- a small report showing stage counts, low-confidence clips, and claim-gate readiness.

## Expected Output of the Next Implementation Plan

The next plan should be scoped to validation infrastructure first:

1. Extend motion diagnostics and confidence labels.
2. Emit validation reports next to recommendations/manifests.
3. Add claim-to-evidence report templates.
4. Add baseline/ablation command templates without launching a large sweep by default.
5. Update `doc/PostTrain_Advice.md` so the documented strategy matches the stricter validation loop.

Only after this infrastructure passes should we run or interpret larger PPO experiments as method evidence.
