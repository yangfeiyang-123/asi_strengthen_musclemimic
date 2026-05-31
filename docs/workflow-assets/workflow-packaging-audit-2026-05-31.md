# Workflow Packaging Audit - 2026-05-31

## Evidence Sources

- Codex raw history: `/home/yangfeiyang/.codex/history.jsonl`, last 30 days.
- Codex thread metadata: `/home/yangfeiyang/.codex/state_5.sqlite`, last 30 days.
- Codex memories: `/home/yangfeiyang/.codex/memories_1.sqlite`; `stage1_outputs` and `jobs` were empty.
- Chronicle: no targeted local Chronicle files were found under `~/.chronicle` or `~/.config`; broad home search was abandoned as too expensive.
- Existing assets: installed skills include `training-check`, `monitor-experiment`, `analyze-results`, `run-experiment`, and Superpowers implementation/review workflows. Repo-local `.agents` and `.codex` directories existed but contained no files.

## Compact Shortlist

### Posttrain Experiment Triage

- Repeated workflow: compare posttrain arms against baseline, inspect WandB/local metrics, review validation videos, diagnose reward/error trends, and decide the next experiment.
- Supporting evidence and dates: raw Codex history includes 143 last-30-day matches for training/validation/baseline terms. Examples: 2026-05-26 "PostTrain ... baseline相比是否有提升"; 2026-05-27 "E1c和baseline谁好"; 2026-05-27 "现在E2c训练效果怎么样"; 2026-05-28 "检查一下，现在的训练效果". Git commits on 2026-05-26 to 2026-05-29 added posttrain evaluation protocol, E1/E2/E3 configs, validation video support, WandB logging, and replay prechecks.
- Frequency / confidence: High frequency / High confidence.
- Recommended form: Skill.
- Rationale: The workflow is stable, high-value, and repo-specific. Existing generic experiment skills do not encode the MuscleMimic config/output conventions.

### MuJoCo Visual Preflight

- Repeated workflow: generate or inspect MuJoCo reset images/videos, verify anatomy visibility, scene orientation, object placement, passive physics, local viewer portability, grip pose, racket handle shape, and shuttle behavior before training.
- Supporting evidence and dates: raw Codex history includes 93 last-30-day matches for visualization/viewer/screenshot terms and 32 matches for grip/racket/handle terms. Examples: 2026-05-25 racket/court/shuttle visual checks; 2026-05-27 viewer failures, anatomy display, passive falling, grip closeups, orientation, and octagonal handle checks; 2026-05-28 initial state facing net. Git commits on 2026-05-27 to 2026-05-29 repeatedly fixed overall viewer visuals, passive physics, court materials, grip seed transfer, octagonal handle bevels, reset videos, and replay prechecks.
- Frequency / confidence: High frequency / High confidence.
- Recommended form: Skill.
- Rationale: This is repeated, expensive to debug late, and has clear pass/fail invariants. It is narrower than a general MuJoCo skill and tailored to this repository.

### Subagent Implementation And Review Loop

- Repeated workflow: implement task from a plan, run spec-compliance review, run code-quality review, fix findings, and re-review.
- Supporting evidence and dates: Codex thread metadata shows 197 last-30-day task/review/implementation threads in this repository, especially on 2026-05-20, 2026-05-21, 2026-05-25, and 2026-05-27.
- Frequency / confidence: High frequency / High confidence.
- Recommended form: Extend Existing / Skip creation.
- Rationale: Already covered by Superpowers `subagent-driven-development`, `requesting-code-review`, `receiving-code-review`, `executing-plans`, and `verification-before-completion`. Creating another local asset would duplicate workflow ownership.

### Retargeting And Dataset Preparation

- Repeated workflow: convert WHAM/SMPL badminton clips to musculoskeletal trajectories, validate frame rate and drift, and document commands.
- Supporting evidence and dates: raw Codex history examples on 2026-05-01, 2026-05-12, and 2026-05-14 mention retargeting WHAM/SMPL files, updating `tools.md`, and debugging 60 Hz vs 100 Hz / trajectory cleaning.
- Frequency / confidence: Medium frequency / Needs validation.
- Recommended form: Skip for now.
- Rationale: It likely deserves a skill later, but the exact stable procedure should be confirmed from `tools.md`, datasets, and current retarget scripts before packaging.

### Research Reading And Strategy Review

- Repeated workflow: read papers or strategy docs, critique ASI/SFV/posttrain ideas, and propose research directions.
- Supporting evidence and dates: examples on 2026-05-11 and 2026-05-12 around SFV/ASI paper and strategy review.
- Frequency / confidence: Medium frequency / Medium confidence.
- Recommended form: Extend Existing / Skip creation.
- Rationale: Already mostly covered by `research-lit`, `research-review`, `auto-review-loop`, and `result-to-claim`; local details are not stable enough to justify a new asset.

## Created Assets

- `docs/workflow-assets/skills/musclemimic-posttrain-triage/`
- `docs/workflow-assets/skills/musclemimic-mujoco-visual-preflight/`

No subagents or automations were created because the high-confidence candidates were best represented as small skills.
