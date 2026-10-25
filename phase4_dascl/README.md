# Phase 4: domain-aware batches for supervised contrastive learning (low resource)

Everything specific to phase 4 (E17–E19) lives in this folder. It builds on the phase-3 split and
helpers (`phase3_scl/`) and on the core package (`03_src/clinc_domain_aux/`, where
`DomainAwareBatchSampler` and the `training.sampler: domain_aware` option were added; existing configs
behave exactly as before).

**Question.** With 10 training utterances per intent, does a batch composition with more same-domain,
different-intent hard negatives (domain-aware, DA) make SupCon *more* useful than with intent-balanced (IB)
batches, and does the extra gain come from fewer within-domain confusions?
Plan, hypotheses and decision rules: [`00_docs/01_experiment_plan.md`](00_docs/01_experiment_plan.md)
(revision 2 of the student's draft `phase3_scl/00_docs/02_domain_aware_sampling_plan.md`).

## Design: 2 × 2 factorial, paired by seed

| | CE | CE+SupCon (λ=0.5, τ=0.1) |
|---|---|---|
| IB (8 intents × 4) | `arm_n10_ib_ce.yaml` | `arm_n10_ib_scl.yaml` |
| DA (2 domains × 4 intents × 4) | `arm_n10_da_ce.yaml` | `arm_n10_da_scl.yaml` |

Primary (10-shot, 10 seeds, evaluated on the unseen pool U):
H1 interaction on macro-F1 > 0; H2 interaction on within-domain error < 0 (Bonferroni α = 0.025).
25-shot (5 seeds) is a descriptive replication. All arms: max 20 epochs, patience 3, weights-only checkpoints.

## Evaluation data (CLINC only)

* **U (primary)**: phase-3 T3 minus the n25 subset — 55 per intent, 8,250 utterances; never trained on by
  phase-4 models and never evaluated. Fixed and hashed by step 01 *before* training.
* **IHT (replication)** and **OOS-251**: phase-3 internal test sets, used for the second time (disclosed).

## How to run (Windows + RTX 4090, Git Bash, `trans` environment, from the repository root)

Prerequisite: the phase-3 split exists under `../dataset/clinc_phase3/` (it does after phase 3).

```bash
conda activate trans
export PYTHON_BIN=python
bash phase4_dascl/05_jobs/01_train.sh     # J1: unit tests, sampler audit + fixed U, 40 n10 runs and 20 n25 runs
#   -> push runs, 07_metadata/ and job logs; supervisor reviews and approves in phase4_dascl/01_configs/study.yaml
bash phase4_dascl/05_jobs/02_evaluate.sh  # J2: ONE evaluation, statistics, analysis (refuses a second run)
```

Rules: never pass `--data-root`, never edit a config after its runs started, never drop or rerun a seed,
and report errors instead of editing and rerunning. Estimated time: about 3 GPU hours; disk about 30 GB.

## Steps

| Step | File | What it does |
|---|---|---|
| P4-S01 | `04_steps/01_audit_and_fix_unseen.py` | replays epoch-1 batches of both samplers (same seeds as training) to document same-domain negatives; fixes and hashes U |
| P4-S02 | `04_steps/02_evaluate_test.py` | approval gate, single evaluation on U / IHT / OOS-251, val thresholds, centroid geometry |
| P4-S03 | `04_steps/03_statistics.py` | H1, H2 and descriptive contrasts (simple effects, cross-domain error, OOS, IHT, n25) |
| P4-S04 | `04_steps/04_analyze.py` | per-domain table, predeclared confusable pairs, geometry, sampler summary, cases, interaction plots |

## What to commit

Configs, `07_metadata/` (sampler audit and U manifest), run records under `08_runs/*/` (checkpoints and
`.npz` are git-ignored), `08_runs/job_logs/`, and `09_results/` (predictions, statistics, analysis, figures).
Fill in the `interpretation` column of `09_results/analysis/cases.csv` by hand.
