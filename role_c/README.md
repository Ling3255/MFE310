# Role C: rejector design and selection

Plan: [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) (revision 3 + section 14 addendum). All steps read saved
predictions and frozen thresholds; c2/c4 additionally use the locally saved CLS features / FP32 logits of
`BERT_Rejection_Study_20261019` (git-ignored `*.npy`) when they exist. Nothing is trained and no existing
directory is modified. Outputs go to `role_c/outputs/` (the `cache/` subfolder holds git-ignored `.npz`
score caches).

**Without the `*.npy` assets and checkpoints** (current snapshot): c2 records the fitted Ledoit-Wolf
shrinkage only (no ablation), c4 compares raw with fitted-temperature scores from the historical per-row
outputs (no temperature sweep), and the demo is `demo_replay.py` (replays saved outputs, no live BERT).

## Run (repository root, Git Bash)

```bash
PY=BERT_Rejection_Study_20261019/.venv/Scripts/python.exe

$PY -m unittest discover -s role_c/tests -v          # 18 tests, synthetic data only
$PY role_c/run_role_c_analysis.py c0                 # must print PASS; if FAIL stop and report
$PY role_c/run_role_c_analysis.py c1                 # mechanism + complementarity (keeps written interpretations)
$PY role_c/run_role_c_analysis.py c4                 # calibration vs separation (FP32 sweep, or historical fallback)
$PY role_c/run_role_c_analysis.py c2                 # covariance ablation, or LW diagnostics without features
$PY role_c/run_role_c_analysis.py c3                 # threshold bootstrap (a few minutes)
$PY role_c/run_role_c_analysis.py c5                 # cost-driven choice incl. combinations (run after c1, c2)

# demo with checkpoints present:
$PY role_c/demo_rejector.py --text "what is my account balance" --text "who won the game last night"
# demo without checkpoints (replay of saved outputs; say so on screen):
$PY role_c/demo_replay.py --cases
$PY role_c/demo_replay.py --text "how do i make my android phone more secure"
```

`all` runs c0-c5 in order. `--bootstrap N` changes the number of draws for c3/c5 (default 2000; do
not change it for reported results). Commit `role_c/outputs/` (except `cache/`) and push.

## Outputs used in the report and presentation

| Question | Files |
|---|---|
| Inputs reproduce | `c0_verification.json`, `input_hashes.json` |
| Why reject (mechanism) | `score_distributions.png`, `separation_clean*.csv` |
| Complementarity | `score_correlations.csv`, `rejection_overlap.csv`, `complementarity_decision.json`, `disagreement_cases.csv` (fill `interpretation`) |
| Distribution fitting | `covariance_ablation*.csv`, `covariance_selection.json`, `covariance_ablation.png`; without features `covariance_diagnostics.json` |
| Threshold stability | `threshold_bootstrap_distribution.csv`, `threshold_frozen_test.csv`, `threshold_stability_summary.json`, `threshold_stability.png` |
| Calibration vs separation | `calibration_vs_separation.csv`, `calibration_bins.csv`, `calibration_summary.json`, `calibration_vs_separation.png`; fallback adds `calibration_rank_flips.csv` |
| Why this method (cost) | `cost_candidates.csv`, `method_choice_by_cost.csv`, `method_choice_heatmap.png`, `candidate_separation*.csv` (incl. combinations) |
| Hand-over to role D | `rejector_candidates.json` |
| Evidence trail | `evidence_map.json` |
