# Role C: rejector design and selection

Plan: [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) (revision 3). All steps read saved predictions, frozen
thresholds and the locally saved CLS features / FP32 logits of `BERT_Rejection_Study_20261019`
(git-ignored `*.npy`); nothing is trained and no existing directory is modified. Outputs go to
`role_c/outputs/` (the `cache/` subfolder holds git-ignored `.npz` score caches).

## Run (repository root, Git Bash)

```bash
PY=BERT_Rejection_Study_20261019/.venv/Scripts/python.exe

$PY -m unittest discover -s role_c/tests -v          # 16 tests, synthetic data only
$PY role_c/run_role_c_analysis.py c0                 # must print PASS; if FAIL stop and report
$PY role_c/run_role_c_analysis.py c1                 # mechanism + complementarity
$PY role_c/run_role_c_analysis.py c4                 # calibration vs separation (needs logits)
$PY role_c/run_role_c_analysis.py c2                 # covariance ablation (needs CLS features)
$PY role_c/run_role_c_analysis.py c3                 # threshold bootstrap (a few minutes)
$PY role_c/run_role_c_analysis.py c5                 # cost-driven choice (run after c2)
$PY role_c/demo_rejector.py --text "what is my account balance" --text "who won the game last night"
```

`all` runs c0-c5 in order. `--bootstrap N` changes the number of draws for c3/c5 (default 2000; do
not change it for reported results). Commit `role_c/outputs/` (except `cache/`) and push.

## Outputs used in the report and presentation

| Question | Files |
|---|---|
| Inputs reproduce | `c0_verification.json`, `input_hashes.json` |
| Why reject (mechanism) | `score_distributions.png`, `separation_clean*.csv` |
| Complementarity | `score_correlations.csv`, `rejection_overlap.csv`, `complementarity_decision.json`, `disagreement_cases.csv` (fill `interpretation`) |
| Distribution fitting | `covariance_ablation*.csv`, `covariance_selection.json`, `covariance_ablation.png` |
| Threshold stability | `threshold_bootstrap_distribution.csv`, `threshold_frozen_test.csv`, `threshold_stability_summary.json`, `threshold_stability.png` |
| Calibration vs separation | `calibration_vs_separation.csv`, `calibration_bins.csv`, `calibration_summary.json`, `calibration_vs_separation.png` |
| Why this method (cost) | `cost_candidates.csv`, `method_choice_by_cost.csv`, `method_choice_heatmap.png` |
| Hand-over to role D | `rejector_candidates.json` |
| Evidence trail | `evidence_map.json` |
