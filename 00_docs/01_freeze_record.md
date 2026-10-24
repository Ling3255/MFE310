# E5 freeze record

Created: 2026-10-24T08:47:19.810539+00:00

Status: **awaiting supervisor approval; test evaluation is blocked**.

Selected lambda: `0.1` (validation only).

The domain-auxiliary design did not use test results. Other group members had previously inspected this benchmark's test set. Final results must be described as a one-time evaluation under a frozen protocol, not fully independent confirmation.

## Model and detector hashes

- `bert_ce` seed `0`: checkpoint SHA-256 `c7ea329ae1ba7399065427cf9a48e1ce2fc3cb65f7073d6a2949e3d26e472dc2`, detector SHA-256 `46449f749135394c6b93712df48d004ff11530d4bf0ca843e2a0e85de0953d73`.
- `bert_ce` seed `1`: checkpoint SHA-256 `089a4ef6890bdd7f124d4c0bb832208633663eec349353ca9eb028addf5aab8b`, detector SHA-256 `a2877ee67f9a041c1e9540736d23037145ec7c63f5b6cb1243ce7ced981d5c20`.
- `bert_ce` seed `2`: checkpoint SHA-256 `fbf351c77575df787952900b001dcf9ebf75e6eb971e748e8136cbf508e90619`, detector SHA-256 `513a3696d14f1c5114aeb8cf8ad65d52e97311d2ad6e650bc9539c08d9e31d68`.
- `bert_dom` seed `0`: checkpoint SHA-256 `5b8d627a65859a5ef161121f0954defb470bb56ef3d214977df760a43ccae81b`, detector SHA-256 `c030663da090dd2bc803481c40270c3b5c8dacbf5a269707f61656bd70ace221`.
- `bert_dom` seed `1`: checkpoint SHA-256 `5dfcfd503f9ffd73777129b96f18688b8e5b31f3711aee4409dffcc62471e9a2`, detector SHA-256 `5b4ba58a8e5d82e0ee7031df3e07074bc84fd821ffcabf36e558fd59bd48a0dd`.
- `bert_dom` seed `2`: checkpoint SHA-256 `499ea4821450e6ce672d8772f22f566b7eecf2b9b86d407d5c0ea57e9693bbc0`, detector SHA-256 `903ee1914f28b57957b5163707605c23106d7402e437042945d013d01a7bc5cb`.

## Metric direction frozen before test

The primary report metric is `fpr95_is_positive`: OOS false acceptance when IS acceptance reaches at least 95%. The supplementary metric is `fpr95_oos_positive`: IS false rejection when OOS recall reaches at least 95%. Both are computed, and the primary direction must not be changed after inspecting test results.

## Supervisor confirmation

Set `supervisor_approval.approved: true` in `01_configs/04_frozen.yaml` only after actual confirmation, and fill in `confirmed_by`, ISO `confirmed_at`, and `evidence`. S06 checks all four fields and all artifact hashes.

Any protocol change after test inspection requires a new recorded revision and disclosure; do not overwrite this record.
