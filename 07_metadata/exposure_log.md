# Test exposure log

- The domain-auxiliary model design does not use test predictions, test performance, or test case errors for model, checkpoint, lambda, detector, threshold, or candidate-confusion selection.
- Other group members previously inspected this benchmark's test set and used it during their spelling-augmentation project. This history is disclosed; the project does not claim fully independent confirmation for the whole group.
- S01 accesses all six official splits only for predeclared integrity and data-quality statistics. It does not run a model or generate predictions.
- Candidate confusions are predeclared from intent-label semantics and checked for presence in train/val only. Test examples and predictions are not inputs to that list.
- Official rows, splits, and text remain unchanged. Any future duplicate-removal analysis is a separate sensitivity analysis, not a replacement for official results.
- Final results are to be reported as a one-time evaluation under a frozen protocol. Any later test-informed design change must be logged here with affected artifacts and rerun reasons.
