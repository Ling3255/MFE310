# GPU determinism record

Formal local runs use the existing `trans` environment on an RTX 4090. `training.deterministic` is enabled, but PyTorch emitted two CUDA warnings during CE seed 0:

- Deterministic CuBLAS behavior requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` or `:16:8`, which was not set.
- Memory-efficient attention backward can use a non-deterministic algorithm.

To keep the execution environment consistent, the remaining runs in this matrix use the same environment rather than changing settings mid-matrix. Exact bitwise GPU reproducibility is therefore not claimed. Seeds, paired initializations, sample-order hashes, package versions, and checkpoints are still recorded. Any future determinism change requires a new configuration and rerun record, not silent replacement of these runs.
