# Local GPU approval record

- Recorded at: 2026-10-24T07:07:12Z
- Decision: The local RTX 4090 may be used for the formal domain-auxiliary experiments.
- Basis: In the project conversation, the user explicitly selected "已获本地 GPU 批准" when asked whether the supervisor had approved the local-GPU exception.
- Hardware: NVIDIA GeForce RTX 4090, 49,140 MiB reported by `nvidia-smi`.
- Runtime: Existing Windows Conda environment `H:\miniconda3_package\envs\trans`, CUDA-enabled PyTorch 2.6.0+cu124.
- Scope: Local training and validation under the frozen experiment protocol. Supervisor confirmation for E5/E6 remains a separate gate and is not implied by this record.

This file records the user's report; it does not invent a supervisor name, timestamp, or external approval document.
