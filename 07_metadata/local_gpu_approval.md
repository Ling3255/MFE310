# Local GPU approval record

- Initially recorded at: 2026-10-24T07:07:12Z.
- Decision: The local RTX 4090 may be used for the formal domain-auxiliary experiments.
- Initial basis: In the project conversation, the user explicitly selected "已获本地 GPU 批准" when asked whether the supervisor had approved the local-GPU exception. This was a user report, not an external approval document.
- Subsequent actual communication: [supervisor PR #2](https://github.com/Ling3255/MFE310/pull/2), created 2026-10-24T08:28:19Z, instructs the student to rerun S01/S02/S05 in the existing `trans` environment and explicitly refers to using the GPU machine's `best.pt` files. The same PR states that all 15 completed runs and lambda=0.1 remain valid and do not need retraining.
- Hardware: NVIDIA GeForce RTX 4090, 49,140 MiB reported by `nvidia-smi`.
- Runtime: Existing Windows Conda environment `H:\miniconda3_package\envs\trans`, CUDA-enabled PyTorch 2.6.0+cu124.
- Scope: Local training and validation under the experiment protocol. Supervisor confirmation of each revised E5 freeze remains a separate gate and is not implied by this record.

This file distinguishes the user's initial report from the later verifiable GitHub communication. It does not invent a supervisor name, external timestamp, or approval document.
