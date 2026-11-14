# AuditKIT Live Rollout Report (real model checkpoints)

> Live rollouts of real model checkpoints against DoubleEntryLedgerEnv.

- **AuditKIT version:** `1.1.11`
- **Evaluation mode:** `live_model_rollout`
- **Dataset:** `agenttune_sft_val.jsonl` (25 held-out episodes, 5 workflow archetypes)
- **`episodes_from_agenttune` bridge verified:** `True`

| Metric | Baseline (`Zero-Shot`) | AgentTune Stage-1 (`SFT`) | AgentTune Stage-2 (`SFT + GRPO`) |
| :--- | :---: | :---: | :---: |
| Tool-call syntax accuracy | 56.97% | 94.00% | 97.00% |
| AuditKIT tool-call F1 | 10.98% | 24.99% | 26.22% |
| Avg steps (optimal = 6.60) | 5.88 | 1.20 | 1.16 |
| Trajectory efficiency | 84.13% | 100.00% | 100.00% |
| Double-entry invariant rate | 36.00% | 4.00% | 4.00% |
| Statechart transition accuracy | 0.00% | 0.00% | 0.00% |
| Unsafe commit violation rate (lower is better) | 0.00% | 0.00% | 0.00% |
| Terminal audit seal rate | 8.00% | 0.00% | 0.00% |
| Mean deterministic reward [-0.5, 1.0] | +0.1008 | +0.2480 | +0.2605 |

## Config fingerprints
SHA-256 hashes of each run's evaluation configuration. They identify the run setup; they do not certify results.

- `Baseline-ZeroShot-tiny-aya-global`: `SHA256:84c54f2add52de12`
- `AgentTune-Stage1-SFT-tiny-aya-global`: `SHA256:b0a52b277ca69f3a`
- `AgentTune-Stage2-GRPO-tiny-aya-global`: `SHA256:f7c9293ddc0fc8e1`
