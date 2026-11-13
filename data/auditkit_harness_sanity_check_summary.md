# AuditKIT Harness Sanity Check (scripted policies, no LLM)

> Harness sanity check only. No language model is evaluated: the three policies are hand-written replays of the teacher trajectory (with injected faults, with one redundant step, and unmodified). Results verify the metrics, not any trained model.

- **AuditKIT version:** `standalone`
- **Evaluation mode:** `harness_sanity_check_scripted_policies`
- **Dataset:** `agenttune_sft_val.jsonl` (25 held-out episodes, 5 workflow archetypes)
- **`episodes_from_agenttune` bridge verified:** `False`

| Metric | Scripted Fault-Injected | Scripted Redundant-Step | Teacher Oracle Replay |
| :--- | :---: | :---: | :---: |
| Tool-call syntax accuracy | 97.78% | 100.00% | 100.00% |
| Tool-call F1 (FALLBACK: AuditKIT not installed, equals syntax validity) | 97.78% | 100.00% | 100.00% |
| Avg steps (optimal = 6.60) | 7.80 | 6.76 | 6.60 |
| Trajectory efficiency | 85.28% | 98.00% | 100.00% |
| Double-entry invariant rate | 80.00% | 100.00% | 100.00% |
| Statechart transition accuracy | 46.67% | 100.00% | 100.00% |
| Unsafe commit violation rate (lower is better) | 20.00% | 0.00% | 0.00% |
| Terminal audit seal rate | 20.00% | 100.00% | 100.00% |
| Mean deterministic reward [-0.5, 1.0] | +0.4244 | +0.9952 | +1.0000 |

## Config fingerprints
SHA-256 hashes of each run's evaluation configuration. They identify the run setup; they do not certify results.

- `Scripted-FaultInjected-Policy`: `SHA256:253f88335eaaf455`
- `Scripted-RedundantStep-Policy`: `SHA256:b95f8ea9f97e8f26`
- `Teacher-Oracle-Replay`: `SHA256:63101a68916b492e`
