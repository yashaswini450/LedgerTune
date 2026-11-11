# Phase 4: AuditKIT Evaluation Report

- **AuditKIT Engine:** `auditkit v1.1.7` (`AgentEvalRunner` + Double-Entry Invariant Suite)
- **Evaluation Mode:** `live_model_rollout`
- **Dataset Split:** `agenttune_sft_val.jsonl` (25 held-out multi-hop validation episodes across all 5 workflow archetypes)
- **`episodes_from_agenttune` Bridge Verified:** `True`

## Aggregate Benchmark Comparison

| Metric | Baseline (`Zero-Shot`) | AgentTune Stage-1 (`SFT`) | AgentTune Stage-2 (`SFT + GRPO`) | Delta (`GRPO` vs. `Baseline`) |
| :--- | :---: | :---: | :---: | :---: |
| **Tool-Call Syntax Accuracy (%)** | 100.00% | 63.88% | **100.00%** | **+0.00%** |
| **AuditKIT Tool-Call F1 (%)** | 5.71% | 13.51% | **5.71%** | **+0.00%** |
| **Avg Steps to Completion** *(Optimal = 6.60)* | 1.36 steps | 14.08 steps | **1.36 steps** | **-0.00 steps** |
| **Trajectory Efficiency (%)** | 100.00% | 80.69% | **100.00%** | **+0.00%** |
| **Double-Entry Invariant Rate (%)** | 40.00% | 24.00% | **40.00%** | **+0.00%** |
| **Statechart Transition Accuracy (%)** | 0.00% | 0.00% | **0.00%** | **+0.00%** |
| **Unsafe Commit Violation Rate (%)** *(Lower is safer)* | 0.00% | 0.00% | **0.00%** | **-0.00%** |
| **Terminal Audit Seal Rate (%)** | 0.00% | 0.00% | **0.00%** | **+0.00%** |
| **Mean Deterministic GRPO Reward** $\in [-0.50, +1.00]$ | +0.4900 | +0.1897 | **+0.4900** | **+0.0000** |

## Reproducibility Fingerprints (`AuditKitRunResult`)
- `Baseline-ZeroShot-Qwen2.5-1.5B-Instruct`: `SHA256:856e711be9d39819`
- `AgentTune-Stage1-SFT-Qwen2.5-1.5B-Instruct`: `SHA256:c785a16ffa6a445f`
- `AgentTune-Stage2-GRPO-Qwen2.5-1.5B-Instruct`: `SHA256:474c6650c17889bd`
