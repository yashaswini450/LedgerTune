# Phase 4: AuditKIT Evaluation Report

- **AuditKIT Engine:** `auditkit vstandalone` (`AgentEvalRunner` + Double-Entry Invariant Suite)
- **Evaluation Mode:** `controlled_policy_ablation`
- **Dataset Split:** `agenttune_sft_val.jsonl` (25 held-out multi-hop validation episodes across all 5 workflow archetypes)
- **`episodes_from_agenttune` Bridge Verified:** `False`

## Aggregate Benchmark Comparison

| Metric | Baseline (`Zero-Shot`) | AgentTune Stage-1 (`SFT`) | AgentTune Stage-2 (`SFT + GRPO`) | Delta (`GRPO` vs. `Baseline`) |
| :--- | :---: | :---: | :---: | :---: |
| **Tool-Call Syntax Accuracy (%)** | 97.78% | 100.00% | **100.00%** | **+2.22%** |
| **AuditKIT Tool-Call F1 (%)** | 97.78% | 100.00% | **100.00%** | **+2.22%** |
| **Avg Steps to Completion** *(Optimal = 6.60)* | 7.80 steps | 6.76 steps | **6.60 steps** | **-1.20 steps** |
| **Trajectory Efficiency (%)** | 85.28% | 98.00% | **100.00%** | **+14.72%** |
| **Double-Entry Invariant Rate (%)** | 80.00% | 100.00% | **100.00%** | **+20.00%** |
| **Statechart Transition Accuracy (%)** | 46.67% | 100.00% | **100.00%** | **+53.33%** |
| **Unsafe Commit Violation Rate (%)** *(Lower is safer)* | 20.00% | 0.00% | **0.00%** | **-20.00%** |
| **Terminal Audit Seal Rate (%)** | 20.00% | 100.00% | **100.00%** | **+80.00%** |
| **Mean Deterministic GRPO Reward** $\in [-0.50, +1.00]$ | +0.4244 | +0.9952 | **+1.0000** | **+0.5756** |

## Reproducibility Fingerprints (`AuditKitRunResult`)
- `Baseline-ZeroShot-Qwen2.5-1.5B-Instruct`: `SHA256:856e711be9d39819`
- `AgentTune-Stage1-SFT-Qwen2.5-1.5B-Instruct`: `SHA256:c785a16ffa6a445f`
- `AgentTune-Stage2-GRPO-Qwen2.5-1.5B-Instruct`: `SHA256:474c6650c17889bd`
