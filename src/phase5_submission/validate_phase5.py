"""Phase 5 Validation Gate: Verifies README 'Show, Don't Tell' AuditKIT Metrics & First-Principles Docs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def validate_phase5_submission() -> None:
    print("=== Running Phase 5 Submission Packaging Validation Gate ===")
    readme_path = ROOT_DIR / "README.md"
    report_path = ROOT_DIR / "data" / "auditkit_evaluation_report.json"

    assert readme_path.exists(), "README.md is missing!"
    assert report_path.exists(), "auditkit_evaluation_report.json is missing!"

    readme_text = readme_path.read_text(encoding="utf-8")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    label = report["model_family"]

    # 1. Verify "Show, Don't Tell" empirical AuditKIT metrics match the JSON report
    print("\n=== [1/3] Validating 'Show, Don't Tell' Empirical AuditKIT Metrics in README.md ===")
    base_metrics = report["models"][f"Baseline-ZeroShot-{label}"]["aggregate_metrics"]
    grpo_metrics = report["models"][f"AgentTune-Stage2-GRPO-{label}"]["aggregate_metrics"]
    base_fp = report["models"][f"Baseline-ZeroShot-{label}"]["config_fingerprint"]
    grpo_fp = report["models"][f"AgentTune-Stage2-GRPO-{label}"]["config_fingerprint"]

    required_empirical_snippets = [
        f"{base_metrics['tool_call_syntax_accuracy_pct']:.2f}%",
        f"{grpo_metrics['tool_call_syntax_accuracy_pct']:.2f}%",
        f"{base_metrics['auditkit_tool_call_f1_pct']:.2f}%",
        f"{grpo_metrics['auditkit_tool_call_f1_pct']:.2f}%",
        f"{base_metrics['avg_steps_to_completion']:.2f} steps",
        f"{grpo_metrics['avg_steps_to_completion']:.2f} steps",
        f"{base_metrics['trajectory_efficiency_pct']:.2f}%",
        f"{grpo_metrics['trajectory_efficiency_pct']:.2f}%",
        f"{base_metrics['terminal_audit_seal_rate_pct']:.2f}%",
        base_fp,
        grpo_fp,
    ]
    for snippet in required_empirical_snippets:
        assert snippet in readme_text, f"README.md missing empirical AuditKIT metric/fingerprint: {snippet}"
    print("  [PASS] All empirical AuditKIT metrics & SHA-256 RunResult fingerprints verified in README.md")

    # 2. Verify First-Principles Software Engineering & Statechart Explanations
    print("\n=== [2/3] Validating First-Principles Accounting & Statechart Documentation ===")
    required_concepts = [
        "Total Assets",
        "Total Liabilities",
        "stateDiagram-v2",
        "PENDING --> VALIDATED",
        "PENDING --> FAILED",
        "FAILED --> ROLLED_BACK",
        "VALIDATED --> COMMITTED",
        "Architectural Trade-Offs",
        "Normal Balance Polarity Matrix",
    ]
    for concept in required_concepts:
        assert concept in readme_text, f"README.md missing architectural concept: {concept}"
    print("  [PASS] Double-entry conservation laws, normal balance polarity, and Mermaid statecharts verified")

    # 3. Verify Complete Deliverable Manifest
    print("\n=== [3/3] Validating Complete Repository File Manifest ===")
    expected_files = [
        "README.md",
        "LL-2.ipynb",
        "requirements.txt",
        "run_all_validations.py",
        "src/phase1_architecture/tool_schemas.json",
        "src/phase1_architecture/ledger_env.py",
        "src/phase1_architecture/validate_phase1.py",
        "src/phase2_trajectories/generate_trajectories.py",
        "src/phase2_trajectories/validate_phase2.py",
        "src/phase3_agenttune/prepare_agenttune_sft.py",
        "src/phase3_agenttune/reward_functions.py",
        "src/phase3_agenttune/train_real_agenttune.py",
        "src/phase3_agenttune/agenttune_config.yaml",
        "src/phase3_agenttune/validate_phase3.py",
        "src/phase4_auditkit/live_rollout.py",
        "src/phase4_auditkit/auditkit_harness.py",
        "src/phase4_auditkit/validate_phase4.py",
        "src/phase5_submission/validate_phase5.py",
        "data/phase2_canonical_5_trajectories.json",
        "data/phase2_canonical_5_readable.md",
        "data/phase2_synthetic_dataset_150.jsonl",
        "data/agenttune_canonical_5.jsonl",
        "data/agenttune_sft_train.jsonl",
        "data/agenttune_sft_val.jsonl",
        "data/agenttune_native_val_trajectories.jsonl",
        "data/agenttune_eventlog_sft_train.jsonl",
        "data/auditkit_evaluation_report.json",
        "data/auditkit_evaluation_summary.md",
        "data/auditkit_live_gpu_rollout_report.json",
        "data/auditkit_live_gpu_rollout_summary.md",
        "checkpoints/sft_ledger/checkpoint-51/trainer_state.json",
        "checkpoints/grpo_ledger/lexsi_provenance.json",
        "checkpoints/grpo_ledger/training_stats.json",
        "checkpoints/grpo_ledger/trajectories.jsonl",
    ]
    for rel_path in expected_files:
        p = ROOT_DIR / rel_path
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty deliverable: {rel_path}"
    print(f"  [PASS] All {len(expected_files)} project files and datasets verified and non-empty")


if __name__ == "__main__":
    validate_phase5_submission()
    print("\nALL PHASE 5 VALIDATION CHECKS PASSED PERFECTLY!")
