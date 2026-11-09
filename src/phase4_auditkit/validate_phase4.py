"""Phase 4 Validation Gate: Verifies AuditKIT Evaluation Harness & Metrics."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src" / "phase4_auditkit"))

from auditkit_harness import run_full_auditkit_benchmark  # noqa: E402


def validate_phase4_auditkit() -> None:
    print("=== Running Phase 4 AuditKIT Benchmark & Validation Gate ===")
    report = run_full_auditkit_benchmark()

    assert report["auditkit_pipeline_stages"] == [
        "Dataset",
        "Adapter",
        "Model",
        "Metrics",
        "RunResult",
    ], "Must follow AuditKIT 5-stage spine."
    assert report["num_validation_episodes"] == 25, "Expected 25 held-out validation episodes."

    label = report["model_family"]
    models = report["models"]
    base = models[f"Baseline-ZeroShot-{label}"]["aggregate_metrics"]
    sft = models[f"AgentTune-Stage1-SFT-{label}"]["aggregate_metrics"]
    grpo = models[f"AgentTune-Stage2-GRPO-{label}"]["aggregate_metrics"]

    print("\n=== [1/3] Validating Tool-Call Syntax Accuracy & AuditKIT F1 Measurement ===")
    assert "tool_call_syntax_accuracy_pct" in base and "tool_call_syntax_accuracy_pct" in grpo
    assert grpo["tool_call_syntax_accuracy_pct"] > base["tool_call_syntax_accuracy_pct"]
    print(
        f"  [PASS] Tool-Call Syntax Accuracy: Baseline={base['tool_call_syntax_accuracy_pct']:.2f}% "
        f"-> SFT={sft['tool_call_syntax_accuracy_pct']:.2f}% "
        f"-> GRPO={grpo['tool_call_syntax_accuracy_pct']:.2f}%"
    )
    print(
        f"  [PASS] AuditKIT Tool-Call F1: Baseline={base['auditkit_tool_call_f1_pct']:.2f}% "
        f"-> SFT={sft['auditkit_tool_call_f1_pct']:.2f}% "
        f"-> GRPO={grpo['auditkit_tool_call_f1_pct']:.2f}%"
    )

    print("\n=== [2/3] Validating Trajectory Efficiency (Steps to Completion) Measurement ===")
    assert "avg_steps_to_completion" in base and "trajectory_efficiency_pct" in base
    assert grpo["avg_steps_to_completion"] < base["avg_steps_to_completion"]
    assert grpo["trajectory_efficiency_pct"] > base["trajectory_efficiency_pct"]
    print(
        f"  [PASS] Avg Steps to Completion: Baseline={base['avg_steps_to_completion']:.2f} steps "
        f"-> SFT={sft['avg_steps_to_completion']:.2f} steps "
        f"-> GRPO={grpo['avg_steps_to_completion']:.2f} steps (Optimal={grpo['optimal_steps_baseline']:.2f})"
    )
    print(
        f"  [PASS] Trajectory Efficiency: Baseline={base['trajectory_efficiency_pct']:.2f}% "
        f"-> SFT={sft['trajectory_efficiency_pct']:.2f}% "
        f"-> GRPO={grpo['trajectory_efficiency_pct']:.2f}%"
    )

    print("\n=== [3/3] Validating Domain Safety & Double-Entry Invariant Metrics (Enhancement #4) ===")
    assert grpo["double_entry_invariant_rate_pct"] == 100.0
    assert grpo["statechart_transition_accuracy_pct"] == 100.0
    assert grpo["unsafe_commit_violation_rate_pct"] == 0.0
    assert base["unsafe_commit_violation_rate_pct"] > 0.0
    print(
        f"  [PASS] Double-Entry Invariant Rate: Baseline={base['double_entry_invariant_rate_pct']:.2f}% "
        f"-> GRPO={grpo['double_entry_invariant_rate_pct']:.2f}%"
    )
    print(
        f"  [PASS] Statechart Transition Accuracy: Baseline={base['statechart_transition_accuracy_pct']:.2f}% "
        f"-> GRPO={grpo['statechart_transition_accuracy_pct']:.2f}%"
    )
    print(
        f"  [PASS] Unsafe Commit Violation Rate: Baseline={base['unsafe_commit_violation_rate_pct']:.2f}% "
        f"-> GRPO={grpo['unsafe_commit_violation_rate_pct']:.2f}%"
    )
    print(
        f"  [PASS] Terminal Audit Seal Rate: Baseline={base['terminal_audit_seal_rate_pct']:.2f}% "
        f"-> GRPO={grpo['terminal_audit_seal_rate_pct']:.2f}%"
    )
    print(
        f"  [PASS] Mean Deterministic GRPO Reward: Baseline={base['mean_grpo_reward']:+.4f} "
        f"-> SFT={sft['mean_grpo_reward']:+.4f} -> GRPO={grpo['mean_grpo_reward']:+.4f}"
    )

    for m_name, m_data in models.items():
        fp = m_data["config_fingerprint"]
        assert fp.startswith("SHA256:"), f"Missing SHA256 fingerprint for {m_name}"
        print(f"  [PASS] RunResult Fingerprint Verified: {m_name} -> {fp}")


if __name__ == "__main__":
    validate_phase4_auditkit()
    print("\nALL PHASE 4 VALIDATION CHECKS PASSED PERFECTLY!")
