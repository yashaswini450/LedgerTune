"""Phase 4 Validation Gate: AuditKIT harness sanity check.

Runs the harness in its default offline mode, where NO language model is loaded. Three
hand-written policies are scored on the 25 held-out episodes:

  * Teacher-Oracle-Replay          - the unmodified teacher trajectory (upper bound)
  * Scripted-RedundantStep-Policy  - teacher trajectory plus a redundant/retried call
  * Scripted-FaultInjected-Policy  - teacher trajectory with injected faults (illegal
                                     shortcuts, commit-after-rejection, unbalanced entries,
                                     overdraft retry loops)

The gate checks that each metric detects the fault it is designed to detect. It does not
evaluate any trained model; real model results live in data/auditkit_live_gpu_rollout_report.json.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src" / "phase4_auditkit"))

from auditkit_harness import run_full_auditkit_benchmark  # noqa: E402


def validate_phase4_auditkit() -> None:
    print("=== Running Phase 4 AuditKIT Harness Sanity Check (scripted policies, no LLM) ===")
    report = run_full_auditkit_benchmark()

    assert report["evaluation_mode"] == "harness_sanity_check_scripted_policies"
    assert report["auditkit_pipeline_stages"] == ["Dataset", "Adapter", "Model", "Metrics", "RunResult"]
    assert report["num_validation_episodes"] == 25, "Expected 25 held-out validation episodes."

    labels = report["policy_labels"]
    models = report["models"]
    fault = models[labels["baseline"]]["aggregate_metrics"]
    redundant = models[labels["sft"]]["aggregate_metrics"]
    oracle = models[labels["grpo"]]["aggregate_metrics"]

    print("\n=== [1/3] Oracle replay scores perfectly (metrics have no false positives) ===")
    assert oracle["double_entry_invariant_rate_pct"] == 100.0
    assert oracle["statechart_transition_accuracy_pct"] == 100.0
    assert oracle["unsafe_commit_violation_rate_pct"] == 0.0
    assert oracle["terminal_audit_seal_rate_pct"] == 100.0
    assert oracle["mean_grpo_reward"] == 1.0
    print("  [PASS] Teacher-Oracle-Replay: invariant=100%, statechart=100%, unsafe=0%, seal=100%, reward=+1.0")

    print("\n=== [2/3] Redundant-step policy is penalised on efficiency only ===")
    assert redundant["avg_steps_to_completion"] > oracle["avg_steps_to_completion"]
    assert redundant["trajectory_efficiency_pct"] < oracle["trajectory_efficiency_pct"]
    assert redundant["mean_grpo_reward"] < oracle["mean_grpo_reward"]
    print(
        f"  [PASS] Steps {oracle['avg_steps_to_completion']:.2f} -> {redundant['avg_steps_to_completion']:.2f}, "
        f"efficiency {oracle['trajectory_efficiency_pct']:.2f}% -> {redundant['trajectory_efficiency_pct']:.2f}%"
    )

    print("\n=== [3/3] Fault-injected policy is caught by every safety metric ===")
    assert fault["unsafe_commit_violation_rate_pct"] > 0.0
    assert fault["double_entry_invariant_rate_pct"] < 100.0
    assert fault["statechart_transition_accuracy_pct"] < 100.0
    assert fault["terminal_audit_seal_rate_pct"] < 100.0
    assert fault["mean_grpo_reward"] < redundant["mean_grpo_reward"]
    print(
        f"  [PASS] unsafe commits={fault['unsafe_commit_violation_rate_pct']:.2f}%, "
        f"invariant={fault['double_entry_invariant_rate_pct']:.2f}%, "
        f"statechart={fault['statechart_transition_accuracy_pct']:.2f}%, "
        f"seal={fault['terminal_audit_seal_rate_pct']:.2f}%, reward={fault['mean_grpo_reward']:+.4f}"
    )

    for m_name, m_data in models.items():
        assert m_data["config_fingerprint"].startswith("SHA256:"), f"Missing config fingerprint for {m_name}"


if __name__ == "__main__":
    validate_phase4_auditkit()
    print("\nALL PHASE 4 VALIDATION CHECKS PASSED!")
