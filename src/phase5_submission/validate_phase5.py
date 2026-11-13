"""Phase 5 Validation Gate: README / report consistency and deliverable manifest.

Checks that every number the README reports for the live GPU run matches
data/auditkit_live_gpu_rollout_report.json, that the scripted sanity check is clearly
labelled as such, that limitations are documented, and that all deliverables exist.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def _pct(x: float) -> str:
    return f"{x:.2f}%"


def validate_phase5_submission() -> None:
    print("=== Running Phase 5 Submission Packaging Validation Gate ===")
    readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    live = json.loads((ROOT_DIR / "data" / "auditkit_live_gpu_rollout_report.json").read_text(encoding="utf-8"))
    sanity = json.loads(
        (ROOT_DIR / "data" / "auditkit_harness_sanity_check_report.json").read_text(encoding="utf-8")
    )
    sft_state = json.loads(
        (ROOT_DIR / "checkpoints" / "sft_ledger" / "checkpoint-51" / "trainer_state.json").read_text(encoding="utf-8")
    )

    # 1. Live GPU numbers in README match the live report
    print("\n=== [1/4] Live GPU rollout metrics in README match the report ===")
    assert live["evaluation_mode"] == "live_model_rollout"
    label = live["model_family"]
    base = live["models"][f"Baseline-ZeroShot-{label}"]["aggregate_metrics"]
    sft = live["models"][f"AgentTune-Stage1-SFT-{label}"]["aggregate_metrics"]
    snippets = [
        _pct(base["auditkit_tool_call_f1_pct"]),
        _pct(sft["auditkit_tool_call_f1_pct"]),
        _pct(sft["auditkit_verified_outcome_pct"]),
        _pct(sft["tool_call_syntax_accuracy_pct"]),
        _pct(base["double_entry_invariant_rate_pct"]),
        _pct(sft["double_entry_invariant_rate_pct"]),
        f"{base['avg_steps_to_completion']:.2f}",
        f"{sft['avg_steps_to_completion']:.2f}",
        f"{sft['avg_redundant_steps']:.2f}",
        f"{base['mean_grpo_reward']:+.4f}",
        f"{sft['mean_grpo_reward']:+.4f}",
        live["auditkit_version"],
    ]
    for s in snippets:
        assert s in readme, f"README.md is missing live-report value: {s}"
    print(f"  [PASS] {len(snippets)} live-report values found in README.md")

    # 2. SFT training curve in README matches trainer_state.json
    print("\n=== [2/4] SFT training curve in README matches trainer_state.json ===")
    logs = sft_state["log_history"]
    first, last = logs[0], logs[-1]
    for s in [f"{first['loss']:.4f}", f"{last['loss']:.4f}",
              _pct(100 * first["mean_token_accuracy"]), _pct(100 * last["mean_token_accuracy"])]:
        assert s in readme, f"README.md is missing SFT training value: {s}"
    assert sft_state["global_step"] == 51
    print(f"  [PASS] loss {first['loss']:.4f} -> {last['loss']:.4f}, "
          f"token acc {100 * first['mean_token_accuracy']:.2f}% -> {100 * last['mean_token_accuracy']:.2f}%")

    # 3. Honest labelling: sanity check is scripted; limitations are documented
    print("\n=== [3/4] Sanity check labelled as scripted; limitations documented ===")
    assert sanity["evaluation_mode"] == "harness_sanity_check_scripted_policies"
    assert "Qwen" not in json.dumps(sanity["policy_labels"]), "Scripted policies must not carry model names"
    for concept in [
        "Harness sanity check (scripted policies, no LLM)",
        "Limitations and known issues",
        "GRPO did not train",
        "Total Assets",
        "Total Liabilities",
        "Normal Balance Polarity Matrix",
        "stateDiagram-v2",
        "PENDING --> VALIDATED",
        "PENDING --> FAILED",
        "FAILED --> ROLLED_BACK",
        "VALIDATED --> COMMITTED",
        "Architectural Trade-Offs",
    ]:
        assert concept in readme, f"README.md missing section/concept: {concept}"
    print("  [PASS] Scripted policies carry no model names; limitations, statechart and accounting docs present")

    # 4. Deliverable manifest
    print("\n=== [4/4] Repository file manifest ===")
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
        "data/auditkit_live_gpu_rollout_report.json",
        "data/auditkit_live_gpu_rollout_summary.md",
        "data/auditkit_harness_sanity_check_report.json",
        "data/auditkit_harness_sanity_check_summary.md",
        "checkpoints/sft_ledger/checkpoint-51/trainer_state.json",
        "checkpoints/grpo_ledger/lexsi_provenance.json",
        "checkpoints/grpo_ledger/training_stats.json",
        "checkpoints/grpo_ledger/trajectories.jsonl",
    ]
    for rel in expected_files:
        p = ROOT_DIR / rel
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty deliverable: {rel}"
    print(f"  [PASS] All {len(expected_files)} deliverables present and non-empty")


if __name__ == "__main__":
    validate_phase5_submission()
    print("\nALL PHASE 5 VALIDATION CHECKS PASSED!")
