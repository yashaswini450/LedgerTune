"""Phase 2 Validation Gate: Verifies Canonical & Synthesized Teacher Trajectories.

Checks:
1. Formatting Check: Every step in every trajectory has non-empty [THOUGHT], [TOOL_CALL],
   [ARGUMENTS], and [OBSERVATION] blocks, and `formatted_trace` parses cleanly.
2. Double-Entry Accounting Invariant Check: Sum of Debits == Sum of Credits in every
   journal entry, and Fundamental Accounting Equation (Assets == Liabilities + Equity + Revenue - Expense)
   is balanced at completion.
3. Safety Halt Check: Failed compliance (`failed_kyc_rollback`) and overdraft (`overdraft_halt`)
   trajectories strictly follow `PENDING -> FAILED -> ROLLED_BACK` and terminate in `audit_transaction`
   with zero retry loops or unauthorized commits.
"""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src" / "phase1_architecture"))
sys.path.insert(0, str(ROOT_DIR / "src" / "phase2_trajectories"))

from ledger_env import to_decimal  # noqa: E402
from generate_trajectories import export_all_trajectories  # noqa: E402


def verify_trajectory_collection(trajectories: List[Dict[str, Any]], label: str) -> None:
    print(f"\n=== Validating {label} ({len(trajectories)} trajectories) ===")
    required_tags = ["[THOUGHT]", "[TOOL_CALL]", "[ARGUMENTS]", "[OBSERVATION]"]

    happy_count = 0
    halt_count = 0

    for traj in trajectories:
        t_id = traj["trajectory_id"]
        w_type = traj["workflow_type"]
        trace_str = traj["formatted_trace"]

        # 1. Check tag presence in formatted trace and structured steps
        for tag in required_tags:
            assert tag in trace_str, f"{t_id}: Missing {tag} in formatted_trace"

        steps = traj["steps"]
        assert len(steps) >= 6, f"{t_id}: Expected multi-hop workflow (>=6 steps), got {len(steps)}"

        for s in steps:
            assert s.get("THOUGHT") and isinstance(s["THOUGHT"], str), f"{t_id}: Empty THOUGHT"
            assert s.get("TOOL_CALL") and isinstance(s["TOOL_CALL"], str), f"{t_id}: Empty TOOL_CALL"
            assert isinstance(s.get("ARGUMENTS"), dict), f"{t_id}: ARGUMENTS must be dict"
            assert isinstance(s.get("OBSERVATION"), dict), f"{t_id}: OBSERVATION must be dict"

            # 2. Check Debits == Credits on every post_journal_entry call
            if s["TOOL_CALL"] == "post_journal_entry":
                entries = s["ARGUMENTS"]["entries"]
                assert len(entries) >= 2, f"{t_id}: Single-entry violation in post_journal_entry"
                debits = sum(
                    to_decimal(e["amount"]) for e in entries if e["entry_type"] == "DEBIT"
                )
                credits = sum(
                    to_decimal(e["amount"]) for e in entries if e["entry_type"] == "CREDIT"
                )
                assert debits == credits and debits > Decimal("0.00"), (
                    f"{t_id}: Unbalanced journal entry: Debits={debits} != Credits={credits}"
                )

        # Verify fundamental accounting equation holds at end of trajectory
        eq = traj["fundamental_equation_after"]
        assert eq["balanced"] is True, f"{t_id}: Fundamental equation unbalanced: {eq}"

        # 3. Check Statechart progression and no infinite loops on compliance/overdraft halts
        last_step = steps[-1]
        assert last_step["TOOL_CALL"] == "audit_transaction", (
            f"{t_id}: Final step must be audit_transaction, got {last_step['TOOL_CALL']}"
        )
        audit_obs = last_step["OBSERVATION"]
        assert audit_obs["status"] == "AUDITED", f"{t_id}: Audit did not succeed: {audit_obs}"

        if w_type in {"failed_kyc_rollback", "overdraft_halt"}:
            halt_count += 1
            assert audit_obs["terminal_state"] == "ROLLED_BACK", (
                f"{t_id}: Expected ROLLED_BACK for {w_type}, got {audit_obs['terminal_state']}"
            )
            assert audit_obs["state_history"] == ["PENDING", "FAILED", "ROLLED_BACK"], (
                f"{t_id}: Unexpected rollback state history: {audit_obs['state_history']}"
            )
            # Ensure no repeated FAILED or VALIDATED retry loop occurred
            target_states_attempted = [
                s["ARGUMENTS"]["target_state"]
                for s in steps
                if s["TOOL_CALL"] == "post_journal_entry"
            ]
            assert target_states_attempted.count("VALIDATED") <= 1, (
                f"{t_id}: Agent looped on VALIDATED after failure: {target_states_attempted}"
            )
            assert "COMMITTED" not in target_states_attempted, (
                f"{t_id}: Agent attempted COMMITTED during halt workflow!"
            )
        else:
            happy_count += 1
            assert audit_obs["terminal_state"] == "COMMITTED", (
                f"{t_id}: Expected COMMITTED for {w_type}, got {audit_obs['terminal_state']}"
            )
            assert audit_obs["state_history"] == ["PENDING", "VALIDATED", "COMMITTED"], (
                f"{t_id}: Unexpected happy state history: {audit_obs['state_history']}"
            )

    print(f"  [PASS] All {len(trajectories)} trajectories formatted with [THOUGHT], [TOOL_CALL], [ARGUMENTS], [OBSERVATION]")
    print(f"  [PASS] Debits == Credits & Assets == Liabilities + Equity verified across all {len(trajectories)} trajectories")
    print(f"  [PASS] Happy-path commits ({happy_count}) & Halted rollbacks ({halt_count}) verified with 0 infinite loops")


if __name__ == "__main__":
    canonical_5, synthetic_150 = export_all_trajectories()
    verify_trajectory_collection(canonical_5, "5 Canonical Multi-Hop Workflows")
    verify_trajectory_collection(synthetic_150, "150 Parameterized Synthetic Trajectories")
    print("\nALL PHASE 2 VALIDATION CHECKS PASSED PERFECTLY!")

