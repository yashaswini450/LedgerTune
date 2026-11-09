"""Phase 3 Validation Gate: Verifies AgentTune SFT JSONL Alignment & GRPO Reward Function."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src" / "phase1_architecture"))
sys.path.insert(0, str(ROOT_DIR / "src" / "phase3_agenttune"))

from prepare_agenttune_sft import build_agenttune_datasets  # noqa: E402
from reward_functions import compute_double_entry_grpo_reward  # noqa: E402


def verify_agenttune_jsonl_file(jsonl_path: Path, label: str) -> Dict[str, int]:
    print(f"\n=== Validating {label}: {jsonl_path.name} ===")
    assert jsonl_path.exists(), f"Missing JSONL file: {jsonl_path}"

    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            assert line.strip(), f"Empty line at {line_num}"
            rec = json.loads(line)
            records.append(rec)

    happy_executions = 0
    graceful_error_recoveries = 0

    for idx, rec in enumerate(records, start=1):
        # 1. Verify top-level AgentTune JSONL keys
        for req_key in ("id", "tools", "messages", "metadata"):
            assert req_key in rec, f"Record {idx} missing top-level key '{req_key}'"

        tools = rec["tools"]
        assert isinstance(tools, list) and len(tools) == 5, f"Record {idx} must include 5 tools"
        schema_names = {t["function"]["name"] for t in tools}

        messages = rec["messages"]
        assert len(messages) >= 5, f"Record {idx} must have multi-turn messages"
        assert messages[0]["role"] == "system" and messages[0]["content"], "Msg 0 must be system"
        assert messages[1]["role"] == "user" and messages[1]["content"], "Msg 1 must be user"
        assert messages[-1]["role"] == "assistant" and messages[-1].get("content"), (
            "Final message must be assistant summary"
        )
        assert not messages[-1].get("tool_calls"), "Final message must not contain tool_calls"

        # Verify strict alternating pairs of (assistant[tool_calls], tool[tool_call_id])
        intermediate = messages[2:-1]
        assert len(intermediate) % 2 == 0, "Intermediate turns must be (assistant, tool) pairs"

        saw_error_or_rejection = False
        for i in range(0, len(intermediate), 2):
            asst_msg = intermediate[i]
            tool_msg = intermediate[i + 1]

            assert asst_msg["role"] == "assistant", f"Expected assistant role at pair {i}"
            tcs = asst_msg.get("tool_calls")
            assert isinstance(tcs, list) and len(tcs) == 1, "Expected 1 tool_call per step"
            tc = tcs[0]
            assert tc["type"] == "function"
            call_id = tc["id"]
            fn_name = tc["function"]["name"]
            assert fn_name in schema_names, f"Unknown tool '{fn_name}'"

            # Arguments must be a valid JSON string
            args_str = tc["function"]["arguments"]
            assert isinstance(args_str, str), "tool_calls.function.arguments must be JSON string"
            parsed_args = json.loads(args_str)
            assert isinstance(parsed_args, dict)

            # Verify matching tool observation message
            assert tool_msg["role"] == "tool", f"Expected tool role following assistant at {i+1}"
            assert tool_msg["tool_call_id"] == call_id, "Mismatched tool_call_id"
            assert tool_msg["name"] == fn_name, "Mismatched tool name"
            obs_parsed = json.loads(tool_msg["content"])
            if obs_parsed.get("status") in {"ERROR", "REJECTED"}:
                saw_error_or_rejection = True

        # Verify reward is +1.0 for every gold-standard trajectory
        reward_res = compute_double_entry_grpo_reward(messages, tools, rec["metadata"])
        assert reward_res["total_reward"] == 1.0, (
            f"Record {rec['id']} did not achieve 1.0 reward: {reward_res}"
        )

        if saw_error_or_rejection:
            graceful_error_recoveries += 1
        else:
            happy_executions += 1

    print(f"  [PASS] Structural JSONL alignment verified across {len(records)} episodes")
    print(f"  [PASS] Successful happy-path tool executions: {happy_executions} episodes")
    print(f"  [PASS] Graceful error-handling / rollback executions: {graceful_error_recoveries} episodes")
    assert happy_executions > 0 and graceful_error_recoveries > 0, (
        "Dataset must contain both successful executions and graceful error handling!"
    )
    return {"total": len(records), "happy": happy_executions, "error_handling": graceful_error_recoveries}


def verify_grpo_reward_discrimination(canonical_path: Path) -> None:
    print("\n=== Validating Enhancement #3: GRPO Deterministic Reward Discrimination ===")
    with open(canonical_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # 1. Gold-standard trajectory score == +1.00
    gold_rec = records[0]
    r_gold = compute_double_entry_grpo_reward(
        gold_rec["messages"], gold_rec["tools"], gold_rec["metadata"]
    )
    print(f"  [PASS] Gold Teacher Trajectory Reward: {r_gold['total_reward']:+.2f} | {r_gold['diagnostics']}")
    assert r_gold["total_reward"] == 1.0

    # 2. Corrupted Trajectory A: Unbalanced Journal Entry (Debits != Credits)
    bad_balance = copy.deepcopy(gold_rec)
    for msg in bad_balance["messages"]:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            if tc["function"]["name"] == "post_journal_entry":
                args = json.loads(tc["function"]["arguments"])
                args["entries"][0]["amount"] = 9999.99  # Break Debits == Credits
                tc["function"]["arguments"] = json.dumps(args)
                break
    r_unbalanced = compute_double_entry_grpo_reward(
        bad_balance["messages"], bad_balance["tools"], bad_balance["metadata"]
    )
    print(
        f"  [PASS] Unbalanced Entry Penalty Verified: Reward={r_unbalanced['total_reward']:+.2f} "
        f"(r_accounting={r_unbalanced['r_accounting']:.2f}) | {r_unbalanced['diagnostics']}"
    )
    assert r_unbalanced["r_accounting"] < 0.30

    # 3. Corrupted Trajectory B: Unsafe Commit After KYC/AML Rejection
    kyc_rec = copy.deepcopy(records[1])  # TRAJ_CANONICAL_02 (failed_kyc_rollback)
    for msg in kyc_rec["messages"]:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            if tc["function"]["name"] == "post_journal_entry":
                args = json.loads(tc["function"]["arguments"])
                if args["target_state"] == "FAILED":
                    args["target_state"] = "COMMITTED"  # Unsafe commit after compliance rejection!
                    tc["function"]["arguments"] = json.dumps(args)
                    break
    r_unsafe = compute_double_entry_grpo_reward(
        kyc_rec["messages"], kyc_rec["tools"], kyc_rec["metadata"]
    )
    print(
        f"  [PASS] Unsafe Commit Safety Penalty Verified: Reward={r_unsafe['total_reward']:+.2f} "
        f"(r_statechart={r_unsafe['r_statechart']:+.2f}) | {r_unsafe['diagnostics']}"
    )
    assert r_unsafe["r_statechart"] == -0.50


if __name__ == "__main__":
    canonical_out, train_out, val_out = build_agenttune_datasets()
    verify_agenttune_jsonl_file(canonical_out, "Canonical 5 Reference Set")
    verify_agenttune_jsonl_file(train_out, "AgentTune SFT Train Set (130 Episodes)")
    verify_agenttune_jsonl_file(val_out, "AgentTune SFT Validation Set (25 Episodes)")
    verify_grpo_reward_discrimination(canonical_out)
    print("\nALL PHASE 3 VALIDATION CHECKS PASSED PERFECTLY!")

