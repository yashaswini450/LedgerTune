"""Phase 3: AgentTune SFT, EventLog & GRPO Dataset Preparation.

Converts Phase 2 Teacher Trajectories into:
1. Multi-turn OpenAI/TRL tool-calling JSONL (`data/agenttune_sft_train.jsonl`,
   `data/agenttune_sft_val.jsonl`, `data/agenttune_canonical_5.jsonl`) with initial ledger
   snapshots for live environment replay.
2. Native AgentTune `Trajectory` JSONL (`data/agenttune_native_trajectories.jsonl`),
   matching `agenttune.agentic.trajectory.dataset.Trajectory.to_dict()` and directly
   importable by AuditKIT's `episodes_from_agenttune()`.
3. Native AgentTune `EventLog(tier="full").as_dataset_rows(fmt="sft")` dataset
   (`data/agenttune_eventlog_sft_rows.jsonl`) with Segment-Aware Loss weights
   (`1.0` for reasoning, `2.0` for `<tool_call>` syntax/actions, `0.0` for tool/user observations).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src" / "phase1_architecture"))
if Path("/tmp/lexsi_ref/AgentTune/src").exists():
    sys.path.insert(0, "/tmp/lexsi_ref/AgentTune/src")

from ledger_env import load_tool_schemas  # noqa: E402

SYSTEM_PROMPT = (
    "You are an autonomous Financial Systems Business Workflow Agent managing a Digital Wallet "
    "Financial Ledger API under strict double-entry accounting principles.\n"
    "Core Invariants:\n"
    "1. Every journal entry must contain at least two lines (DEBIT and CREDIT) where sum(DEBIT) == sum(CREDIT).\n"
    "2. The fundamental accounting equation must always hold: Total Assets = Total Liabilities + Equity + (Revenue - Expense).\n"
    "3. Follow the strict transaction statechart:\n"
    "   - Happy Path: [*] -> PENDING -> VALIDATED -> COMMITTED -> audit_transaction\n"
    "   - Rollback/Failure Path: [*] -> PENDING -> FAILED -> ROLLED_BACK -> audit_transaction\n"
    "4. Always inspect account balances before staging, run check_compliance while PENDING, "
    "and immediately halt & roll back (PENDING -> FAILED -> ROLLED_BACK) without retrying if "
    "compliance is rejected or an overdraft is detected."
)


def make_tool_call_id(trajectory_id: str, step_index: int, tool_name: str) -> str:
    digest = hashlib.md5(f"{trajectory_id}:{step_index}:{tool_name}".encode()).hexdigest()[:10]
    return f"call_{digest}"


def trajectory_to_agenttune_record(
    traj: Dict[str, Any], tool_schemas: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Converts a single Phase 2 trajectory into an AgentTune SFT/GRPO JSONL record."""
    traj_id = traj["trajectory_id"]
    w_type = traj["workflow_type"]
    steps = traj["steps"]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": traj["user_prompt"]},
    ]

    has_error_observation = False
    for s in steps:
        idx = s["step_index"]
        tool_name = s["TOOL_CALL"]
        args_dict = s["ARGUMENTS"]
        obs_dict = s["OBSERVATION"]
        thought_str = s["THOUGHT"]

        if obs_dict.get("status") in {"ERROR", "REJECTED"}:
            has_error_observation = True

        call_id = make_tool_call_id(traj_id, idx, tool_name)

        messages.append(
            {
                "role": "assistant",
                "content": f"[THOUGHT]\n{thought_str}",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args_dict, separators=(",", ":")),
                        },
                    }
                ],
            }
        )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": json.dumps(obs_dict, separators=(",", ":")),
            }
        )

    messages.append(
        {
            "role": "assistant",
            "content": traj["final_response"],
        }
    )

    terminal_state = (
        "ROLLED_BACK"
        if w_type in {"failed_kyc_rollback", "overdraft_halt"}
        else "COMMITTED"
    )

    return {
        "id": traj_id,
        "tools": tool_schemas,
        "messages": messages,
        "metadata": {
            "workflow_type": w_type,
            "expected_terminal_state": terminal_state,
            "num_tool_steps": len(steps),
            "is_error_recovery_trajectory": has_error_observation,
            "fundamental_equation_balanced": traj["fundamental_equation_after"]["balanced"],
            "initial_accounts": traj.get("initial_accounts", []),
        },
    }


def trajectory_to_native_agenttune_dict(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Builds the exact `Trajectory.to_dict()` schema produced by AgentTune rollouts
    and consumed natively by `auditkit.agent_eval.episodes_from_agenttune()`."""
    traj_id = traj["trajectory_id"]
    w_type = traj["workflow_type"]
    terminal_state = (
        "ROLLED_BACK"
        if w_type in {"failed_kyc_rollback", "overdraft_halt"}
        else "COMMITTED"
    )
    native_steps = []
    for s in traj["steps"]:
        idx = s["step_index"]
        call_id = make_tool_call_id(traj_id, idx, s["TOOL_CALL"])
        native_steps.append(
            {
                "step_number": idx,
                "state": traj["user_prompt"] if idx == 1 else f"step_{idx}",
                "action": {
                    "id": call_id,
                    "name": s["TOOL_CALL"],
                    "arguments": s["ARGUMENTS"],
                },
                "observation": json.dumps(s["OBSERVATION"], separators=(",", ":")),
                "thought": s["THOUGHT"],
                "reward": 1.0 if idx == len(traj["steps"]) else 0.0,
            }
        )

    return {
        "trajectory_id": traj_id,
        "task": traj["user_prompt"],
        "steps": native_steps,
        "final_response": traj["final_response"],
        "reward": 1.0,
        "metadata": {
            "case_id": traj_id,
            "workflow_type": w_type,
            "expected_terminal_state": terminal_state,
            "final_state": {
                "terminal_state": terminal_state,
                "equation_balanced": traj["fundamental_equation_after"]["balanced"],
                "audited": True,
            },
        },
    }


def trajectory_to_eventlog_sft_row(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Produces an AgentTune `EventLog.as_dataset_rows(fmt='sft')` row with
    Segment-Aware Loss weights (1.0 reasoning, 2.0 tool_call, 0.0 observation)."""
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nTask: {traj['user_prompt']}"}
    ]
    segment_weights: List[float] = [0.0]

    for s in traj["steps"]:
        # Reasoning segment (weight 1.0)
        messages.append({"role": "assistant", "content": s["THOUGHT"]})
        segment_weights.append(1.0)
        # Tool Call segment (weight 2.0 — double weight for tool syntax & arguments)
        call_payload = {"name": s["TOOL_CALL"], "arguments": s["ARGUMENTS"]}
        messages.append(
            {
                "role": "assistant",
                "content": f"<tool_call>{json.dumps(call_payload, separators=(',', ':'))}</tool_call>",
            }
        )
        segment_weights.append(2.0)
        # Tool Result observation segment (weight 0.0 — E1 observation masking)
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(s["OBSERVATION"], separators=(",", ":")),
            }
        )
        segment_weights.append(0.0)

    # Final response segment (weight 1.0)
    messages.append({"role": "assistant", "content": traj["final_response"]})
    segment_weights.append(1.0)

    return {
        "id": traj["trajectory_id"],
        "messages": messages,
        "segment_weights": segment_weights,
        "loss_mask": [w == 0.0 for w in segment_weights],
    }


def build_agenttune_datasets() -> Tuple[Path, Path, Path]:
    data_dir = ROOT_DIR / "data"
    tool_schemas = load_tool_schemas()

    with open(data_dir / "phase2_canonical_5_trajectories.json", "r", encoding="utf-8") as f:
        canonical_5 = json.load(f)

    synthetic_150: List[Dict[str, Any]] = []
    with open(data_dir / "phase2_synthetic_dataset_150.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                synthetic_150.append(json.loads(line))

    canonical_records = [trajectory_to_agenttune_record(t, tool_schemas) for t in canonical_5]
    synthetic_records = [trajectory_to_agenttune_record(t, tool_schemas) for t in synthetic_150]

    canonical_out = data_dir / "agenttune_canonical_5.jsonl"
    with open(canonical_out, "w", encoding="utf-8") as f:
        for rec in canonical_records:
            f.write(json.dumps(rec) + "\n")

    train_raw = canonical_5 + synthetic_150[:125]
    val_raw = synthetic_150[125:]

    train_records = canonical_records + synthetic_records[:125]
    val_records = synthetic_records[125:]

    train_out = data_dir / "agenttune_sft_train.jsonl"
    with open(train_out, "w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec) + "\n")

    val_out = data_dir / "agenttune_sft_val.jsonl"
    with open(val_out, "w", encoding="utf-8") as f:
        for rec in val_records:
            f.write(json.dumps(rec) + "\n")

    # Export native AgentTune `trajectories.jsonl` format for the 25 validation episodes
    native_val_out = data_dir / "agenttune_native_val_trajectories.jsonl"
    with open(native_val_out, "w", encoding="utf-8") as f:
        for t in val_raw:
            f.write(json.dumps(trajectory_to_native_agenttune_dict(t)) + "\n")

    # Export native AgentTune EventLog Segment-Aware SFT rows for the 130 train episodes
    eventlog_sft_out = data_dir / "agenttune_eventlog_sft_train.jsonl"
    with open(eventlog_sft_out, "w", encoding="utf-8") as f:
        for t in train_raw:
            f.write(json.dumps(trajectory_to_eventlog_sft_row(t)) + "\n")

    print(f"Created AgentTune Canonical JSONL ({len(canonical_records)} episodes) -> {canonical_out}")
    print(f"Created AgentTune SFT Train JSONL ({len(train_records)} episodes) -> {train_out}")
    print(f"Created AgentTune SFT Val JSONL   ({len(val_records)} episodes) -> {val_out}")
    print(f"Created Native AgentTune Trajectory JSONL ({len(val_raw)} episodes) -> {native_val_out}")
    print(f"Created Native EventLog Segment-Weighted SFT JSONL ({len(train_raw)} rows) -> {eventlog_sft_out}")
    return canonical_out, train_out, val_out


if __name__ == "__main__":
    build_agenttune_datasets()
