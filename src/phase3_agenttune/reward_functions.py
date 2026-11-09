"""Phase 3 (Enhancement #3): Deterministic Verifiable Reward Function for AgentTune GRPO.

In Lexsi Labs' AgentTune / AlignTune post-training pipeline, Group-Relative Policy
Optimization (GRPO) scores groups of rollout trajectories using deterministic,
rule-based outcome and process rewards rather than a brittle neural reward model.

This module implements `compute_double_entry_grpo_reward()`, which decomposes the
trajectory reward R(tau) in [-0.50, +1.00] into 4 verifiable components:
  1. R_syntax     (max +0.25): Strict JSON schema & parameter adherence.
  2. R_accounting (max +0.30): Double-entry invariant (sum(Debits) == sum(Credits)).
  3. R_statechart (max +0.30, min -0.50): Valid statechart transitions & safety halt compliance.
  4. R_efficiency (max +0.15): Sealed terminal audit & step efficiency (no infinite loops).
"""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src" / "phase1_architecture"))

from ledger_env import VALID_STATE_TRANSITIONS, TxState, load_tool_schemas, to_decimal  # noqa: E402


def compute_double_entry_grpo_reward(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Computes the deterministic multi-objective GRPO reward for an agent trajectory."""
    schemas = tools if tools is not None else load_tool_schemas()
    schema_map = {t["function"]["name"]: t["function"]["parameters"] for t in schemas}

    tool_calls_made: List[Dict[str, Any]] = []
    tool_observations: Dict[str, Dict[str, Any]] = {}

    syntax_valid_calls = 0
    total_calls = 0

    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                total_calls += 1
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                call_id = tc.get("id", "")

                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(parsed_args, dict):
                        raise ValueError("Arguments must decode to a JSON object.")
                except Exception:
                    tool_calls_made.append(
                        {"id": call_id, "name": fn_name, "args": None, "valid_syntax": False}
                    )
                    continue

                # Validate against schema required keys and additionalProperties=False
                param_schema = schema_map.get(fn_name)
                if param_schema is not None:
                    req_keys = set(param_schema.get("required", []))
                    allowed_keys = set(param_schema.get("properties", {}).keys())
                    actual_keys = set(parsed_args.keys())
                    if req_keys.issubset(actual_keys) and actual_keys.issubset(allowed_keys):
                        syntax_valid_calls += 1
                        is_valid = True
                    else:
                        is_valid = False
                else:
                    is_valid = False

                tool_calls_made.append(
                    {"id": call_id, "name": fn_name, "args": parsed_args, "valid_syntax": is_valid}
                )

        elif role == "tool":
            call_id = msg.get("tool_call_id", "")
            try:
                obs_data = json.loads(msg.get("content", "{}"))
            except Exception:
                obs_data = {"status": "INVALID_JSON"}
            tool_observations[call_id] = obs_data

    if total_calls == 0:
        return {
            "total_reward": -0.50,
            "r_syntax": 0.0,
            "r_accounting": 0.0,
            "r_statechart": -0.50,
            "r_efficiency": 0.0,
            "diagnostics": ["NO_TOOL_CALLS_GENERATED"],
        }

    diagnostics: List[str] = []

    # 1. R_syntax (0.0 to 0.25)
    r_syntax = round(0.25 * (syntax_valid_calls / total_calls), 4)
    if syntax_valid_calls < total_calls:
        diagnostics.append(f"SYNTAX_ERRORS:{total_calls - syntax_valid_calls}/{total_calls}")

    # 2. R_accounting (0.0 to 0.30)
    journal_calls = [
        tc for tc in tool_calls_made if tc["name"] == "post_journal_entry" and tc["valid_syntax"]
    ]
    balanced_journal_calls = 0
    for jc in journal_calls:
        entries = jc["args"].get("entries", [])
        if len(entries) >= 2:
            try:
                debits = sum(
                    to_decimal(e["amount"]) for e in entries if e.get("entry_type") == "DEBIT"
                )
                credits = sum(
                    to_decimal(e["amount"]) for e in entries if e.get("entry_type") == "CREDIT"
                )
                if debits == credits and debits > Decimal("0.00"):
                    balanced_journal_calls += 1
                else:
                    diagnostics.append(f"UNBALANCED_ENTRY:D={debits}!=C={credits}")
            except Exception:
                diagnostics.append("MALFORMED_ENTRY_AMOUNT")
        else:
            diagnostics.append("SINGLE_ENTRY_VIOLATION")

    if journal_calls:
        r_accounting = round(0.30 * (balanced_journal_calls / len(journal_calls)), 4)
    else:
        r_accounting = 0.0
        diagnostics.append("NO_JOURNAL_ENTRIES_POSTED")

    # 3. R_statechart (-0.50 to +0.30)
    current_state: TxState | None = None
    illegal_transitions = 0
    unsafe_commit_after_rejection = False
    compliance_rejected_or_overdraft = False
    state_sequence: List[str] = []

    for tc in tool_calls_made:
        obs = tool_observations.get(tc["id"], {})
        if tc["name"] == "check_compliance" and obs.get("status") == "REJECTED":
            compliance_rejected_or_overdraft = True

        if tc["name"] == "post_journal_entry" and isinstance(tc.get("args"), dict):
            target_str = tc["args"].get("target_state", "")
            if obs.get("error_code") == "INSUFFICIENT_FUNDS_OVERDRAFT":
                compliance_rejected_or_overdraft = True
                # Attempting VALIDATED when overdraft is discovered is the expected detection step
                continue

            try:
                desired_state = TxState(target_str)
            except ValueError:
                illegal_transitions += 1
                continue

            if compliance_rejected_or_overdraft and desired_state == TxState.COMMITTED:
                unsafe_commit_after_rejection = True
                diagnostics.append("CRITICAL_SAFETY_VIOLATION:COMMIT_ATTEMPTED_AFTER_REJECTION")

            allowed = VALID_STATE_TRANSITIONS.get(current_state, set())
            if desired_state in allowed and obs.get("status") == "SUCCESS" and tc["valid_syntax"]:
                current_state = desired_state
                state_sequence.append(desired_state.value)
            else:
                illegal_transitions += 1
                diagnostics.append(f"ILLEGAL_TRANSITION:{current_state}->{desired_state}")

    expected_terminal = (metadata or {}).get("expected_terminal_state")
    if unsafe_commit_after_rejection:
        r_statechart = -0.50
    elif illegal_transitions > 0:
        r_statechart = max(-0.25, round(0.30 - 0.15 * illegal_transitions, 4))
    else:
        if expected_terminal and current_state and current_state.value == expected_terminal:
            r_statechart = 0.30
        elif current_state in {TxState.COMMITTED, TxState.ROLLED_BACK}:
            r_statechart = 0.25
        else:
            r_statechart = 0.0
            diagnostics.append("NON_TERMINAL_STATECHART_END")

    # 4. R_efficiency (0.0 to 0.15)
    last_call = tool_calls_made[-1]
    last_obs = tool_observations.get(last_call["id"], {})
    audited_cleanly = (
        last_call["name"] == "audit_transaction" and last_obs.get("status") == "AUDITED"
    )
    optimal_steps = (metadata or {}).get("num_tool_steps", 6)
    extra_steps = max(0, total_calls - optimal_steps)

    if audited_cleanly:
        r_efficiency = max(0.0, round(0.15 - 0.03 * extra_steps, 4))
    else:
        r_efficiency = 0.0
        diagnostics.append("MISSING_FINAL_AUDIT_SEAL")

    total_reward = round(r_syntax + r_accounting + r_statechart + r_efficiency, 4)
    return {
        "total_reward": total_reward,
        "r_syntax": r_syntax,
        "r_accounting": r_accounting,
        "r_statechart": r_statechart,
        "r_efficiency": r_efficiency,
        "state_sequence": state_sequence,
        "total_tool_calls": total_calls,
        "diagnostics": diagnostics if diagnostics else ["OPTIMAL_TRAJECTORY"],
    }


def agenttune_grpo_reward_fn(
    completions: List[Any],
    prompts: List[Any] | None = None,
    trajectories: List[Any] | None = None,
    expected_terminal_state: List[str] | None = None,
    num_tool_steps: List[int] | None = None,
    **kwargs: Any,
) -> List[float]:
    """Native `TrlAgenticGrpo` reward function (`create_agentic_trainer("grpo", reward_funcs=...)`).

    Accepts `completions` as a list of multi-turn message lists (or `trajectories` from
    `create_rollout_fn`) and returns a `list[float]` of deterministic double-entry rewards.
    """
    schemas = load_tool_schemas()
    rewards: List[float] = []

    for i, comp in enumerate(completions):
        meta: Dict[str, Any] = {}
        if expected_terminal_state and i < len(expected_terminal_state):
            meta["expected_terminal_state"] = expected_terminal_state[i]
        if num_tool_steps and i < len(num_tool_steps):
            meta["num_tool_steps"] = num_tool_steps[i]

        if isinstance(comp, list) and comp and isinstance(comp[0], dict):
            msgs = comp
        elif trajectories and i < len(trajectories) and hasattr(trajectories[i], "to_trl_format"):
            msgs = trajectories[i].to_trl_format()["messages"]
        else:
            rewards.append(-0.50)
            continue

        res = compute_double_entry_grpo_reward(msgs, schemas, meta)
        rewards.append(float(res["total_reward"]))

    return rewards

