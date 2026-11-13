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
import ast
import json
from pathlib import Path
import re
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
    last_call_id = ""

    for msg_idx, msg in enumerate(messages):
        role = msg.get("role")
        tcs = list(msg.get("tool_calls") or [])
        if role == "assistant" and not tcs and isinstance(msg.get("content"), str):
            tcs = _extract_inline_tool_calls(msg["content"], prefix=f"inline_{msg_idx}")

        if role == "assistant" and tcs:
            for tc in tcs:
                total_calls += 1
                fn = tc.get("function", tc)
                fn_name = fn.get("name") or fn.get("tool_name") or ""
                raw_args = fn.get("arguments", fn.get("parameters", "{}"))
                call_id = tc.get("id") or f"call_{msg_idx}_{total_calls}"
                last_call_id = call_id

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
                        if fn_name == "post_journal_entry":
                            ents = parsed_args.get("entries")
                            is_valid = isinstance(ents, list) and len(ents) > 0 and all(
                                isinstance(e, dict) and {"account_id", "entry_type", "amount"}.issubset(e.keys())
                                for e in ents
                            )
                        else:
                            is_valid = True
                        if is_valid:
                            syntax_valid_calls += 1
                    else:
                        is_valid = False
                else:
                    is_valid = False

                tool_calls_made.append(
                    {"id": call_id, "name": fn_name, "args": parsed_args, "valid_syntax": is_valid}
                )

        elif role == "tool" or (
            role == "user"
            and isinstance(msg.get("content"), str)
            and ("Tool result" in msg["content"] or msg["content"].strip().startswith("{"))
        ):
            call_id = msg.get("tool_call_id") or last_call_id
            obs_data = _parse_obs_payload(msg.get("content", "{}"))
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


def _parse_obs_payload(raw_content: str) -> Dict[str, Any]:
    """Parses tool observation JSON or Python repr dict string (from AgentTune tool execution)."""
    text = raw_content.strip()
    if ":" in text and text.lower().startswith("tool result"):
        text = text.split(":", 1)[1].strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            val = parser(text)
            if isinstance(val, dict):
                # Unwrap nested single-tool observation dict if wrapped as {'tool_name': '{...}'}
                if len(val) == 1 and "status" not in val:
                    inner = next(iter(val.values()))
                    if isinstance(inner, dict):
                        return inner
                    if isinstance(inner, str):
                        for p2 in (json.loads, ast.literal_eval):
                            try:
                                v2 = p2(inner)
                                if isinstance(v2, dict):
                                    return v2
                            except Exception:
                                pass
                return val
        except Exception:
            pass
    return {"status": "INVALID_JSON"}


def _extract_inline_tool_calls(text: str, prefix: str = "call") -> List[Dict[str, Any]]:
    """Extracts tool calls from `<tool_call>{...}</tool_call>`, `<|START_ACTION|>[...]<|END_ACTION|>`,
    or AgentTune's Cohere fallback `[{"tool_name": ..., "parameters": {...}}]`."""
    calls: List[Dict[str, Any]] = []
    tc_blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    for idx, blk in enumerate(tc_blocks):
        try:
            obj = json.loads(blk)
            name = obj.get("name") or obj.get("tool_name") or "unknown"
            args = obj.get("arguments", obj.get("parameters", {}))
            args_str = json.dumps(args) if isinstance(args, dict) else str(args)
        except Exception:
            name, args_str = "invalid_json", blk
        calls.append({"id": f"{prefix}_{idx}", "type": "function", "function": {"name": name, "arguments": args_str}})
    if calls:
        return calls

    action_blocks = re.findall(r"<\|START_ACTION\|>\s*(.*?)\s*<\|END_ACTION\|>", text, re.DOTALL)
    candidates = action_blocks if action_blocks else re.findall(r"(\[\s*\{.*?\}\s*\])", text, re.DOTALL)
    for blk in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                arr = parser(blk)
                if isinstance(arr, dict):
                    arr = [arr]
                if isinstance(arr, list):
                    for idx, item in enumerate(arr):
                        if isinstance(item, dict) and ("tool_name" in item or "name" in item):
                            name = item.get("tool_name") or item.get("name") or "unknown"
                            args = item.get("parameters", item.get("arguments", {}))
                            args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                            calls.append(
                                {
                                    "id": f"{prefix}_arr_{idx}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": args_str},
                                }
                            )
                    if calls:
                        return calls
            except Exception:
                pass
    return calls


def _parse_text_completion_to_messages(text: str) -> List[Dict[str, Any]]:
    """Parse flattened multi-turn `<tool_call>` / `<tool_response>` / Cohere fallback text into messages."""
    msgs: List[Dict[str, Any]] = []
    calls = _extract_inline_tool_calls(text, prefix="call_parsed")
    resps = re.findall(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", text, re.DOTALL)
    if not resps:
        resps = re.findall(r"Tool result[^:\n]*:\s*(\{.*?\})(?:\n|$)", text)
    for idx, tc in enumerate(calls):
        call_id = tc["id"]
        msgs.append({"role": "assistant", "content": "", "tool_calls": [tc]})
        if idx < len(resps):
            msgs.append({"role": "tool", "tool_call_id": call_id, "content": resps[idx]})
    return msgs


def _messages_from_trajectory_obj(traj: Any) -> List[Dict[str, Any]]:
    """Extracts multi-turn OpenAI conversation messages from an AgentTune Trajectory object or dict."""
    if hasattr(traj, "metadata") and isinstance(traj.metadata, dict) and "conversation" in traj.metadata:
        return traj.metadata["conversation"]
    if isinstance(traj, dict):
        if "metadata" in traj and isinstance(traj["metadata"], dict) and "conversation" in traj["metadata"]:
            return traj["metadata"]["conversation"]
        if "messages" in traj and isinstance(traj["messages"], list):
            return traj["messages"]
    if hasattr(traj, "to_trl_format"):
        return traj.to_trl_format().get("messages", [])
    return []


def agenttune_grpo_reward_fn(
    completions: Any,
    prompts: List[Any] | None = None,
    trajectories: List[Any] | None = None,
    expected_terminal_state: List[str] | None = None,
    num_tool_steps: List[int] | None = None,
    **kwargs: Any,
) -> Any:
    """Native `TrlAgenticGrpo` reward function (`create_agentic_trainer("grpo", reward_funcs=...)`).

    Supports all three AgentTune reward call signatures:
      1. `fn(trajectory) -> float` (single Trajectory object passed during rollout collection)
      2. `fn(completions, prompts=..., trajectories=..., **kwargs) -> list[float]` (TRL batch call)
      3. `fn(responses, prompts) -> list[float]` (text completion batch call)
    """
    schemas = load_tool_schemas()

    # Signature 1: Single Trajectory object or single trajectory dict -> return a single float
    if not isinstance(completions, list):
        msgs = _messages_from_trajectory_obj(completions)
        if not msgs and isinstance(completions, str):
            msgs = _parse_text_completion_to_messages(completions)
        if not msgs:
            return -0.50
        res = compute_double_entry_grpo_reward(msgs, schemas)
        return float(res["total_reward"])

    # Signature 2 & 3: Batch list of completions/trajectories -> return list[float]
    rewards: List[float] = []
    for i, comp in enumerate(completions):
        meta: Dict[str, Any] = {}
        if expected_terminal_state and i < len(expected_terminal_state):
            meta["expected_terminal_state"] = expected_terminal_state[i]
        if num_tool_steps and i < len(num_tool_steps):
            meta["num_tool_steps"] = num_tool_steps[i]

        msgs: List[Dict[str, Any]] = []
        # Always prioritize `trajectories[i]` first because TRL's `comp` often contains only the
        # final turn text (`final_response`), whereas `trajectories[i].metadata["conversation"]`
        # holds the full multi-turn tool-calling trajectory!
        if trajectories and i < len(trajectories):
            msgs = _messages_from_trajectory_obj(trajectories[i])

        if not msgs:
            if isinstance(comp, list) and comp and isinstance(comp[0], dict):
                msgs = comp
            elif isinstance(comp, dict) or hasattr(comp, "metadata"):
                msgs = _messages_from_trajectory_obj(comp)
            elif isinstance(comp, str):
                msgs = _parse_text_completion_to_messages(comp)

        if not msgs:
            rewards.append(-0.50)
            continue

        res = compute_double_entry_grpo_reward(msgs, schemas, meta)
        rewards.append(float(res["total_reward"]))

    return rewards

