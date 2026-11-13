"""Live multi-turn rollouts of a REAL model against DoubleEntryLedgerEnv.

Nothing here is scripted: the policy (an actual LLM) chooses every tool call,
the environment executes it, and the observation is fed back to the model.

Policy backends:
  * HFPolicy             - local HuggingFace transformers model (+ optional LoRA adapter)
  * OpenAICompatPolicy   - any OpenAI-compatible /chat/completions endpoint
                           (vLLM server, Ollama, OpenAI, Groq, OpenRouter, ...)
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src" / "phase1_architecture"))

from ledger_env import DoubleEntryLedgerEnv, TxState, load_tool_schemas  # noqa: E402

# A policy maps (messages, tools) -> one assistant message dict:
#   {"role": "assistant", "content": str, "tool_calls": [ {id, type, function:{name, arguments(str)}} ]}
Policy = Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, Any]]

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


# -----------------------------------------------------------------------------
# Environment reconstruction & tool execution
# -----------------------------------------------------------------------------
def build_env_from_snapshot(initial_accounts: List[Dict[str, Any]]) -> DoubleEntryLedgerEnv:
    env = DoubleEntryLedgerEnv()
    for acc in initial_accounts:
        env.create_account(
            acc["account_id"],
            acc["owner_name"],
            acc["account_type"],
            acc["currency"],
            acc["kyc_tier"],
            acc["initial_balance"],
        )
    return env


def execute_tool_call(
    env: DoubleEntryLedgerEnv, name: str, raw_args: Any, schema_map: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Validates a model-issued tool call against the strict schema, then executes it."""
    if name not in schema_map:
        return {"status": "ERROR", "error_code": "UNKNOWN_TOOL", "message": f"No tool named '{name}'."}
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except Exception as exc:  # noqa: BLE001
        return {"status": "ERROR", "error_code": "MALFORMED_JSON_ARGUMENTS", "message": str(exc)}

    params = schema_map[name]
    required = set(params.get("required", []))
    allowed = set(params.get("properties", {}).keys())
    keys = set(args.keys())
    if not required.issubset(keys) or not keys.issubset(allowed):
        return {
            "status": "ERROR",
            "error_code": "SCHEMA_VALIDATION_ERROR",
            "missing": sorted(required - keys),
            "unexpected": sorted(keys - allowed),
        }
    try:
        return getattr(env, name)(**args)
    except Exception as exc:  # noqa: BLE001  (bad types/values inside valid keys)
        return {"status": "ERROR", "error_code": "TOOL_EXECUTION_ERROR", "message": str(exc)}


def run_live_episode(
    policy: Policy, episode: Dict[str, Any], max_turns: int = 14
) -> Dict[str, Any]:
    """Runs one held-out episode live. Returns the generated messages + ground-truth outcome."""
    tools = episode["tools"]
    schema_map = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    env = build_env_from_snapshot(episode["metadata"]["initial_accounts"])

    messages: List[Dict[str, Any]] = [dict(episode["messages"][0]), dict(episode["messages"][1])]
    hit_turn_limit = True
    for turn in range(max_turns):
        asst = policy(messages, tools)
        messages.append(asst)
        tool_calls = asst.get("tool_calls") or []
        if not tool_calls:
            hit_turn_limit = False
            break
        for i, tc in enumerate(tool_calls):
            tc.setdefault("id", f"call_{turn}_{i}")
            fn = tc.get("function", {})
            obs = execute_tool_call(env, fn.get("name", ""), fn.get("arguments", "{}"), schema_map)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": fn.get("name", ""),
                    "content": json.dumps(obs, separators=(",", ":")),
                }
            )

    # Ground-truth outcome straight from the environment state (not from the transcript)
    expected = episode["metadata"]["expected_terminal_state"]
    txs = list(env.transactions.values())
    tx = txs[0] if txs else None
    task_success = bool(
        tx is not None
        and len(txs) == 1
        and tx.state.value == expected
        and tx.audited
        and env.get_equation_summary()["balanced"]
    )
    return {
        "messages": messages,
        "hit_turn_limit": hit_turn_limit,
        "task_success": task_success,
        "final_tx_state": tx.state.value if tx else "NONE",
        "equation_balanced": env.get_equation_summary()["balanced"],
    }


# -----------------------------------------------------------------------------
# Helpers shared by backends
# -----------------------------------------------------------------------------
TOOL_CALL_PROMPT_GUIDE = """Available Tools (emit exactly one tool call per turn wrapped in <tool_call>{"name": "...", "arguments": {...}}</tool_call>):
1. get_account_balance: {"account_id": str}
2. create_account: {"account_id": str, "owner_name": str, "account_type": "ASSET"|"LIABILITY"|"EQUITY"|"REVENUE"|"EXPENSE", "currency": str, "kyc_tier": str}
3. check_compliance: {"transaction_id": str, "sender_account_id": str, "recipient_account_ids": [str, ...], "total_amount": float, "jurisdiction": "DOMESTIC"|"CROSS_BORDER"|"HIGH_RISK_SANCTIONED"}
4. post_journal_entry: {"transaction_id": str, "target_state": "PENDING"|"VALIDATED"|"COMMITTED"|"FAILED"|"ROLLED_BACK", "entries": [{"account_id": str, "entry_type": "DEBIT"|"CREDIT", "amount": float}, ...], "memo": str}
5. audit_transaction: {"transaction_id": str, "expected_terminal_state": "COMMITTED"|"ROLLED_BACK", "compliance_token": str, "auditor_notes": str}"""


def messages_for_chat_template(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Qwen/Llama chat templates expect tool_call arguments as dicts, not JSON strings."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            fixed = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                args = tc["function"].get("arguments")
                if isinstance(args, str):
                    try:
                        tc["function"]["arguments"] = json.loads(args)
                    except Exception:  # noqa: BLE001
                        pass
                fixed.append(tc)
            m["tool_calls"] = fixed
        out.append(m)
    return out


def messages_for_eventlog_template(
    messages: List[Dict[str, Any]], include_tool_guide: bool = False
) -> List[Dict[str, str]]:
    """Formats conversation into strict alternating user/assistant turns matching AgentTune EventLog SFT.
    Also prevents Cohere Tiny Aya's default chat template from dropping `role: 'tool'` turns."""
    sys_text = messages[0].get("content", "") if messages and messages[0].get("role") == "system" else ""
    user_text = messages[1].get("content", "") if len(messages) > 1 else ""
    if include_tool_guide:
        first_user = f"{sys_text}\n\n{TOOL_CALL_PROMPT_GUIDE}\n\nTask: {user_text}".strip()
    else:
        first_user = f"{sys_text}\n\nTask: {user_text}".strip()

    out: List[Dict[str, str]] = [{"role": "user", "content": first_user}]
    for m in messages[2:]:
        role = m.get("role")
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                fn = tcs[0].get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                try:
                    args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args_obj = raw_args
                payload = json.dumps({"name": name, "arguments": args_obj}, separators=(",", ":"))
                out.append({"role": "assistant", "content": f"<tool_call>{payload}</tool_call>"})
            else:
                out.append({"role": "assistant", "content": m.get("content", "")})
        elif role == "tool":
            t_name = m.get("name", "tool")
            out.append({"role": "user", "content": f"Tool result ({t_name}): {m.get('content', '{}')}"})
    return out


def parse_hermes_tool_calls(text: str) -> Dict[str, Any]:
    """Parses `<tool_call>{json}</tool_call>`, Cohere `<|START_ACTION|>[...]<|END_ACTION|>`,
    or AgentTune's Cohere fallback `[{"tool_name": ..., "parameters": {...}}]`."""
    calls = []
    for i, block in enumerate(TOOL_CALL_RE.findall(text)):
        try:
            obj = json.loads(block)
            name = obj.get("name") or obj.get("tool_name") or ""
            args = obj.get("arguments", obj.get("parameters", {}))
            args_str = args if isinstance(args, str) else json.dumps(args, separators=(",", ":"))
        except Exception:  # noqa: BLE001  -> keep it so the syntax metric counts it as invalid
            name, args_str = "__unparseable__", block
        calls.append({"id": f"call_gen_{i}", "type": "function", "function": {"name": name, "arguments": args_str}})

    if not calls:
        action_blocks = re.findall(r"<\|START_ACTION\|>\s*(.*?)\s*<\|END_ACTION\|>", text, re.DOTALL)
        candidates = action_blocks if action_blocks else re.findall(r"(\[\s*\{.*?\}\s*\])", text, re.DOTALL)
        for blk in candidates:
            for parser in (json.loads, ast.literal_eval):
                try:
                    arr = parser(blk)
                    if isinstance(arr, dict):
                        arr = [arr]
                    if isinstance(arr, list):
                        for i, item in enumerate(arr):
                            if isinstance(item, dict) and ("tool_name" in item or "name" in item):
                                name = item.get("tool_name") or item.get("name") or ""
                                args = item.get("parameters", item.get("arguments", {}))
                                args_str = args if isinstance(args, str) else json.dumps(args, separators=(",", ":"))
                                calls.append(
                                    {
                                        "id": f"call_gen_{i}",
                                        "type": "function",
                                        "function": {"name": name, "arguments": args_str},
                                    }
                                )
                        if calls:
                            break
                except Exception:
                    pass
            if calls:
                break

    content = TOOL_CALL_RE.sub("", text).strip()
    msg: Dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
    return msg


# -----------------------------------------------------------------------------
# Backend 1: local HuggingFace model (+ optional LoRA adapter) with lazy load & VRAM cleanup
# -----------------------------------------------------------------------------
def _resolve_adapter_dir(adapter_path: Optional[str]) -> Optional[str]:
    """Resolves adapter directory (or latest checkpoint-* subdirectory if saved by Trainer)."""
    if not adapter_path:
        return None
    p = Path(adapter_path)
    if not p.exists():
        return None
    if (p / "adapter_config.json").exists() or (p / "config.json").exists():
        return str(p)
    ckpts = sorted(
        [d for d in p.glob("checkpoint-*") if d.is_dir()],
        key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0,
    )
    if ckpts:
        return str(ckpts[-1])
    return None


class HFPolicy:
    def __init__(self, model_name_or_path: str, adapter_path: Optional[str] = None,
                 max_new_tokens: int = 512, load_in_4bit: bool = False, lazy: bool = True) -> None:
        self.model_name_or_path = model_name_or_path
        self.adapter_path = _resolve_adapter_dir(adapter_path)
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.tok = None
        self.model = None
        if not lazy:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto",
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.tok = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **kwargs)
        if self.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()

    def unload(self) -> None:
        """Frees GPU VRAM immediately after evaluating this checkpoint."""
        import gc
        self.model = None
        self.tok = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __call__(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        self._ensure_loaded()
        is_cohere_or_aya = any(k in self.model_name_or_path.lower() for k in ("aya", "cohere"))
        use_eventlog_fmt = bool(self.adapter_path) or is_cohere_or_aya

        if use_eventlog_fmt:
            ev_msgs = messages_for_eventlog_template(messages, include_tool_guide=True)
            prompt = self.tok.apply_chat_template(ev_msgs, add_generation_prompt=True, tokenize=False)
        else:
            try:
                prompt = self.tok.apply_chat_template(
                    messages_for_chat_template(messages), tools=tools, add_generation_prompt=True, tokenize=False
                )
            except Exception:
                ev_msgs = messages_for_eventlog_template(messages, include_tool_guide=True)
                prompt = self.tok.apply_chat_template(ev_msgs, add_generation_prompt=True, tokenize=False)

        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = parse_hermes_tool_calls(text)

        # If a two-segment EventLog adapter emitted a reasoning thought first without `<tool_call>`
        # and the episode hasn't audited the transaction yet, prompt the `<tool_call>` continuation.
        already_audited = any(
            m.get("role") == "tool" and m.get("name") == "audit_transaction" for m in messages
        )
        if self.adapter_path and not parsed.get("tool_calls") and not already_audited:
            cont_prompt = prompt + text + "\n<tool_call>"
            c_inputs = self.tok(cont_prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                c_out = self.model.generate(
                    **c_inputs, max_new_tokens=256, do_sample=False, pad_token_id=self.tok.eos_token_id
                )
            c_text = "<tool_call>" + self.tok.decode(
                c_out[0][c_inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            c_parsed = parse_hermes_tool_calls(c_text)
            if c_parsed.get("tool_calls"):
                c_parsed["content"] = text
                return c_parsed

        return parsed


# -----------------------------------------------------------------------------
# Backend 2: any OpenAI-compatible server (vLLM, Ollama, OpenAI, Groq, ...)
# -----------------------------------------------------------------------------
class OpenAICompatPolicy:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", temperature: float = 0.0) -> None:
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature

    def __call__(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools, temperature=self.temperature
        )
        m = resp.choices[0].message
        msg: Dict[str, Any] = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in m.tool_calls
            ]
        # Some servers return Hermes text instead of structured tool_calls
        elif "<tool_call>" in msg["content"]:
            return parse_hermes_tool_calls(msg["content"])
        return msg

