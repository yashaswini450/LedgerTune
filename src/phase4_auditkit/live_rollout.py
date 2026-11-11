"""Live multi-turn rollouts of a REAL model against DoubleEntryLedgerEnv.

Nothing here is scripted: the policy (an actual LLM) chooses every tool call,
the environment executes it, and the observation is fed back to the model.

Policy backends:
  * HFPolicy             - local HuggingFace transformers model (+ optional LoRA adapter)
  * OpenAICompatPolicy   - any OpenAI-compatible /chat/completions endpoint
                           (vLLM server, Ollama, OpenAI, Groq, OpenRouter, ...)
"""

from __future__ import annotations

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


def parse_hermes_tool_calls(text: str) -> Dict[str, Any]:
    """Parses Qwen2.5 / Hermes-style `<tool_call>{json}</tool_call>` output."""
    calls = []
    for i, block in enumerate(TOOL_CALL_RE.findall(text)):
        try:
            obj = json.loads(block)
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            args_str = args if isinstance(args, str) else json.dumps(args, separators=(",", ":"))
        except Exception:  # noqa: BLE001  -> keep it so the syntax metric counts it as invalid
            name, args_str = "__unparseable__", block
        calls.append({"id": f"call_gen_{i}", "type": "function", "function": {"name": name, "arguments": args_str}})
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
        prompt = self.tok.apply_chat_template(
            messages_for_chat_template(messages), tools=tools, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_hermes_tool_calls(text)


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

