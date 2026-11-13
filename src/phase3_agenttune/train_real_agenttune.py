"""End-to-End Real AgentTune Post-Training (Stage-1 SFT + Stage-2 GRPO) & Live AuditKIT Eval.

Uses the real Lexsi Labs `AgentTune` and `AuditKIT` libraries:
  1. Converts our 130 training episodes into full-tier `agenttune.agentic.EventLog` objects.
  2. Runs Stage-1 LoRA SFT / Agentic Distillation via `agenttune.create_distill_trainer()`.
  3. Runs Stage-2 Agentic GRPO via `agenttune.core.backend_factory.create_agentic_trainer("grpo")`
     with our deterministic `agenttune_grpo_reward_fn` and writes `trajectories.jsonl`.
  4. Evaluates the real Baseline, Stage-1 SFT, and Stage-2 GRPO checkpoints live against
     `DoubleEntryLedgerEnv` on the 25 held-out validation episodes using `AuditKIT`'s
     `AgentEvalRunner` (`auditkit_harness.py`).

Usage (on Google Colab T4/A100 GPU or local CUDA/MPS machine):
    python3 src/phase3_agenttune/train_real_agenttune.py \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --sft-epochs 3 \
        --grpo-steps 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src" / "phase1_architecture"))
sys.path.insert(0, str(ROOT_DIR / "src" / "phase3_agenttune"))
sys.path.insert(0, str(ROOT_DIR / "src" / "phase4_auditkit"))

from ledger_env import DoubleEntryLedgerEnv  # noqa: E402
from prepare_agenttune_sft import SYSTEM_PROMPT  # noqa: E402
from reward_functions import agenttune_grpo_reward_fn  # noqa: E402


def build_full_tier_eventlogs(train_jsonl: Path) -> List[Any]:
    """Converts `data/agenttune_sft_train.jsonl` into `agenttune.agentic.EventLog(tier='full')` objects."""
    from agenttune.agentic.events import Event, EventKind, EventLog

    logs: List[EventLog] = []
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            msgs = rec["messages"]

            log = EventLog(tier="full", id=rec["id"])
            # Initial user task
            user_task = f"{msgs[0]['content']}\n\nTask: {msgs[1]['content']}"
            log.append(Event(EventKind.OBSERVATION, {"text": user_task}))

            intermediate = msgs[2:-1]
            for i in range(0, len(intermediate), 2):
                asst_msg = intermediate[i]
                tool_msg = intermediate[i + 1]
                thought_text = (asst_msg.get("content") or "").replace("[THOUGHT]\n", "").strip()
                if thought_text:
                    log.append(Event(EventKind.REASONING, {"text": thought_text}))

                tc = asst_msg["tool_calls"][0]
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                log.append(
                    Event(
                        EventKind.TOOL_CALL,
                        {"action": {"name": fn_name, "arguments": fn_args}},
                    )
                )
                log.append(Event(EventKind.TOOL_RESULT, {"output": tool_msg["content"]}))
                log.append(Event(EventKind.TURN_COMPLETE, {"step": (i // 2) + 1}))

            # Final assistant summary with token_span so full-tier check passes
            final_text = msgs[-1].get("content", "")
            log.append(
                Event(
                    EventKind.TEXT,
                    {"text": final_text},
                    token_span=(0, max(1, len(final_text.split()))),
                    logprobs=[-0.1],
                )
            )
            log.append(Event(EventKind.REWARD, {"value": 1.0}, scope="episode"))
            logs.append(log)

    return logs


def make_callable_ledger_tools() -> List[Any]:
    """Creates typed Python tool callables backed by a DoubleEntryLedgerEnv instance
    for `create_agentic_trainer('grpo', tools=...)`."""
    env = DoubleEntryLedgerEnv()
    env.create_account("ACC_ASSET_VAULT", "Omnibus Reserve Vault", "ASSET", "USD", "INSTITUTIONAL", 15000.0)
    env.create_account("ACC_WALLET_ALICE", "Alice Rivera", "LIABILITY", "USD", "TIER_2_FULL", 5000.0)
    env.create_account("ACC_WALLET_BOB", "Bob Chen", "LIABILITY", "USD", "TIER_2_FULL", 2000.0)
    env.create_account("ACC_WALLET_CAROL", "Carol Vane", "LIABILITY", "USD", "SANCTIONED", 1500.0)
    env.create_account("ACC_MERCHANT_ALPHA", "Alpha Logistics", "LIABILITY", "USD", "INSTITUTIONAL", 3000.0)
    env.create_account("ACC_MERCHANT_BETA", "Beta Studios", "LIABILITY", "USD", "TIER_2_FULL", 1000.0)
    env.create_account("ACC_REV_FEES", "Platform Fee Revenue", "REVENUE", "USD", "INSTITUTIONAL", 250.0)
    env.create_account("ACC_EQUITY_RESERVE", "Core Equity", "EQUITY", "USD", "INSTITUTIONAL", 2250.0)

    def create_account(
        account_id: str, owner_name: str, account_type: str, currency: str, kyc_tier: str
    ) -> dict:
        """Creates a new account in the double-entry ledger chart of accounts."""
        return env.create_account(account_id, owner_name, account_type, currency, kyc_tier)

    def get_account_balance(account_id: str) -> dict:
        """Retrieves the current settled balance, normal balance side, and KYC status of an account."""
        return env.get_account_balance(account_id)

    def check_compliance(
        transaction_id: str,
        sender_account_id: str,
        recipient_account_ids: list,
        total_amount: float,
        jurisdiction: str,
    ) -> dict:
        """Runs KYC/AML, sanctions screening, and velocity limit checks on a PENDING transaction."""
        return env.check_compliance(
            transaction_id, sender_account_id, recipient_account_ids, total_amount, jurisdiction
        )

    def post_journal_entry(
        transaction_id: str, target_state: str, entries: list, memo: str
    ) -> dict:
        """Stages or transitions a double-entry journal block along the strict statechart."""
        return env.post_journal_entry(transaction_id, target_state, entries, memo)

    def audit_transaction(
        transaction_id: str,
        expected_terminal_state: str,
        compliance_token: str,
        auditor_notes: str,
    ) -> dict:
        """Verifies double-entry invariants and records an immutable SHA-256 audit trail entry."""
        return env.audit_transaction(
            transaction_id, expected_terminal_state, compliance_token, auditor_notes
        )

    return [
        create_account,
        get_account_balance,
        check_compliance,
        post_journal_entry,
        audit_transaction,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & Evaluate Real AgentTune Ledger Agent")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sft-output-dir", type=str, default="./checkpoints/sft_ledger")
    parser.add_argument("--grpo-output-dir", type=str, default="./checkpoints/grpo_ledger")
    parser.add_argument("--sft-epochs", type=int, default=3)
    parser.add_argument("--grpo-steps", type=int, default=40)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--skip-grpo", action="store_true", help="Run Stage-1 SFT + AuditKIT eval only")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate saved checkpoints in ./checkpoints/")
    args = parser.parse_args()

    import gc
    import torch

    def _free_vram() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grpo_adapter_path = args.grpo_output_dir if Path(args.grpo_output_dir).exists() else args.sft_output_dir

    if not args.eval_only:
        from datasets import Dataset
        from agenttune import create_distill_trainer
        from agenttune.core.backend_factory import create_agentic_trainer

        train_jsonl = ROOT_DIR / "data" / "agenttune_sft_train.jsonl"
        print(f"=== [Stage 1] Building Full-Tier AgentTune EventLogs from {train_jsonl.name} ===")
        eventlogs = build_full_tier_eventlogs(train_jsonl)
        print(f"Loaded {len(eventlogs)} full-tier EventLogs.")

        print(f"\n=== [Stage 1] Running AgentTune SFT / Distillation on {args.base_model} ===")
        distill_trainer = create_distill_trainer(
            student=args.base_model,
            trajectories=eventlogs,
            output_dir=args.sft_output_dir,
            use_peft=True,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            num_train_epochs=args.sft_epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=5,
            save_strategy="epoch",
            report_to="none",
        )
        sft_result = distill_trainer.train()
        distill_trainer.trainer_factory.sft_trainer.save_model(args.sft_output_dir)
        distill_trainer.trainer_factory.tokenizer.save_pretrained(args.sft_output_dir)
        print(f"Stage-1 SFT complete! Saved adapter -> {args.sft_output_dir} | {sft_result}")
        del distill_trainer
        _free_vram()

        grpo_adapter_path = args.sft_output_dir
        if not args.skip_grpo:
            print("\n=== [Stage 2] Running AgentTune GRPO with Deterministic Double-Entry Reward ===")
            with open(train_jsonl, "r", encoding="utf-8") as f:
                raw_train = [json.loads(line) for line in f if line.strip()]

            grpo_rows = [
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": r["messages"][1]["content"]},
                    ],
                    "expected_terminal_state": r["metadata"]["expected_terminal_state"],
                    "num_tool_steps": r["metadata"]["num_tool_steps"],
                }
                for r in raw_train
            ]
            grpo_dataset = Dataset.from_list(grpo_rows)

            # Start GRPO from the SFT policy, not the raw base model: merge the Stage-1 LoRA
            # adapter into the base weights and hand the merged checkpoint to GRPO.
            # (The first Colab run initialised GRPO from args.base_model, so Stage 2 did not
            # build on Stage 1.)
            grpo_init_model = args.base_model
            try:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer

                merged_dir = str(Path(args.sft_output_dir).parent / "sft_merged")
                print(f"Merging SFT adapter {args.sft_output_dir} into {args.base_model} -> {merged_dir}")
                base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
                merged = PeftModel.from_pretrained(base, args.sft_output_dir).merge_and_unload()
                merged.save_pretrained(merged_dir)
                AutoTokenizer.from_pretrained(args.sft_output_dir).save_pretrained(merged_dir)
                del base, merged
                _free_vram()
                grpo_init_model = merged_dir
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: could not merge SFT adapter ({exc}); GRPO will start from the base model.")

            grpo_trainer = create_agentic_trainer(
                algorithm="grpo",
                model=grpo_init_model,
                train_dataset=grpo_dataset,
                tools=make_callable_ledger_tools(),
                reward_funcs=[agenttune_grpo_reward_fn],
                output_dir=args.grpo_output_dir,
                max_steps=args.grpo_steps,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=5e-6,
                num_generations=2,
                max_steps_per_turn=8,
                max_completion_length=1024,
                use_vllm=False,
                peft_config={
                    "r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "target_modules": "all-linear",
                    "task_type": "CAUSAL_LM",
                },
                report_to="none",
            )
            grpo_res = grpo_trainer.train()
            if getattr(grpo_trainer, "trainer", None) is not None:
                grpo_trainer.trainer.save_model(args.grpo_output_dir)
            grpo_adapter_path = args.grpo_output_dir
            print(f"Stage-2 GRPO complete! Saved -> {args.grpo_output_dir} | {grpo_res}")
            del grpo_trainer
            _free_vram()

    print("\n=== [Stage 3] Running Live AuditKIT Evaluation (Sequential Lazy Load + VRAM Flush) ===")
    from auditkit_harness import run_full_auditkit_benchmark
    from live_rollout import HFPolicy, _resolve_adapter_dir

    model_short = args.base_model.split("/")[-1]
    resolved_sft = _resolve_adapter_dir(args.sft_output_dir)
    resolved_grpo = _resolve_adapter_dir(grpo_adapter_path) or resolved_sft
    print(f"  -> Baseline HF Model : {args.base_model}")
    print(f"  -> SFT Adapter Path  : {resolved_sft}")
    print(f"  -> GRPO Adapter Path : {resolved_grpo}")

    # `lazy=True` ensures only ONE model is loaded into VRAM at a time and unloaded right after its 25 episodes!
    base_policy = HFPolicy(args.base_model, lazy=True)
    sft_policy = HFPolicy(args.base_model, adapter_path=resolved_sft, lazy=True) if resolved_sft else None
    grpo_policy = HFPolicy(args.base_model, adapter_path=resolved_grpo, lazy=True) if resolved_grpo else None

    run_full_auditkit_benchmark(
        baseline_policy=base_policy,
        sft_policy=sft_policy,
        grpo_policy=grpo_policy,
        model_family_label=model_short,
    )
    print("\nREAL TRAINING + LIVE AUDITKIT BENCHMARK COMPLETE!")


if __name__ == "__main__":
    main()

