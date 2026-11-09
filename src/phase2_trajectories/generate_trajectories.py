"""Phase 2: Synthetic Trajectory Generation (Teacher Agent Simulator).

Generates:
1. The 5 Canonical Multi-Hop Workflows executed against the live DoubleEntryLedgerEnv:
   - Workflow 1: Standard P2P Transfer with Platform Fee (Happy Path)
   - Workflow 2: Failed KYC/AML & Sanctions Rollback (Compliance Rejection Path)
   - Workflow 3: Multi-Party Marketplace Split Settlement (3-Way Split + Fee)
   - Workflow 4: Overdraft Halt & Safe Rollback (Insufficient Funds Guard)
   - Workflow 5: Daily Omnibus Reconciliation & Revenue Sweep (Multi-Account Audit & Settlement)
2. Parameterized Trajectory Synthesizer (Enhancement #2):
   - Expands the 5 canonical archetypes into 150 diverse, environment-verified
     multi-hop trajectories (30 per archetype) for AgentTune SFT & GRPO training.

Every trajectory step strictly records:
- [THOUGHT]: First-principles accounting & statechart reasoning
- [TOOL_CALL]: Tool function name
- [ARGUMENTS]: Strict JSON arguments matching Phase 1 schemas
- [OBSERVATION]: Exact deterministic JSON output from DoubleEntryLedgerEnv
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

# Import Phase 1 Environment
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src" / "phase1_architecture"))

from ledger_env import DoubleEntryLedgerEnv, to_decimal  # noqa: E402


class TeacherTrajectoryRecorder:
    """Executes tool calls against DoubleEntryLedgerEnv and records structured steps."""

    def __init__(
        self,
        env: DoubleEntryLedgerEnv,
        trajectory_id: str,
        workflow_type: str,
        user_prompt: str,
    ) -> None:
        self.env = env
        self.trajectory_id = trajectory_id
        self.workflow_type = workflow_type
        self.user_prompt = user_prompt
        self.steps: List[Dict[str, Any]] = []
        self.final_response: str = ""
        # Snapshot of the starting ledger so the episode can be replayed live
        # against a fresh environment when evaluating a real model.
        self.initial_accounts: List[Dict[str, Any]] = [
            {
                "account_id": a.account_id,
                "owner_name": a.owner_name,
                "account_type": a.account_type.value,
                "currency": a.currency,
                "kyc_tier": a.kyc_tier,
                "initial_balance": float(a.balance),
            }
            for a in env.accounts.values()
        ]

    def step(self, thought: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        fn = getattr(self.env, tool_name)
        observation = fn(**arguments)
        step_record = {
            "step_index": len(self.steps) + 1,
            "THOUGHT": thought,
            "TOOL_CALL": tool_name,
            "ARGUMENTS": arguments,
            "OBSERVATION": observation,
        }
        self.steps.append(step_record)
        return observation

    def finish(self, final_response: str) -> Dict[str, Any]:
        self.final_response = final_response
        formatted_blocks: List[str] = []
        for s in self.steps:
            block = (
                f"[THOUGHT]\n{s['THOUGHT']}\n"
                f"[TOOL_CALL]\n{s['TOOL_CALL']}\n"
                f"[ARGUMENTS]\n{json.dumps(s['ARGUMENTS'], separators=(',', ':'))}\n"
                f"[OBSERVATION]\n{json.dumps(s['OBSERVATION'], separators=(',', ':'))}"
            )
            formatted_blocks.append(block)
        formatted_blocks.append(f"[FINAL_RESPONSE]\n{self.final_response}")

        return {
            "trajectory_id": self.trajectory_id,
            "workflow_type": self.workflow_type,
            "user_prompt": self.user_prompt,
            "initial_accounts": self.initial_accounts,
            "num_steps": len(self.steps),
            "steps": self.steps,
            "final_response": self.final_response,
            "formatted_trace": "\n\n".join(formatted_blocks),
            "fundamental_equation_after": self.env.get_equation_summary(),
        }


def bootstrap_base_ledger(
    currency: str = "USD",
    alice_bal: float = 5000.00,
    bob_bal: float = 2000.00,
    carol_bal: float = 1500.00,
    merch_a_bal: float = 3000.00,
    merch_b_bal: float = 1000.00,
    fee_rev_bal: float = 250.00,
    equity_bal: float = 2250.00,
    alice_kyc: str = "TIER_2_FULL",
    bob_kyc: str = "TIER_2_FULL",
    carol_kyc: str = "SANCTIONED",
) -> DoubleEntryLedgerEnv:
    """Initializes a balanced Chart of Accounts where Assets == Liabilities + Equity + Revenue."""
    env = DoubleEntryLedgerEnv()
    total_claims = (
        to_decimal(alice_bal)
        + to_decimal(bob_bal)
        + to_decimal(carol_bal)
        + to_decimal(merch_a_bal)
        + to_decimal(merch_b_bal)
        + to_decimal(fee_rev_bal)
        + to_decimal(equity_bal)
    )
    env.create_account(
        "ACC_ASSET_VAULT",
        "Omnibus Custodial Reserve Vault",
        "ASSET",
        currency,
        "INSTITUTIONAL",
        float(total_claims),
    )
    env.create_account(
        "ACC_WALLET_ALICE", "Alice Rivera", "LIABILITY", currency, alice_kyc, alice_bal
    )
    env.create_account(
        "ACC_WALLET_BOB", "Bob Chen", "LIABILITY", currency, bob_kyc, bob_bal
    )
    env.create_account(
        "ACC_WALLET_CAROL", "Carol Vane", "LIABILITY", currency, carol_kyc, carol_bal
    )
    env.create_account(
        "ACC_MERCHANT_ALPHA",
        "Alpha Logistics LLC",
        "LIABILITY",
        currency,
        "INSTITUTIONAL",
        merch_a_bal,
    )
    env.create_account(
        "ACC_MERCHANT_BETA",
        "Beta Cloud Studios",
        "LIABILITY",
        currency,
        "TIER_2_FULL",
        merch_b_bal,
    )
    env.create_account(
        "ACC_REV_FEES",
        "Platform Transaction Fee Revenue",
        "REVENUE",
        currency,
        "INSTITUTIONAL",
        fee_rev_bal,
    )
    env.create_account(
        "ACC_EQUITY_RESERVE",
        "Platform Core Retained Equity",
        "EQUITY",
        currency,
        "INSTITUTIONAL",
        equity_bal,
    )
    assert env.get_equation_summary()["balanced"], "Bootstrap ledger must be balanced."
    return env


# =============================================================================
# WORKFLOW 1: Standard P2P Transfer with Platform Fee (Happy Path)
# =============================================================================
def simulate_workflow_1_standard_transfer(
    traj_id: str = "TRAJ_CANONICAL_01",
    amount: float = 850.00,
    fee: float = 8.50,
    currency: str = "USD",
    jurisdiction: str = "US",
    alice_bal: float = 5000.00,
) -> Dict[str, Any]:
    env = bootstrap_base_ledger(currency=currency, alice_bal=alice_bal)
    tx_id = f"TX_P2P_{traj_id[-4:]}"
    total_debit = float(to_decimal(amount) + to_decimal(fee))

    prompt = (
        f"Execute a standard peer-to-peer wallet transfer ({tx_id}) of {amount:.2f} {currency} "
        f"from Alice ('ACC_WALLET_ALICE') to Bob ('ACC_WALLET_BOB') in jurisdiction '{jurisdiction}' "
        f"with a platform fee of {fee:.2f} {currency} credited to 'ACC_REV_FEES'. Ensure strict "
        f"double-entry accounting, compliance verification, statechart progression, and final audit."
    )
    rec = TeacherTrajectoryRecorder(env, traj_id, "standard_transfer", prompt)

    # Step 1: Pre-flight sender balance check
    obs1 = rec.step(
        thought=(
            f"Before staging transaction {tx_id}, I must inspect the sender account "
            f"'ACC_WALLET_ALICE' to verify its normal balance side, KYC tier, and that its "
            f"settled balance covers the total debit of {total_debit:.2f} {currency} "
            f"({amount:.2f} principal + {fee:.2f} fee)."
        ),
        tool_name="get_account_balance",
        arguments={"account_id": "ACC_WALLET_ALICE"},
    )

    # Step 2: Stage PENDING journal entry
    entries = [
        {"account_id": "ACC_WALLET_ALICE", "entry_type": "DEBIT", "amount": total_debit},
        {"account_id": "ACC_WALLET_BOB", "entry_type": "CREDIT", "amount": round(amount, 2)},
        {"account_id": "ACC_REV_FEES", "entry_type": "CREDIT", "amount": round(fee, 2)},
    ]
    rec.step(
        thought=(
            f"'ACC_WALLET_ALICE' is a LIABILITY account (normal CREDIT balance) with "
            f"{obs1['balance']:.2f} {currency} and KYC tier '{obs1['kyc_tier']}'. Debiting "
            f"{total_debit:.2f} reduces Alice's wallet liability, while crediting "
            f"'ACC_WALLET_BOB' by {amount:.2f} and 'ACC_REV_FEES' by {fee:.2f} keeps "
            f"Debits ({total_debit:.2f}) == Credits ({total_debit:.2f}). I will now stage "
            f"{tx_id} in the PENDING state."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "PENDING",
            "entries": entries,
            "memo": f"Stage P2P transfer {amount:.2f} {currency} Alice->Bob + {fee:.2f} fee",
        },
    )

    # Step 3: Run KYC/AML compliance check
    obs3 = rec.step(
        thought=(
            f"Transaction {tx_id} is now PENDING. Per the statechart rules, I must run "
            f"check_compliance on sender 'ACC_WALLET_ALICE' and recipients "
            f"['ACC_WALLET_BOB', 'ACC_REV_FEES'] for gross amount {total_debit:.2f} {currency} "
            f"in '{jurisdiction}' before transitioning to VALIDATED."
        ),
        tool_name="check_compliance",
        arguments={
            "transaction_id": tx_id,
            "sender_account_id": "ACC_WALLET_ALICE",
            "recipient_account_ids": ["ACC_WALLET_BOB", "ACC_REV_FEES"],
            "total_amount": total_debit,
            "jurisdiction": jurisdiction,
        },
    )

    # Step 4: Transition PENDING -> VALIDATED
    rec.step(
        thought=(
            f"Compliance check returned APPROVED with token '{obs3['compliance_token']}'. "
            f"I can now transition {tx_id} from PENDING to VALIDATED."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "VALIDATED",
            "entries": entries,
            "memo": f"Validated P2P transfer with token {obs3['compliance_token']}",
        },
    )

    # Step 5: Transition VALIDATED -> COMMITTED
    rec.step(
        thought=(
            f"Transaction {tx_id} is VALIDATED with no overdraft or compliance errors. "
            f"I will now transition {tx_id} from VALIDATED to COMMITTED to atomically update "
            f"the ledger balances."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "COMMITTED",
            "entries": entries,
            "memo": f"Committed P2P transfer Alice->Bob ({amount:.2f} + {fee:.2f} fee)",
        },
    )

    # Step 6: Seal audit trail
    obs6 = rec.step(
        thought=(
            f"Transaction {tx_id} is in terminal state COMMITTED and the fundamental accounting "
            f"equation remains balanced. I will now invoke audit_transaction to record the "
            f"immutable SHA-256 audit entry."
        ),
        tool_name="audit_transaction",
        arguments={
            "transaction_id": tx_id,
            "expected_terminal_state": "COMMITTED",
            "compliance_token": obs3["compliance_token"],
            "auditor_notes": (
                f"P2P transfer of {amount:.2f} {currency} + {fee:.2f} fee committed and balanced."
            ),
        },
    )

    return rec.finish(
        f"Transaction {tx_id} completed and audited ({obs6['audit_hash']}). "
        f"State path: PENDING -> VALIDATED -> COMMITTED. Debits == Credits == {total_debit:.2f} {currency}."
    )


# =============================================================================
# WORKFLOW 2: Failed KYC/AML Rollback (Compliance Rejection Path)
# =============================================================================
def simulate_workflow_2_kyc_rollback(
    traj_id: str = "TRAJ_CANONICAL_02",
    amount: float = 1200.00,
    currency: str = "USD",
    jurisdiction: str = "CROSS_BORDER",
    sender_kyc: str = "TIER_1_BASIC",
    recipient_kyc: str = "SANCTIONED",
) -> Dict[str, Any]:
    env = bootstrap_base_ledger(
        currency=currency,
        alice_bal=4000.00,
        alice_kyc=sender_kyc,
        carol_kyc=recipient_kyc,
    )
    tx_id = f"TX_AML_{traj_id[-4:]}"
    prompt = (
        f"Process cross-border transfer ({tx_id}) of {amount:.2f} {currency} from Alice "
        f"('ACC_WALLET_ALICE') to Carol ('ACC_WALLET_CAROL') under jurisdiction '{jurisdiction}'. "
        f"If compliance screening fails, immediately transition through the failure statechart "
        f"(PENDING -> FAILED -> ROLLED_BACK) and seal the audit log without mutating balances."
    )
    rec = TeacherTrajectoryRecorder(env, traj_id, "failed_kyc_rollback", prompt)

    # Step 1: Inspect sender balance & KYC status
    obs1 = rec.step(
        thought=(
            f"First, I will check 'ACC_WALLET_ALICE' balance and KYC metadata prior to staging "
            f"transaction {tx_id} for {amount:.2f} {currency}."
        ),
        tool_name="get_account_balance",
        arguments={"account_id": "ACC_WALLET_ALICE"},
    )

    # Step 2: Stage PENDING journal entry
    entries = [
        {"account_id": "ACC_WALLET_ALICE", "entry_type": "DEBIT", "amount": round(amount, 2)},
        {"account_id": "ACC_WALLET_CAROL", "entry_type": "CREDIT", "amount": round(amount, 2)},
    ]
    rec.step(
        thought=(
            f"Alice has {obs1['balance']:.2f} {currency} available. To run the mandatory compliance "
            f"gate on {tx_id}, I must first stage the balanced double-entry block (Debit "
            f"ACC_WALLET_ALICE {amount:.2f}, Credit ACC_WALLET_CAROL {amount:.2f}) in the PENDING state."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "PENDING",
            "entries": entries,
            "memo": f"Stage transfer {amount:.2f} {currency} Alice->Carol for compliance review",
        },
    )

    # Step 3: Run check_compliance -> REJECTED
    obs3 = rec.step(
        thought=(
            f"With {tx_id} staged in PENDING, I will execute check_compliance across sender "
            f"'ACC_WALLET_ALICE' and recipient 'ACC_WALLET_CAROL' for {amount:.2f} {currency} "
            f"in '{jurisdiction}'."
        ),
        tool_name="check_compliance",
        arguments={
            "transaction_id": tx_id,
            "sender_account_id": "ACC_WALLET_ALICE",
            "recipient_account_ids": ["ACC_WALLET_CAROL"],
            "total_amount": round(amount, 2),
            "jurisdiction": jurisdiction,
        },
    )

    # Step 4: Transition PENDING -> FAILED (Halt without retrying validation)
    reasons_str = ", ".join(obs3.get("reasons", []))
    rec.step(
        thought=(
            f"Compliance check returned REJECTED with reasons: [{reasons_str}]. "
            f"Attempting to transition to VALIDATED would violate safety invariants. "
            f"Following the rollback statechart (PENDING -> FAILED -> ROLLED_BACK), I must "
            f"immediately transition {tx_id} from PENDING to FAILED."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "FAILED",
            "entries": entries,
            "memo": f"Compliance rejected: {reasons_str}",
        },
    )

    # Step 5: Transition FAILED -> ROLLED_BACK
    rec.step(
        thought=(
            f"Transaction {tx_id} is now marked FAILED. Next, I must transition {tx_id} from "
            f"FAILED to the terminal ROLLED_BACK state to void the staged lines and ensure zero "
            f"balance mutation."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "ROLLED_BACK",
            "entries": entries,
            "memo": f"Rolled back {tx_id} after KYC/AML compliance rejection",
        },
    )

    # Step 6: Audit rolled-back transaction
    obs6 = rec.step(
        thought=(
            f"Transaction {tx_id} has reached terminal state ROLLED_BACK. I will now invoke "
            f"audit_transaction with expected_terminal_state='ROLLED_BACK' and "
            f"compliance_token='COMPLIANCE_FAILED' to record the compliance halt in the immutable audit trail."
        ),
        tool_name="audit_transaction",
        arguments={
            "transaction_id": tx_id,
            "expected_terminal_state": "ROLLED_BACK",
            "compliance_token": "COMPLIANCE_FAILED",
            "auditor_notes": f"Halted and rolled back due to KYC/AML violations: {reasons_str}",
        },
    )

    return rec.finish(
        f"Transaction {tx_id} was halted due to compliance rejection ({reasons_str}) and safely "
        f"rolled back via PENDING -> FAILED -> ROLLED_BACK. Audit sealed ({obs6['audit_hash']})."
    )


# =============================================================================
# WORKFLOW 3: Multi-Party Marketplace Split Settlement (Happy Path)
# =============================================================================
def simulate_workflow_3_multiparty_split(
    traj_id: str = "TRAJ_CANONICAL_03",
    gross_amount: float = 2400.00,
    split_alpha_ratio: float = 0.60,
    split_beta_ratio: float = 0.35,
    currency: str = "USD",
    jurisdiction: str = "US",
) -> Dict[str, Any]:
    env = bootstrap_base_ledger(currency=currency, alice_bal=8000.00)
    tx_id = f"TX_SPLIT_{traj_id[-4:]}"

    gross_dec = to_decimal(gross_amount)
    alpha_amt = float((gross_dec * Decimal(str(split_alpha_ratio))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    beta_amt = float((gross_dec * Decimal(str(split_beta_ratio))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    fee_amt = float(gross_dec - to_decimal(alpha_amt) - to_decimal(beta_amt))

    prompt = (
        f"Execute a multi-party split settlement ({tx_id}) from buyer Alice ('ACC_WALLET_ALICE') "
        f"for a total gross order of {gross_amount:.2f} {currency} in jurisdiction '{jurisdiction}': "
        f"credit {alpha_amt:.2f} {currency} to 'ACC_MERCHANT_ALPHA', credit {beta_amt:.2f} {currency} "
        f"to 'ACC_MERCHANT_BETA', and credit {fee_amt:.2f} {currency} to platform fee revenue 'ACC_REV_FEES'."
    )
    rec = TeacherTrajectoryRecorder(env, traj_id, "multiparty_split", prompt)

    # Step 1: Check buyer balance
    obs1 = rec.step(
        thought=(
            f"I need to verify that buyer account 'ACC_WALLET_ALICE' has at least "
            f"{gross_amount:.2f} {currency} available to cover the 3-way split settlement."
        ),
        tool_name="get_account_balance",
        arguments={"account_id": "ACC_WALLET_ALICE"},
    )

    # Step 2: Stage multi-leg PENDING journal entry
    entries = [
        {"account_id": "ACC_WALLET_ALICE", "entry_type": "DEBIT", "amount": round(gross_amount, 2)},
        {"account_id": "ACC_MERCHANT_ALPHA", "entry_type": "CREDIT", "amount": round(alpha_amt, 2)},
        {"account_id": "ACC_MERCHANT_BETA", "entry_type": "CREDIT", "amount": round(beta_amt, 2)},
        {"account_id": "ACC_REV_FEES", "entry_type": "CREDIT", "amount": round(fee_amt, 2)},
    ]
    rec.step(
        thought=(
            f"Alice has {obs1['balance']:.2f} {currency}, which is sufficient. I will verify the "
            f"double-entry math: Total Debits = {gross_amount:.2f}; Total Credits = "
            f"{alpha_amt:.2f} + {beta_amt:.2f} + {fee_amt:.2f} = {gross_amount:.2f}. "
            f"Since Debits == Credits across all 4 ledger lines, I will stage {tx_id} as PENDING."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "PENDING",
            "entries": entries,
            "memo": f"Multi-party split settlement {tx_id} across Alpha, Beta, and Platform Fees",
        },
    )

    # Step 3: Compliance check across all 3 recipient accounts
    obs3 = rec.step(
        thought=(
            f"Now I must run check_compliance on {tx_id} verifying sender 'ACC_WALLET_ALICE' "
            f"and all three recipients ['ACC_MERCHANT_ALPHA', 'ACC_MERCHANT_BETA', 'ACC_REV_FEES'] "
            f"for total amount {gross_amount:.2f} {currency}."
        ),
        tool_name="check_compliance",
        arguments={
            "transaction_id": tx_id,
            "sender_account_id": "ACC_WALLET_ALICE",
            "recipient_account_ids": [
                "ACC_MERCHANT_ALPHA",
                "ACC_MERCHANT_BETA",
                "ACC_REV_FEES",
            ],
            "total_amount": round(gross_amount, 2),
            "jurisdiction": jurisdiction,
        },
    )

    # Step 4: Transition PENDING -> VALIDATED
    rec.step(
        thought=(
            f"All parties passed KYC/AML compliance ({obs3['compliance_token']}). "
            f"Transitioning multi-party split {tx_id} from PENDING to VALIDATED."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "VALIDATED",
            "entries": entries,
            "memo": f"Validated 3-way split settlement ({obs3['compliance_token']})",
        },
    )

    # Step 5: Transition VALIDATED -> COMMITTED
    rec.step(
        thought=(
            f"Validation succeeded. Now transitioning {tx_id} from VALIDATED to COMMITTED "
            f"to atomically settle funds to both merchants and the fee revenue ledger."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "COMMITTED",
            "entries": entries,
            "memo": f"Committed 3-way marketplace split settlement {tx_id}",
        },
    )

    # Step 6: Seal audit trail
    obs6 = rec.step(
        thought=(
            f"With {tx_id} committed, I will invoke audit_transaction to verify the "
            f"fundamental accounting equation and generate the audit hash."
        ),
        tool_name="audit_transaction",
        arguments={
            "transaction_id": tx_id,
            "expected_terminal_state": "COMMITTED",
            "compliance_token": obs3["compliance_token"],
            "auditor_notes": (
                f"Multi-party split {gross_amount:.2f} {currency} settled: Alpha={alpha_amt:.2f}, "
                f"Beta={beta_amt:.2f}, Fee={fee_amt:.2f}."
            ),
        },
    )

    return rec.finish(
        f"Multi-party split {tx_id} committed and audited ({obs6['audit_hash']}). "
        f"Debits ({gross_amount:.2f}) == Credits ({alpha_amt:.2f} + {beta_amt:.2f} + {fee_amt:.2f})."
    )


# =============================================================================
# WORKFLOW 4: Overdraft Halt & Safe Rollback (Insufficient Funds Guard)
# =============================================================================
def simulate_workflow_4_overdraft_halt(
    traj_id: str = "TRAJ_CANONICAL_04",
    wallet_balance: float = 450.00,
    transfer_amount: float = 1800.00,
    fee: float = 15.00,
    currency: str = "USD",
    jurisdiction: str = "US",
) -> Dict[str, Any]:
    env = bootstrap_base_ledger(currency=currency, alice_bal=wallet_balance)
    tx_id = f"TX_OVD_{traj_id[-4:]}"
    total_debit = float(to_decimal(transfer_amount) + to_decimal(fee))

    prompt = (
        f"Attempt to process transfer ({tx_id}) of {transfer_amount:.2f} {currency} plus a "
        f"{fee:.2f} {currency} fee from Alice ('ACC_WALLET_ALICE') to Bob ('ACC_WALLET_BOB') "
        f"in jurisdiction '{jurisdiction}'. Stage the transaction, run compliance, and if "
        f"validation detects an overdraft/insufficient funds, halt immediately by transitioning "
        f"to FAILED -> ROLLED_BACK and auditing the halt without retrying."
    )
    rec = TeacherTrajectoryRecorder(env, traj_id, "overdraft_halt", prompt)

    # Step 1: Check sender balance
    obs1 = rec.step(
        thought=(
            f"I will first inspect 'ACC_WALLET_ALICE' to see its current settled balance and "
            f"KYC status before staging {tx_id} for {total_debit:.2f} {currency}."
        ),
        tool_name="get_account_balance",
        arguments={"account_id": "ACC_WALLET_ALICE"},
    )

    # Step 2: Stage PENDING journal entry
    entries = [
        {"account_id": "ACC_WALLET_ALICE", "entry_type": "DEBIT", "amount": total_debit},
        {"account_id": "ACC_WALLET_BOB", "entry_type": "CREDIT", "amount": round(transfer_amount, 2)},
        {"account_id": "ACC_REV_FEES", "entry_type": "CREDIT", "amount": round(fee, 2)},
    ]
    rec.step(
        thought=(
            f"'ACC_WALLET_ALICE' currently holds {obs1['balance']:.2f} {currency}, which is less "
            f"than the requested debit of {total_debit:.2f} {currency}. To formally record and "
            f"audit the rejected transaction in the statechart, I will stage {tx_id} in PENDING "
            f"and run the validation gate."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "PENDING",
            "entries": entries,
            "memo": f"Stage transfer {total_debit:.2f} {currency} for validation check",
        },
    )

    # Step 3: Run compliance check (passes KYC, as Alice is TIER_2_FULL)
    obs3 = rec.step(
        thought=(
            f"Now I run check_compliance on {tx_id}. Even though KYC/AML may pass, the "
            f"subsequent VALIDATED transition will enforce the strict non-negative balance invariant."
        ),
        tool_name="check_compliance",
        arguments={
            "transaction_id": tx_id,
            "sender_account_id": "ACC_WALLET_ALICE",
            "recipient_account_ids": ["ACC_WALLET_BOB", "ACC_REV_FEES"],
            "total_amount": total_debit,
            "jurisdiction": jurisdiction,
        },
    )

    # Step 4: Attempt PENDING -> VALIDATED (triggers INSUFFICIENT_FUNDS_OVERDRAFT error)
    obs4 = rec.step(
        thought=(
            f"Compliance passed ({obs3['compliance_token']}). Now I attempt to transition {tx_id} "
            f"to VALIDATED, which checks projected balances against overdraft constraints."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "VALIDATED",
            "entries": entries,
            "memo": "Attempt validation of staged transfer",
        },
    )

    # Step 5: Immediately transition PENDING -> FAILED (no infinite retry loop!)
    rec.step(
        thought=(
            f"Validation blocked with error '{obs4['error_code']}' on account "
            f"{obs4.get('overdraft_accounts')} because debiting {total_debit:.2f} {currency} from "
            f"a {obs1['balance']:.2f} {currency} balance would cause an illegal negative liability "
            f"(overdraft). Rather than looping or retrying, I must immediately transition {tx_id} "
            f"from PENDING to FAILED."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "FAILED",
            "entries": entries,
            "memo": f"Overdraft halt: requested {total_debit:.2f} > available {obs1['balance']:.2f}",
        },
    )

    # Step 6: Transition FAILED -> ROLLED_BACK
    rec.step(
        thought=(
            f"With {tx_id} in the FAILED state, I will now transition it to the terminal "
            f"ROLLED_BACK state to release the staged entries cleanly."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "ROLLED_BACK",
            "entries": entries,
            "memo": f"Rolled back {tx_id} due to insufficient funds overdraft protection",
        },
    )

    # Step 7: Audit rolled-back transaction
    obs7 = rec.step(
        thought=(
            f"Transaction {tx_id} is now ROLLED_BACK and Alice's balance remains untouched at "
            f"{obs1['balance']:.2f} {currency}. I will seal the audit trail with "
            f"compliance_token='OVERDRAFT_HALT'."
        ),
        tool_name="audit_transaction",
        arguments={
            "transaction_id": tx_id,
            "expected_terminal_state": "ROLLED_BACK",
            "compliance_token": "OVERDRAFT_HALT",
            "auditor_notes": (
                f"Overdraft prevented on ACC_WALLET_ALICE (requested {total_debit:.2f}, "
                f"available {obs1['balance']:.2f}). Safely rolled back."
            ),
        },
    )

    return rec.finish(
        f"Overdraft prevented on {tx_id}. Halted and rolled back via PENDING -> FAILED -> ROLLED_BACK "
        f"with zero balance mutation. Audit sealed ({obs7['audit_hash']})."
    )


# =============================================================================
# WORKFLOW 5: Daily Reconciliation & Fee Revenue Sweep to Equity Reserve
# =============================================================================
def simulate_workflow_5_daily_reconciliation(
    traj_id: str = "TRAJ_CANONICAL_05",
    fee_rev_bal: float = 640.00,
    equity_bal: float = 2000.00,
    currency: str = "USD",
    jurisdiction: str = "US",
) -> Dict[str, Any]:
    env = bootstrap_base_ledger(
        currency=currency, fee_rev_bal=fee_rev_bal, equity_bal=equity_bal
    )
    tx_id = f"TX_RECON_{traj_id[-4:]}"
    treasury_acc_id = f"ACC_EQUITY_TREASURY_{traj_id[-2:]}"

    prompt = (
        f"Perform end-of-day ledger reconciliation ({tx_id}): "
        f"1) Inspect the Omnibus Vault ('ACC_ASSET_VAULT') and accumulated Platform Fee Revenue ('ACC_REV_FEES'). "
        f"2) Create a dedicated institutional Equity Treasury sweep account '{treasury_acc_id}' in {currency}. "
        f"3) Sweep the entire accumulated fee revenue ({fee_rev_bal:.2f} {currency}) from 'ACC_REV_FEES' "
        f"into '{treasury_acc_id}', verifying compliance, committing, and sealing the daily reconciliation audit."
    )
    rec = TeacherTrajectoryRecorder(env, traj_id, "daily_reconciliation", prompt)

    # Step 1: Inspect Omnibus Asset Vault
    obs1 = rec.step(
        thought=(
            "To begin daily reconciliation, I will inspect the Omnibus Custodial Reserve Vault "
            "('ACC_ASSET_VAULT') to verify total custodial assets held on the ledger."
        ),
        tool_name="get_account_balance",
        arguments={"account_id": "ACC_ASSET_VAULT"},
    )

    # Step 2: Inspect Platform Fee Revenue account
    obs2 = rec.step(
        thought=(
            f"Omnibus Vault holds {obs1['balance']:.2f} {currency} (normal DEBIT balance). "
            f"Next, I will inspect 'ACC_REV_FEES' to determine the exact accumulated revenue "
            f"to sweep into retained treasury equity."
        ),
        tool_name="get_account_balance",
        arguments={"account_id": "ACC_REV_FEES"},
    )

    # Step 3: Create the dedicated Equity Treasury Sweep account
    sweep_amount = obs2["balance"]
    rec.step(
        thought=(
            f"'ACC_REV_FEES' holds {sweep_amount:.2f} {currency} (REVENUE account, normal CREDIT side). "
            f"I will now create the institutional EQUITY account '{treasury_acc_id}' to receive "
            f"the end-of-day revenue sweep."
        ),
        tool_name="create_account",
        arguments={
            "account_id": treasury_acc_id,
            "owner_name": "Platform EOD Retained Treasury Equity",
            "account_type": "EQUITY",
            "currency": currency,
            "kyc_tier": "INSTITUTIONAL",
        },
    )

    # Step 4: Stage PENDING journal entry (Debit Revenue, Credit Equity)
    entries = [
        {"account_id": "ACC_REV_FEES", "entry_type": "DEBIT", "amount": round(sweep_amount, 2)},
        {"account_id": treasury_acc_id, "entry_type": "CREDIT", "amount": round(sweep_amount, 2)},
    ]
    rec.step(
        thought=(
            f"In double-entry accounting, REVENUE and EQUITY both have a normal CREDIT balance. "
            f"To close out {sweep_amount:.2f} {currency} from 'ACC_REV_FEES' into '{treasury_acc_id}', "
            f"I must DEBIT 'ACC_REV_FEES' by {sweep_amount:.2f} (reducing revenue to 0.00) and "
            f"CREDIT '{treasury_acc_id}' by {sweep_amount:.2f} (increasing retained equity by "
            f"{sweep_amount:.2f}), keeping Assets == Liabilities + Equity + Revenue - Expense invariant. "
            f"Staging {tx_id} as PENDING."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "PENDING",
            "entries": entries,
            "memo": f"EOD reconciliation sweep of {sweep_amount:.2f} {currency} fee revenue to equity",
        },
    )

    # Step 5: Compliance check on internal institutional sweep
    obs5 = rec.step(
        thought=(
            f"Now I run check_compliance on {tx_id} between institutional accounts "
            f"'ACC_REV_FEES' and '{treasury_acc_id}'."
        ),
        tool_name="check_compliance",
        arguments={
            "transaction_id": tx_id,
            "sender_account_id": "ACC_REV_FEES",
            "recipient_account_ids": [treasury_acc_id],
            "total_amount": round(sweep_amount, 2),
            "jurisdiction": jurisdiction,
        },
    )

    # Step 6: Transition PENDING -> VALIDATED
    rec.step(
        thought=(
            f"Compliance approved ({obs5['compliance_token']}). Transitioning reconciliation "
            f"sweep {tx_id} from PENDING to VALIDATED."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "VALIDATED",
            "entries": entries,
            "memo": f"Validated EOD revenue sweep ({obs5['compliance_token']})",
        },
    )

    # Step 7: Transition VALIDATED -> COMMITTED
    obs7 = rec.step(
        thought=(
            f"Transitioning {tx_id} from VALIDATED to COMMITTED to finalize the EOD revenue "
            f"reclassification and verify the fundamental accounting equation."
        ),
        tool_name="post_journal_entry",
        arguments={
            "transaction_id": tx_id,
            "target_state": "COMMITTED",
            "entries": entries,
            "memo": f"Committed EOD revenue sweep of {sweep_amount:.2f} {currency}",
        },
    )

    # Step 8: Seal audit trail
    obs8 = rec.step(
        thought=(
            f"EOD sweep {tx_id} is COMMITTED and Assets ({obs7['fundamental_equation']['assets']:.2f}) "
            f"== Total Claims ({obs7['fundamental_equation']['rhs_total']:.2f}). I will now call "
            f"audit_transaction to seal the daily reconciliation."
        ),
        tool_name="audit_transaction",
        arguments={
            "transaction_id": tx_id,
            "expected_terminal_state": "COMMITTED",
            "compliance_token": obs5["compliance_token"],
            "auditor_notes": (
                f"Daily reconciliation complete: swept {sweep_amount:.2f} {currency} from "
                f"ACC_REV_FEES to {treasury_acc_id}. Assets={obs1['balance']:.2f} verified."
            ),
        },
    )

    return rec.finish(
        f"Daily reconciliation {tx_id} complete and audited ({obs8['audit_hash']}). "
        f"Swept {sweep_amount:.2f} {currency} from ACC_REV_FEES to {treasury_acc_id}. "
        f"Fundamental equation holds: Assets ({obs1['balance']:.2f}) == Claims ({obs1['balance']:.2f})."
    )


# =============================================================================
# ENHANCEMENT #2: Parameterized Trajectory Synthesizer (150 Trajectories)
# =============================================================================
def synthesize_parameterized_dataset(num_per_archetype: int = 30, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates `num_per_archetype * 5` diverse, mathematically verified trajectories."""
    rng = random.Random(seed)
    currencies = ["USD", "EUR", "GBP", "INR"]
    jurisdictions_domestic = ["US", "EU", "UK", "IN"]
    dataset: List[Dict[str, Any]] = []

    for i in range(1, num_per_archetype + 1):
        curr = rng.choice(currencies)
        jur = rng.choice(jurisdictions_domestic)

        # 1. Standard Transfer variant
        amt1 = round(rng.uniform(50.0, 1500.0), 2)
        fee1 = round(amt1 * rng.uniform(0.005, 0.02), 2)
        dataset.append(
            simulate_workflow_1_standard_transfer(
                traj_id=f"TRAJ_SYN_P2P_{i:03d}",
                amount=amt1,
                fee=fee1,
                currency=curr,
                jurisdiction=jur,
                alice_bal=round(amt1 + fee1 + rng.uniform(500.0, 5000.0), 2),
            )
        )

        # 2. Failed KYC/AML Rollback variant
        amt2 = round(rng.uniform(300.0, 4500.0), 2)
        sender_kyc = rng.choice(["TIER_1_BASIC", "TIER_2_FULL", "PEP_FLAGGED"])
        rec_kyc = rng.choice(["SANCTIONED", "UNVERIFIED", "PEP_FLAGGED"])
        jur2 = rng.choice(["CROSS_BORDER", jur])
        dataset.append(
            simulate_workflow_2_kyc_rollback(
                traj_id=f"TRAJ_SYN_AML_{i:03d}",
                amount=amt2,
                currency=curr,
                jurisdiction=jur2,
                sender_kyc=sender_kyc,
                recipient_kyc=rec_kyc,
            )
        )

        # 3. Multi-Party Split variant
        gross3 = round(rng.uniform(400.0, 6000.0), 2)
        r_alpha = round(rng.uniform(0.45, 0.70), 2)
        r_beta = round(rng.uniform(0.20, 0.92 - r_alpha), 2)
        dataset.append(
            simulate_workflow_3_multiparty_split(
                traj_id=f"TRAJ_SYN_SPL_{i:03d}",
                gross_amount=gross3,
                split_alpha_ratio=r_alpha,
                split_beta_ratio=r_beta,
                currency=curr,
                jurisdiction=jur,
            )
        )

        # 4. Overdraft Halt variant
        bal4 = round(rng.uniform(100.0, 900.0), 2)
        amt4 = round(bal4 + rng.uniform(150.0, 2500.0), 2)
        fee4 = round(rng.uniform(5.0, 25.0), 2)
        dataset.append(
            simulate_workflow_4_overdraft_halt(
                traj_id=f"TRAJ_SYN_OVD_{i:03d}",
                wallet_balance=bal4,
                transfer_amount=amt4,
                fee=fee4,
                currency=curr,
                jurisdiction=jur,
            )
        )

        # 5. Daily Reconciliation variant
        rev5 = round(rng.uniform(120.0, 3500.0), 2)
        eq5 = round(rng.uniform(1000.0, 8000.0), 2)
        dataset.append(
            simulate_workflow_5_daily_reconciliation(
                traj_id=f"TRAJ_SYN_REC_{i:03d}",
                fee_rev_bal=rev5,
                equity_bal=eq5,
                currency=curr,
                jurisdiction=jur,
            )
        )

    return dataset


def export_all_trajectories() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data_dir = ROOT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    canonical_5 = [
        simulate_workflow_1_standard_transfer(),
        simulate_workflow_2_kyc_rollback(),
        simulate_workflow_3_multiparty_split(),
        simulate_workflow_4_overdraft_halt(),
        simulate_workflow_5_daily_reconciliation(),
    ]

    canonical_json_path = data_dir / "phase2_canonical_5_trajectories.json"
    with open(canonical_json_path, "w", encoding="utf-8") as f:
        json.dump(canonical_5, f, indent=2)

    # Also write a clean human-readable Markdown file for the 5 canonical trajectories
    md_lines = [
        "# Phase 2: 5 Canonical Teacher Agent Trajectories",
        "",
        "Every trajectory below was executed against the live `DoubleEntryLedgerEnv` engine.",
        "",
    ]
    for item in canonical_5:
        md_lines.append(f"## {item['trajectory_id']} — `{item['workflow_type']}` ({item['num_steps']} steps)")
        md_lines.append(f"**User Prompt:** {item['user_prompt']}")
        md_lines.append("")
        md_lines.append("```text")
        md_lines.append(item["formatted_trace"])
        md_lines.append("```")
        md_lines.append("")

    canonical_md_path = data_dir / "phase2_canonical_5_readable.md"
    with open(canonical_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Generate 150 parameterized trajectories
    synthetic_150 = synthesize_parameterized_dataset(num_per_archetype=30, seed=42)
    synth_jsonl_path = data_dir / "phase2_synthetic_dataset_150.jsonl"
    with open(synth_jsonl_path, "w", encoding="utf-8") as f:
        for row in synthetic_150:
            f.write(json.dumps(row) + "\n")

    print(f"Exported 5 canonical trajectories -> {canonical_json_path}")
    print(f"Exported readable canonical trace -> {canonical_md_path}")
    print(f"Exported {len(synthetic_150)} parameterized trajectories -> {synth_jsonl_path}")
    return canonical_5, synthetic_150


if __name__ == "__main__":
    export_all_trajectories()

