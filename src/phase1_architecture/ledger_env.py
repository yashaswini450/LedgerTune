"""Phase 1: Double-Entry Financial Ledger Environment & Statechart Engine.

Adheres strictly to:
1. Double-Entry Invariant: Every transaction has >= 2 entries with sum(DEBIT) == sum(CREDIT).
2. Fundamental Accounting Equation: Total Assets = Total Liabilities + Equity + (Revenue - Expense).
3. Explicit Statechart:
   - Happy Path: [*] -> PENDING -> VALIDATED -> COMMITTED
   - Failure Path: [*] -> PENDING -> FAILED -> ROLLED_BACK
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class AccountType(str, Enum):
    ASSET = "ASSET"
    LIABILITY = "LIABILITY"
    EQUITY = "EQUITY"
    REVENUE = "REVENUE"
    EXPENSE = "EXPENSE"


# Normal balance sides in standard double-entry bookkeeping:
# ASSET & EXPENSE accounts increase with DEBIT and decrease with CREDIT.
# LIABILITY, EQUITY & REVENUE accounts increase with CREDIT and decrease with DEBIT.
DEBIT_NORMAL_TYPES: Set[AccountType] = {AccountType.ASSET, AccountType.EXPENSE}
CREDIT_NORMAL_TYPES: Set[AccountType] = {
    AccountType.LIABILITY,
    AccountType.EQUITY,
    AccountType.REVENUE,
}


class TxState(str, Enum):
    PENDING = "PENDING"
    VALIDATED = "VALIDATED"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


VALID_STATE_TRANSITIONS: Dict[Optional[TxState], Set[TxState]] = {
    None: {TxState.PENDING},
    TxState.PENDING: {TxState.VALIDATED, TxState.FAILED},
    TxState.VALIDATED: {TxState.COMMITTED, TxState.FAILED},
    TxState.FAILED: {TxState.ROLLED_BACK},
    TxState.COMMITTED: set(),
    TxState.ROLLED_BACK: set(),
}

KYC_VELOCITY_LIMITS: Dict[str, Decimal] = {
    "UNVERIFIED": Decimal("0.00"),
    "SANCTIONED": Decimal("0.00"),
    "PEP_FLAGGED": Decimal("0.00"),
    "TIER_1_BASIC": Decimal("1000.00"),
    "TIER_2_FULL": Decimal("50000.00"),
    "INSTITUTIONAL": Decimal("10000000.00"),
}


def to_decimal(val: float | int | str | Decimal) -> Decimal:
    return Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@dataclass
class Account:
    account_id: str
    owner_name: str
    account_type: AccountType
    currency: str
    kyc_tier: str
    balance: Decimal = Decimal("0.00")

    @property
    def normal_side(self) -> str:
        return "DEBIT" if self.account_type in DEBIT_NORMAL_TYPES else "CREDIT"

    def compute_projected_balance(self, entry_type: str, amount: Decimal) -> Decimal:
        """Calculates resulting balance after applying a DEBIT or CREDIT."""
        if self.normal_side == entry_type:
            return self.balance + amount
        return self.balance - amount


@dataclass
class TransactionRecord:
    transaction_id: str
    state: TxState
    entries: List[Dict[str, Any]]
    memo: str
    state_history: List[str] = field(default_factory=list)
    compliance_checked: bool = False
    compliance_passed: bool = False
    compliance_token: str = ""
    failure_reason: Optional[str] = None
    audited: bool = False
    audit_hash: Optional[str] = None


class DoubleEntryLedgerEnv:
    """Deterministic Mock Environment for the Digital Wallet Financial Ledger API."""

    def __init__(self) -> None:
        self.accounts: Dict[str, Account] = {}
        self.transactions: Dict[str, TransactionRecord] = {}
        self.audit_log: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Fundamental Accounting Invariant Verification
    # -------------------------------------------------------------------------
    def get_equation_summary(self) -> Dict[str, Any]:
        """Computes Assets = Liabilities + Equity + (Revenue - Expense)."""
        totals: Dict[AccountType, Decimal] = {t: Decimal("0.00") for t in AccountType}
        for acc in self.accounts.values():
            totals[acc.account_type] += acc.balance

        lhs_assets = totals[AccountType.ASSET]
        rhs_claims = (
            totals[AccountType.LIABILITY]
            + totals[AccountType.EQUITY]
            + totals[AccountType.REVENUE]
            - totals[AccountType.EXPENSE]
        )
        return {
            "assets": float(lhs_assets),
            "liabilities": float(totals[AccountType.LIABILITY]),
            "equity": float(totals[AccountType.EQUITY]),
            "revenue": float(totals[AccountType.REVENUE]),
            "expense": float(totals[AccountType.EXPENSE]),
            "rhs_total": float(rhs_claims),
            "balanced": lhs_assets == rhs_claims,
        }

    # -------------------------------------------------------------------------
    # Tool 1: create_account
    # -------------------------------------------------------------------------
    def create_account(
        self,
        account_id: str,
        owner_name: str,
        account_type: str,
        currency: str,
        kyc_tier: str,
        initial_balance: float = 0.0,
    ) -> Dict[str, Any]:
        if account_id in self.accounts:
            return {
                "status": "ERROR",
                "error_code": "ACCOUNT_ALREADY_EXISTS",
                "message": f"Account '{account_id}' already exists in the chart of accounts.",
            }

        try:
            acc_type = AccountType(account_type)
        except ValueError:
            return {
                "status": "ERROR",
                "error_code": "INVALID_ACCOUNT_TYPE",
                "message": f"Unsupported account_type '{account_type}'.",
            }

        acc = Account(
            account_id=account_id,
            owner_name=owner_name,
            account_type=acc_type,
            currency=currency,
            kyc_tier=kyc_tier,
            balance=to_decimal(initial_balance),
        )
        self.accounts[account_id] = acc
        return {
            "status": "SUCCESS",
            "account_id": acc.account_id,
            "owner_name": acc.owner_name,
            "account_type": acc.account_type.value,
            "normal_balance_side": acc.normal_side,
            "currency": acc.currency,
            "kyc_tier": acc.kyc_tier,
            "balance": float(acc.balance),
        }

    # -------------------------------------------------------------------------
    # Tool 2: get_account_balance
    # -------------------------------------------------------------------------
    def get_account_balance(self, account_id: str) -> Dict[str, Any]:
        acc = self.accounts.get(account_id)
        if not acc:
            return {
                "status": "ERROR",
                "error_code": "ACCOUNT_NOT_FOUND",
                "message": f"Account '{account_id}' does not exist.",
            }
        return {
            "status": "SUCCESS",
            "account_id": acc.account_id,
            "owner_name": acc.owner_name,
            "account_type": acc.account_type.value,
            "normal_balance_side": acc.normal_side,
            "currency": acc.currency,
            "kyc_tier": acc.kyc_tier,
            "balance": float(acc.balance),
            "velocity_limit": float(KYC_VELOCITY_LIMITS.get(acc.kyc_tier, Decimal("0.00"))),
        }

    # -------------------------------------------------------------------------
    # Tool 3: check_compliance
    # -------------------------------------------------------------------------
    def check_compliance(
        self,
        transaction_id: str,
        sender_account_id: str,
        recipient_account_ids: List[str],
        total_amount: float,
        jurisdiction: str,
    ) -> Dict[str, Any]:
        tx = self.transactions.get(transaction_id)
        if not tx:
            return {
                "status": "ERROR",
                "error_code": "TRANSACTION_NOT_STAGED",
                "message": f"Transaction '{transaction_id}' must be staged in PENDING state before compliance check.",
            }

        if tx.state != TxState.PENDING:
            return {
                "status": "ERROR",
                "error_code": "INVALID_STATE_FOR_COMPLIANCE",
                "message": f"Compliance check only valid in PENDING state; current state is {tx.state.value}.",
            }

        sender = self.accounts.get(sender_account_id)
        if not sender:
            return {
                "status": "ERROR",
                "error_code": "SENDER_NOT_FOUND",
                "message": f"Sender account '{sender_account_id}' not found.",
            }

        amt = to_decimal(total_amount)
        reasons: List[str] = []

        # Check sender KYC / sanctions status
        if sender.kyc_tier in {"UNVERIFIED", "SANCTIONED", "PEP_FLAGGED"}:
            reasons.append(f"SENDER_KYC_BLOCKED:{sender.kyc_tier}")

        limit = KYC_VELOCITY_LIMITS.get(sender.kyc_tier, Decimal("0.00"))
        if amt > limit:
            reasons.append(
                f"VELOCITY_LIMIT_EXCEEDED:amount={amt}>limit={limit}_for_{sender.kyc_tier}"
            )

        # Check recipients
        for r_id in recipient_account_ids:
            rec = self.accounts.get(r_id)
            if not rec:
                reasons.append(f"RECIPIENT_NOT_FOUND:{r_id}")
            elif rec.kyc_tier in {"UNVERIFIED", "SANCTIONED", "PEP_FLAGGED"}:
                reasons.append(f"RECIPIENT_KYC_BLOCKED:{r_id}:{rec.kyc_tier}")

        # Cross-border restriction for Tier 1 Basic accounts
        if jurisdiction == "CROSS_BORDER" and sender.kyc_tier == "TIER_1_BASIC":
            reasons.append("CROSS_BORDER_REQUIRES_TIER_2_FULL_OR_INSTITUTIONAL")

        tx.compliance_checked = True
        if reasons:
            tx.compliance_passed = False
            tx.compliance_token = "COMPLIANCE_FAILED"
            tx.failure_reason = "; ".join(reasons)
            return {
                "status": "REJECTED",
                "transaction_id": transaction_id,
                "compliance_passed": False,
                "compliance_token": "COMPLIANCE_FAILED",
                "reasons": reasons,
                "required_next_state": "FAILED",
            }

        token_digest = hashlib.sha256(
            f"{transaction_id}:{sender_account_id}:{amt}:{jurisdiction}".encode()
        ).hexdigest()[:12].upper()
        tx.compliance_passed = True
        tx.compliance_token = f"CMP_OK_{token_digest}"
        return {
            "status": "APPROVED",
            "transaction_id": transaction_id,
            "compliance_passed": True,
            "compliance_token": tx.compliance_token,
            "jurisdiction": jurisdiction,
            "required_next_state": "VALIDATED",
        }

    # -------------------------------------------------------------------------
    # Tool 4: post_journal_entry
    # -------------------------------------------------------------------------
    def post_journal_entry(
        self,
        transaction_id: str,
        target_state: str,
        entries: List[Dict[str, Any]],
        memo: str,
    ) -> Dict[str, Any]:
        try:
            desired_state = TxState(target_state)
        except ValueError:
            return {
                "status": "ERROR",
                "error_code": "INVALID_TARGET_STATE",
                "message": f"Unknown state '{target_state}'.",
            }

        existing_tx = self.transactions.get(transaction_id)
        current_state = existing_tx.state if existing_tx else None

        # 1. Verify Statechart Transition Validity
        allowed_next = VALID_STATE_TRANSITIONS.get(current_state, set())
        if desired_state not in allowed_next:
            return {
                "status": "ERROR",
                "error_code": "ILLEGAL_STATE_TRANSITION",
                "current_state": current_state.value if current_state else "NONE",
                "attempted_state": desired_state.value,
                "allowed_next_states": sorted(s.value for s in allowed_next),
                "message": (
                    f"Statechart violation: Cannot transition '{transaction_id}' from "
                    f"{current_state.value if current_state else 'NONE'} to {desired_state.value}."
                ),
            }

        # 2. Verify Double-Entry Structure (At least 2 lines, Debits == Credits)
        if len(entries) < 2:
            return {
                "status": "ERROR",
                "error_code": "SINGLE_ENTRY_VIOLATION",
                "message": "Every double-entry transaction must contain at least 2 ledger lines.",
            }

        total_debits = Decimal("0.00")
        total_credits = Decimal("0.00")
        for line in entries:
            amt = to_decimal(line["amount"])
            if amt <= Decimal("0.00"):
                return {
                    "status": "ERROR",
                    "error_code": "NON_POSITIVE_AMOUNT",
                    "message": f"Entry amount must be strictly positive, got {amt}.",
                }
            if line["account_id"] not in self.accounts:
                return {
                    "status": "ERROR",
                    "error_code": "ACCOUNT_NOT_FOUND",
                    "message": f"Account '{line['account_id']}' does not exist.",
                }
            if line["entry_type"] == "DEBIT":
                total_debits += amt
            elif line["entry_type"] == "CREDIT":
                total_credits += amt
            else:
                return {
                    "status": "ERROR",
                    "error_code": "INVALID_ENTRY_TYPE",
                    "message": f"Unknown entry_type '{line['entry_type']}'.",
                }

        is_balanced = total_debits == total_credits

        # 3. Handle State-Specific Guard Conditions
        if desired_state == TxState.PENDING:
            if not is_balanced:
                return {
                    "status": "ERROR",
                    "error_code": "UNBALANCED_JOURNAL_ENTRY",
                    "total_debits": float(total_debits),
                    "total_credits": float(total_credits),
                    "message": f"Debits ({total_debits}) != Credits ({total_credits}). Cannot stage.",
                }
            tx = TransactionRecord(
                transaction_id=transaction_id,
                state=TxState.PENDING,
                entries=entries,
                memo=memo,
                state_history=[TxState.PENDING.value],
            )
            self.transactions[transaction_id] = tx
            return {
                "status": "SUCCESS",
                "transaction_id": transaction_id,
                "state": tx.state.value,
                "state_history": tx.state_history,
                "total_debits": float(total_debits),
                "total_credits": float(total_credits),
                "message": "Transaction staged in PENDING state. Run check_compliance before VALIDATED.",
            }

        assert existing_tx is not None

        if desired_state == TxState.VALIDATED:
            if not is_balanced:
                return {
                    "status": "ERROR",
                    "error_code": "UNBALANCED_JOURNAL_ENTRY",
                    "message": "Debits must equal Credits to validate.",
                }
            if not existing_tx.compliance_checked or not existing_tx.compliance_passed:
                return {
                    "status": "ERROR",
                    "error_code": "COMPLIANCE_GATE_NOT_PASSED",
                    "message": "Cannot transition to VALIDATED without an APPROVED check_compliance result.",
                }

            # Check overdraft / negative balance constraints across all affected accounts
            projected_map: Dict[str, Decimal] = {
                acc_id: acc.balance for acc_id, acc in self.accounts.items()
            }
            for line in entries:
                acc = self.accounts[line["account_id"]]
                amt = to_decimal(line["amount"])
                if acc.normal_side == line["entry_type"]:
                    projected_map[acc.account_id] += amt
                else:
                    projected_map[acc.account_id] -= amt

            overdraft_accounts = [
                acc_id for acc_id, bal in projected_map.items() if bal < Decimal("0.00")
            ]
            if overdraft_accounts:
                existing_tx.failure_reason = f"OVERDRAFT_DETECTED:{','.join(overdraft_accounts)}"
                return {
                    "status": "ERROR",
                    "error_code": "INSUFFICIENT_FUNDS_OVERDRAFT",
                    "overdraft_accounts": overdraft_accounts,
                    "required_next_state": "FAILED",
                    "message": (
                        f"Overdraft prevented on {overdraft_accounts}. Transition transaction "
                        "to FAILED and then ROLLED_BACK."
                    ),
                }

            existing_tx.state = TxState.VALIDATED
            existing_tx.entries = entries
            existing_tx.memo = memo
            existing_tx.state_history.append(TxState.VALIDATED.value)
            return {
                "status": "SUCCESS",
                "transaction_id": transaction_id,
                "state": existing_tx.state.value,
                "state_history": existing_tx.state_history,
                "message": "Transaction VALIDATED. Ready to transition to COMMITTED.",
            }

        if desired_state == TxState.COMMITTED:
            # Apply atomic balance updates
            for line in existing_tx.entries:
                acc = self.accounts[line["account_id"]]
                amt = to_decimal(line["amount"])
                acc.balance = acc.compute_projected_balance(line["entry_type"], amt)

            eq = self.get_equation_summary()
            if not eq["balanced"]:
                return {
                    "status": "ERROR",
                    "error_code": "FUNDAMENTAL_EQUATION_VIOLATION",
                    "equation": eq,
                    "message": "Assets != Liabilities + Equity after commit.",
                }

            existing_tx.state = TxState.COMMITTED
            existing_tx.memo = memo
            existing_tx.state_history.append(TxState.COMMITTED.value)
            return {
                "status": "SUCCESS",
                "transaction_id": transaction_id,
                "state": existing_tx.state.value,
                "state_history": existing_tx.state_history,
                "total_debits": float(total_debits),
                "total_credits": float(total_credits),
                "fundamental_equation": eq,
                "message": "Transaction COMMITTED atomically. Call audit_transaction to seal.",
            }

        if desired_state == TxState.FAILED:
            existing_tx.state = TxState.FAILED
            existing_tx.memo = memo
            if not existing_tx.failure_reason:
                existing_tx.failure_reason = memo
            existing_tx.state_history.append(TxState.FAILED.value)
            return {
                "status": "SUCCESS",
                "transaction_id": transaction_id,
                "state": existing_tx.state.value,
                "state_history": existing_tx.state_history,
                "failure_reason": existing_tx.failure_reason,
                "required_next_state": "ROLLED_BACK",
                "message": "Transaction marked FAILED. Transition to ROLLED_BACK to release holds.",
            }

        if desired_state == TxState.ROLLED_BACK:
            existing_tx.state = TxState.ROLLED_BACK
            existing_tx.memo = memo
            existing_tx.state_history.append(TxState.ROLLED_BACK.value)
            return {
                "status": "SUCCESS",
                "transaction_id": transaction_id,
                "state": existing_tx.state.value,
                "state_history": existing_tx.state_history,
                "message": "Transaction ROLLED_BACK safely. Zero balances mutated. Call audit_transaction to seal.",
            }

        return {"status": "ERROR", "message": "Unhandled state transition."}

    # -------------------------------------------------------------------------
    # Tool 5: audit_transaction
    # -------------------------------------------------------------------------
    def audit_transaction(
        self,
        transaction_id: str,
        expected_terminal_state: str,
        compliance_token: str,
        auditor_notes: str,
    ) -> Dict[str, Any]:
        tx = self.transactions.get(transaction_id)
        if not tx:
            return {
                "status": "ERROR",
                "error_code": "TRANSACTION_NOT_FOUND",
                "message": f"Transaction '{transaction_id}' does not exist.",
            }

        if tx.state not in {TxState.COMMITTED, TxState.ROLLED_BACK}:
            return {
                "status": "ERROR",
                "error_code": "NON_TERMINAL_STATE_AUDIT",
                "current_state": tx.state.value,
                "message": f"Cannot seal audit for non-terminal state '{tx.state.value}'. Must be COMMITTED or ROLLED_BACK.",
            }

        if tx.state.value != expected_terminal_state:
            return {
                "status": "ERROR",
                "error_code": "TERMINAL_STATE_MISMATCH",
                "actual_state": tx.state.value,
                "expected_state": expected_terminal_state,
            }

        eq = self.get_equation_summary()
        total_debits = sum(
            to_decimal(line["amount"]) for line in tx.entries if line["entry_type"] == "DEBIT"
        )
        total_credits = sum(
            to_decimal(line["amount"]) for line in tx.entries if line["entry_type"] == "CREDIT"
        )

        payload = json.dumps(
            {
                "tx_id": transaction_id,
                "state_history": tx.state_history,
                "debits": str(total_debits),
                "credits": str(total_credits),
                "compliance_token": compliance_token,
                "equation_balanced": eq["balanced"],
            },
            sort_keys=True,
        )
        audit_hash = hashlib.sha256(payload.encode()).hexdigest()
        tx.audited = True
        tx.audit_hash = audit_hash

        record = {
            "status": "AUDITED",
            "transaction_id": transaction_id,
            "terminal_state": tx.state.value,
            "state_history": tx.state_history,
            "debits_equal_credits": total_debits == total_credits,
            "fundamental_equation_balanced": eq["balanced"],
            "compliance_token": compliance_token,
            "audit_hash": f"SHA256:{audit_hash[:16]}",
            "auditor_notes": auditor_notes,
        }
        self.audit_log.append(record)
        return record


def load_tool_schemas() -> List[Dict[str, Any]]:
    schema_path = Path(__file__).parent / "tool_schemas.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

