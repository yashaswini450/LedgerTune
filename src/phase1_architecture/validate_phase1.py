"""Phase 1 Validation Gate: Verifies JSON Tool Schemas and Statechart Invariants."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ledger_env import DoubleEntryLedgerEnv, TxState, VALID_STATE_TRANSITIONS, load_tool_schemas


def validate_tool_schemas() -> None:
    print("=== [1/3] Validating JSON Tool Schemas ===")
    schemas = load_tool_schemas()
    assert isinstance(schemas, list) and len(schemas) == 5, "Expected 5 tool schemas."

    expected_names = {
        "create_account",
        "get_account_balance",
        "check_compliance",
        "post_journal_entry",
        "audit_transaction",
    }
    found_names = set()
    for tool in schemas:
        assert tool.get("type") == "function", "Each schema must have type='function'."
        fn = tool.get("function", {})
        name = fn.get("name")
        found_names.add(name)
        assert fn.get("strict") is True, f"Tool '{name}' must have strict=True."
        params = fn.get("parameters", {})
        assert params.get("type") == "object", f"Tool '{name}' parameters must be an object."
        assert params.get("additionalProperties") is False, (
            f"Tool '{name}' must forbid additionalProperties."
        )
        assert "required" in params and "properties" in params, (
            f"Tool '{name}' must define properties and required fields."
        )
        print(f"  [PASS] Tool Schema Verified: {name} (required={params['required']})")

    assert found_names == expected_names, f"Missing expected tools: {expected_names - found_names}"


def validate_happy_path_and_equation() -> None:
    print("\n=== [2/3] Validating Statechart Happy Path (PENDING -> VALIDATED -> COMMITTED) ===")
    env = DoubleEntryLedgerEnv()

    # Bootstrap Chart of Accounts adhering to Assets = Liabilities + Equity
    # Platform Vault Cash (ASSET) = $10,000.00
    # Alice Wallet (LIABILITY)   = $6,000.00
    # Bob Wallet (LIABILITY)     = $3,000.00
    # Platform Equity (EQUITY)   = $1,000.00
    # Platform Fee Revenue (REVENUE) = $0.00
    env.create_account("ACC_ASSET_VAULT_USD", "Platform Omnibus Vault", "ASSET", "USD", "INSTITUTIONAL", 10000.00)
    env.create_account("ACC_WALLET_ALICE", "Alice Rivera", "LIABILITY", "USD", "TIER_2_FULL", 6000.00)
    env.create_account("ACC_WALLET_BOB", "Bob Chen", "LIABILITY", "USD", "TIER_2_FULL", 3000.00)
    env.create_account("ACC_EQUITY_RESERVE", "Platform Core Equity", "EQUITY", "USD", "INSTITUTIONAL", 1000.00)
    env.create_account("ACC_REV_FEES", "Transaction Fee Revenue", "REVENUE", "USD", "INSTITUTIONAL", 0.00)

    eq_init = env.get_equation_summary()
    assert eq_init["balanced"] is True, f"Initial equation unbalanced: {eq_init}"
    print(f"  [PASS] Initial Fundamental Equation Balanced: Assets=${eq_init['assets']} == Claims=${eq_init['rhs_total']}")

    # Alice transfers $500 to Bob with a $5 platform fee (Total Debit to Alice = $505, Credit Bob = $500, Credit Fee = $5)
    entries = [
        {"account_id": "ACC_WALLET_ALICE", "entry_type": "DEBIT", "amount": 505.00},
        {"account_id": "ACC_WALLET_BOB", "entry_type": "CREDIT", "amount": 500.00},
        {"account_id": "ACC_REV_FEES", "entry_type": "CREDIT", "amount": 5.00},
    ]

    # Step 1: Stage PENDING
    r1 = env.post_journal_entry("TX_HAPPY_001", "PENDING", entries, "P2P Transfer Alice -> Bob with fee")
    assert r1["status"] == "SUCCESS" and r1["state"] == "PENDING"

    # Illegal shortcut test: PENDING -> COMMITTED must fail
    r_illegal = env.post_journal_entry("TX_HAPPY_001", "COMMITTED", entries, "Try skipping validation")
    assert r_illegal["status"] == "ERROR" and r_illegal["error_code"] == "ILLEGAL_STATE_TRANSITION"
    print("  [PASS] Guarded against illegal shortcut (PENDING -> COMMITTED blocked)")

    # Step 2: Compliance Check
    r_comp = env.check_compliance("TX_HAPPY_001", "ACC_WALLET_ALICE", ["ACC_WALLET_BOB", "ACC_REV_FEES"], 505.00, "US")
    assert r_comp["status"] == "APPROVED"

    # Step 3: Transition PENDING -> VALIDATED
    r2 = env.post_journal_entry("TX_HAPPY_001", "VALIDATED", entries, "Validated after KYC check")
    assert r2["status"] == "SUCCESS" and r2["state"] == "VALIDATED"

    # Step 4: Transition VALIDATED -> COMMITTED
    r3 = env.post_journal_entry("TX_HAPPY_001", "COMMITTED", entries, "Committed to ledger")
    assert r3["status"] == "SUCCESS" and r3["state"] == "COMMITTED"
    assert r3["state_history"] == ["PENDING", "VALIDATED", "COMMITTED"]

    # Step 5: Audit Transaction
    r_audit = env.audit_transaction("TX_HAPPY_001", "COMMITTED", r_comp["compliance_token"], "Verified P2P transfer")
    assert r_audit["status"] == "AUDITED" and r_audit["fundamental_equation_balanced"] is True
    print(f"  [PASS] Happy Path Trajectory Verified: {r_audit['state_history']} | Hash={r_audit['audit_hash']}")


def validate_rollback_and_failure_paths() -> None:
    print("\n=== [3/3] Validating Statechart Failure & Rollback Path (PENDING -> FAILED -> ROLLED_BACK) ===")
    env = DoubleEntryLedgerEnv()
    env.create_account("ACC_ASSET_VAULT_USD", "Platform Omnibus Vault", "ASSET", "USD", "INSTITUTIONAL", 5000.00)
    env.create_account("ACC_WALLET_CAROL", "Carol Danver", "LIABILITY", "USD", "TIER_1_BASIC", 3000.00)
    env.create_account("ACC_WALLET_SANCTIONED", "Offshore Shell LLC", "LIABILITY", "USD", "SANCTIONED", 2000.00)

    entries = [
        {"account_id": "ACC_WALLET_CAROL", "entry_type": "DEBIT", "amount": 1500.00},
        {"account_id": "ACC_WALLET_SANCTIONED", "entry_type": "CREDIT", "amount": 1500.00},
    ]

    # Step 1: Stage PENDING
    r1 = env.post_journal_entry("TX_FAIL_001", "PENDING", entries, "Transfer to sanctioned entity")
    assert r1["status"] == "SUCCESS" and r1["state"] == "PENDING"

    # Step 2: Check Compliance (Fails due to SANCTIONED recipient + velocity limit > $1000 for TIER_1_BASIC)
    r_comp = env.check_compliance("TX_FAIL_001", "ACC_WALLET_CAROL", ["ACC_WALLET_SANCTIONED"], 1500.00, "US")
    assert r_comp["status"] == "REJECTED" and r_comp["required_next_state"] == "FAILED"

    # Guard check: Trying to transition PENDING -> VALIDATED when compliance failed must error
    r_bad_val = env.post_journal_entry("TX_FAIL_001", "VALIDATED", entries, "Force validate")
    assert r_bad_val["status"] == "ERROR" and r_bad_val["error_code"] == "COMPLIANCE_GATE_NOT_PASSED"
    print("  [PASS] Guarded against unauthorized validation after failed compliance")

    # Step 3: Transition PENDING -> FAILED
    r2 = env.post_journal_entry("TX_FAIL_001", "FAILED", entries, "Halted due to sanctions and velocity limit")
    assert r2["status"] == "SUCCESS" and r2["state"] == "FAILED"

    # Step 4: Transition FAILED -> ROLLED_BACK
    r3 = env.post_journal_entry("TX_FAIL_001", "ROLLED_BACK", entries, "Rolled back staged entries")
    assert r3["status"] == "SUCCESS" and r3["state"] == "ROLLED_BACK"
    assert r3["state_history"] == ["PENDING", "FAILED", "ROLLED_BACK"]

    # Step 5: Audit Rollback
    r_audit = env.audit_transaction("TX_FAIL_001", "ROLLED_BACK", "COMPLIANCE_FAILED", "Halted & rolled back safely")
    assert r_audit["status"] == "AUDITED"
    assert env.get_account_balance("ACC_WALLET_CAROL")["balance"] == 3000.00
    print(f"  [PASS] Failure & Rollback Path Verified: {r_audit['state_history']} | Hash={r_audit['audit_hash']}")


if __name__ == "__main__":
    validate_tool_schemas()
    validate_happy_path_and_equation()
    validate_rollback_and_failure_paths()
    print("\nALL PHASE 1 VALIDATION CHECKS PASSED PERFECTLY!")

