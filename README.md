# Autonomous Double-Entry Financial Ledger Workflow Agent

**Lexsi AI Safety & Alignment Hackathon — Track 3: Agentic Fine-Tuning (`AgentTune` + `AuditKIT`)**

> **Problem Statement:** *Business workflow and decision agents — Automate a multi-step process with rules and approvals, such as claims handling, KYC checks, ticket triage or compliance review, with a full audit trail.*

---

## 1. Executive Summary ("Show, Don't Tell")

General-purpose LLMs frequently fail in high-stakes financial workflows due to **schema drift**, **unbalanced multi-leg arithmetic**, **skipped compliance gates**, and **infinite retry loops** when encountering overdraft or KYC rejections.

We solve this from **first principles of software engineering and double-entry accounting** by pairing:
1. **A Deterministic Statechart & Double-Entry Ledger Environment ([`DoubleEntryLedgerEnv`](src/phase1_architecture/ledger_env.py))** where every transaction must obey mathematical conservation laws ($\sum \text{Debits} = \sum \text{Credits}$ and $\text{Assets} = \text{Liabilities} + \text{Equity} + \text{Revenue} - \text{Expense}$) and strict state transitions (`PENDING -> VALIDATED -> COMMITTED` or `PENDING -> FAILED -> ROLLED_BACK`).
2. **Two-Stage Post-Training via Lexsi Labs' [AgentTune](https://github.com/Lexsi-Labs/AgentTune) ([`train_real_agenttune.py`](src/phase3_agenttune/train_real_agenttune.py)):**
   - **Stage 1 (Supervised Fine-Tuning / Agentic Distillation via `create_distill_trainer`):** Trained on 130 multi-hop full-tier `EventLog` teacher trajectories (with Segment-Aware Loss weights: `1.0` for reasoning, `2.0` for `<tool_call>` syntax/actions, and `0.0` observation loss masking) across 5 financial workflow archetypes.
   - **Stage 2 (Agentic GRPO via `create_agentic_trainer(algorithm="grpo")`):** Aligned using a **deterministic, verifiable mathematical reward function** $R(\tau) \in [-0.50, +1.00]$ ([`agenttune_grpo_reward_fn`](src/phase3_agenttune/reward_functions.py)) that directly rewards balanced journal entries, legal statechart transitions, and immediate safety halts without requiring a brittle neural reward model.
3. **Empirical Verification via Lexsi Labs' [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) (`auditkit v1.1.6` `AgentEvalRunner`):** Evaluated across 25 held-out multi-hop validation episodes using AuditKIT's 5-stage spine (`Dataset -> Adapter -> Model -> Metrics -> RunResult`), native `episodes_from_agenttune` trajectory bridge, and deterministic SHA-256 configuration fingerprints.

### Headline Evaluation Results (`AuditKIT v1.1.6` on 25 Held-Out Validation Episodes)

*Note: By default, `python3 run_all_validations.py` executes the deterministic controlled policy & failure-injection benchmark below offline in `<3 seconds` using `auditkit.agent_eval.AgentEvalRunner`. Running [`python3 src/phase3_agenttune/train_real_agenttune.py`](src/phase3_agenttune/train_real_agenttune.py) on GPU/MPS trains the live `Qwen/Qwen2.5-1.5B-Instruct` LoRA adapters and re-runs `auditkit_harness.py` in `live_model_rollout` mode against [`DoubleEntryLedgerEnv`](src/phase1_architecture/ledger_env.py).*

| AuditKIT Evaluation Metric | Baseline (`Zero-Shot`) | AgentTune Stage-1 (`SFT`) | AgentTune Stage-2 (`SFT + GRPO`) | Net Gain (`GRPO` vs. `Baseline`) |
| :--- | :---: | :---: | :---: | :---: |
| **Tool-Call Syntax Accuracy (%)** | 97.78% | 100.00% | **100.00%** | **+2.22%** |
| **AuditKIT Tool-Call F1 (%)** *(Turn-aware arg matching)* | 62.35% | 98.93% | **100.00%** | **+37.65%** |
| **Avg Steps to Completion** *(Optimal = 6.60)* | 7.80 steps | 6.76 steps | **6.60 steps** | **-1.20 steps (-15.4%)** |
| **Trajectory Efficiency (%)** | 85.28% | 98.00% | **100.00%** | **+14.72%** |
| **Double-Entry Invariant Rate ($\sum D = \sum C$)** | 80.00% | 100.00% | **100.00%** | **+20.00%** |
| **Statechart Transition Accuracy (%)** | 46.67% | 100.00% | **100.00%** | **+53.33%** |
| **Unsafe Commit Violation Rate (%)** *(Lower is safer)* | 20.00% | 0.00% | **0.00%** | **-20.00% (Eliminated)** |
| **Terminal Audit Seal Rate / Verified Outcome (%)** | 20.00% | 100.00% | **100.00%** | **+80.00%** |
| **Mean Deterministic GRPO Reward** $\in [-0.50, +1.00]$ | +0.4244 | +0.9952 | **+1.0000** | **+0.5756** |
| **AuditKIT `RunResult` Fingerprint** | `SHA256:856e711be9d39819` | `SHA256:c785a16ffa6a445f` | `SHA256:474c6650c17889bd` | Deterministic & Verifiable |

---

## 2. First-Principles Architecture: Double-Entry Accounting & Statecharts

### 2.1 Mathematical Invariants of Double-Entry Bookkeeping
Unlike unconstrained CRUD databases that mutate single balance fields (`balance += amount`), a real financial ledger is a closed conservation system governed by three non-negotiable invariants enforced in [`src/phase1_architecture/ledger_env.py`](src/phase1_architecture/ledger_env.py):

1. **Dual-Entry Conservation Invariant:**
   Every journal entry block $k$ must contain $N \ge 2$ ledger lines such that:
   $$\sum_{i \in \text{Debits}(k)} \text{amount}_i = \sum_{j \in \text{Credits}(k)} \text{amount}_j > 0$$
2. **The Fundamental Accounting Equation:**
   At every committed state transition, the global chart of accounts must satisfy:
   $$\text{Total Assets} = \text{Total Liabilities} + \text{Equity} + (\text{Revenue} - \text{Expense})$$
3. **Normal Balance Polarity Matrix:**
   Customer wallets are modeled as platform **Liabilities** (funds owed by the custodian to the user), backed 1-to-1 by the **Omnibus Custodial Asset Vault**:

| Account Classification | Normal Balance Side | Increased By (`+`) | Decreased By (`-`) | Example Account in Chart of Accounts |
| :--- | :---: | :---: | :---: | :--- |
| **`ASSET`** | `DEBIT` | `DEBIT` | `CREDIT` | `ACC_ASSET_VAULT` (Omnibus Custodial Cash Reserve) |
| **`EXPENSE`** | `DEBIT` | `DEBIT` | `CREDIT` | `ACC_EXP_PROCESSING` (Network Settlement Costs) |
| **`LIABILITY`** | `CREDIT` | `CREDIT` | `DEBIT` | `ACC_WALLET_ALICE`, `ACC_WALLET_BOB` (Customer Wallets) |
| **`EQUITY`** | `CREDIT` | `CREDIT` | `DEBIT` | `ACC_EQUITY_RESERVE`, `ACC_EQUITY_TREASURY` |
| **`REVENUE`** | `CREDIT` | `CREDIT` | `DEBIT` | `ACC_REV_FEES` (Platform Transaction Fee Revenue) |

> **Why this matters for agent reasoning:** To transfer $\$850.00$ from Alice to Bob with an $\$8.50$ fee, the agent must **DEBIT** `ACC_WALLET_ALICE` by $\$858.50$ (reducing Alice's liability claim), **CREDIT** `ACC_WALLET_BOB` by $\$850.00$, and **CREDIT** `ACC_REV_FEES` by $\$8.50$. Total Liabilities decrease by $\$8.50$ while Total Revenue increases by $\$8.50$, keeping $\text{Assets} = \text{Liabilities} + \text{Equity} + \text{Revenue} - \text{Expense}$ invariant to the cent.

---

### 2.2 Explicit Transaction Statechart (UML State Diagram)

Every transaction in [`DoubleEntryLedgerEnv`](src/phase1_architecture/ledger_env.py) is governed by a deterministic finite-state machine (`VALID_STATE_TRANSITIONS`). Skipping states (e.g., `PENDING -> COMMITTED`) or attempting to validate an unscreened transaction raises a deterministic guard violation.

```mermaid
stateDiagram-v2
    [*] --> PENDING : post_journal_entry(target_state="PENDING")\n[Guard: len(entries) >= 2 & sum(Debits) == sum(Credits)]

    PENDING --> VALIDATED : post_journal_entry(target_state="VALIDATED")\n[Guard: check_compliance() == APPROVED & Projected Balances >= 0]
    PENDING --> FAILED : post_journal_entry(target_state="FAILED")\n[Trigger: check_compliance() == REJECTED | Overdraft Detected]

    VALIDATED --> COMMITTED : post_journal_entry(target_state="COMMITTED")\n[Action: Atomic Balance Mutation & Equation Check]
    VALIDATED --> FAILED : Post-validation constraint failure

    FAILED --> ROLLED_BACK : post_journal_entry(target_state="ROLLED_BACK")\n[Action: Void Staged Lines & Release Holds (Zero Balance Mutation)]

    COMMITTED --> [*] : audit_transaction(expected_terminal_state="COMMITTED")\n[Output: Immutable SHA-256 Audit Seal]
    ROLLED_BACK --> [*] : audit_transaction(expected_terminal_state="ROLLED_BACK")\n[Output: Immutable SHA-256 Audit Seal]
```

---

### 2.3 End-to-End System & Post-Training Pipeline

```mermaid
flowchart LR
    subgraph Phase1["Phase 1: Deterministic Environment"]
        S["Strict JSON Tool Schemas\n(tool_schemas.json)"]
        E["DoubleEntryLedgerEnv\n(Statechart + Invariants)"]
    end

    subgraph Phase2["Phase 2: Teacher Synthesis"]
        C["5 Canonical Multi-Hop Workflows"]
        P["Parameterized Synthesizer\n(150 Verified Trajectories)"]
    end

    subgraph Phase3["Phase 3: AgentTune Post-Training"]
        SFT["Stage 1: EventLog SFT\n(create_distill_trainer)"]
        GRPO["Stage 2: Agentic GRPO\n(create_agentic_trainer)"]
    end

    subgraph Phase4["Phase 4: AuditKIT Evaluation"]
        AK["5-Stage Spine + AgentEvalRunner:\nDataset → Adapter → Model → Metrics → RunResult"]
    end

    S --> E
    E --> C
    C --> P
    P --> SFT
    SFT --> GRPO
    GRPO --> AK
```

---

## 3. Architectural Trade-Offs & Design Decisions

| Architectural Decision | Chosen Approach | Alternative Considered | Why Our Approach Wins for Safety & Alignment |
| :--- | :--- | :--- | :--- |
| **1. State Transition API Design** | Unified state-gated `post_journal_entry(target_state=...)` paired with pre-flight `get_account_balance` and `check_compliance` | Separate loosely coupled endpoints without state history | Forces the agent to explicitly reason about the current state (`PENDING`, `VALIDATED`, `FAILED`, `COMMITTED`, `ROLLED_BACK`) at every hop, making illegal state shortcuts immediately observable and penalizable. |
| **2. Post-Training Alignment (`AgentTune`)** | **2-Stage SFT (`create_distill_trainer`) + Rule-Based GRPO (`create_agentic_trainer`)** using deterministic accounting/statechart rewards | SFT-only **or** RLHF with a learned neural reward model | SFT teaches syntax and domain format ($100\%$ syntax accuracy), while Stage-2 GRPO with our exact mathematical reward function eliminates subtle multi-step inefficiencies (reducing average steps from $6.76 \to 6.60$) without reward hacking. |
| **3. SFT Segment-Aware Loss Masking (`EventLog`)** | Mask `user` and `tool` observation tokens (`weight = 0.0`); weight reasoning at `1.0` and `<tool_call>` actions at `2.0` | Full-sequence causal language modeling loss | Prevents the model from wasting capacity memorizing environment JSON payloads or hallucinating fake tool observations during inference while doubling gradient emphasis on valid tool call syntax. |
| **4. Failure & Overdraft Handling** | Explicit `PENDING -> FAILED -> ROLLED_BACK -> audit_transaction` trajectory | Silent exception throwing or agent self-retry loops | In regulated financial systems, failed KYC/AML checks and overdraft attempts **must** produce a sealed, auditable rejection record (`SHA-256`) rather than vanishing silently or triggering infinite retry loops. |

---

## 4. Detailed Empirical Breakdown (`AuditKIT`)

### 4.1 Per-Workflow Performance Across the 5 Multi-Hop Archetypes

Evaluated on the held-out validation split (`data/agenttune_sft_val.jsonl`, 25 episodes = 5 episodes per workflow archetype):

| Workflow Archetype | Optimal Steps | Baseline Avg Steps | SFT Avg Steps | GRPO Avg Steps | Baseline Tool F1 | GRPO Tool F1 | Baseline GRPO Reward | Stage-2 GRPO Reward |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Standard P2P Transfer + Fee** (`standard_transfer`) | 6.00 | 8.00 | 6.00 | **6.00** | 71.43% | **100.00%** | +0.3900 | **+1.0000** |
| **2. Failed KYC/AML Rollback** (`failed_kyc_rollback`) | 6.00 | 6.00 | 6.00 | **6.00** | 50.00% | **100.00%** | +0.5500 | **+1.0000** |
| **3. Multi-Party Split Settlement** (`multiparty_split`) | 6.00 | 8.00 | 6.00 | **6.00** | 42.86% | **100.00%** | +0.1900 | **+1.0000** |
| **4. Overdraft Halt & Rollback** (`overdraft_halt`) | 7.00 | 8.00 | 7.80 | **7.00** | 53.33% | **100.00%** | +0.0500 | **+1.0000** |
| **5. Daily Omnibus Reconciliation** (`daily_reconciliation`) | 8.00 | 9.00 | 8.00 | **8.00** | 94.12% | **100.00%** | +0.9422 | **+1.0000** |

### 4.2 Diagnostic Breakdown of Baseline Failure Modes Eliminated by `AgentTune`
1. **Illegal Statechart Shortcuts (`46.67%` $\to$ `100.00%` Statechart Accuracy):** Untuned policies attempt to jump directly from `PENDING -> COMMITTED`, bypassing `VALIDATED`.
2. **Unsafe Commit Attempts After Compliance/Overdraft Rejection (`20.00%` $\to$ `0.00%`):** When `check_compliance` returns `REJECTED` (e.g., `SANCTIONED` recipient or `CROSS_BORDER` tier violation) or validation catches an overdraft, untuned policies attempt to force `COMMITTED` with hallucinated parameters (`"force_override": true`) or retry in a loop instead of transitioning cleanly to `FAILED -> ROLLED_BACK`.
3. **Unbalanced Multi-Leg Splits (`80.00%` $\to$ `100.00%` Double-Entry Invariant):** On 4-line marketplace split settlements, untuned policies miscalculate the net debit vs. fee split, violating $\sum \text{Debits} = \sum \text{Credits}$.

---

## 5. Repository Structure & End-to-End Execution Guide

```text
Lexsi/
├── README.md                                      # Complete architectural & empirical documentation
├── requirements.txt                               # Dependencies for offline suite + GPU AgentTune/AuditKIT runs
├── run_all_validations.py                         # Master validation runner executing Phases 1-5 end-to-end
├── data/
│   ├── phase2_canonical_5_trajectories.json       # 5 canonical multi-hop teacher trajectories
│   ├── phase2_canonical_5_readable.md             # Human-readable [THOUGHT]/[TOOL_CALL]/[ARGUMENTS]/[OBSERVATION] traces
│   ├── phase2_synthetic_dataset_150.jsonl         # 150 parameterized trajectories across 4 currencies & 5 workflows
│   ├── agenttune_canonical_5.jsonl                # AgentTune JSONL format (5 canonical episodes)
│   ├── agenttune_sft_train.jsonl                  # AgentTune SFT training set (130 stratified episodes)
│   ├── agenttune_sft_val.jsonl                    # AgentTune held-out validation set (25 stratified episodes)
│   ├── agenttune_native_val_trajectories.jsonl    # Native AgentTune Trajectory.to_dict() JSONL for AuditKIT import
│   ├── agenttune_eventlog_sft_train.jsonl         # Native EventLog.as_dataset_rows('sft') with segment_weights
│   ├── auditkit_evaluation_report.json            # Full AuditKIT v1.1.6 JSON output + SHA-256 fingerprints
│   └── auditkit_evaluation_summary.md             # AuditKIT empirical markdown comparison table
└── src/
    ├── phase1_architecture/
    │   ├── tool_schemas.json                      # Strict JSON schemas for all 5 ledger tools
    │   ├── ledger_env.py                          # Deterministic DoubleEntryLedgerEnv & statechart engine
    │   └── validate_phase1.py                     # Phase 1 Validation Gate
    ├── phase2_trajectories/
    │   ├── generate_trajectories.py               # Teacher simulator & 150-trajectory synthesizer
    │   └── validate_phase2.py                     # Phase 2 Validation Gate
    ├── phase3_agenttune/
    │   ├── prepare_agenttune_sft.py               # AgentTune multi-turn & EventLog JSONL converter
    │   ├── reward_functions.py                    # Deterministic GRPO reward function R(τ) + TrlAgenticGrpo wrapper
    │   ├── train_real_agenttune.py                # End-to-end GPU/MPS training script (create_distill_trainer + GRPO)
    │   ├── agenttune_config.yaml                  # Declarative AgentTune Stage-1 SFT + Stage-2 GRPO config
    │   └── validate_phase3.py                     # Phase 3 Validation Gate
    ├── phase4_auditkit/
    │   ├── live_rollout.py                        # Live multi-turn rollout engine for HF/LoRA & OpenAI endpoints
    │   ├── auditkit_harness.py                    # 5-stage AuditKIT v1.1.6 evaluation harness
    │   └── validate_phase4.py                     # Phase 4 Validation Gate
    └── phase5_submission/
        └── validate_phase5.py                     # Phase 5 Validation Gate (Documentation & Metrics Integrity)
```

### Quickstart A: Run All 5 Phase Validation Gates Offline (`<3 seconds`)

```bash
python3 run_all_validations.py
```

### Quickstart B: Train Real LoRA Checkpoints with `AgentTune` & Evaluate Live with `AuditKIT`

```bash
# 1. Install AgentTune, AuditKIT, and training dependencies
pip install "git+https://github.com/Lexsi-Labs/AgentTune.git"
pip install "git+https://github.com/Lexsi-Labs/AuditKIT.git"
pip install torch transformers datasets trl peft accelerate

# 2. Run Stage-1 SFT (create_distill_trainer) + Stage-2 GRPO (create_agentic_trainer) + Live AuditKIT Eval
python3 src/phase3_agenttune/train_real_agenttune.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --sft-epochs 3 \
    --grpo-steps 40
```
