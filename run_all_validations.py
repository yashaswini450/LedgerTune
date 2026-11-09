"""Master End-to-End Validation Runner for the Lexsi Track 3 Hackathon Submission.

Executes all 5 Phase Validation Gates in order:
  - Phase 1: Environment & Architecture Definition (`validate_phase1.py`)
  - Phase 2: Synthetic Trajectory Generation (`validate_phase2.py`)
  - Phase 3: AgentTune Post-Training Prep & GRPO Reward (`validate_phase3.py`)
  - Phase 4: AuditKIT 5-Stage Evaluation Benchmark (`validate_phase4.py`)
  - Phase 5: Submission Packaging & Documentation Verification (`validate_phase5.py`)
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent

VALIDATION_SCRIPTS = [
    ("Phase 1: Environment & Architecture Definition", "src/phase1_architecture/validate_phase1.py"),
    ("Phase 2: Synthetic Trajectory Generation", "src/phase2_trajectories/validate_phase2.py"),
    ("Phase 3: AgentTune Post-Training Prep", "src/phase3_agenttune/validate_phase3.py"),
    ("Phase 4: AuditKIT Evaluation Scripting", "src/phase4_auditkit/validate_phase4.py"),
    ("Phase 5: Submission Packaging", "src/phase5_submission/validate_phase5.py"),
]


def main() -> None:
    print("=" * 78)
    print("LEXSI AI SAFETY & ALIGNMENT HACKATHON (TRACK 3: AGENTIC FINE-TUNING)")
    print("Executing Full 5-Phase Validation Suite...")
    print("=" * 78)

    for phase_title, rel_script in VALIDATION_SCRIPTS:
        print(f"\n>>> RUNNING {phase_title} ({rel_script})")
        script_path = ROOT_DIR / rel_script
        res = subprocess.run([sys.executable, str(script_path)], check=False)
        if res.returncode != 0:
            print(f"\n[FAIL] {phase_title} exited with code {res.returncode}")
            sys.exit(res.returncode)

    print("\n" + "=" * 78)
    print("ALL 5 PHASES & VALIDATION GATES PASSED (100% READY FOR SUBMISSION)")
    print("=" * 78)


if __name__ == "__main__":
    main()

