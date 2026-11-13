"""Uploads all LedgerTune Models, Adapters, Logs, Datasets, and AuditKIT Reports to Hugging Face Hub
and automatically creates the two public Hugging Face Collections required for submission:
  1. Hugging Face Model Collection (models, LoRA adapters, training logs, provenance)
  2. Hugging Face Dataset Collection (datasets, EventLog SFT splits, GRPO traces, AuditKIT reports)

Usage (on Colab or local machine after `huggingface_cli login` or setting `HF_TOKEN`):
    python3 src/phase5_submission/upload_to_hf.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

ROOT_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload LedgerTune artifacts & create HF Collections")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face WRITE token (or set HF_TOKEN)")
    parser.add_argument("--model-repo-name", type=str, default="ledgertune-tiny-aya-global-agent")
    parser.add_argument("--dataset-repo-name", type=str, default="ledgertune-double-entry-ledger-dataset")
    args = parser.parse_args()

    from huggingface_hub import HfApi, create_collection

    api = HfApi(token=args.token)
    user_info = api.whoami()
    username = user_info["name"]
    print(f"Authenticated on Hugging Face as: {username}")

    model_repo_id = f"{username}/{args.model_repo_name}"
    dataset_repo_id = f"{username}/{args.dataset_repo_name}"

    # -------------------------------------------------------------------------
    # 1. Create & Upload Public Model Repository (Adapters + Checkpoints + Logs)
    # -------------------------------------------------------------------------
    print(f"\n[1/3] Creating & uploading public Model Repo -> https://huggingface.co/{model_repo_id}")
    api.create_repo(repo_id=model_repo_id, repo_type="model", private=False, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_model_dir:
        tmp_m = Path(tmp_model_dir)
        # Copy README.md as Model Card with YAML frontmatter
        readme_body = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
        model_card = (
            "---\n"
            "library_name: peft\n"
            "base_model:\n"
            "- CohereLabs/tiny-aya-global\n"
            "- Qwen/Qwen2.5-1.5B-Instruct\n"
            "tags:\n"
            "- agenttune\n"
            "- auditkit\n"
            "- double-entry-ledger\n"
            "- tool-calling\n"
            "- grpo\n"
            "- sft\n"
            "---\n\n"
            + readme_body
        )
        (tmp_m / "README.md").write_text(model_card, encoding="utf-8")

        ckpt_src = ROOT_DIR / "checkpoints"
        if ckpt_src.exists():
            # Copy all adapters (.safetensors), configs, trainer_state, provenance, and rollout logs
            # Skip intermediate *_merged full base model copies so upload is fast
            for item in ckpt_src.iterdir():
                if item.is_dir() and not item.name.endswith("_merged"):
                    shutil.copytree(item, tmp_m / "checkpoints" / item.name, dirs_exist_ok=True)

        api.upload_folder(
            folder_path=str(tmp_m),
            repo_id=model_repo_id,
            repo_type="model",
            commit_message="Upload LedgerTune SFT & GRPO adapters, trainer logs, and Lexsi provenance",
        )
    print(f"  [OK] Uploaded Model Repo: https://huggingface.co/{model_repo_id}")

    # -------------------------------------------------------------------------
    # 2. Create & Upload Public Dataset Repository (Datasets + Traces + Reports)
    # -------------------------------------------------------------------------
    print(f"\n[2/3] Creating & uploading public Dataset Repo -> https://huggingface.co/{dataset_repo_id}")
    api.create_repo(repo_id=dataset_repo_id, repo_type="dataset", private=False, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_data_dir:
        tmp_d = Path(tmp_data_dir)
        readme_body = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
        dataset_card = (
            "---\n"
            "task_categories:\n"
            "- text-generation\n"
            "language:\n"
            "- en\n"
            "tags:\n"
            "- agenttune\n"
            "- auditkit\n"
            "- financial-ledger\n"
            "- tool-calling\n"
            "- synthetic-trajectories\n"
            "---\n\n"
            + readme_body
        )
        (tmp_d / "README.md").write_text(dataset_card, encoding="utf-8")

        data_src = ROOT_DIR / "data"
        if data_src.exists():
            shutil.copytree(data_src, tmp_d / "data", dirs_exist_ok=True)

        # Also include the GRPO rollout trajectory traces in the dataset repo
        for trace_dir in ("grpo_tiny_aya", "grpo_ledger"):
            trace_file = ROOT_DIR / "checkpoints" / trace_dir / "trajectories.jsonl"
            if trace_file.exists():
                (tmp_d / "traces").mkdir(exist_ok=True)
                shutil.copy2(trace_file, tmp_d / "traces" / f"{trace_dir}_trajectories.jsonl")

        api.upload_folder(
            folder_path=str(tmp_d),
            repo_id=dataset_repo_id,
            repo_type="dataset",
            commit_message="Upload LedgerTune synthetic datasets, EventLog splits, GRPO traces, and AuditKIT reports",
        )
    print(f"  [OK] Uploaded Dataset Repo: https://huggingface.co/{dataset_repo_id}")

    # -------------------------------------------------------------------------
    # 3. Create Public Hugging Face Model & Dataset Collections
    # -------------------------------------------------------------------------
    print("\n[3/3] Creating Public Hugging Face Collections...")
    try:
        model_col = create_collection(
            title="LedgerTune - Double-Entry Ledger Agent Models & Adapters",
            namespace=username,
            description="Fine-tuned Stage-1 SFT and Stage-2 GRPO LoRA adapters, training logs, and provenance for the Double-Entry Financial Ledger Workflow Agent (AgentTune + AuditKIT).",
            private=False,
            token=args.token,
            exists_ok=True,
        )
        api.add_collection_item(
            collection_slug=model_col.slug,
            item_id=model_repo_id,
            item_type="model",
            exists_ok=True,
        )
        api.add_collection_item(
            collection_slug=model_col.slug,
            item_id="CohereLabs/tiny-aya-global",
            item_type="model",
            exists_ok=True,
        )
        model_col_url = model_col.url
    except Exception as exc:  # noqa: BLE001
        print(f"  [Note] Could not auto-create model collection ({exc}); use repo URL directly.")
        model_col_url = f"https://huggingface.co/{model_repo_id}"

    try:
        data_col = create_collection(
            title="LedgerTune - Double-Entry Ledger Datasets, Traces & Benchmarks",
            namespace=username,
            description="150 verified double-entry trajectories, EventLog SFT splits, GRPO rollout traces, and AuditKIT evaluation reports.",
            private=False,
            token=args.token,
            exists_ok=True,
        )
        api.add_collection_item(
            collection_slug=data_col.slug,
            item_id=dataset_repo_id,
            item_type="dataset",
            exists_ok=True,
        )
        data_col_url = data_col.url
    except Exception as exc:  # noqa: BLE001
        print(f"  [Note] Could not auto-create dataset collection ({exc}); use repo URL directly.")
        data_col_url = f"https://huggingface.co/datasets/{dataset_repo_id}"

    print("\n" + "=" * 78)
    print("HUGGING FACE UPLOAD COMPLETE! COPY THESE LINKS FOR YOUR SUBMISSION FORM:")
    print(f"  * HF Model Repo:         https://huggingface.co/{model_repo_id}")
    print(f"  * HF Dataset Repo:       https://huggingface.co/datasets/{dataset_repo_id}")
    print(f"  * HF Model Collection:   {model_col_url}")
    print(f"  * HF Dataset Collection: {data_col_url}")
    print("=" * 78)


if __name__ == "__main__":
    main()

