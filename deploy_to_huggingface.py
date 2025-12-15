#!/usr/bin/env python3
"""
Deploy ODRV2 to Hugging Face Spaces.

Steps:
1. Create a Hugging Face account at https://huggingface.co/join
2. Create an access token at https://huggingface.co/settings/tokens
3. Run: huggingface-cli login
4. Run this script: python deploy_to_huggingface.py

This will:
- Create a model repository for the weights
- Create a Space for the Gradio app
- Upload everything
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SPACE_DIR = PROJECT_ROOT / 'huggingface_space'
MODEL_DIR = PROJECT_ROOT / 'models' / 'unified_v3_retrain'
FOLDS = [0, 1, 4]

# Change these to your username
HF_USERNAME = "fdbprojects"
MODEL_REPO = f"{HF_USERNAME}/odrv2-models"
SPACE_REPO = f"{HF_USERNAME}/odrv2"


def check_hf_cli():
    """Check if huggingface-cli is installed and logged in."""
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], capture_output=True, text=True)
        if result.returncode == 0:
            username = result.stdout.strip()
            print(f"✓ Logged in as: {username}")
            return username
        else:
            print("❌ Not logged in to Hugging Face")
            print("Run: huggingface-cli login")
            return None
    except FileNotFoundError:
        print("❌ huggingface-cli not found")
        print("Run: pip install huggingface_hub")
        return None


def create_model_repo():
    """Create and upload model weights to HF Hub."""
    from huggingface_hub import HfApi, create_repo
    
    api = HfApi()
    
    # Create repo
    try:
        create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
        print(f"✓ Created model repo: {MODEL_REPO}")
    except Exception as e:
        print(f"Repo may already exist: {e}")
    
    # Upload model files
    for fold in FOLDS:
        model_path = MODEL_DIR / f'fold_{fold}' / 'best_model.pth'
        if model_path.exists():
            print(f"Uploading fold {fold} model (~350MB)...")
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=f"fold_{fold}/best_model.pth",
                repo_id=MODEL_REPO,
                repo_type="model"
            )
            print(f"✓ Uploaded fold {fold}")
        else:
            print(f"⚠️ Model not found: {model_path}")


def create_space():
    """Create and deploy the Gradio Space."""
    from huggingface_hub import HfApi, create_repo, upload_folder
    
    api = HfApi()
    
    # Create Space
    try:
        create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", exist_ok=True)
        print(f"✓ Created Space: {SPACE_REPO}")
    except Exception as e:
        print(f"Space may already exist: {e}")
    
    # Update app.py to use correct model repo
    app_path = SPACE_DIR / 'app.py'
    content = app_path.read_text()
    content = content.replace('fdbadmin/odrv2-models', MODEL_REPO)
    app_path.write_text(content)
    
    # Upload Space files
    print("Uploading Space files...")
    upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=SPACE_REPO,
        repo_type="space"
    )
    print(f"✓ Deployed to: https://huggingface.co/spaces/{SPACE_REPO}")


def main():
    print("=" * 50)
    print("ODRV2 Hugging Face Deployment")
    print("=" * 50)
    
    if HF_USERNAME == "YOUR_USERNAME":
        print("\n❌ Please edit this script and set HF_USERNAME to your Hugging Face username")
        sys.exit(1)
    
    username = check_hf_cli()
    if not username:
        print("\nPlease login first:")
        print("  pip install huggingface_hub")
        print("  huggingface-cli login")
        sys.exit(1)
    
    print(f"\nWill deploy to:")
    print(f"  Models: https://huggingface.co/{MODEL_REPO}")
    print(f"  Space:  https://huggingface.co/spaces/{SPACE_REPO}")
    
    response = input("\nContinue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled")
        sys.exit(0)
    
    print("\n📦 Uploading models (this may take a while)...")
    create_model_repo()
    
    print("\n🚀 Deploying Space...")
    create_space()
    
    print("\n" + "=" * 50)
    print("✅ Deployment complete!")
    print(f"View your app at: https://huggingface.co/spaces/{SPACE_REPO}")
    print("=" * 50)


if __name__ == '__main__':
    main()
