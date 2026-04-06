"""
Script to upload ASIC-RAG-CHIMERA to Hugging Face Hub.
Requires `huggingface_hub` library.
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo

def upload_to_hf(repo_id: str, token: str):
    print(f"Uploading to Hugging Face: {repo_id}")
    
    api = HfApi(token=token)
    
    # 1. Create Repo (if not exists)
    try:
        create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)
        print("Repository created/verified.")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # 2. Upload Files
    # Upload everything in current dir except git/venv
    files_to_upload = [
        "ASIC_RAG_CHIMERA_Paper.html",
        "paper.md",
        "README.md",
        "requirements.txt",
        "asic_rag_repository.zip"
    ]
    
    for filename in files_to_upload:
        if os.path.exists(filename):
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
    
    print("Upload complete! Check your repo at: https://huggingface.co/" + repo_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="Agnuxo/ASIC-RAG-CHIMERA", help="HF Repo ID")
    parser.add_argument("--token", type=str, required=True, help="HF Write Token")
    args = parser.parse_args()
    
    upload_to_hf(args.repo, args.token)
