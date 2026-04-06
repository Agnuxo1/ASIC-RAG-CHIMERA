#!/usr/bin/env python3
"""
Master Upload Script - Execute all platform uploads
Uploads ASIC-RAG-CHIMERA to all scientific platforms
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

class MasterUploader:
    def __init__(self):
        self.upload_dir = Path(__file__).parent
        self.results = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def run_script(self, script_name, platform_name, required=True):
        """Run an upload script and track results"""
        print("\n" + "=" * 80)
        print(f"UPLOADING TO: {platform_name}")
        print("=" * 80)

        script_path = self.upload_dir / script_name

        if not script_path.exists():
            print(f"[WARNING] Script not found: {script_path}")
            self.results.append({
                "platform": platform_name,
                "status": "SKIPPED",
                "reason": "Script not found"
            })
            return False

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            print(result.stdout)

            if result.returncode == 0:
                print(f"\n[SUCCESS] {platform_name} upload completed!")
                self.results.append({
                    "platform": platform_name,
                    "status": "SUCCESS",
                    "timestamp": self.timestamp
                })
                return True
            else:
                print(f"\n[ERROR] {platform_name} upload failed!")
                print(result.stderr)
                self.results.append({
                    "platform": platform_name,
                    "status": "FAILED",
                    "error": result.stderr[:200]
                })
                return False

        except subprocess.TimeoutExpired:
            print(f"\n[TIMEOUT] {platform_name} upload timed out!")
            self.results.append({
                "platform": platform_name,
                "status": "TIMEOUT",
                "reason": "Exceeded 5 minute timeout"
            })
            return False

        except Exception as e:
            print(f"\n[ERROR] {platform_name} upload exception: {e}")
            self.results.append({
                "platform": platform_name,
                "status": "ERROR",
                "error": str(e)
            })
            return False

    def run_all_uploads(self, skip_manual=False):
        """Execute all upload scripts"""

        print("=" * 80)
        print("ASIC-RAG-CHIMERA - MASTER UPLOAD SCRIPT")
        print("=" * 80)
        print(f"Started at: {self.timestamp}")
        print("\nThis will upload to all configured platforms.")
        print("Make sure you have:")
        print("  - Generated packages (python publication_toolkit.py)")
        print("  - Configured all API keys")
        print("  - Authenticated with CLI tools (wandb, huggingface-cli, etc.)")

        print("\n[AUTO-MODE] Proceeding with automatic upload...")

        # Core scientific repositories
        print("\n" + "=" * 80)
        print("PHASE 1: CORE SCIENTIFIC REPOSITORIES")
        print("=" * 80)

        self.run_script("upload_to_wandb.py", "Weights & Biases")
        self.run_script("upload_to_zenodo.py", "Zenodo")
        self.run_script("upload_to_figshare.py", "Figshare")
        self.run_script("upload_to_osf.py", "Open Science Framework")

        # Dataset platforms
        print("\n" + "=" * 80)
        print("PHASE 2: DATASET PLATFORMS")
        print("=" * 80)

        self.run_script("upload_to_kaggle.py", "Kaggle")
        self.run_script("upload_to_huggingface.py", "HuggingFace Hub")

        # ML/AI platforms
        print("\n" + "=" * 80)
        print("PHASE 3: ML/AI PLATFORMS")
        print("=" * 80)

        self.run_script("upload_to_openml.py", "OpenML", required=False)
        self.run_script("upload_to_datahub.py", "DataHub", required=False)

        # Generate summary
        self.generate_summary()

        # Manual platforms reminder
        if not skip_manual:
            self.show_manual_platforms()

    def generate_summary(self):
        """Generate upload summary report"""
        print("\n" + "=" * 80)
        print("UPLOAD SUMMARY")
        print("=" * 80)

        success_count = sum(1 for r in self.results if r["status"] == "SUCCESS")
        failed_count = sum(1 for r in self.results if r["status"] == "FAILED")
        skipped_count = sum(1 for r in self.results if r["status"] == "SKIPPED")

        print(f"\nTotal Platforms: {len(self.results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"Skipped: {skipped_count}")

        print("\nDetailed Results:")
        print("-" * 80)
        for result in self.results:
            status_symbol = {
                "SUCCESS": "[OK]",
                "FAILED": "[FAIL]",
                "SKIPPED": "[SKIP]",
                "TIMEOUT": "[TIME]",
                "ERROR": "[ERR]"
            }.get(result["status"], "[???]")

            print(f"{status_symbol} {result['platform']:<30} - {result['status']}")

            if "error" in result:
                print(f"     Error: {result['error'][:100]}")

        # Save summary to file
        summary_file = self.upload_dir.parent / "upload_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"ASIC-RAG-CHIMERA Upload Summary\n")
            f.write(f"Generated: {self.timestamp}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Success: {success_count}/{len(self.results)}\n\n")

            for result in self.results:
                f.write(f"{result['platform']}: {result['status']}\n")
                if "error" in result:
                    f.write(f"  Error: {result['error']}\n")

        print(f"\n[OK] Summary saved to: {summary_file}")

    def show_manual_platforms(self):
        """Show platforms that require manual upload"""
        print("\n" + "=" * 80)
        print("MANUAL UPLOAD REQUIRED")
        print("=" * 80)

        manual_platforms = [
            {
                "name": "Academia.edu",
                "url": "https://www.academia.edu/",
                "steps": [
                    "1. Log in to your account",
                    "2. Click 'Upload' → 'Paper'",
                    "3. Upload: ASIC-RAG-CHIMERA_Unified.pdf",
                    "4. Add metadata and publish"
                ]
            },
            {
                "name": "ResearchGate",
                "url": "https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3",
                "steps": [
                    "1. Log in to your profile",
                    "2. Add new publication",
                    "3. Upload: ASIC-RAG-CHIMERA_Unified.pdf",
                    "4. Add links to GitHub and datasets",
                    "5. Publish"
                ]
            },
            {
                "name": "OpenAIRE Explore",
                "url": "https://explore.openaire.eu/",
                "steps": [
                    "Note: OpenAIRE harvests from Zenodo automatically",
                    "1. Verify Zenodo upload is published",
                    "2. Wait 24-48 hours for OpenAIRE indexing",
                    "3. Search for your publication on explore.openaire.eu"
                ]
            },
            {
                "name": "arXiv (Optional)",
                "url": "https://arxiv.org/",
                "steps": [
                    "If you want to submit to arXiv:",
                    "1. Register for arXiv account",
                    "2. Submit paper to cs.AI or cs.CR category",
                    "3. Include link to code repository",
                    "4. Wait for moderation"
                ]
            }
        ]

        for platform in manual_platforms:
            print(f"\n{platform['name']}")
            print(f"URL: {platform['url']}")
            print("Steps:")
            for step in platform['steps']:
                print(f"  {step}")

        print("\n" + "=" * 80)

    def generate_urls_document(self):
        """Generate document with all publication URLs"""
        urls_doc = f"""# ASIC-RAG-CHIMERA Publication URLs

Generated: {self.timestamp}

## Primary Repository
- **GitHub**: https://github.com/Agnuxo1/ASIC-RAG-CHIMERA

## Scientific Repositories

### Zenodo (with DOI)
- **URL**: https://zenodo.org/search?q=ASIC-RAG-CHIMERA
- **Status**: {self._get_status("Zenodo")}
- **Note**: Check Zenodo dashboard for DOI after publishing

### Figshare
- **URL**: https://figshare.com/search?q=ASIC-RAG-CHIMERA
- **Status**: {self._get_status("Figshare")}

### Open Science Framework (OSF)
- **URL**: https://osf.io/search/?q=ASIC-RAG-CHIMERA
- **Status**: {self._get_status("Open Science Framework")}

### OpenAIRE
- **URL**: https://explore.openaire.eu/search/find/research-outcomes?q=ASIC-RAG-CHIMERA
- **Status**: Indexed from Zenodo (24-48h delay)

## Dataset Platforms

### Kaggle
- **URL**: https://www.kaggle.com/datasets/franciscoangulo/asic-rag-chimera
- **Profile**: https://www.kaggle.com/franciscoangulo
- **Status**: {self._get_status("Kaggle")}

### HuggingFace
- **URL**: https://huggingface.co/datasets/Agnuxo/ASIC-RAG-CHIMERA
- **Profile**: https://huggingface.co/Agnuxo
- **Status**: {self._get_status("HuggingFace Hub")}

### OpenML
- **URL**: https://www.openml.org/search?q=ASIC-RAG-CHIMERA
- **Profile**: https://www.openml.org/u/[your_id]
- **Status**: {self._get_status("OpenML")}

### DataHub
- **URL**: https://datahub.io/[username]/asic-rag-chimera
- **Status**: {self._get_status("DataHub")}

## Experiment Tracking

### Weights & Biases
- **URL**: https://wandb.ai/lareliquia-angulo/asic-rag-chimera
- **Status**: {self._get_status("Weights & Biases")}

## Author Profiles

### Academia
- **URL**: https://www.academia.edu/
- **Profile**: [Your Academia.edu profile]
- **Status**: Manual upload required

### ResearchGate
- **URL**: https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3
- **Status**: Manual upload required

### Other Profiles
- **Wikipedia**: https://es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente
- **DrivenData**: https://www.drivendata.org/users/Francisco_Angulo_Lafuente/
- **Gravatar**: https://gravatar.com/ecofa

## Citation

```bibtex
@article{{angulo2024asicrag,
  title={{ASIC-RAG-CHIMERA: Hardware-Accelerated Cryptographic Framework for Secure Retrieval-Augmented Generation}},
  author={{Angulo de Lafuente, Francisco and Tej, Nirmal}},
  year={{2024}},
  doi={{[DOI from Zenodo]}},
  url={{https://github.com/Agnuxo1/ASIC-RAG-CHIMERA}}
}}
```

## Next Steps

1. ✅ Verify all automated uploads completed successfully
2. ⏳ Complete manual uploads (Academia.edu, ResearchGate)
3. ⏳ Wait for DOI from Zenodo
4. ⏳ Update GitHub README with DOI and all publication links
5. ⏳ Share on social media and academic networks
6. ⏳ Monitor downloads, citations, and usage

## Impact Tracking

Track your research impact:
- Citations: Google Scholar, ResearchGate
- Downloads: Zenodo, Figshare, Kaggle statistics
- GitHub Stars: Watch repository growth
- Community: Issues, discussions, pull requests

---

Last updated: {self.timestamp}
"""

        urls_file = self.upload_dir.parent / "PUBLICATION_URLS.md"
        with open(urls_file, 'w', encoding='utf-8') as f:
            f.write(urls_doc)

        print(f"[OK] URLs document saved to: {urls_file}")

    def _get_status(self, platform_name):
        """Get upload status for a platform"""
        for result in self.results:
            if result["platform"] == platform_name:
                return result["status"]
        return "NOT ATTEMPTED"


def main():
    uploader = MasterUploader()

    # Run all uploads
    uploader.run_all_uploads()

    # Generate URLs document
    uploader.generate_urls_document()

    print("\n" + "=" * 80)
    print("[SUCCESS] MASTER UPLOAD COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review upload_summary.txt for details")
    print("2. Check PUBLICATION_URLS.md for all links")
    print("3. Complete manual uploads (Academia.edu, ResearchGate)")
    print("4. Publish on Zenodo/Figshare web interfaces to get DOIs")
    print("5. Update README with DOI and publication links")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Upload process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
