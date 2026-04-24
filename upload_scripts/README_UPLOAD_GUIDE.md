# ASIC-RAG-CHIMERA Upload Guide

Complete guide for publishing ASIC-RAG-CHIMERA to all scientific platforms.

## 📋 Table of Contents

1. [Preparation](#preparation)
2. [Platform-by-Platform Guide](#platform-by-platform-guide)
3. [Credentials Reference](#credentials-reference)
4. [Quick Upload Commands](#quick-upload-commands)
5. [Verification Checklist](#verification-checklist)

---

## 🔧 Preparation

### Step 1: Generate All Packages

```bash
cd D:\ASIC_RAG
python publication_toolkit.py
```

This creates:
- `publication_packages/` - Distribution packages with checksums
- `publication_results/` - Benchmark summaries
- `upload_scripts/` - Platform-specific upload scripts

### Step 2: Install Required Tools

```bash
# Python packages
pip install wandb zenodo_upload openml kaggle huggingface-hub data

# CLI tools
pip install data  # For DataHub
huggingface-cli login  # For HuggingFace
kaggle  # For Kaggle
```

---

## 📤 Platform-by-Platform Guide

### 1. Weights & Biases (W&B)

**Purpose**: Track experiments and benchmarks
**URL**: https://wandb.ai/lareliquia-angulo

**Steps**:
```bash
# Option A: Run upload script
python upload_scripts/upload_to_wandb.py

# Option B: Manual via web interface
# 1. Go to https://wandb.ai/lareliquia-angulo
# 2. Create new project: "asic-rag-chimera"
# 3. Upload files manually
```

**What to upload**:
- Benchmark results (`benchmark_summary.json`)
- Performance metrics
- System configuration
- Paper PDF

**API Key**: `<REDACTED - use environment variable>`

---

### 2. Zenodo

**Purpose**: Long-term archival with DOI
**URL**: https://zenodo.org/

**Steps**:
```bash
# Run upload script
python upload_scripts/upload_to_zenodo.py

# Then go to Zenodo web interface to:
# 1. Review metadata
# 2. Click "Publish" to get DOI
```

**What to upload**:
- Complete package ZIP
- Paper PDF
- README and documentation

**API Key**: `<REDACTED - use environment variable>`

**Important**: You must manually publish on Zenodo website to get DOI!

---

### 3. Figshare

**Purpose**: Research data repository
**URL**: https://figshare.com/

**Steps**:
```bash
# Option A: Run upload script
python upload_scripts/upload_to_figshare.py

# Option B: FTP Upload
# Host: ftp.figshare.com
# Username: 5292188
# Password: <REDACTED - use environment variable>
```

**What to upload**:
- Complete package ZIP
- Paper PDF
- Benchmark results

**FTP Credentials**:
- Username: `5292188`
- Password: `<REDACTED - use environment variable>`

---

### 4. Open Science Framework (OSF)

**Purpose**: Open science project hosting
**URL**: https://osf.io/

**Steps**:
```bash
# Run upload script
python upload_scripts/upload_to_osf.py

# Then make project public on OSF website
```

**What to upload**:
- Complete package
- Paper PDF
- Documentation
- Benchmark results

**API Key**: `<REDACTED - use environment variable>`

---

### 5. Kaggle

**Purpose**: Dataset sharing and competitions
**URL**: https://www.kaggle.com/franciscoangulo

**Steps**:
```bash
# Setup Kaggle credentials first
# Windows: C:\Users\<username>\.kaggle\kaggle.json
# Linux/Mac: ~/.kaggle/kaggle.json

# Run upload script
python upload_scripts/upload_to_kaggle.py
```

**What to upload**:
- Complete package ZIP
- Benchmark results
- Paper PDF
- README

**Profile**: https://www.kaggle.com/franciscoangulo

---

### 6. HuggingFace

**Purpose**: ML model and dataset hub
**URL**: https://huggingface.co/Agnuxo

**Steps**:
```bash
# Login first
huggingface-cli login

# Run upload script
python upload_scripts/upload_to_huggingface.py
```

**What to upload**:
- Complete source code
- Paper PDF
- Benchmarks
- Model card (README)

**Profile**: https://huggingface.co/Agnuxo

---

### 7. OpenML

**Purpose**: Machine learning experiment database
**URL**: https://www.openml.org/

**Steps**:
```bash
# Get API key from: https://www.openml.org/auth/profile-page
# Add to script, then run:
python upload_scripts/upload_to_openml.py
```

**What to upload**:
- Benchmark dataset (CSV format)
- Performance metrics
- Experimental results

---

### 8. DataHub

**Purpose**: Data package publishing
**URL**: https://datahub.io/

**Steps**:
```bash
# Run preparation script
python upload_scripts/upload_to_datahub.py

# Then manually:
# 1. Create account at https://datahub.io/
# 2. data login
# 3. cd datahub_upload/
# 4. data push
```

**What to upload**:
- Dataset with datapackage.json
- Benchmark results
- Documentation

---

### 9. OpenAIRE / Explore

**Purpose**: European Open Science platform
**URL**: https://explore.openaire.eu/

**Steps**:
1. Upload to Zenodo first (OpenAIRE harvests from Zenodo)
2. Or use direct deposit form at OpenAIRE
3. Ensure DOI is assigned

**Note**: OpenAIRE aggregates from other repositories, so uploading to Zenodo is sufficient.

---

### 10. Additional Platforms

#### Academia.edu
**URL**: https://www.academia.edu/

**Steps**:
1. Log in to your account
2. Go to "Upload" → "Paper"
3. Upload `ASIC-RAG-CHIMERA_Unified.pdf`
4. Add title, abstract, keywords
5. Publish

#### ResearchGate
**URL**: https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3

**Steps**:
1. Log in to your profile
2. Add new publication
3. Upload paper PDF
4. Add metadata and links to code/data

#### DrivenData
**URL**: https://www.drivendata.org/users/Francisco_Angulo_Lafuente/

**Steps**:
1. Check for relevant competitions
2. If applicable, submit solution using ASIC-RAG-CHIMERA
3. Share methodology and code

---

## 🔑 Credentials Reference

Quick reference for all API keys and credentials:

| Platform | Credential Type | Value |
|----------|----------------|-------|
| W&B | API Key | `<REDACTED - use environment variable>` |
| Zenodo | Access Token | `<REDACTED - use environment variable>` |
| Figshare | FTP Username | `5292188` |
| Figshare | FTP Password | `<REDACTED - use environment variable>` |
| OSF | Personal Token | `<REDACTED - use environment variable>` |
| Kaggle | JSON File | `~/.kaggle/kaggle.json` |
| HuggingFace | CLI Login | Use `huggingface-cli login` |
| OpenML | Web Profile | Get from profile page |

---

## 🚀 Quick Upload Commands

Run all uploads sequentially:

```bash
# Core scientific repositories
python upload_scripts/upload_to_wandb.py
python upload_scripts/upload_to_zenodo.py
python upload_scripts/upload_to_figshare.py
python upload_scripts/upload_to_osf.py

# Dataset platforms
python upload_scripts/upload_to_kaggle.py
python upload_scripts/upload_to_huggingface.py
python upload_scripts/upload_to_openml.py
python upload_scripts/upload_to_datahub.py
```

---

## ✅ Verification Checklist

After uploading, verify each platform:

- [ ] **W&B**: Project visible at https://wandb.ai/lareliquia-angulo/asic-rag-chimera
- [ ] **Zenodo**: DOI assigned and publication live
- [ ] **Figshare**: Article published and publicly accessible
- [ ] **OSF**: Project set to PUBLIC
- [ ] **Kaggle**: Dataset at https://kaggle.com/datasets/franciscoangulo/asic-rag-chimera
- [ ] **HuggingFace**: Dataset at https://huggingface.co/datasets/Agnuxo/ASIC-RAG-CHIMERA
- [ ] **OpenML**: Dataset searchable on OpenML
- [ ] **DataHub**: Package published and validated
- [ ] **Academia.edu**: Paper uploaded to profile
- [ ] **ResearchGate**: Publication added with links

---

## 📊 Impact Tracking

After publication, track impact:

1. **Citations**: Google Scholar, ResearchGate
2. **Downloads**: Zenodo, Figshare, Kaggle
3. **Stars**: GitHub repository
4. **Views**: HuggingFace, OpenML
5. **Usage**: W&B project views

---

## 🆘 Troubleshooting

### API Authentication Issues
```bash
# Clear cached credentials
rm ~/.wandb_api_key
rm ~/.kaggle/kaggle.json
huggingface-cli logout

# Re-authenticate
wandb login
huggingface-cli login
# Add kaggle.json manually
```

### Upload Failures
1. Check internet connection
2. Verify API keys are current
3. Check file size limits (Zenodo: 50GB, Figshare: 20GB per file)
4. Try manual upload as fallback

### Missing Files
```bash
# Re-generate packages
python publication_toolkit.py

# Verify files exist
ls -la publication_packages/
ls -la publication_results/
```

---

## 📞 Support

If you encounter issues:

1. Check platform-specific documentation
2. Review API status pages
3. Contact platform support
4. Check GitHub issues for similar problems

---

## 🎯 Summary

You now have:
- ✅ Complete distribution packages
- ✅ Platform-specific upload scripts
- ✅ Credentials for all platforms
- ✅ Step-by-step instructions

**Next**: Execute uploads and verify publications!

**Goal**: Maximum visibility and impact for ASIC-RAG-CHIMERA across the scientific community.

---

**Last Updated**: 2024-12-09
**Version**: 1.0
