# NHS Clinical Coding & Letter Classifier Platform

An enterprise clinical coding and letter classification platform that combines a **3-Model Ensemble** (ClinicalBERT, MedCAT SNOMED CT Engine, and Qwen 2.5 LLM) to process clinical documentation, extract entity diagnostic categories, map SNOMED CT concepts, and provide consensus validation.

---

## 🌟 Overview

The **NHS Ensemble Platform** brings together three specialized medical NLP and AI models:
1. **ClinicalBERT** (`nlpie/clinical-distilbert-i2b2-2010`): Token classification for medical entity extraction.
2. **MedCAT Engine**: Concept Extraction & SNOMED CT mapping using MetaCAT negation and subject checks.
3. **Qwen 2.5 LLM** (`qwen2.5:7b-instruct` via Ollama): Context-aware extraction, status classification, and consensus verification.

---

## 📋 System Requirements & Prerequisites

### 1. Python Environment
- Python `3.9+` or `3.10+` recommended.

### 2. External Tools
- **[Ollama](https://ollama.com/)** installed and running locally on port `11434`.

---

## ⚙️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/Alpha2Omega-PVT-LTD/letter-classifier.git
cd letter-classifier
```

### Step 2: Install Python Dependencies
Install all required packages via `pip`:

```bash
pip install fastapi uvicorn pandas openpyxl requests torch transformers spacy negspacy medcat ollama pydantic
```

### Step 3: Download spaCy Language Model
Download the small English model for spaCy NLP negation parsing:

```bash
python -m spacy download en_core_web_sm
```

### Step 4: Pull Required Ollama Model
Start your local Ollama instance and pull the Qwen model:

```bash
ollama pull qwen2.5:7b-instruct
```

### Step 5: Model Packs & Models
- **ClinicalBERT Model**: Downloaded automatically by HuggingFace `transformers` on first run (`nlpie/clinical-distilbert-i2b2-2010`).
- **MedCAT SNOMED Model**: Ensure the `v2_Snomed2025_MIMIC_IV_bbe806e192df009f` directory is placed in the project root directory.

---

## 🚀 Running the Application

To start the Web UI launcher, run:

```bash
python run_ensemble_ui.py
```

The script automatically performs dependency checks, verifies Ollama connection, starts the Uvicorn web server at `http://127.0.0.1:8000`, and opens the interactive dashboard in your default browser.

---

## 📂 Project Structure

```text
letter-classifier/
├── ensemble_static/           # Web UI frontend assets (HTML, CSS, JS)
├── medgemma/                  # SLM pipeline modules & prompts
│   └── slms/
│       ├── config.py
│       ├── pipeline/          # Classifier, Extractor, Validator, SNOMED modules
│       └── prompts/           # LLM prompt templates
├── clinicalbert_multi_category.py  # ClinicalBERT extraction script
├── medcat_multi_category.py        # MedCAT extraction script
├── nhs_ensemble_pipeline.py        # Core 3-model consensus & ensemble logic
├── run_ensemble_ui.py              # FastAPI server & browser launcher
├── datavalid.xlsx                  # Reference data validation sheet
├── .gitignore                      # Git ignore rules (excluding large model binaries)
└── README.md                       # Project documentation
```

---

## 📄 License
Internal repository belonging to **Alpha2Omega-PVT-LTD**. All rights reserved.
