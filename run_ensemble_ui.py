import os
import sys
import json
import re
import datetime
import traceback
import webbrowser
import time
import threading
import torch
import pandas as pd
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure current folder is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SLMS_DIR = os.path.join(BASE_DIR, "medgemma", "slms")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SLMS_DIR not in sys.path:
    sys.path.insert(0, SLMS_DIR)

# Import ensemble pipeline functions
try:
    from nhs_ensemble_pipeline import (
        extract_clinicalbert_diagnoses,
        extract_medcat_diagnoses,
        extract_clinicalbert_all_categories,
        extract_medcat_all_categories,
        qwen_process_report,
        build_3_model_consensus,
        build_3_model_multi_category_consensus
    )
except ImportError as e:
    print(f"[ERROR] Failed to import ensemble pipeline: {e}")
    sys.exit(1)

# Import SLMS pipeline modules for full entity extraction + SNOMED
_snomed_mapper = None
def get_snomed_mapper():
    global _snomed_mapper
    if _snomed_mapper is None:
        try:
            from pipeline.snomed import SNOMEDMapper
            _snomed_mapper = SNOMEDMapper()
        except Exception as e:
            print(f"[WARN] SNOMEDMapper not loaded: {e}")
    return _snomed_mapper

# ─────────────────────────────────────────────────────────
# Qwen full entity extraction (all categories at once)
# ─────────────────────────────────────────────────────────
def qwen_extract_all_entities(text: str) -> Dict[str, List[str]]:
    """
    Uses the SLMS pipeline components (MedicalEntityExtractor + EntityClassifier)
    which run Qwen via Ollama using step1_extraction.txt and step2_classification.txt
    to extract and classify all clinical entities (Procedures, Symptoms, Meds, Vitals).
    """
    try:
        SLMS_DIR = os.path.join(BASE_DIR, "medgemma", "slms")
        if SLMS_DIR not in sys.path:
            sys.path.insert(0, SLMS_DIR)

        orig_cwd = os.getcwd()
        os.chdir(SLMS_DIR)
        try:
            from pipeline.extractor import MedicalEntityExtractor
            from pipeline.classifier import EntityClassifier

            extractor = MedicalEntityExtractor()
            classifier = EntityClassifier()

            raw_entities = extractor.extract(text)

            result = {"diagnoses": [], "symptoms": [], "procedures": [], "medications": [], "vitals": []}

            for ent in raw_entities:
                if ent.provisional_type.lower() == "vital":
                    result["vitals"].append(ent.entity)
                    continue

                category = classifier.classify(ent)
                cat_lower = category.lower()

                if cat_lower == "procedure":
                    result["procedures"].append(ent.entity)
                elif cat_lower == "symptom":
                    result["symptoms"].append(ent.entity)
                elif cat_lower == "medication":
                    result["medications"].append(ent.entity)
                elif cat_lower == "diagnosis":
                    result["diagnoses"].append(ent.entity)

            return result
        finally:
            os.chdir(orig_cwd)

    except Exception as e:
        print(f"[WARN] SLMS pipeline Qwen extraction failed: {e}")
        return {"diagnoses": [], "symptoms": [], "procedures": [], "medications": [], "vitals": []}


# ─────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────
app = FastAPI(title="NHS 3-Model Ensemble Clinical Coding Platform")
SESSION_FILE = os.path.join(BASE_DIR, "ensemble_web_session_progress.json")

class EntityDecision(BaseModel):
    id: str
    text: str
    category: str       # Diagnosis / Symptom / Procedure / Medication / Vital
    confidence: float
    validation_status: str   # e.g. "Confirmed by all 3 models" or "Qwen only"
    status: str         # Clinical status: Current / Historical / Performed etc.
    snomed: str         # SNOMED code or empty
    decision: str       # "Yes", "No", "Do Not Need"

class RowProcessingRequest(BaseModel):
    row_index: int
    clinical_text: str
    decisions: List[EntityDecision]


def load_session() -> Dict[str, Any]:
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"file_path": "", "column": "", "records": [], "current_index": 0}


def save_session(session_data: Dict[str, Any]):
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save session progress: {e}")


def default_status_for_category(category: str) -> str:
    return {
        "Diagnosis": "Current",
        "Symptom": "Current",
        "Procedure": "Performed",
        "Medication": "Current",
        "Vital": "N/A"
    }.get(category, "Current")


# ─────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────

@app.get("/api/files")
def get_files():
    """Lists available Excel files in the workspace root."""
    files = []
    seen = set()
    for f in os.listdir(BASE_DIR):
        if f.endswith(".xlsx") and not f.startswith("~$") and "ensemble_export" not in f:
            full_path = os.path.abspath(os.path.join(BASE_DIR, f))
            if full_path not in seen:
                files.append({"name": f, "path": full_path, "folder": "Workspace Root"})
                seen.add(full_path)
    return files


@app.get("/api/columns")
def get_columns(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="File path does not exist.")
    try:
        df = pd.read_excel(file_path, nrows=2)
        return list(df.columns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file columns: {str(e)}")


@app.post("/api/load-records")
def load_records(payload: Dict[str, str]):
    file_path = payload.get("file_path")
    column = payload.get("column", "Cleaned Data")

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="Valid file path required.")

    try:
        df = pd.read_excel(file_path)
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found.")

        session = load_session()
        if session.get("file_path") != file_path or session.get("column") != column:
            records = []
            for idx, row in df.iterrows():
                text = row[column]
                records.append({
                    "index": idx,
                    "text": str(text) if pd.notna(text) else "",
                    "extracted_entities": None,
                    "final_results": None,
                    "reviewed": False
                })
            session = {"file_path": file_path, "column": column, "records": records, "current_index": 0}
            save_session(session)

        return {
            "success": True,
            "total_records": len(session["records"]),
            "current_index": session.get("current_index", 0),
            "records": [
                {"index": r["index"], "reviewed": r.get("reviewed", False),
                 "has_extractions": r.get("extracted_entities") is not None}
                for r in session["records"]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading file: {str(e)}")


@app.get("/api/record/{index}")
def get_record(index: int):
    session = load_session()
    if not session.get("records") or index < 0 or index >= len(session["records"]):
        raise HTTPException(status_code=404, detail="Record not found")
    session["current_index"] = index
    save_session(session)
    record = session["records"][index]
    return {
        "index": index,
        "text": record["text"],
        "extracted_entities": record.get("extracted_entities"),
        "final_results": record.get("final_results"),
        "reviewed": record.get("reviewed", False)
    }


@app.post("/api/extract/{index}")
def extract_entities(index: int):
    """
    Full extraction: diagnoses via 3-model consensus, then symptoms/procedures/
    medications/vitals via Qwen.
    """
    session = load_session()
    if not session.get("records") or index < 0 or index >= len(session["records"]):
        raise HTTPException(status_code=404, detail="Record not found")

    record = session["records"][index]
    if record.get("extracted_entities") is not None:
        return {"entities": record["extracted_entities"]}

    clinical_text = record["text"]
    if not clinical_text.strip():
        entities = []
    else:
        try:
            # ── 1. Independent extraction from all 3 models across all categories ──
            cb_all = extract_clinicalbert_all_categories(clinical_text)
            mc_all = extract_medcat_all_categories(clinical_text)
            qw_all = qwen_extract_all_entities(clinical_text)

            # ── 2. 3-Model Cross-Checking Consensus across all 5 categories ────────
            entities = build_3_model_multi_category_consensus(cb_all, mc_all, qw_all, clinical_text)

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Ensemble extraction error: {str(e)}")

    record["extracted_entities"] = entities
    session["records"][index] = record
    save_session(session)
    return {"entities": entities}


@app.get("/api/snomed-search")
def snomed_search(term: str = Query(..., min_length=2), category: Optional[str] = None):
    """Searches the SNOMED FHIR server for possible matching concepts."""
    try:
        mapper = get_snomed_mapper()
        if not mapper:
            return {"success": False, "results": [], "message": "SNOMED mapper not available."}
        possible = mapper.get_snomed_possible_codes_for_term(term, category)
        if possible:
            return {"success": True, "results": possible}
        return {"success": False, "results": [], "message": "No matching codes found."}
    except Exception as e:
        return {"success": False, "results": [], "error": str(e)}


@app.post("/api/process-row")
def process_row(request: RowProcessingRequest):
    """Saves all entity coding decisions for the row and marks it reviewed."""
    session = load_session()
    idx = request.row_index
    if not session.get("records") or idx < 0 or idx >= len(session["records"]):
        raise HTTPException(status_code=404, detail="Record not found")

    diagnoses, symptoms, procedures, medications, vitals = [], [], [], [], []
    processed_entities = []

    for dec in request.decisions:
        if dec.decision != "Yes":
            processed_entities.append({
                "id": dec.id, "text": dec.text, "category": dec.category,
                "decision": dec.decision, "status": "Skipped", "snomed": ""
            })
            continue

        # Auto-fetch SNOMED if not already set and category warrants it
        snomed_code = dec.snomed
        if dec.category in ("Diagnosis", "Symptom", "Procedure") and not snomed_code:
            mapper = get_snomed_mapper()
            if mapper:
                try:
                    snomed_code = mapper.get_snomed_code(dec.text, dec.category.lower(), dec.status)
                except Exception:
                    snomed_code = ""

        entry = {"entity": dec.text, "status": dec.status, "snomed": snomed_code or "N/A"}

        if dec.category == "Diagnosis":
            diagnoses.append(entry)
        elif dec.category == "Symptom":
            symptoms.append(entry)
        elif dec.category == "Procedure":
            procedures.append(entry)
        elif dec.category == "Medication":
            medications.append({"entity": dec.text, "status": dec.status})
        elif dec.category == "Vital":
            vitals.append({"entity": dec.text})

        processed_entities.append({
            "id": dec.id, "text": dec.text, "category": dec.category,
            "decision": "Yes", "status": dec.status, "snomed": snomed_code or "N/A"
        })

    record = session["records"][idx]
    record["extracted_entities"] = [dec.model_dump() for dec in request.decisions]
    record["final_results"] = {
        "diagnoses": diagnoses,
        "symptoms": symptoms,
        "procedures": procedures,
        "medications": medications,
        "vitals": vitals
    }
    record["reviewed"] = True
    session["records"][idx] = record
    save_session(session)

    return {"success": True, "processed_entities": processed_entities}


@app.post("/api/export")
def export_file():
    """Generates the final Excel with all 5 clinical categories."""
    session = load_session()
    if not session.get("file_path"):
        raise HTTPException(status_code=400, detail="No active session loaded")

    try:
        df_input = pd.read_excel(session["file_path"])

        def fmt(items, key="entity"):
            return "\n".join(f"{i+1}. {e.get(key,'')}" for i, e in enumerate(items)) if items else ""

        diag_col, symp_col, proc_col, med_col, vit_col = [], [], [], [], []

        for idx in range(len(df_input)):
            rec = next((r for r in session["records"] if r["index"] == idx), None)
            if rec and rec.get("final_results"):
                res = rec["final_results"]
                diag_col.append(fmt(res.get("diagnoses", [])))
                symp_col.append(fmt(res.get("symptoms", [])))
                proc_col.append(fmt(res.get("procedures", [])))
                med_col.append(fmt(res.get("medications", [])))
                vit_col.append(fmt(res.get("vitals", [])))
            else:
                diag_col.append(""); symp_col.append(""); proc_col.append("")
                med_col.append(""); vit_col.append("")

        df_input["Diagnosis"] = diag_col
        df_input["Symptoms"] = symp_col
        df_input["Procedures"] = proc_col
        df_input["Medications"] = med_col
        df_input["Vitals"] = vit_col

        os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"ensemble_export_{timestamp}.xlsx"
        output_path = os.path.abspath(os.path.join(BASE_DIR, "output", output_filename))
        df_input.to_excel(output_path, index=False)

        return {"success": True, "filename": output_filename,
                "path": output_path, "download_url": f"/api/download/{output_filename}"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.get("/api/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join(BASE_DIR, "output", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ─────────────────────────────────────────────────────────
# Static Frontend
# ─────────────────────────────────────────────────────────
static_dir = os.path.join(BASE_DIR, "ensemble_static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(content={"error": "Frontend not built yet"}, status_code=503)


# ─────────────────────────────────────────────────────────
# Launcher
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" NHS Clinical Coding 3-Model Ensemble Platform Web UI Launcher")
    print("=" * 60)

    print("Checking dependencies...")
    try:
        import fastapi, uvicorn, pandas, openpyxl, requests, ollama
        print("[OK] All primary Python dependencies are present.")
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Please run: pip install fastapi uvicorn pandas openpyxl requests ollama")
        sys.exit(1)

    print("Checking Ollama model status...")
    try:
        import ollama
        response = ollama.list()
        models = []
        models_list = getattr(response, 'models', [])
        if not models_list and isinstance(response, dict):
            models_list = response.get('models', [])
        for m in models_list:
            name = getattr(m, 'name', None) or (m.get('name') if isinstance(m, dict) else None)
            if name:
                models.append(name)
        required_model = "qwen2.5:7b-instruct"
        if required_model in models or any(required_model in m for m in models):
            print(f"[OK] Found Ollama model '{required_model}'.")
        else:
            print(f"[WARN] Ollama model '{required_model}' not found. Run: ollama pull {required_model}")
    except Exception as e:
        print(f"[WARN] Could not connect to Ollama: {e}")

    print("\nStarting Uvicorn server on http://127.0.0.1:8000 ...")

    def open_browser():
        time.sleep(2.0)
        print("Opening Web UI in your default browser...")
        webbrowser.open("http://127.0.0.1:8000")

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        import uvicorn
        uvicorn.run("run_ensemble_ui:app", host="127.0.0.1", port=8000, reload=True, reload_dirs=[BASE_DIR])
    except KeyboardInterrupt:
        print("\nStopping Web UI server. Goodbye!")
    except Exception as e:
        print(f"\n❌ Server crashed: {e}")


if __name__ == "__main__":
    main()
