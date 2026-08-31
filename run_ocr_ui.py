import os
import sys
import json
import re
import datetime
import traceback
import webbrowser
import time
import threading
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────
# System Path Setup (Strictly final_pipeline)
# ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_PIPELINE_DIR = os.path.join(BASE_DIR, "final_pipeline")
SLMS_DIR = os.path.join(FINAL_PIPELINE_DIR, "slms", "slms")

if FINAL_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, FINAL_PIPELINE_DIR)
if SLMS_DIR not in sys.path:
    sys.path.insert(0, SLMS_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Fix stdout/stderr encoding on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ─────────────────────────────────────────────────────────
# Import Models & Pipeline Modules strictly from final_pipeline
# ─────────────────────────────────────────────────────────
try:
    from paddle_ocr import extract_text_from_pdf_paddle, initialize_paddle_ocr, PADDLE_AVAILABLE
except ImportError:
    print("[WARN] Could not import paddle_ocr.py from final_pipeline.")
    PADDLE_AVAILABLE = False

try:
    from clinical_bert import extract_entities_from_letter as extract_clinicalbert_entities
except ImportError:
    print("[WARN] Could not import clinical_bert.py from final_pipeline.")
    extract_clinicalbert_entities = None

try:
    from medcatex import extract_entities_with_medcat, load_medcat_engine
except ImportError:
    print("[WARN] Could not import medcatex.py from final_pipeline.")
    extract_entities_with_medcat = None
    load_medcat_engine = None

_ocr_instance = None
def get_paddle_ocr():
    global _ocr_instance
    if _ocr_instance is None and PADDLE_AVAILABLE:
        try:
            print("[OCR] Initializing PaddleOCR model...")
            _ocr_instance = initialize_paddle_ocr()
        except Exception as e:
            print(f"[WARN] Failed to initialize PaddleOCR: {e}")
    return _ocr_instance

_clinical_pipeline = None
def get_slms_pipeline():
    global _clinical_pipeline
    if _clinical_pipeline is None:
        try:
            orig_cwd = os.getcwd()
            os.chdir(SLMS_DIR)
            try:
                from pipeline.orchestrator import ClinicalPipeline
                _clinical_pipeline = ClinicalPipeline()
                print("[SLMS] Loaded Qwen 6-Stage Clinical Pipeline & SNOMED CT Mapper.")
            finally:
                os.chdir(orig_cwd)
        except Exception as e:
            print(f"[WARN] Failed to load SLMS ClinicalPipeline: {e}")
_snomed_mapper = None
def get_snomed_mapper():
    global _snomed_mapper
    if _snomed_mapper is None:
        try:
            orig_cwd = os.getcwd()
            os.chdir(SLMS_DIR)
            try:
                from pipeline.snomed import SNOMEDMapper
                _snomed_mapper = SNOMEDMapper()
            finally:
                os.chdir(orig_cwd)
        except Exception as e:
            print(f"[WARN] Could not load SNOMEDMapper: {e}")
    return _snomed_mapper

_medcat_engine = None
def get_medcat():
    global _medcat_engine
    if _medcat_engine is None and load_medcat_engine is not None:
        try:
            print("[MedCAT] Loading MedCAT engine from final_pipeline/medcatex.py...")
            _medcat_engine = load_medcat_engine()
        except Exception as e:
            print(f"[WARN] Failed to load MedCAT model pack: {e}")
    return _medcat_engine

# ─────────────────────────────────────────────────────────
# 3-Model Cross-Checking Consensus Engine (final_pipeline)
# ─────────────────────────────────────────────────────────
def get_fuzzy_similarity(str1: str, str2: str) -> float:
    s1, s2 = str1.lower().strip(), str2.lower().strip()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    words1 = set(re.findall(r'\w+', s1))
    words2 = set(re.findall(r'\w+', s2))
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)

def calculate_consensus_confidence(models_dict: dict):
    active_models = [m for m, val in models_dict.items() if val]
    n_matches = len(active_models)

    if n_matches >= 3:
        confidence = 0.98
        validation_status = "High Consensus"
    elif n_matches == 2:
        validation_status = "Consensus"
        confidence = 0.85
    else:
        validation_status = "Validated"
        confidence = 0.75

    return confidence, validation_status, n_matches


def qwen_filter_junk_entities(merged_pool: list, clinical_text: str = "") -> list:
    """
    Dynamically calls Qwen (via Ollama) to review extracted candidate entities and filter out
    random English words, pronouns, incomplete tokens, or administrative filler.
    Retains terms validated by Qwen OR terms with multi-model consensus (>=2 models).
    """
    if not merged_pool:
        return merged_pool

    single_model_items = [item for item in merged_pool if sum(item["models"].values()) < 2]
    high_consensus_items = [item for item in merged_pool if sum(item["models"].values()) >= 2]

    if not single_model_items:
        return merged_pool

    candidate_texts = [item["text"] for item in single_model_items]

    prompt = f"""You are an expert clinical quality assurance agent.
Review the following candidate extracted terms from a clinical letter.

CLINICAL TEXT:
{clinical_text[:2000] if clinical_text else "N/A"}

CANDIDATE TERMS:
{json.dumps(candidate_texts)}

TASK:
Filter out any random English words, pronouns, verbs, incomplete sub-words, or non-clinical administrative filler (such as 'his', 'side', 'air', 'carries', 'related', 'support', 'mel', 'the medication', 'these medic', 'presentation', 'discharge').
Return ONLY a JSON array of valid clinical entity strings representing genuine medical Diagnoses, Symptoms, Procedures, Medications, or Vitals.

JSON OUTPUT FORMAT (Return ONLY a JSON array):
["valid_term_1", "valid_term_2"]
"""
    try:
        import ollama
        res = ollama.chat(
            model='qwen2.5:7b-instruct',
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={'temperature': 0.05}
        )
        content = res.get('message', {}).get('content', '')
        data = json.loads(content)
        
        valid_list = []
        if isinstance(data, list):
            valid_list = data
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    valid_list = v
                    break

        valid_set = set(str(t).lower().strip() for t in valid_list if t)

        validated_single_items = []
        for item in single_model_items:
            t_lower = item["text"].lower().strip()
            if t_lower in valid_set:
                validated_single_items.append(item)
            else:
                print(f"[QWEN JUNK FILTER REJECTED] '{item['text']}' (Single-Model Candidate)")

        return high_consensus_items + validated_single_items
    except Exception as e:
        print(f"[WARN] Qwen dynamic junk filter execution skipped/failed: {e}")
        return merged_pool


def build_3_model_consensus_from_final_pipeline(cb_cats: dict, mc_cats: dict, qw_cats: dict, clinical_text: str = ""):
    categories = ["Diagnosis", "Symptom", "Procedure", "Medication", "Vital"]
    qw_key_map = {
        "Diagnosis": "diagnoses",
        "Symptom": "symptoms",
        "Procedure": "procedures",
        "Medication": "medications",
        "Vital": "vitals"
    }

    all_entities = []
    counter = 0

    for cat in categories:
        q_key = qw_key_map[cat]

        cb_raw = cb_cats.get(cat, []) if isinstance(cb_cats, dict) else []
        mc_raw = mc_cats.get(cat, []) if isinstance(mc_cats, dict) else []
        qw_raw = qw_cats.get(q_key, []) if isinstance(qw_cats, dict) else []

        model_candidates = []

        # 1. ClinicalBERT candidates
        for c in cb_raw:
            t = c.get("text") if isinstance(c, dict) else str(c)
            if t:
                model_candidates.append({"text": t.strip(), "snomed": "", "model": "ClinicalBERT"})

        # 2. MedCAT candidates
        for c in mc_raw:
            t = c.get("text") if isinstance(c, dict) else str(c)
            cui = c.get("cui") if isinstance(c, dict) else ""
            if t:
                model_candidates.append({"text": t.strip(), "snomed": cui or "", "model": "MedCAT"})

        # 3. Qwen candidates
        for c in qw_raw:
            t = c.get("entity", "") if isinstance(c, dict) else (c.get("text", "") if isinstance(c, dict) else str(c))
            snomed_val = c.get("snomed", "") if isinstance(c, dict) else ""
            if snomed_val == "Not found":
                snomed_val = ""
            if t:
                model_candidates.append({"text": t.strip(), "snomed": snomed_val, "model": "Qwen"})

        # Group & Merge candidates across all 3 models
        merged_entities = []
        for cand in model_candidates:
            m_name = cand["model"]
            c_text = cand["text"]
            c_snomed = cand["snomed"]

            matched = None
            for target in merged_entities:
                t_text = target["text"]

                # Exact match
                if c_text.lower() == t_text.lower():
                    matched = target
                    break
                # Substring containment
                if len(c_text) > 3 and len(t_text) > 3 and (c_text.lower() in t_text.lower() or t_text.lower() in c_text.lower()):
                    matched = target
                    break
                # Fuzzy match
                if get_fuzzy_similarity(c_text, t_text) >= 0.70:
                    matched = target
                    break

            if matched:
                matched["models"][m_name] = True
                if c_snomed and not matched.get("snomed"):
                    matched["snomed"] = c_snomed
                if len(c_text) > len(matched["text"]):
                    matched["text"] = c_text
            else:
                entry = {
                    "text": c_text,
                    "category": cat,
                    "snomed": c_snomed,
                    "models": {"ClinicalBERT": False, "MedCAT": False, "Qwen": False}
                }
                entry["models"][m_name] = True
                merged_entities.append(entry)

        # Qwen Dynamic LLM Junk & Noise Filter
        merged_entities = qwen_filter_junk_entities(merged_entities, clinical_text)

        for entry in merged_entities:
            counter += 1
            confidence, validation_status, n_matches = calculate_consensus_confidence(entry["models"])

            status_val = {
                "Diagnosis": "Current",
                "Symptom": "Current",
                "Procedure": "Performed",
                "Medication": "Current",
                "Vital": "N/A"
            }.get(cat, "Current")

            all_entities.append({
                "id": f"E{counter}",
                "text": entry["text"],
                "category": cat,
                "confidence": float(confidence),
                "validation_status": validation_status,
                "status": status_val,
                "snomed": entry.get("snomed", ""),
                "models": entry["models"],
                "decision": "Yes"
            })

    all_entities.sort(key=lambda e: e["confidence"], reverse=True)
    return all_entities

# ─────────────────────────────────────────────────────────
# Load Pre-computed SLMS Extraction Results if available
# ─────────────────────────────────────────────────────────
def load_slms_precomputed_results() -> Dict[str, Any]:
    slms_json_path = os.path.join(SLMS_DIR, "slms_extraction_results.json")
    if os.path.exists(slms_json_path):
        try:
            with open(slms_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

# ─────────────────────────────────────────────────────────
# FastAPI Application & Global Session State
# ─────────────────────────────────────────────────────────
app = FastAPI(title="NHS PDF Letter Clinical Extraction & Coding Platform")

@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

SESSION_FILE = os.path.join(BASE_DIR, "pdf_folder_session_progress.json")

def load_session() -> Dict[str, Any]:
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"folder_path": "", "records": [], "current_index": 0}

def save_session(session_data: Dict[str, Any]):
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Failed to save session: {e}")

# ─────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────
class LoadFolderRequest(BaseModel):
    folder_path: str

class EntityDecision(BaseModel):
    id: str
    text: str
    category: str
    confidence: Optional[float] = 0.95
    validation_status: Optional[str] = "Validated"
    status: str = "Current"
    snomed: str = ""
    decision: str = "Yes"

class ProcessRowRequest(BaseModel):
    row_index: int
    decisions: List[EntityDecision]

# ─────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────
@app.get("/api/default-folder")
def get_default_folder():
    default_dir = os.path.join(FINAL_PIPELINE_DIR, "sample_letters")
    if not os.path.exists(default_dir):
        default_dir = BASE_DIR
    return {"default_folder": default_dir}

@app.post("/api/load-folder")
def load_folder(req: LoadFolderRequest):
    folder = req.folder_path.strip()
    if not os.path.exists(folder) or not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")

    pdf_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.pdf', '.txt'))])
    if not pdf_files:
        raise HTTPException(status_code=400, detail=f"No PDF or TXT files found in: {folder}")

    session = load_session()
    existing_records_by_name = {r["filename"]: r for r in session.get("records", [])}
    precomputed = load_slms_precomputed_results()

    new_records = []
    for idx, fname in enumerate(pdf_files):
        fpath = os.path.join(folder, fname)
        if fname in existing_records_by_name:
            rec = existing_records_by_name[fname]
            rec["index"] = idx
            rec["file_path"] = fpath
            new_records.append(rec)
        else:
            # Format precomputed extractions if available for instant display
            formatted = []
            if fname in precomputed:
                data = precomputed[fname]
                counter = 1
                cat_mapping = [
                    ("diagnoses", "Diagnosis", "Current"),
                    ("symptoms", "Symptom", "Current"),
                    ("procedures", "Procedure", "Performed"),
                    ("medications", "Medication", "Current"),
                    ("vitals", "Vital", "N/A")
                ]
                for key, category_name, default_status in cat_mapping:
                    for item in data.get(key, []):
                        ent_text = item.get("entity", "") if isinstance(item, dict) else str(item)
                        if not ent_text:
                            continue
                        status_val = item.get("status", default_status) if isinstance(item, dict) else default_status
                        snomed_val = item.get("snomed", "") if isinstance(item, dict) else ""
                        if snomed_val == "Not found":
                            snomed_val = ""
                        formatted.append({
                            "id": f"E{counter}",
                            "text": ent_text,
                            "category": category_name,
                            "confidence": 0.98,
                            "validation_status": "Validated",
                            "status": status_val,
                            "snomed": snomed_val,
                            "decision": "Yes"
                        })
                        counter += 1

            new_records.append({
                "index": idx,
                "filename": fname,
                "file_path": fpath,
                "text": "",
                "extracted_entities": formatted if formatted else None,
                "final_results": None,
                "reviewed": False,
                "has_extractions": bool(formatted)
            })

    session["folder_path"] = folder
    session["records"] = new_records
    session["current_index"] = 0
    save_session(session)

    return {
        "success": True,
        "folder_path": folder,
        "total_records": len(new_records),
        "records": new_records,
        "current_index": 0
    }

@app.get("/api/record/{index}")
def get_record(index: int):
    session = load_session()
    records = session.get("records", [])
    if index < 0 or index >= len(records):
        raise HTTPException(status_code=404, detail="Record index out of bounds")

    rec = records[index]
    # If text is not extracted yet, run PaddleOCR
    if not rec["text"] or not rec["text"].strip():
        fpath = rec["file_path"]
        if fpath.lower().endswith('.pdf') and PADDLE_AVAILABLE:
            try:
                ocr = get_paddle_ocr()
                print(f"[OCR] Extracting text for: {rec['filename']}...")
                res = extract_text_from_pdf_paddle(fpath, ocr_instance=ocr)
                rec["text"] = res.get("full_text", "")
            except Exception as e:
                print(f"[ERROR] PaddleOCR failed for {fpath}: {e}")
                rec["text"] = f"[OCR Error: Could not extract text from {rec['filename']}]"
        else:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    rec["text"] = f.read()
            except Exception as e:
                rec["text"] = f"[Error reading file: {e}]"

        session["records"][index] = rec
        save_session(session)

    return rec

@app.post("/api/extract/{index}")
def extract_pdf_entities(index: int, force: bool = Query(False)):
    session = load_session()
    records = session.get("records", [])
    if index < 0 or index >= len(records):
        raise HTTPException(status_code=404, detail="Record not found")

    rec = records[index]
    if not force and rec.get("extracted_entities") is not None:
        return {"entities": rec["extracted_entities"]}

    clinical_text = rec.get("text", "")
    if not clinical_text or not clinical_text.strip():
        fpath = rec["file_path"]
        if fpath.lower().endswith('.pdf') and PADDLE_AVAILABLE:
            ocr = get_paddle_ocr()
            res = extract_text_from_pdf_paddle(fpath, ocr_instance=ocr)
            clinical_text = res.get("full_text", "")
            rec["text"] = clinical_text

    if not clinical_text or not clinical_text.strip():
        return {"entities": []}

    print(f"[EXTRACT] Running 3-Model Ensemble (strictly final_pipeline) for: {rec['filename']}...")
    
    # 1. ClinicalBERT (from final_pipeline/clinical_bert.py)
    cb_all = {"Diagnosis": [], "Symptom": [], "Procedure": [], "Medication": [], "Vital": []}
    if extract_clinicalbert_entities is not None:
        try:
            cb_all = extract_clinicalbert_entities(clinical_text)
        except Exception as e_cb:
            print(f"[WARN] final_pipeline/clinical_bert extraction error: {e_cb}")

    # 2. MedCAT (from final_pipeline/medcatex.py)
    mc_all = {"Diagnosis": [], "Symptom": [], "Procedure": [], "Medication": [], "Vital": []}
    if extract_entities_with_medcat is not None:
        try:
            get_medcat()
            mc_all = extract_entities_with_medcat(clinical_text)
        except Exception as e_mc:
            print(f"[WARN] final_pipeline/medcatex extraction error: {e_mc}")

    # 3. Qwen (from final_pipeline/slms)
    qw_all = {"diagnoses": [], "symptoms": [], "procedures": [], "medications": [], "vitals": []}
    pipeline = get_slms_pipeline()
    if pipeline:
        try:
            orig_cwd = os.getcwd()
            os.chdir(SLMS_DIR)
            try:
                res_slms = pipeline.process_letter(clinical_text)
                qw_all = {
                    "diagnoses": [i.get("entity", "") if isinstance(i, dict) else str(i) for i in res_slms.get("diagnoses", [])],
                    "symptoms": [i.get("entity", "") if isinstance(i, dict) else str(i) for i in res_slms.get("symptoms", [])],
                    "procedures": [i.get("entity", "") if isinstance(i, dict) else str(i) for i in res_slms.get("procedures", [])],
                    "medications": [i.get("entity", "") if isinstance(i, dict) else str(i) for i in res_slms.get("medications", [])],
                    "vitals": [i.get("entity", "") if isinstance(i, dict) else str(i) for i in res_slms.get("vitals", [])]
                }
            finally:
                os.chdir(orig_cwd)
        except Exception as e_slms:
            print(f"[WARN] final_pipeline/slms Qwen extraction error: {e_slms}")

    if not any(qw_all.values()):
        try:
            import ollama
            prompt = f"Extract all diagnoses, symptoms, procedures, medications, vitals from:\n{clinical_text}"
            res = ollama.chat(model='qwen2.5:7b-instruct', messages=[{'role':'user','content':prompt}], format='json')
            content = res['message']['content']
            data = json.loads(content)
            qw_all = {
                "diagnoses": data.get("diagnoses", []),
                "symptoms": data.get("symptoms", []),
                "procedures": data.get("procedures", []),
                "medications": data.get("medications", []),
                "vitals": data.get("vitals", [])
            }
        except Exception as e_ollama:
            print(f"[ERROR] Direct Ollama fallback failed: {e_ollama}")

    # 4. Build 3-Model Cross-Checking Consensus strictly using final_pipeline models
    formatted_entities = build_3_model_consensus_from_final_pipeline(cb_all, mc_all, qw_all, clinical_text)

    # Clean up IDs and decision values
    for idx_ent, ent in enumerate(formatted_entities):
        ent["id"] = f"E{idx_ent + 1}"
        if not ent.get("decision"):
            ent["decision"] = "Yes"

    rec["extracted_entities"] = formatted_entities
    rec["has_extractions"] = True
    session["records"][index] = rec
    save_session(session)

    return {"entities": formatted_entities}

@app.get("/api/snomed-search")
def snomed_search(term: str = Query(..., min_length=2), category: Optional[str] = None):
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
def process_row(request: ProcessRowRequest):
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
    session = load_session()
    folder_path = session.get("folder_path")
    records = session.get("records", [])

    if not folder_path or not records:
        raise HTTPException(status_code=400, detail="No folder session active")

    try:
        def fmt(items, key="entity"):
            if not items:
                return ""
            return "\n".join([f"{i+1}. {e.get(key,'')} [{e.get('snomed','')}]" if e.get('snomed') else f"{i+1}. {e.get(key,'')}" for i, e in enumerate(items)])

        export_rows = []
        for rec in records:
            res = rec.get("final_results") or {}
            export_rows.append({
                "Filename": rec["filename"],
                "Diagnoses": fmt(res.get("diagnoses", [])),
                "Symptoms": fmt(res.get("symptoms", [])),
                "Procedures": fmt(res.get("procedures", [])),
                "Medications": fmt(res.get("medications", [])),
                "Vitals": fmt(res.get("vitals", []))
            })

        df_export = pd.DataFrame(export_rows)
        output_dir = os.path.join(BASE_DIR, "output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"pdf_folder_extraction_{timestamp}.xlsx"
        output_path = os.path.abspath(os.path.join(output_dir, output_filename))
        df_export.to_excel(output_path, index=False)

        return {
            "success": True,
            "filename": output_filename,
            "path": output_path,
            "download_url": f"/api/download/{output_filename}"
        }
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

@app.get("/api/view-pdf/{index}")
def view_pdf(index: int):
    session = load_session()
    records = session.get("records", [])
    if index < 0 or index >= len(records):
        raise HTTPException(status_code=404, detail="Record index out of bounds")
    fpath = records[index]["file_path"]
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"File not found: {fpath}")
    if fpath.lower().endswith('.pdf'):
        return FileResponse(
            path=fpath,
            media_type="application/pdf",
            content_disposition_type="inline"
        )
    else:
        return FileResponse(
            path=fpath,
            media_type="text/plain",
            content_disposition_type="inline"
        )

# ─────────────────────────────────────────────────────────
# Static Frontend Files Mount
# ─────────────────────────────────────────────────────────
static_dir = os.path.join(BASE_DIR, "ocr_static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(content={"error": "Frontend not found"}, status_code=503)

# ─────────────────────────────────────────────────────────
# Launcher
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" NHS PDF Letter Clinical Extraction & Coding Platform Web UI")
    print("=" * 70)
    print("[OK] PaddleOCR available:", PADDLE_AVAILABLE)
    print("Starting Uvicorn server on http://127.0.0.1:8000 ...")

    def open_browser():
        time.sleep(2.0)
        print("Opening Web UI in default browser...")
        webbrowser.open("http://127.0.0.1:8000")

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        import uvicorn
        uvicorn.run("run_ocr_ui:app", host="127.0.0.1", port=8000, reload=True, reload_dirs=[BASE_DIR])
    except KeyboardInterrupt:
        print("\nStopping Web UI server. Goodbye!")
    except Exception as e:
        print(f"\n❌ Server crashed: {e}")

if __name__ == "__main__":
    main()
