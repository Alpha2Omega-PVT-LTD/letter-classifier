import os
import sys
import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Ensure local module imports work regardless of CWD
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Import PaddleOCR module functions
try:
    from paddle_ocr import extract_text_from_pdf_paddle, initialize_paddle_ocr, PADDLE_AVAILABLE
except ImportError:
    try:
        from .paddle_ocr import extract_text_from_pdf_paddle, initialize_paddle_ocr, PADDLE_AVAILABLE
    except ImportError:
        PADDLE_AVAILABLE = False

# Import pypdf as fallback
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# -----------------------------
# MedCAT Model Pack Initialization
# -----------------------------
try:
    from medcat.cat import CAT
    MEDCAT_AVAILABLE = True
except ImportError:
    MEDCAT_AVAILABLE = False

MODEL_PACK_DIR = os.path.join(WORKSPACE_DIR, "v2_Snomed2025_MIMIC_IV_bbe806e192df009f")
if not os.path.exists(MODEL_PACK_DIR):
    MODEL_PACK_DIR = os.path.join(CURRENT_DIR, "v2_Snomed2025_MIMIC_IV_bbe806e192df009f")

cat_model = None

def load_medcat_engine():
    global cat_model
    if not MEDCAT_AVAILABLE:
        raise RuntimeError("MedCAT library is not installed in the python environment.")
    if not os.path.exists(MODEL_PACK_DIR):
        raise FileNotFoundError(f"MedCAT model pack directory not found at {MODEL_PACK_DIR}")
    
    print(f"Loading MedCAT model pack: {MODEL_PACK_DIR}...")
    cat_model = CAT.load_model_pack(MODEL_PACK_DIR)
    print("MedCAT model loaded successfully.")
    return cat_model


# -----------------------------
# Administrative & Document Structure Pattern Filter (Regex)
ADMIN_HEADER_PATTERN = re.compile(
    r'\b(dob|date\s+of\s+birth|address|admission\s+date|discharge\s+date|ref|ref\s*:|tel|phone|email|dear|yours|practice\s+manager|medical\s+centre|hospital|patient|doctor|gp|nhs\s+no|nhs\s+number)\b',
    re.IGNORECASE
)

NEGATED_VALUES = {"false", "negated", "absent", "no", "negative"}

def passes_general_filter(entity_text: str) -> bool:
    """General structural filter based on token length and standard document header prefixes."""
    text = entity_text.strip()
    if len(text) <= 2:
        return False
    if ADMIN_HEADER_PATTERN.search(text):
        return False
    return True


# -----------------------------
# Dynamic Regex Vital Signs Scanner
# -----------------------------
VITALS_REGEX_PATTERNS = [
    ("Blood Pressure", re.compile(r'\b(?:BP|Blood\s+Pressure)?\s*[:=]?\s*([7-9]\d|1\d{2}|2[0-4]\d)\s*/\s*([4-9]\d|1[0-4]\d)\b', re.IGNORECASE)),
    ("Heart Rate", re.compile(r'\b(?:HR|Heart\s+Rate|Pulse)\s*[:=]?\s*([3-9]\d|1\d{2}|200)\s*(?:bpm)?\b', re.IGNORECASE)),
    ("Temperature", re.compile(r'\b(?:Temp|Temperature)\s*[:=]?\s*([34]\d(?:\.\d)?|9[5-9](?:\.\d)?|10[0-6](?:\.\d)?)\s*(?:°?[CF]|degrees)?\b', re.IGNORECASE)),
    ("O2 Saturation", re.compile(r'\b(?:SpO2|O2\s+Sat|Oxygen\s+Saturation)\s*[:=]?\s*([89]\d|100)\s*%?\b', re.IGNORECASE)),
    ("BMI", re.compile(r'\b(?:BMI|Body\s+Mass\s+Index)\s*[:=]?\s*([1-6]\d(?:\.\d{1,2})?)\b', re.IGNORECASE)),
    ("Weight", re.compile(r'\b(?:Weight|Wt)\s*[:=]?\s*(\d{2,3}(?:\.\d{1,2})?)\s*(?:kg|lbs|st)?\b', re.IGNORECASE)),
]

def extract_vitals_via_regex(text: str) -> List[Dict[str, Any]]:
    """Extracts numeric vitals dynamically using universal medical regex patterns."""
    vitals = []
    lines = text.splitlines()

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        # Skip date header lines (e.g., DOB, admission date) to avoid false positives
        if re.search(r'\b(dob|date\s+of\s+birth|admission|discharge|received)\b', line_clean, re.IGNORECASE):
            continue

        for vital_label, pattern in VITALS_REGEX_PATTERNS:
            for match in pattern.finditer(line_clean):
                matched_str = match.group(0).strip()
                vitals.append({
                    "text": f"{vital_label}: {matched_str}" if not matched_str.lower().startswith(vital_label.lower()) else matched_str,
                    "confidence": 1.0
                })
    return vitals


# -----------------------------
# Dynamic MedCAT SNOMED Semantic Categorization
# -----------------------------
def get_snomed_type_names(type_ids: List[str]) -> List[str]:
    """Resolves SNOMED CT type IDs to human-readable semantic type names using MedCAT CDB."""
    names = []
    if cat_model and hasattr(cat_model.cdb, "type_id2info"):
        for tid in type_ids:
            info = cat_model.cdb.type_id2info.get(tid)
            if info:
                name = getattr(info, "name", str(tid)).lower().strip()
                names.append(name)
            else:
                names.append(str(tid).lower().strip())
    else:
        names = [str(tid).lower().strip() for tid in type_ids]
    return names


# SNOMED CT Semantic Type Categories
EXCLUDED_NON_CLINICAL_TYPES = {
    'geographic location', 'environment', 'occupation', 'person',
    'record artifact', 'organism', 'social concept', 'qualifier value',
    'spatial concept', 'attribute', 'event', 'unit of measure'
}

DIAGNOSIS_TYPES = {
    'disorder', 'disease', 'morphologic abnormality', 'pathologic function'
}
SYMPTOM_TYPES = {
    'finding', 'sign', 'symptom', 'clinical finding'
}
PROCEDURE_TYPES = {
    'procedure', 'regime/therapy', 'regime therapy', 'intervention',
    'health care activity', 'diagnostic procedure', 'surgical procedure'
}
MEDICATION_TYPES = {
    'substance', 'pharmaceutical / biologic product', 'clinical drug',
    'medicinal product', 'drug', 'supplement'
}


def classify_entity_by_snomed_semantics(entity_text: str, type_ids: List[str], preferred_name: str) -> str:
    """
    Pure semantic classification driven entirely by MedCAT SNOMED CT type definitions.
    Returns category string ('Diagnosis', 'Symptom', 'Procedure', 'Medication', 'Vital') or None if non-clinical.
    """
    type_names = set(get_snomed_type_names(type_ids))

    # 1. Discard non-clinical SNOMED types (locations, places, occupations, artifacts)
    if type_names & EXCLUDED_NON_CLINICAL_TYPES:
        # Check if disorder/procedure override exists
        if not (type_names & (DIAGNOSIS_TYPES | PROCEDURE_TYPES | MEDICATION_TYPES)):
            return None

    # 2. Medication (Substance / Clinical Drug / Product)
    if type_names & MEDICATION_TYPES:
        return 'Medication'

    # 3. Procedure (Procedure / Regime / Intervention / Activity)
    if type_names & PROCEDURE_TYPES:
        return 'Procedure'

    # 4. Diagnosis (Disorder / Disease / Morphologic Abnormality)
    if type_names & DIAGNOSIS_TYPES:
        return 'Diagnosis'

    # 5. Symptom (Sign / Symptom / Clinical Finding)
    if type_names & SYMPTOM_TYPES:
        # Guard against administrative findings
        text_lower = entity_text.lower().strip()
        if any(admin_word in text_lower for admin_word in ['date', 'dob', 'address', 'reach', 'attend', 'request', 'status', 'fit']):
            return None
        return 'Symptom'

    # 6. Never default unmapped words to Diagnosis!
    return None


# -----------------------------
# PDF Text Reading via PaddleOCR
# -----------------------------
def read_letter_file(filepath: str, ocr_instance: Any = None) -> str:
    """Calls PaddleOCR to extract text from PDF letters."""
    if filepath.endswith('.pdf'):
        if PADDLE_AVAILABLE:
            try:
                print(f"  [PaddleOCR] Extracting OCR text from: {os.path.basename(filepath)}...")
                ocr_res = extract_text_from_pdf_paddle(filepath, ocr_instance=ocr_instance)
                if ocr_res.get("full_text") and ocr_res["full_text"].strip():
                    return ocr_res["full_text"]
            except Exception as e:
                print(f"  [warn] PaddleOCR extraction failed on {filepath}, using pypdf: {e}")

        if PYPDF_AVAILABLE:
            reader = pypdf.PdfReader(filepath)
            pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
            return "\n".join(pages_text)
        raise RuntimeError("No PDF reader available.")
    else:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


# -----------------------------
# MedCAT Entity Extraction Engine
# -----------------------------
def extract_entities_with_medcat(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Runs MedCAT on full document text and organizes entities into categories."""
    extracted = {
        "Diagnosis": [],
        "Symptom": [],
        "Procedure": [],
        "Medication": [],
        "Vital": []
    }
    seen = {k: set() for k in extracted}

    # 1. Dynamic Regex Vitals Extraction
    for rv in extract_vitals_via_regex(text):
        key = rv["text"].lower()
        if key not in seen["Vital"]:
            seen["Vital"].add(key)
            extracted["Vital"].append(rv)

    # 2. MedCAT NER & Linking
    doc_result = cat_model.get_entities(text)
    entities = doc_result.get("entities", {}) if doc_result else {}

    for ent in entities.values():
        entity_text = ent.get("source_value") or ent.get("pretty_name") or ""
        if not passes_general_filter(entity_text):
            continue

        # Filter out negated entities from MetaCAT
        meta_anns = ent.get("meta_anns", {}) or {}
        presence = meta_anns.get("Presence", {}).get("value", "").lower()
        if presence in NEGATED_VALUES:
            continue

        # MedCAT context confidence score threshold
        acc = ent.get("acc", 0.0) or ent.get("context_similarity", 0.0)
        if acc < 0.20:
            continue

        type_ids = ent.get("type_ids", [])
        pref_name = ent.get("pretty_name", entity_text)

        # Dynamic Classification using SNOMED CT semantics
        target_cat = classify_entity_by_snomed_semantics(entity_text, type_ids, pref_name)
        if target_cat is None or target_cat not in extracted:
            continue

        key = entity_text.lower()
        if key not in seen[target_cat]:
            seen[target_cat].add(key)
            extracted[target_cat].append({
                "text": entity_text,
                "preferred_name": pref_name,
                "cui": ent.get("cui"),
                "confidence": round(float(acc), 4) if acc else None,
            })

    return extracted


# -----------------------------
# Batch Processing & Output Generation
# -----------------------------
def process_sample_letters(sample_dir: str = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if sample_dir is None:
        sample_dir = os.path.join(CURRENT_DIR, "sample_letters")
        if not os.path.exists(sample_dir):
            sample_dir = os.path.join(WORKSPACE_DIR, "final_pipeline", "sample_letters")

    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"Sample letters directory not found: {sample_dir}")

    files = sorted([f for f in os.listdir(sample_dir) if f.endswith(('.pdf', '.txt'))])
    if not files:
        print(f"No .pdf or .txt sample letters found in {sample_dir}")
        return {}, pd.DataFrame()

    print(f"\n[Pipeline] Found {len(files)} sample letters in '{sample_dir}'")
    
    # Initialize PaddleOCR engine
    ocr_instance = None
    if PADDLE_AVAILABLE:
        print("[PaddleOCR] Initializing PaddleOCR engine...")
        ocr_instance = initialize_paddle_ocr()

    # Load MedCAT model engine
    load_medcat_engine()

    all_results = {}
    rows = []

    for fname in files:
        fpath = os.path.join(sample_dir, fname)
        print(f"\n--- Processing Document: {fname} ---")

        # 1. Get OCR text from PaddleOCR
        ocr_text = read_letter_file(fpath, ocr_instance=ocr_instance)
        
        # 2. Extract entities using MedCAT + Dynamic Regex
        print(f"  [MedCAT] Extracting entities (Diagnosis, Symptom, Procedure)...")
        extracted = extract_entities_with_medcat(ocr_text)
        all_results[fname] = extracted

        row = {
            "Filename": fname,
            "Diagnosis": "; ".join([e["text"] for e in extracted["Diagnosis"]]),
            "Symptom": "; ".join([e["text"] for e in extracted["Symptom"]]),
            "Procedure": "; ".join([e["text"] for e in extracted["Procedure"]]),
            "Medication": "; ".join([e["text"] for e in extracted["Medication"]]),
            "Vital": "; ".join([e["text"] for e in extracted["Vital"]])
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return all_results, df


# -----------------------------
# Main Execution CLI
# -----------------------------
if __name__ == "__main__":
    results, df = process_sample_letters()

    # 1. Output JSON File
    json_path = os.path.join(CURRENT_DIR, "medcat_extraction_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved extracted results to JSON: {json_path}")

    # 2. Output Excel File
    excel_path = os.path.join(CURRENT_DIR, "medcat_extraction_results.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Saved extracted results to Excel: {excel_path}\n")

    # 3. Print Console Summary Table
    print("=" * 80)
    print("PADDLE OCR + MEDCAT EXTRACTION SUMMARY")
    print("=" * 80)
    for fname, ext in results.items():
        print(f"\n[Letter] {fname}")
        print(f"  - Diagnosis  : {', '.join([e['text'] for e in ext['Diagnosis']]) or 'None'}")
        print(f"  - Symptom    : {', '.join([e['text'] for e in ext['Symptom']]) or 'None'}")
        print(f"  - Procedure  : {', '.join([e['text'] for e in ext['Procedure']]) or 'None'}")
        if ext['Medication']:
            print(f"  - Medication : {', '.join([e['text'] for e in ext['Medication']])}")
        if ext['Vital']:
            print(f"  - Vitals     : {', '.join([e['text'] for e in ext['Vital']])}")
    print("=" * 80)