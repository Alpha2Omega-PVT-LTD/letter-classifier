import os
import sys
import re
import json
import torch
import pandas as pd
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Ensure local module imports work regardless of CWD
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
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

# Try optional spaCy / NegEx
SPACY_AVAILABLE = False
nlp = None
try:
    import spacy
    from negspacy.negation import Negex
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("negex", config={"ent_types": ["CLINICAL_ENTITY"]})
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False


# -----------------------------
# ClinicalBERT Model Initialization
# -----------------------------
MODEL_NAME = "nlpie/clinical-distilbert-i2b2-2010"

print(f"Loading ClinicalBERT model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    model = model.to("cuda")
model.eval()
id2label = model.config.id2label
print("ClinicalBERT model loaded successfully.")


# -----------------------------
# Guardrails & Filters
# -----------------------------
SEMANTIC_BLOCKLIST = {
    "any concerns", "concerns", "concern", "none", "version", "chan", "problems",
    "service", "doctor", "gp", "patient", "medication changes", "discharge diagnosis",
    "gp actions", "all regular medications", "exclusion conditions", "screening questionnaire",
    "access policy", "contact details", "practice manager", "medical practice",
    "medical centre", "home blood pressure diary", "patient record", "discharge summary"
}

def passes_semantic_filter(entity_text: str) -> bool:
    """Filters out short tokens and structural/conversational filler."""
    text_lower = entity_text.lower().strip()
    if len(text_lower) <= 2 or text_lower in SEMANTIC_BLOCKLIST:
        return False
    if text_lower.startswith(("dear ", "yours ", "ref:", "dob:", "nhs ")):
        return False
    return True


def check_spacy_negation(entity_text: str, sentence_text: str) -> bool:
    if not SPACY_AVAILABLE or nlp is None:
        return False
    try:
        doc = nlp(sentence_text)
        start_char = sentence_text.find(entity_text)
        if start_char == -1:
            return False
        end_char = start_char + len(entity_text)
        span = doc.char_span(start_char, end_char, label="CLINICAL_ENTITY")
        if span is not None:
            try:
                doc.ents = [span]
            except Exception:
                pass
        for ent in doc.ents:
            if ent.label_ == "CLINICAL_ENTITY" and ent.text == entity_text:
                if getattr(ent._, 'negex', False):
                    return True
    except Exception:
        pass
    return False


def is_negated_or_hypothetical(entity_text: str, sentence_text: str) -> bool:
    """Checks if an extracted entity is negated or hypothetical in context."""
    escaped_ent = re.escape(entity_text)
    pattern = re.compile(
        rf"{escaped_ent}\s*\??:\s*(none|no)\b|\bno\s+{escaped_ent}\b|\bdenies\s+{escaped_ent}\b|\bno\s+history\s+of\s+{escaped_ent}\b|\bnone\s+required\b",
        re.IGNORECASE
    )
    if pattern.search(sentence_text):
        return True
    return check_spacy_negation(entity_text, sentence_text)


# -----------------------------
# Entity Classification Logic
# -----------------------------
def classify_entity(entity_text: str, model_category: str) -> str:
    """
    Classifies entity into 'Diagnosis', 'Symptom', 'Procedure', 'Medication', or 'Vital'.
    Enforces strict guardrails so symptoms (e.g. chronic back pain) and procedures (e.g. physiotherapy)
    are NEVER misclassified as Medication.
    """
    text_lower = entity_text.lower().strip()

    # 1. Vital Signs
    vitals_terms = [
        'blood pressure', 'temperature', 'pulse', 'heart rate', 'respiratory rate',
        'oxygen saturation', 'weight', 'height', 'creatinine', 'platelets', 'hemoglobin',
        'bmi', 'systolic', 'diastolic', 'o2 sat', 'bp diary'
    ]
    vitals_abbr_pattern = re.compile(r'\b(bp|hr|rr|wt|ht|hb|crp|esr|wbc|bmi)\b', re.IGNORECASE)

    if vitals_abbr_pattern.search(text_lower) or any(term in text_lower for term in vitals_terms):
        if 'weight loss' not in text_lower and 'weight gain' not in text_lower:
            return 'Vital'

    # 2. Procedure Terms (High Priority - physiotherapy, scans, referrals, surgery, tests)
    procedure_terms = [
        'physiotherapy', 'physio', 'surgery', 'biopsy', 'scan', 'mri', 'ct', 'ultrasound', 'x-ray',
        'referral', 'referred', 'appointment', 'checkup', 'ecg', 'endoscopy', 'colonoscopy',
        'blood test', 'urine test', 'injection', 'rehabilitation', 'follow-up', 'investigation',
        'gp review', 'examination', 'pathway', 'dressing', 'treatment'
    ]
    if any(term in text_lower for term in procedure_terms) or text_lower.endswith(('ectomy', 'otomy', 'plasty', 'scopy', 'graphy')):
        return 'Procedure'

    # 3. Symptom Terms (High Priority - pain, cough, fever, dizziness, nausea, SOB)
    symptom_terms = [
        'pain', 'nocturnal pain', 'back pain', 'chest pain', 'ache', 'aching', 'cough', 'fever', 'pyrexia', 'nausea',
        'vomiting', 'dizziness', 'fatigue', 'shortness of breath', 'sob', 'breathlessness',
        'rash', 'headache', 'swelling', 'edema', 'weakness', 'diarrhoea', 'constipation',
        'numbness', 'tingling', 'stiffness', 'cramp', 'spasm'
    ]
    if any(term in text_lower for term in symptom_terms):
        return 'Symptom'

    # 4. Explicit Diagnosis Terms
    diagnosis_terms = [
        'infection', 'urinary tract infection', 'uti', 'delirium', 'cancer', 'carcinoma',
        'hypertension', 'diabetes', 'asthma', 'copd', 'arthritis', 'osteoarthritis', 'disease', 'syndrome',
        'disorder', 'suspected cancer'
    ]
    if any(term in text_lower for term in diagnosis_terms):
        return 'Diagnosis'

    # 5. Medication (Strict check on drug names & specific suffixes, excluding non-med words)
    med_keywords = [
        'mounjaro', 'aspirin', 'paracetamol', 'ibuprofen', 'metformin',
        'atorvastatin', 'amlodipine', 'lisinopril', 'levothyroxine', 'albuterol',
        'ventolin', 'omeprazole', 'simvastatin', 'ramipril', 'bisoprolol', 'prednisolone',
        'warfarin', 'apixaban', 'clopidogrel', 'gabapentin', 'sertraline', 'amoxicillin',
        'codeine', 'morphine', 'furosemide', 'tramadol', 'co-codamol', 'naproxen'
    ]
    med_suffixes = ('pam', 'ol', 'ine', 'ide', 'pril', 'sartan', 'statin', 'asone', 'olol', 'cillin')
    non_med_words = {'pain', 'brain', 'skin', 'stain', 'vein', 'strain', 'sprain', 'drain', 'grain', 'main', 'chain', 'domain'}

    if not any(w in text_lower for w in non_med_words):
        if any(k in text_lower for k in med_keywords):
            return 'Medication'
        if text_lower.endswith(med_suffixes) and len(text_lower) > 4:
            return 'Medication'

    # Fallback to model's predicted category
    if model_category == 'problem':
        return 'Diagnosis'
    elif model_category in ['treatment', 'test']:
        return 'Procedure'

    return 'Diagnosis'


# -----------------------------
# NER Engine & Span Merger
# -----------------------------
def run_ner_with_offsets(sentence: str):
    """Runs model to get token predictions with character offsets."""
    encoding = tokenizer(
        sentence,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    if device == 0:
        encoding = {k: v.to("cuda") for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits[0]
    scores = torch.softmax(logits, dim=-1)
    pred_ids = torch.argmax(scores, dim=-1).tolist()
    pred_scores = scores.max(dim=-1).values.tolist()

    tokens = []
    for (start, end), pred_id, score in zip(offset_mapping, pred_ids, pred_scores):
        if start == end:
            continue
        tokens.append({"start": start, "end": end, "label": id2label[pred_id], "score": score})

    return tokens


def merge_spans(tokens, sentence: str, target_cat: str, min_score: float = 0.5):
    """Merges adjacent sub-word tokens into single entity phrases."""
    spans = []
    current_start, current_end, current_scores = None, None, []

    def flush():
        if current_start is not None:
            spans.append({
                "start": current_start,
                "end": current_end,
                "score": sum(current_scores) / len(current_scores)
            })

    for tok in tokens:
        is_candidate = (target_cat in tok["label"].lower()) and tok["score"] >= min_score
        if is_candidate:
            if current_start is None:
                current_start, current_end = tok["start"], tok["end"]
                current_scores = [tok["score"]]
            else:
                gap = sentence[current_end:tok["start"]]
                if gap.strip() == "":
                    current_end = tok["end"]
                    current_scores.append(tok["score"])
                else:
                    flush()
                    current_start, current_end = tok["start"], tok["end"]
                    current_scores = [tok["score"]]
        else:
            if current_start is not None:
                flush()
                current_start, current_end, current_scores = None, None, []

    flush()

    results = []
    for sp in spans:
        txt = sentence[sp["start"]:sp["end"]].strip()
        txt = re.sub(r'[\s,:\-_]+(of|with|in|for|and|or|to|a|an|the)$', '', txt, flags=re.IGNORECASE).strip()
        if txt:
            results.append({"text": txt, "start": sp["start"], "end": sp["end"], "score": sp["score"]})
    return results


# -----------------------------
# Rule-based Vital Signs Scanner
# -----------------------------
def extract_vitals_rules(sentence: str) -> List[Dict[str, Any]]:
    """Extracts explicit numeric vitals like Blood Pressure, Temperature, BMI, etc."""
    vitals = []
    sent_lower = sentence.lower()

    # Avoid matching dates (DOB, Admission date, Discharge date)
    if any(date_kw in sent_lower for date_kw in ["dob", "date of birth", "admission date", "discharge date", "received / date"]):
        pass
    else:
        # Blood Pressure readings (e.g., 147/97, 150/99)
        bp_matches = re.findall(r'\b([7-9]\d|1\d{2}|2[0-4]\d)/([4-9]\d|1[0-4]\d)\b', sentence)
        if bp_matches:
            for sys_val, dia_val in bp_matches:
                vitals.append({
                    "text": f"Blood Pressure: {sys_val}/{dia_val}",
                    "score": 1.0,
                    "sentence": sentence
                })

    # BMI readings (e.g., BMI 38.82)
    bmi_match = re.search(r'\bbmi\s*[:=]?\s*(\d+(?:\.\d+)?)\b', sent_lower)
    if bmi_match:
        vitals.append({
            "text": f"BMI: {bmi_match.group(1)}",
            "score": 1.0,
            "sentence": sentence
        })

    # Weight / Height
    wt_match = re.search(r'\bweight\s*[:=]?\s*(\d+(?:\.\d+)?\s*(?:kg|lbs)?)\b', sent_lower)
    if wt_match:
        vitals.append({
            "text": f"Weight: {wt_match.group(1)}",
            "score": 1.0,
            "sentence": sentence
        })

    return vitals


# -----------------------------
# Document & Sentence Processing via PaddleOCR
# -----------------------------
def read_letter_file(filepath: str, ocr_instance: Any = None) -> str:
    """
    Calls PaddleOCR to extract text contents from a PDF or text letter.
    Falls back to pypdf or file read if OCR is unavailable.
    """
    if filepath.endswith('.pdf'):
        if PADDLE_AVAILABLE:
            try:
                print(f"  [PaddleOCR] Extracting OCR text from: {os.path.basename(filepath)}...")
                ocr_res = extract_text_from_pdf_paddle(filepath, ocr_instance=ocr_instance)
                if ocr_res.get("full_text") and ocr_res["full_text"].strip():
                    return ocr_res["full_text"]
            except Exception as e:
                print(f"  [warn] PaddleOCR extraction failed on {filepath}, trying pypdf fallback: {e}")

        if PYPDF_AVAILABLE:
            reader = pypdf.PdfReader(filepath)
            pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
            return "\n".join(pages_text)
        raise RuntimeError("No PDF reader available.")
    else:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


def split_into_sentences(text: str) -> List[str]:
    """Splits raw document text into clean line-by-line and sentence fragments."""
    lines = text.splitlines()
    sentences = []
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        parts = [p.strip() for p in re.split(r'\.(?=\s+[A-Z0-9]|$)', line_clean) if p.strip()]
        sentences.extend(parts)
    return sentences


def extract_entities_from_letter(text: str, min_score: float = 0.5) -> Dict[str, List[Dict[str, Any]]]:
    """
    Core extraction function using ClinicalBERT + rule fallback and guardrails.
    Returns dictionary with keys: 'Diagnosis', 'Symptom', 'Procedure', 'Medication', 'Vital'.
    """
    sentences = split_into_sentences(text)

    extracted = {
        "Diagnosis": [],
        "Symptom": [],
        "Procedure": [],
        "Medication": [],
        "Vital": []
    }
    seen = {k: set() for k in extracted}

    for sentence in sentences:
        # Rule-based Vitals
        rule_vitals = extract_vitals_rules(sentence)
        for rv in rule_vitals:
            key = rv["text"].lower()
            if key not in seen["Vital"]:
                seen["Vital"].add(key)
                extracted["Vital"].append(rv)

        # ClinicalBERT NER Token Extraction
        try:
            tokens = run_ner_with_offsets(sentence)
        except Exception as e:
            print(f"  [warn] NER failed on sentence: '{sentence[:30]}...': {e}")
            continue

        for raw_cat in ["problem", "treatment", "test"]:
            spans = merge_spans(tokens, sentence, raw_cat, min_score=min_score)
            for sp in spans:
                entity_text = sp["text"]

                if not passes_semantic_filter(entity_text):
                    continue
                if is_negated_or_hypothetical(entity_text, sentence):
                    continue

                target_cat = classify_entity(entity_text, raw_cat)
                key = entity_text.lower()

                if key not in seen[target_cat]:
                    seen[target_cat].add(key)
                    extracted[target_cat].append({
                        "text": entity_text,
                        "score": round(sp["score"], 4),
                        "sentence": sentence
                    })

    return extracted


# -----------------------------
# Batch Processing & Report Generation
# -----------------------------
def process_sample_letters(sample_dir: str = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Calls PaddleOCR on sample letters, then extracts Diagnosis, Symptom, Procedure using ClinicalBERT.
    """
    if sample_dir is None:
        sample_dir = os.path.join(CURRENT_DIR, "sample_letters")
        if not os.path.exists(sample_dir):
            sample_dir = "final_pipeline/sample_letters"

    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"Sample letters directory not found: {sample_dir}")

    files = sorted([f for f in os.listdir(sample_dir) if f.endswith(('.pdf', '.txt'))])
    if not files:
        print(f"No .pdf or .txt sample letters found in {sample_dir}")
        return {}, pd.DataFrame()

    print(f"\n[Pipeline] Found {len(files)} sample letters in '{sample_dir}'")
    
    # Initialize PaddleOCR engine once for reuse
    ocr_instance = None
    if PADDLE_AVAILABLE:
        print("[PaddleOCR] Initializing PaddleOCR engine...")
        ocr_instance = initialize_paddle_ocr()

    all_results = {}
    rows = []

    for fname in files:
        fpath = os.path.join(sample_dir, fname)
        print(f"\n--- Processing Document: {fname} ---")

        # 1. Get text output from PaddleOCR
        ocr_text = read_letter_file(fpath, ocr_instance=ocr_instance)
        
        # 2. Extract Diagnosis, Symptom, Procedure using ClinicalBERT
        print(f"  [ClinicalBERT] Extracting entities (Diagnosis, Symptom, Procedure)...")
        extracted = extract_entities_from_letter(ocr_text)
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
    json_path = os.path.join(CURRENT_DIR, "clinicalbert_extraction_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved extracted results to JSON: {json_path}")

    # 2. Output Excel File
    excel_path = os.path.join(CURRENT_DIR, "clinicalbert_extraction_results.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Saved extracted results to Excel: {excel_path}\n")

    # 3. Print Console Summary Table
    print("=" * 80)
    print("PADDLE OCR + CLINICALBERT EXTRACTION SUMMARY")
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
