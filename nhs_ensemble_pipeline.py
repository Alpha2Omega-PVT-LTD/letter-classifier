import json
import pandas as pd
import re
import datetime
from typing import Optional, Dict, List, Any
import torch
import ollama
from transformers import AutoTokenizer, AutoModelForTokenClassification


# -----------------------------
# Load ClinicalBERT Model
# -----------------------------
# -----------------------------
# Lazy-loaded Models References
# -----------------------------
_tokenizer = None
_model = None
_cat = None
_device = None
_id2label = None

def get_clinicalbert_model():
    global _tokenizer, _model, _device, _id2label
    if _model is None:
        CLINICALBERT_MODEL = "nlpie/clinical-distilbert-i2b2-2010"
        print("Loading ClinicalBERT NER model (lazy)...")
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        _tokenizer = AutoTokenizer.from_pretrained(CLINICALBERT_MODEL)
        _model = AutoModelForTokenClassification.from_pretrained(CLINICALBERT_MODEL)
        _device = 0 if torch.cuda.is_available() else -1
        if _device == 0:
            _model = _model.to("cuda")
        _model.eval()
        _id2label = _model.config.id2label
        print("ClinicalBERT loaded successfully")
    return _tokenizer, _model, _device, _id2label

def get_medcat_model():
    global _cat
    if _cat is None:
        MEDCAT_MODEL_PATH = "./v2_Snomed2025_MIMIC_IV_bbe806e192df009f"
        print("Loading MedCAT model (lazy)...")
        from medcat.cat import CAT
        _cat = CAT.load_model_pack(MEDCAT_MODEL_PATH)
        print("MedCAT loaded successfully")
    return _cat

NEGATING_VALUES = {"negated", "hypothetical", "ruled out", "absent"}
NON_PATIENT_VALUES = {"family member", "other", "relative"}

BLOCKLIST_TYPES = {
    "body structure", "person", "organism", "qualifier value", 
    "record artifact", "namespace concept", "geographic location", 
    "racial group", "ethnic group", "social concept", 
    "intellectual product", "occupation", "cell", "cell structure", "specimen"
}


def is_excluded_by_meta(meta_anns: dict):
    """
    Checks MetaCAT annotations to exclude negated or non-patient mentions.
    """
    for task_name, task_result in meta_anns.items():
        value = str(task_result.get("value", "")).strip().lower()
        if value in NEGATING_VALUES or value in NON_PATIENT_VALUES:
            return True
    return False


def is_problem_label(label: str) -> bool:
    return "problem" in label.lower()


def run_clinicalbert_ner(sentence: str):
    """
    Runs ClinicalBERT NER manually to get character offsets.
    """
    tokenizer, model, device, id2label = get_clinicalbert_model()
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
        label = id2label[pred_id]
        tokens.append({"start": start, "end": end, "label": label, "score": score})

    return tokens


def merge_problem_spans(tokens, sentence: str, min_score: float = 0.5):
    """
    Stitches token-level ClinicalBERT predictions into phrases based on adjacency.
    """
    spans = []
    current_start = None
    current_end = None
    current_scores = []

    def flush():
        if current_start is not None:
            spans.append({
                "start": current_start,
                "end": current_end,
                "score": sum(current_scores) / len(current_scores),
            })

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        is_candidate = is_problem_label(tok["label"]) and tok["score"] >= min_score

        if is_candidate:
            if current_start is None:
                current_start, current_end = tok["start"], tok["end"]
                current_scores = [tok["score"]]
            else:
                gap_text = sentence[current_end:tok["start"]]
                if gap_text.strip() == "":
                    current_end = tok["end"]
                    current_scores.append(tok["score"])
                else:
                    flush()
                    current_start, current_end = tok["start"], tok["end"]
                    current_scores = [tok["score"]]
        else:
            if current_start is not None and i + 1 < n:
                next_tok = tokens[i + 1]
                gap_before = sentence[current_end:tok["start"]]
                gap_after = sentence[tok["end"]:next_tok["start"]]
                next_is_candidate = is_problem_label(next_tok["label"]) and next_tok["score"] >= min_score
                if gap_before.strip() == "" and gap_after.strip() == "" and next_is_candidate:
                    current_end = tok["end"]
                    current_scores.append(tok["score"])
                    i += 1
                    continue
            flush()
            current_start = None
            current_end = None
            current_scores = []
        i += 1
    flush()
    return spans


def get_clinicalbert_problem_spans(text: str):
    """
    Finds all 'problem' spans in the text using ClinicalBERT, mapped to full-text indices.
    """
    if not text or not text.strip():
        return []

    # Simple sentence splitting and tracking index offset
    # Since splitting on '.' might miss indices, we reconstruct carefully:
    sentence_matches = list(re.finditer(r'[^.]+', text))
    
    all_spans = []
    for match in sentence_matches:
        sentence = match.group(0)
        start_offset = match.start()
        
        tokens = run_clinicalbert_ner(sentence)
        merged = merge_problem_spans(tokens, sentence)
        
        for m_span in merged:
            all_spans.append({
                "start": start_offset + m_span["start"],
                "end": start_offset + m_span["end"],
                "score": m_span["score"]
            })
            
    return all_spans

vitals_abbr_pattern = re.compile(
    r'\b(bp|hr|rr|wt|ht|hb|crp|esr|wbc|ast|alt|ldl|hdl|sat|sats|bmi|egfr)\b',
    re.IGNORECASE
)
vitals_terms = [
    'blood pressure', 'temperature', 'temp', 'pulse', 'heart rate', 'respiratory rate',
    'oxygen saturation', 'weight', 'height', 'creatinine', 'platelets', 'hemoglobin',
    'sodium', 'potassium', 'urea', 'bilirubin', 'cholesterol', 'systolic', 'diastolic',
    'pulse rate', 'respirations', 'o2 sat', 'o2 saturation'
]
med_suffixes = (
    'pam', 'ol', 'in', 'ine', 'ide', 'fil', 'pril', 'sartan', 'oxacin', 'mycin',
    'olol', 'terol', 'statin', 'asone', 'pred', 'floxacin', 'cillin', 'mab', 'nib'
)
med_keywords = [
    'aspirin', 'paracetamol', 'ibuprofen', 'metformin', 'atorvastatin', 'amlodipine',
    'lisinopril', 'levothyroxine', 'albuterol', 'ventolin', 'omeprazole', 'simvastatin',
    'ramipril', 'bisoprolol', 'prednisolone', 'warfarin', 'apixaban', 'clopidogrel',
    'gabapentin', 'sertraline', 'amoxicillin', 'gliclazide', 'insulin', 'lantus', 'novorapid',
    'salbutamol', 'furosemide', 'spironolactone', 'tamsulosin', 'finasteride', 'atorva',
    'prednisone', 'morphine', 'codeine', 'tramadol', 'oxycodone', 'fentanyl', 'naproxen',
    'diclofenac', 'co-codamol', 'co-dydramol', 'methotrexate', 'hydroxychloroquine', 'gabapentine'
]
procedure_terms = [
    'surgery', 'appendectomy', 'biopsy', 'bypass', 'scan', 'mri', 'ct', 'ultrasound',
    'xray', 'x-ray', 'referral', 'referred', 'appointment', 'checkup', 'ecg', 'ekg',
    'endoscopy', 'colonoscopy', 'blood test', 'urine test', 'suture', 'excision',
    'infusion', 'physiotherapy', 'cbt', 'counselling', 'therapy', 'resection',
    'graft', 'stent', 'angioplasty', 'catheter', 'intubation', 'vaccination', 'immunisation',
    'injection', 'transfusion', 'drainage', 'amputation', 'scope', 'rehab', 'rehabilitation', 'therapy'
]

def matches_any_word(text_lower: str, keywords: list) -> bool:
    for kw in keywords:
        pattern = rf"\b{re.escape(kw)}\b"
        if re.search(pattern, text_lower):
            return True
    return False

def is_clinical_noise(text_lower: str) -> bool:
    """Helper to check if the candidate is a procedure, medication, or vital sign."""
    if matches_any_word(text_lower, vitals_terms) or vitals_abbr_pattern.search(text_lower):
        if 'weight loss' not in text_lower and 'weight gain' not in text_lower:
            return True
    if any(text_lower.endswith(sfx) for sfx in med_suffixes) or matches_any_word(text_lower, med_keywords):
        return True
    if matches_any_word(text_lower, procedure_terms) or any(text_lower.endswith(sfx) for sfx in ('ectomy', 'otomy', 'plasty', 'scopy')):
        return True
    return False


def extract_clinicalbert_diagnoses(text: str):
    """Runs ClinicalBERT independently and filters candidates down to diagnoses."""
    spans = get_clinicalbert_problem_spans(text)
    candidates = []
    seen = set()
    for sp in spans:
        span_text = text[sp["start"]:sp["end"]].strip()
        text_lower = span_text.lower()
        
        # Guardrail: skip if it matches clinical noise (procedure, med, vital)
        if len(text_lower) <= 2 or is_clinical_noise(text_lower):
            continue
            
        if text_lower not in seen:
            seen.add(text_lower)
            candidates.append({
                "text": span_text,
                "score": sp["score"]
            })
    return candidates


def extract_medcat_diagnoses(text: str):
    """Runs MedCAT independently and filters candidates down to diagnoses."""
    cat = get_medcat_model()
    mc_result = cat.get_entities(text)
    mc_entities = mc_result.get("entities", {})
    candidates = []
    seen = set()

    for ent_id, ent in mc_entities.items():
        type_ids = ent.get("type_ids", [])
        pretty_name = ent.get("pretty_name") or ent.get("source_value")
        meta_anns = ent.get("meta_anns", {})
        acc = ent.get("acc", 1.0)

        # MetaCAT filter
        if is_excluded_by_meta(meta_anns):
            continue

        # Confidence filter
        if acc is not None and acc < 0.2:
            continue

        # Resolve types
        type_names = set()
        for tid in type_ids:
            type_info = cat.cdb.type_id2info.get(tid)
            if type_info:
                type_names.add(type_info.name.lower())

        # Structural filter
        if type_names & BLOCKLIST_TYPES:
            continue

        text_lower = pretty_name.lower().strip()
        
        is_diag_type = "disorder" in type_names or "morphologic abnormality" in type_names
        is_finding = "finding" in type_names and not is_clinical_noise(text_lower)

        if (is_diag_type or is_finding) and text_lower not in seen:
            seen.add(text_lower)
            candidates.append({
                "text": pretty_name,
                "score": acc if acc is not None else 1.0
            })
            
    return candidates

def qwen_process_report(text: str, candidate_list: list):
    """
    Performs independent zero-shot extraction AND rephrasing/deduplication
    of ClinicalBERT + MedCAT candidates in a single Ollama call to Qwen.
    """
    candidates_str = ", ".join(f'"{c}"' for c in candidate_list) if candidate_list else ""
    
    prompt = f"""You are a clinical expert system. Read the following clinical note and perform two tasks:
1. Extract a clean list of active, confirmed clinical diagnoses from the note (Zero-shot Extraction).
2. Clean, deduplicate, and rephrase/standardize the list of candidates extracted by other models (Rephrase & Deduplicate).

### Clinical Note:
{text}

### Candidates to Clean/Rephrase:
[{candidates_str}]

### Output Format (JSON):
Respond with ONLY a single valid JSON object containing exactly these two keys: "extracted" and "rephrased". No markdown code fences, no commentary.
{{
  "extracted": ["independent diagnosis 1", "independent diagnosis 2"],
  "rephrased": ["standardized candidate 1", "standardized candidate 2"]
}}"""
    try:
        response = ollama.chat(
            model='qwen2.5:7b-instruct',
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={
                'temperature': 0.05,
                'num_predict': 400
            }
        )
        content = response['message']['content']
        content = re.sub(r"^```(?:json)?\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)
        return data.get("extracted", []), data.get("rephrased", [])
    except Exception as e:
        print(f"Error in Qwen processing: {e}")
        return [], candidate_list


from difflib import SequenceMatcher

def get_fuzzy_similarity(s1, s2):
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def check_match_in_list(diagnosis_text: str, candidate_list: list):
    for c in candidate_list:
        sim = get_fuzzy_similarity(diagnosis_text, c)
        if sim >= 0.75 or diagnosis_text.lower() in c.lower() or c.lower() in diagnosis_text.lower():
            return True
    return False


# ─────────────────────────────────────────────────────────
# Exact Source Span Validation & Double-Verified Status Helpers
# ─────────────────────────────────────────────────────────

def find_exact_source_span(clinical_text: str, candidate_text: str) -> Optional[dict]:
    """
    Locates candidate_text within clinical_text.
    Strictly enforces that the candidate MUST exist in the source document.
    Returns dict with exact source_text, start, and end character offsets.
    If candidate is absent or paraphrased, returns None.
    """
    if not clinical_text or not candidate_text:
        return None

    cand = str(candidate_text).strip()
    if not cand:
        return None

    # 1. Exact string match
    idx = clinical_text.find(cand)
    if idx != -1:
        matched_str = clinical_text[idx:idx + len(cand)]
        return {
            "text": matched_str,
            "source_text": matched_str,
            "start": idx,
            "end": idx + len(cand)
        }

    # 2. Case-insensitive exact match
    pattern = re.escape(cand)
    match = re.search(pattern, clinical_text, re.IGNORECASE)
    if match:
        start, end = match.span()
        matched_str = clinical_text[start:end]
        return {
            "text": matched_str,
            "source_text": matched_str,
            "start": start,
            "end": end
        }

    # 3. Flexible punctuation / whitespace normalized match
    words = re.findall(r'\w+', cand)
    if words:
        regex_pattern = r'\b' + r'[\s\-_]+'.join(re.escape(w) for w in words) + r'\b'
        match = re.search(regex_pattern, clinical_text, re.IGNORECASE)
        if match:
            start, end = match.span()
            matched_str = clinical_text[start:end]
            return {
                "text": matched_str,
                "source_text": matched_str,
                "start": start,
                "end": end
            }

    return None


def determine_clinical_status(entity_text: str, category: str, start: int, end: int, clinical_text: str) -> str:
    """
    Double-verifies clinical status using surrounding sentence context,
    negation indicators, historical phrases, temporal date markers, and procedure status.
    """
    if not clinical_text:
        return "Current" if category in ("Diagnosis", "Symptom", "Medication") else ("Performed" if category == "Procedure" else "N/A")

    # Extract sentence / local window around entity
    window_start = max(0, start - 150)
    window_end = min(len(clinical_text), end + 150)
    window = clinical_text[window_start:window_end].lower()

    prefix = clinical_text[window_start:start].lower()
    full_sentence_prefix = prefix.split('.')[-1]

    # 1. Negation Check
    neg_patterns = r'\b(no|denies|denied|absent|ruled out|no evidence of|without|negative for|neither|nor|free of|unremarkable)\b'
    if re.search(neg_patterns, full_sentence_prefix):
        return "Negated"

    # 2. Suspected / Uncertainty Check
    suspect_patterns = r'\b(suggestive of|suspected|possible|probable|query|q\?|features of|likely|differential|to rule out|concerning for|appearances suggestive of)\b'
    if re.search(suspect_patterns, full_sentence_prefix) or re.search(suspect_patterns, window):
        return "Suspected"

    # 3. Historical Check (Phrase-based + Temporal / Dates)
    hist_phrase_patterns = r'\b(history of|past medical history|pmh|previous|h/o|formerly|prior|ex-|diagnosed in|known)\b'
    date_patterns = r'\b(19\d\d|20[0-2]\d)\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b|\b\d+\s+(years|months)\s+ago\b'

    if re.search(hist_phrase_patterns, full_sentence_prefix) or re.search(hist_phrase_patterns, window):
        return "Historical"
    if re.search(date_patterns, full_sentence_prefix) or re.search(date_patterns, window):
        if category in ("Diagnosis", "Symptom"):
            return "Historical"

    # 4. Resolved Check
    resolved_patterns = r'\b(resolved|cured|cleared|settled|discontinued|stopped)\b'
    if re.search(resolved_patterns, full_sentence_prefix) or re.search(resolved_patterns, window):
        return "Resolved"

    # 5. Procedure Specific Status
    if category == "Procedure":
        planned_patterns = r'\b(planned|scheduled|recommended|for|awaiting|to undergo|monitoring|under review|listed for)\b'
        if re.search(planned_patterns, full_sentence_prefix):
            return "Planned"
        return "Performed"

    if category in ("Diagnosis", "Symptom", "Medication"):
        return "Current"
    elif category == "Vital":
        return "N/A"

    return "Current"


def resolve_nested_entities(entities: list) -> list:
    """
    Handles nested source spans within the same category and phrase.
    Prefers the larger/most specific span for overlapping mentions of the same concept.
    Preserves distinct concepts with non-overlapping spans.
    """
    if len(entities) <= 1:
        return entities

    sorted_ents = sorted(entities, key=lambda e: (e["end"] - e["start"]), reverse=True)
    kept = []

    for cand in sorted_ents:
        cand_start = cand["start"]
        cand_end = cand["end"]
        cand_cat = cand["category"]

        is_nested_duplicate = False
        for parent in kept:
            if parent["category"] == cand_cat:
                parent_start = parent["start"]
                parent_end = parent["end"]
                if parent_start <= cand_start and cand_end <= parent_end and (parent_end - parent_start > cand_end - cand_start):
                    for m, val in cand["models"].items():
                        if val:
                            parent["models"][m] = True
                    is_nested_duplicate = True
                    break

        if not is_nested_duplicate:
            kept.append(cand)

    return sorted(kept, key=lambda e: e["start"])


def calculate_model_consensus(models_dict: dict):
    active_models = [m for m, val in models_dict.items() if val]
    n_matches = len(active_models)

    if n_matches >= 3:
        confidence = 0.98
        validation_status = "High Consensus — 3/3 models (ClinicalBERT, MedCAT, Qwen)"
        consensus = "3/3"
    elif n_matches == 2:
        confidence = 0.85
        validation_status = f"Consensus — 2/3 models ({' & '.join(active_models)})"
        consensus = "2/3"
    else:
        confidence = 0.70
        validation_status = f"Single Model — 1/3 models ({active_models[0] if active_models else 'Unknown'})"
        consensus = "1/3"

    return confidence, validation_status, consensus, n_matches


# ─────────────────────────────────────────────────────────
# Multi-Category 3-Model Extraction & Consensus
# ─────────────────────────────────────────────────────────
def extract_clinicalbert_all_categories(text: str):
    """
    Extracts Diagnoses, Symptoms, Procedures, Medications, Vitals using ClinicalBERT.
    """
    try:
        from clinicalbert_multi_category import extract_all_categories
        return extract_all_categories(text)
    except Exception as e:
        print(f"[WARN] ClinicalBERT multi-category extraction failed: {e}")
        return {"Diagnosis": [], "Symptom": [], "Procedure": [], "Medication": [], "Vital": []}


def extract_medcat_all_categories(text: str):
    """
    Extracts Diagnoses, Symptoms, Procedures, Medications, Vitals using MedCAT.
    """
    try:
        from medcat_multi_category import extract_all_categories_medcat
        cat = get_medcat_model()
        return extract_all_categories_medcat(text, cat)
    except Exception as e:
        print(f"[WARN] MedCAT multi-category extraction failed: {e}")
        return {"Diagnosis": [], "Symptom": [], "Procedure": [], "Medication": [], "Vital": []}


def build_3_model_multi_category_consensus(cb_cats: dict, mc_cats: dict, qwen_cats: dict, clinical_text: str = ""):
    """
    Cross-checks ClinicalBERT, MedCAT, and Qwen extractions for ALL 5 categories.
    Validates candidates against original clinical text, deduplicates spans, resolves
    nested entities, double-verifies clinical status, and computes post-validation consensus.
    """
    categories = ["Diagnosis", "Symptom", "Procedure", "Medication", "Vital"]
    qwen_key_map = {
        "Diagnosis": "diagnoses",
        "Symptom": "symptoms",
        "Procedure": "procedures",
        "Medication": "medications",
        "Vital": "vitals"
    }

    all_entities = []
    counter = 0

    for cat in categories:
        q_key = qwen_key_map[cat]

        cb_raw = cb_cats.get(cat, [])
        cb_list = [c["text"] for c in cb_raw] if isinstance(cb_raw, list) and cb_raw and isinstance(cb_raw[0], dict) else list(cb_raw)

        mc_raw = mc_cats.get(cat, [])
        mc_list = [c["text"] for c in mc_raw] if isinstance(mc_raw, list) and mc_raw and isinstance(mc_raw[0], dict) else list(mc_raw)

        qw_raw = qwen_cats.get(q_key, [])
        qw_list = [c["text"] for c in qw_raw] if isinstance(qw_raw, list) and qw_raw and isinstance(qw_raw[0], dict) else list(qw_raw)

        # 1. Source Span Validation & Provenance Collection
        span_map = {}  # key: (start, end) or lower text

        def add_candidate(cand_text, model_name):
            if not cand_text:
                return
            span_info = find_exact_source_span(clinical_text, str(cand_text)) if clinical_text else {
                "text": str(cand_text).strip(), "source_text": str(cand_text).strip(), "start": 0, "end": len(str(cand_text).strip())
            }
            if span_info is None:
                # Candidate not found in source text -> Reject!
                return

            key = (span_info["start"], span_info["end"]) if clinical_text else span_info["text"].lower()
            if key not in span_map:
                span_map[key] = {
                    "text": span_info["text"],
                    "source_text": span_info["source_text"],
                    "start": span_info["start"],
                    "end": span_info["end"],
                    "category": cat,
                    "models": {"ClinicalBERT": False, "MedCAT": False, "Qwen": False}
                }
            span_map[key]["models"][model_name] = True

        for c in cb_list:
            add_candidate(c, "ClinicalBERT")
        for c in mc_list:
            add_candidate(c, "MedCAT")
        for c in qw_list:
            add_candidate(c, "Qwen")

        candidates_for_cat = list(span_map.values())

        # 2. Nested Span Resolution (prefer larger/more specific spans in same phrase)
        resolved_spans = resolve_nested_entities(candidates_for_cat)

        # 3. Status Determination & Model Consensus
        for span in resolved_spans:
            counter += 1
            status = determine_clinical_status(span["text"], cat, span["start"], span["end"], clinical_text)
            confidence, validation_status, consensus, model_count = calculate_model_consensus(span["models"])

            all_entities.append({
                "id": f"ent-{counter}",
                "text": span["text"],
                "source_text": span["source_text"],
                "start": span["start"],
                "end": span["end"],
                "category": cat,
                "confidence": confidence,
                "validation_status": validation_status,
                "consensus": consensus,
                "model_count": model_count,
                "models": span["models"],
                "status": status,
                "normalized_concept": "",
                "snomed": "",
                "decision": "Yes"
            })

    # Sort all entities by confidence score descending (highest confidence first), then by character start
    all_entities.sort(key=lambda e: (e.get("confidence", 0.0), -e.get("start", 0)), reverse=True)

    return all_entities


# Alias for backward compatibility
build_3_model_consensus = build_3_model_multi_category_consensus



def format_numbered_list(items):
    if not items:
        return ""
    return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))


# -----------------------------
# Main Execution: Process Excel File
# -----------------------------
if __name__ == "__main__":
    input_file = "datavalid.xlsx"
    print(f"Loading input file: {input_file}")
    df = pd.read_excel(input_file)

    diagnoses_col = []
    confidence_col = []
    status_col = []

    for index, row in df.iterrows():
        print(f"Processing report {index + 1}/{len(df)}...")
        text = str(row["Cleaned Data"])
        
        if pd.isna(row["Cleaned Data"]) or not text.strip():
            diagnoses_col.append("")
            confidence_col.append("")
            status_col.append("")
        else:
            # 1. Extract independently using ClinicalBERT and MedCAT
            cb_candidates = extract_clinicalbert_diagnoses(text)
            mc_candidates = extract_medcat_diagnoses(text)
            
            # 2. Extract Qwen diagnoses and rephrase candidates in a single Ollama call
            raw_candidates = [c["text"] for c in cb_candidates] + [c["text"] for c in mc_candidates]
            qwen_extracted, rephrased_candidates = qwen_process_report(text, raw_candidates)
            
            # 3. Cross-check & validate confidence
            validated = build_3_model_consensus(cb_candidates, mc_candidates, qwen_extracted, rephrased_candidates)
            
            diag_list = [v["text"] for v in validated]
            conf_list = [f"{v['confidence']:.2f}" for v in validated]
            stat_list = [v["status"] for v in validated]
            
            diagnoses_col.append(format_numbered_list(diag_list))
            confidence_col.append(format_numbered_list(conf_list))
            status_col.append(format_numbered_list(stat_list))

    df["Diagnosis"] = diagnoses_col
    df["Confidence"] = confidence_col
    df["Validation_Status"] = status_col

    output_file = "ensemble_diagnoses_output.xlsx"
    try:
        df.to_excel(output_file, index=False)
        print("\nCompleted successfully!")
        print(f"Saved: {output_file}")
    except PermissionError:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_file = f"ensemble_diagnoses_output_{timestamp}.xlsx"
        df.to_excel(fallback_file, index=False)
        print(f"\n'{output_file}' was locked. Saved instead to: {fallback_file}")
