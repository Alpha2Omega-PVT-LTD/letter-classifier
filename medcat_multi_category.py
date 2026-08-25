import pandas as pd
import re
import datetime
from medcat.cat import CAT

# -----------------------------
# Load MedCAT model to memory
# -----------------------------
MODEL_PATH = "./v2_Snomed2025_MIMIC_IV_bbe806e192df009f"
print("Loading MedCAT model...")
cat = CAT.load_model_pack(MODEL_PATH)
print("Model loaded successfully")

NEGATING_VALUES = {"negated", "hypothetical", "ruled out", "absent"}
NON_PATIENT_VALUES = {"family member", "other", "relative"}


def is_excluded_by_meta(meta_anns: dict):
    """
    Returns (excluded: bool, reason: str or None).
    Checks every key that looks like a status/negation/subject task.
    """
    for task_name, task_result in meta_anns.items():
        value = str(task_result.get("value", "")).strip().lower()
        if value in NEGATING_VALUES:
            return True, f"{task_name}={value}"
        if value in NON_PATIENT_VALUES:
            return True, f"{task_name}={value}"
    return False, None


BLOCKLIST_TYPES = {
    "body structure", "person", "organism", "qualifier value", 
    "record artifact", "namespace concept", "geographic location", 
    "racial group", "ethnic group", "social concept", 
    "intellectual product", "occupation", "cell", "cell structure", "specimen"
}

def matches_any_word(text_lower: str, keywords: list) -> bool:
    """Helper to check if any of the keywords match as a whole word in text."""
    for kw in keywords:
        pattern = rf"\b{re.escape(kw)}\b"
        if re.search(pattern, text_lower):
            return True
    return False

def classify_entity_by_medcat(pretty_name: str, type_names: set) -> str or None:
    """
    Classifies a MedCAT entity into one of: 'Diagnosis', 'Symptom', 'Procedure', 'Medication', 'Vital', or None.
    Uses SNOMED CT semantic types combined with rule/keyword checks.
    """
    # 0. Filter out irrelevant SNOMED structural types
    if type_names & BLOCKLIST_TYPES:
        return None

    text_lower = pretty_name.lower().strip()
    
    # 1. Vital sign matchers
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
    
    is_vital = False
    if vitals_abbr_pattern.search(text_lower):
        is_vital = True
    elif matches_any_word(text_lower, vitals_terms):
        if 'weight loss' in text_lower or 'weight gain' in text_lower:
            is_vital = False
        else:
            is_vital = True
    elif re.search(r'\b\d+(?:\.\d+)?\s*(?:mg/dl|mmol/l|g/l|%|bpm|c|f|kg|cm|ml)\b', text_lower):
        is_vital = True

    # 2. Medication matchers
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
    med_types = {
        "clinical drug", "medicinal product", "medicinal product form",
        "product", "product name"
    }
    
    is_medication = False
    if type_names & med_types:
        is_medication = True
    # Ensure suffix matches are complete words or true suffixes, and keywords match as whole words
    elif any(text_lower.endswith(sfx) for sfx in med_suffixes) or matches_any_word(text_lower, med_keywords):
        is_medication = True
    elif re.search(r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|micrograms|g|ml|units|puff|puffs|tablet|tablets|cap|caps|capsule|capsules)\b', text_lower):
        is_medication = True

    # 3. Procedure matchers
    procedure_terms = [
        'surgery', 'appendectomy', 'biopsy', 'bypass', 'scan', 'mri', 'ct', 'ultrasound',
        'xray', 'x-ray', 'referral', 'referred', 'appointment', 'checkup', 'ecg', 'ekg',
        'endoscopy', 'colonoscopy', 'blood test', 'urine test', 'suture', 'excision',
        'infusion', 'physiotherapy', 'cbt', 'counselling', 'therapy', 'resection',
        'graft', 'stent', 'angioplasty', 'catheter', 'intubation', 'vaccination', 'immunisation',
        'injection', 'transfusion', 'drainage', 'amputation', 'scope', 'rehab', 'rehabilitation', 'therapy'
    ]
    procedure_types = {"procedure", "regime/therapy"}
    
    is_procedure = False
    if type_names & procedure_types:
        is_procedure = True
    elif matches_any_word(text_lower, procedure_terms) or any(text_lower.endswith(sfx) for sfx in ('ectomy', 'otomy', 'plasty', 'scopy')):
        is_procedure = True

    # 4. Symptom matchers
    symptom_terms = [
        'pain', 'ache', 'aching', 'cough', 'coughing', 'fever', 'pyrexia', 'nausea', 'vomiting', 'emesis',
        'dizziness', 'dizzy', 'fatigue', 'shortness of breath', 'sob', 'breathlessness', 'rash', 'headache',
        'swelling', 'edema', 'oedema', 'chills', 'rigors', 'sweating', 'sweats', 'night sweats',
        'diarrhoea', 'diarhea', 'diarrhea', 'constipation', 'numbness', 'tingling', 'paresthesia', 'parasthesia',
        'weakness', 'itch', 'itching', 'pruritus', 'sore throat', 'dyspnoea', 'dyspnea', 'palpitations',
        'chest pain', 'back pain', 'abdominal pain', 'myalgia', 'arthralgia', 'insomnia',
        'lethargy', 'malaise', 'weight loss', 'weight gain', 'wheeze', 'wheezing', 'stridor', 'sputum',
        'hemoptysis', 'haemoptysis', 'hematuria', 'haematuria', 'dysuria', 'nocturia', 'polyuria', 'polydipsia',
        'tremor', 'rigidity', 'bleeding', 'bleed', 'bleeds', 'rectal bleeding', 'vaginal bleeding',
        'gi bleeding', 'gastrointestinal bleeding', 'haemorrhage', 'hemorrhage', 'spotting', 'discharge',
        'cramp', 'cramps', 'cramping', 'spasm', 'spasms', 'stiffness', 'symptom', 'symptoms',
        'nocturnal symptoms', 'b symptoms', 'vertigo', 'syncope', 'fainting', 'lightheadedness',
        'tenderness', 'soreness', 'discomfort', 'lump', 'mass', 'nodule', 'lesion', 'lesions',
        'slurred speech', 'confusion', 'delirium', 'incontinence', 'urgency', 'frequency', 'hoarseness'
    ]
    is_symptom = False
    if matches_any_word(text_lower, symptom_terms):
        is_symptom = True

    # High-priority overrides (Symptom prioritized before Procedure)
    if is_vital:
        return 'Vital'
    if is_medication:
        return None  # MedCAT does not extract medications per requirement
    if is_symptom:
        return 'Symptom'
    if is_procedure:
        return 'Procedure'

    # SNOMED hierarchy logic
    if "disorder" in type_names or "morphologic abnormality" in type_names:
        return 'Diagnosis'
    if "finding" in type_names:
        return 'Symptom'
    if "procedure" in type_names or "regime/therapy" in type_names:
        return 'Procedure'
    if type_names & med_types:
        return None  # MedCAT does not extract medications per requirement
            
    # Discard other generic types to prevent noise
    return None


def extract_vitals_rules(sentence: str):
    """
    Rule-based backup scanner to extract vital sign names and values.
    """
    vitals = []
    sent_lower = sentence.lower()
    
    # 1. Blood Pressure: bp or blood pressure and fraction \d/\d
    bp_match = re.search(r'\b(\d{2,3}/\d{2,3})\b', sentence)
    if bp_match and ('bp' in sent_lower or 'blood pressure' in sent_lower or 'hypertension' in sent_lower):
        val = bp_match.group(1)
        vitals.append({
            "text": f"Blood Pressure: {val}",
            "cui": "",
            "score": 1.0
        })
        
    # 2. Temperature: temp X, temp: X, or X C/F/celsius/fahrenheit
    temp_pattern = re.compile(
        r'\b(?:temp|temperature)\s*(?:of|is|was)?\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\b|\b(\d{2,3}(?:\.\d+)?)\s*(?:c|f|°c|°f|celsius|fahrenheit)\b',
        re.IGNORECASE
    )
    for match in temp_pattern.finditer(sentence):
        val = match.group(1) or match.group(2)
        vitals.append({
            "text": f"Temperature: {val}",
            "cui": "",
            "score": 1.0
        })
            
    # 3. Heart Rate: pulse is X, pulse: X, or X bpm/beats
    hr_pattern = re.compile(
        r'\b(?:pulse|heart rate|hr)\s*(?:of|is|was|at)?\s*[:=]?\s*(\d{2,3})\b|\b(\d{2,3})\s*(?:bpm|beats)\b',
        re.IGNORECASE
    )
    for match in hr_pattern.finditer(sentence):
        val = match.group(1) or match.group(2)
        vitals.append({
            "text": f"Heart Rate: {val}",
            "cui": "",
            "score": 1.0
        })

    # 4. Weight: weight is X, wt: X, or X kg/lbs
    wt_pattern = re.compile(
        r'\b(?:weight|wt)\s*(?:of|is|was|at)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:kg|lbs|stone|st)?\b|\b(\d+(?:\.\d+)?)\s*(?:kg|lbs)\b',
        re.IGNORECASE
    )
    for match in wt_pattern.finditer(sentence):
        val = match.group(1) or match.group(2)
        vitals.append({
            "text": f"Weight: {val}",
            "cui": "",
            "score": 1.0
        })

    # 5. BMI: bmi is X, bmi: X
    bmi_pattern = re.compile(
        r'\bbmi\s*(?:of|is|was|at)?\s*[:=]?\s*(\d+(?:\.\d+)?)\b',
        re.IGNORECASE
    )
    for match in bmi_pattern.finditer(sentence):
        val = match.group(1)
        vitals.append({
            "text": f"BMI: {val}",
            "cui": "",
            "score": 1.0
        })
            
    return vitals


def extract_all_categories_medcat(text: str, cat_model: CAT):
    """
    Runs MedCAT over the clinical text and extracts concepts classified into 5 categories.
    """
    if not text or not text.strip():
        return {
            "Diagnosis": [],
            "Symptom": [],
            "Procedure": [],
            "Medication": [],
            "Vital": []
        }

    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    
    seen = {
        "Diagnosis": set(),
        "Symptom": set(),
        "Procedure": set(),
        "Medication": set(),
        "Vital": set()
    }
    extracted = {
        "Diagnosis": [],
        "Symptom": [],
        "Procedure": [],
        "Medication": [],
        "Vital": []
    }

    # 1. Rule-based vitals backup
    for sentence in sentences:
        rule_vitals = extract_vitals_rules(sentence)
        for rv in rule_vitals:
            key = rv["text"].lower()
            if key not in seen["Vital"]:
                seen["Vital"].add(key)
                extracted["Vital"].append(rv)

    # 2. MedCAT Entity Extraction
    result = cat_model.get_entities(text)
    entities = result.get("entities", {})

    for ent_id, ent in entities.items():
        cui = ent.get("cui")
        type_ids = ent.get("type_ids", [])
        pretty_name = ent.get("pretty_name") or ent.get("source_value")
        meta_anns = ent.get("meta_anns", {})
        acc = ent.get("acc", 1.0)

        # Meta-annotation filter
        excluded, exclude_reason = is_excluded_by_meta(meta_anns)
        if excluded:
            continue

        # Confidence filter
        if acc is not None and acc < 0.2:
            continue

        # Resolve type names
        type_names = set()
        for tid in type_ids:
            type_info = cat_model.cdb.type_id2info.get(tid)
            if type_info:
                type_names.add(type_info.name.lower())

        target_cat = classify_entity_by_medcat(pretty_name, type_names)
        if target_cat is None:
            continue
        
        # Deduplicate per category per report
        key = pretty_name.lower()
        if key not in seen[target_cat]:
            seen[target_cat].add(key)
            
            extracted[target_cat].append({
                "text": pretty_name,
                "cui": cui,
                "score": round(float(acc), 3) if acc is not None else 1.0
            })

    return extracted


def format_category_text(spans):
    """
    Formats the list of extracted concepts for a category as a numbered list of text names.
    """
    if not spans:
        return ""

    # Sort by score
    ranked = sorted(spans, key=lambda s: s["score"] if s["score"] is not None else 0.0, reverse=True)
    return "\n".join(f"{i+1}. {sp['text']}" for i, sp in enumerate(ranked))


def format_category_cui(spans):
    """
    Formats the list of SNOMED codes (CUIs) for a category as a numbered list of codes.
    """
    if not spans:
        return ""

    # Sort by score
    ranked = sorted(spans, key=lambda s: s["score"] if s["score"] is not None else 0.0, reverse=True)
    return "\n".join(f"{i+1}. {sp.get('cui') or ''}" for i, sp in enumerate(ranked))


# -----------------------------
# Process Excel File
# -----------------------------
if __name__ == "__main__":
    input_file = "datavalid.xlsx"
    print(f"Loading input file: {input_file}")
    df = pd.read_excel(input_file)

    # Initialize structure to collect results
    categories = ["Diagnosis", "Symptom", "Procedure", "Medication", "Vital"]
    columns_data = {category: [] for category in categories}
    diagnosis_snomed_data = []

    for index, row in df.iterrows():
        print(f"Processing report {index + 1}/{len(df)}...")
        text = row["Cleaned Data"]
        
        if pd.isna(text) or not str(text).strip():
            for category in categories:
                columns_data[category].append("")
            diagnosis_snomed_data.append("")
        else:
            extracted = extract_all_categories_medcat(str(text), cat)
            
            for category in categories:
                txt_str = format_category_text(extracted[category])
                columns_data[category].append(txt_str)
            
            cui_str = format_category_cui(extracted["Diagnosis"])
            diagnosis_snomed_data.append(cui_str)

    # Assign columns to dataframe
    df["Diagnosis"] = columns_data["Diagnosis"]
    df["Diagnosis_SNOMED"] = diagnosis_snomed_data
    df["Symptom"] = columns_data["Symptom"]
    df["Procedure"] = columns_data["Procedure"]
    df["Medication"] = columns_data["Medication"]
    df["Vital"] = columns_data["Vital"]

    # Save output (handles Windows file-lock case)
    output_file = "medcat_multi_category_output.xlsx"

    try:
        df.to_excel(output_file, index=False)
        print("\nCompleted successfully!")
        print(f"Saved to: {output_file}")
    except PermissionError:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_file = f"medcat_multi_category_output_{timestamp}.xlsx"
        df.to_excel(fallback_file, index=False)
        print(f"\n'{output_file}' was locked (likely open in Excel or another program).")
        print(f"Saved instead to: {fallback_file}")
