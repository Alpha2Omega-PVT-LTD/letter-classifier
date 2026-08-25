import pandas as pd
import torch
import re
import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification
import spacy
from negspacy.negation import Negex

# -----------------------------
# Initialize spaCy and NegEx (Optional)
# -----------------------------
print("Loading spaCy and negspacy (if available)...")
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
except Exception as e:
    print(f"spaCy not available, using fast regex negation fallback: {e}")

# Safe negation check wrapper
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

# -----------------------------
# Guardrail 2: Semantic Vocabulary Filter
# -----------------------------
SEMANTIC_BLOCKLIST = {
    "any concerns", 
    "concerns", 
    "concern", 
    "none", 
    "version", 
    "chan", 
    "problems"
}

def passes_semantic_filter(entity_text: str) -> bool:
    """
    Drops extracted strings that are too short (<=2 chars) or exactly
    match non-informative conversational filler / structural artifacts.
    """
    text_lower = entity_text.lower().strip()
    if len(text_lower) <= 2:
        return False
    if text_lower in SEMANTIC_BLOCKLIST:
        return False
    return True


# -----------------------------
# Guardrail 1: Context & Negation Filter
# -----------------------------
def is_negated_or_hypothetical(entity_text: str, sentence_text: str) -> bool:
    """
    Checks if an extracted entity is negated or hypothetical in context.
    Uses a fast regex for template negative forms, followed by spaCy NegEx if available.
    """
    # 1. Fast Regex Fallback Rule for structured template layouts
    escaped_ent = re.escape(entity_text)
    pattern = re.compile(rf"{escaped_ent}\s*\??:\s*(none|no)\b|\bno\s+{escaped_ent}\b|\bdenies\s+{escaped_ent}\b", re.IGNORECASE)
    if pattern.search(sentence_text):
        return True

    # 2. NegEx via spaCy (if available)
    return check_spacy_negation(entity_text, sentence_text)


# -----------------------------
# Load pretrained ClinicalBERT NER model
# -----------------------------
MODEL_NAME = "nlpie/clinical-distilbert-i2b2-2010"

print("Loading ClinicalBERT NER model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
print("Model loaded")

device = 0 if torch.cuda.is_available() else -1
if device == 0:
    model = model.to("cuda")
model.eval()

id2label = model.config.id2label
print("Model labels:", id2label)


def is_category_label(label: str, target_cat: str) -> bool:
    """
    Checks if the label contains the target category string (problem, treatment, test).
    """
    return target_cat in label.lower()


def run_ner_with_offsets(sentence: str):
    """
    Runs the model manually to get raw per-token predictions WITH character offsets.
    """
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


def merge_spans(tokens, sentence: str, target_cat: str, min_score: float = 0.5):
    """
    Merges token-level predictions into whole-phrase spans using character adjacency.
    Works for any of 'problem', 'treatment', or 'test'.
    """
    spans = []
    current_start = None
    current_end = None
    current_scores = []

    def flush():
        if current_start is not None:
            spans.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "score": sum(current_scores) / len(current_scores),
                }
            )

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        is_candidate = is_category_label(tok["label"], target_cat) and tok["score"] >= min_score

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
                next_is_candidate = (
                    is_category_label(next_tok["label"], target_cat) and next_tok["score"] >= min_score
                )
                if (
                    gap_before.strip() == ""
                    and gap_after.strip() == ""
                    and next_is_candidate
                ):
                    current_end = tok["end"]
                    current_scores.append(tok["score"])
                    i += 1
                    continue

            if current_start is not None:
                flush()
                current_start, current_end = None, None
                current_scores = []

        i += 1

    flush()

    results = []
    for sp in spans:
        text = sentence[sp["start"]:sp["end"]].strip()
        text = re.sub(r'[\s,:\-_]+(of|with|in|for|and|or|to|a|an|the)$', '', text, flags=re.IGNORECASE).strip()
        if text:
            results.append({"text": text, "start": sp["start"], "end": sp["end"], "score": sp["score"]})
    return results


# -----------------------------
# Entity Classification Logic
# -----------------------------
def classify_entity(entity_text: str, model_category: str) -> str:
    """
    Classifies a ClinicalBERT entity into one of: 'Diagnosis', 'Symptom', 'Procedure', 'Medication', 'Vital'.
    """
    text_lower = entity_text.lower().strip()
    
    # 1. Define vital sign matchers
    # Abbreviations must match as whole words
    vitals_abbr_pattern = re.compile(
        r'\b(bp|hr|rr|wt|ht|hb|crp|esr|wbc|ast|alt|ldl|hdl|sat|sats|bmi|egfr)\b',
        re.IGNORECASE
    )
    # Full terms can match as whole phrases/words
    vitals_terms = [
        'blood pressure', 'temperature', 'temp', 'pulse', 'heart rate', 'respiratory rate',
        'oxygen saturation', 'weight', 'height', 'creatinine', 'platelets', 'hemoglobin',
        'sodium', 'potassium', 'urea', 'bilirubin', 'cholesterol', 'systolic', 'diastolic',
        'pulse rate', 'respirations', 'o2 sat', 'o2 saturation'
    ]
    
    is_vital = False
    if vitals_abbr_pattern.search(text_lower):
        is_vital = True
    elif any(term in text_lower for term in vitals_terms):
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
    
    is_medication = False
    if text_lower.endswith(med_suffixes) or any(k in text_lower for k in med_keywords):
        is_medication = True
    elif re.search(r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|micrograms|g|ml|units|puff|puffs|tablet|tablets|cap|caps|capsule|capsules)\b', text_lower):
        is_medication = True

    # 3. Symptom matchers (prioritised over general procedure fallbacks)
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
    is_symptom = any(term in text_lower for term in symptom_terms)

    # 4. Procedure matchers
    procedure_terms = [
        'surgery', 'appendectomy', 'biopsy', 'bypass', 'scan', 'mri', 'ct', 'ultrasound',
        'xray', 'x-ray', 'referral', 'referred', 'appointment', 'checkup', 'ecg', 'ekg',
        'endoscopy', 'colonoscopy', 'blood test', 'urine test', 'suture', 'excision',
        'infusion', 'physiotherapy', 'cbt', 'counselling', 'therapy', 'resection',
        'graft', 'stent', 'angioplasty', 'catheter', 'intubation', 'vaccination', 'immunisation',
        'injection', 'transfusion', 'drainage', 'amputation', 'scope', 'rehab', 'rehabilitation'
    ]
    is_procedure = False
    if any(term in text_lower for term in procedure_terms) or text_lower.endswith(('ectomy', 'otomy', 'plasty', 'scopy')):
        is_procedure = True

    # High-priority overrides (Symptom prioritized before Procedure so symptoms are not mislabeled)
    if is_vital:
        return 'Vital'
    if is_medication:
        return 'Medication'
    if is_symptom:
        return 'Symptom'
    if is_procedure:
        return 'Procedure'

    # Fallback matches based on ClinicalBERT category
    if model_category == 'problem':
        return 'Diagnosis'
    elif model_category == 'treatment':
        return 'Procedure'
    elif model_category == 'test':
        return 'Procedure'
            
    return 'Diagnosis'


def extract_vitals_rules(sentence: str):
    """
    Rule-based backup scanner to extract vital sign names and values.
    """
    vitals = []
    sent_lower = sentence.lower()
    
    # 1. Look for BP pattern
    bp_match = re.search(r'\b(\d{2,3}/\d{2,3})\b', sentence)
    if bp_match and ('bp' in sent_lower or 'blood pressure' in sent_lower or 'hypertension' in sent_lower):
        val = bp_match.group(1)
        vitals.append({
            "text": f"Blood Pressure: {val}",
            "score": 1.0,
            "sentence": sentence
        })
        
    # 2. Look for temperature: temp is X, or temp: X, or X C/F/celsius/fahrenheit
    temp_pattern = re.compile(
        r'\b(?:temp|temperature)\s*(?:of|is|was)?\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\b|\b(\d{2,3}(?:\.\d+)?)\s*(?:c|f|°c|°f|celsius|fahrenheit)\b',
        re.IGNORECASE
    )
    for match in temp_pattern.finditer(sentence):
        val = match.group(1) or match.group(2)
        vitals.append({
            "text": f"Temperature: {val}",
            "score": 1.0,
            "sentence": sentence
        })
            
    # 3. Look for heart rate / pulse: pulse is X, or pulse: X, or X bpm/beats
    hr_pattern = re.compile(
        r'\b(?:pulse|heart rate|hr)\s*(?:of|is|was|at)?\s*[:=]?\s*(\d{2,3})\b|\b(\d{2,3})\s*(?:bpm|beats)\b',
        re.IGNORECASE
    )
    for match in hr_pattern.finditer(sentence):
        val = match.group(1) or match.group(2)
        vitals.append({
            "text": f"Heart Rate: {val}",
            "score": 1.0,
            "sentence": sentence
        })

    # 4. Look for weight: weight is X, wt: X, or X kg/lbs
    wt_pattern = re.compile(
        r'\b(?:weight|wt)\s*(?:of|is|was|at)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:kg|lbs|stone|st)?\b|\b(\d+(?:\.\d+)?)\s*(?:kg|lbs)\b',
        re.IGNORECASE
    )
    for match in wt_pattern.finditer(sentence):
        val = match.group(1) or match.group(2)
        vitals.append({
            "text": f"Weight: {val}",
            "score": 1.0,
            "sentence": sentence
        })

    # 5. Look for BMI: bmi is X, bmi: X
    bmi_pattern = re.compile(
        r'\bbmi\s*(?:of|is|was|at)?\s*[:=]?\s*(\d+(?:\.\d+)?)\b',
        re.IGNORECASE
    )
    for match in bmi_pattern.finditer(sentence):
        val = match.group(1)
        vitals.append({
            "text": f"BMI: {val}",
            "score": 1.0,
            "sentence": sentence
        })
            
    return vitals


def extract_all_categories(text: str, min_score: float = 0.5):
    """
    Extracts all clinical categories using ClinicalBERT and rule-based mapping,
    applying semantic and negation guardrails.
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

    # To deduplicate per category per report
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

    for sentence in sentences:
        # --- RULE-BASED VITALS EXTRACTION ---
        rule_vitals = extract_vitals_rules(sentence)
        for rv in rule_vitals:
            key = rv["text"].lower()
            if key not in seen["Vital"]:
                seen["Vital"].add(key)
                extracted["Vital"].append(rv)

        try:
            tokens = run_ner_with_offsets(sentence)
        except Exception as e:
            print(f"  [warn] NER failed on a sentence: {e}")
            continue

        # Extract for each raw category in ClinicalBERT
        for raw_cat in ["problem", "treatment", "test"]:
            spans = merge_spans(tokens, sentence, raw_cat, min_score=min_score)
            for sp in spans:
                entity_text = sp["text"]
                
                # --- GUARDRAIL 2: SEMANTIC FILTER ---
                if not passes_semantic_filter(entity_text):
                    continue
                    
                # --- GUARDRAIL 1: NEGATION FILTER ---
                if is_negated_or_hypothetical(entity_text, sentence):
                    continue

                # Classify into one of 5 target categories
                target_cat = classify_entity(entity_text, raw_cat)
                
                key = entity_text.lower()
                if key not in seen[target_cat]:
                    seen[target_cat].add(key)
                    extracted[target_cat].append({
                        "text": entity_text,
                        "sentence": sentence,
                        "score": sp["score"]
                    })

    return extracted


def format_category_outputs(spans, category_name: str):
    """
    Formats the list of extracted spans for a category into three output strings.
    """
    if not spans:
        empty = f"No {category_name.lower()} extracted"
        return empty, "", ""

    ranked = sorted(spans, key=lambda s: s["score"], reverse=True)

    lines = []
    conf_lines = []
    reason_lines = []
    for i, sp in enumerate(ranked, start=1):
        lines.append(f"{i}. {sp['text']}")
        conf_lines.append(f"{i}. {sp['score']:.2f}")
        reason_lines.append(
            f'{i}. Extracted from: "{sp["sentence"]}" (confidence {sp["score"]:.2f})'
        )

    return "\n".join(lines), "\n".join(conf_lines), "\n".join(reason_lines)


# -----------------------------
# Process Excel File
# -----------------------------
if __name__ == "__main__":
    input_file = "datavalid.xlsx"
    print(f"Loading input file: {input_file}")
    df = pd.read_excel(input_file)

    # Initialize columns list
    columns_data = {
        "Diagnosis": {"text": [], "conf": [], "reason": []},
        "Symptom": {"text": [], "conf": [], "reason": []},
        "Procedure": {"text": [], "conf": [], "reason": []},
        "Medication": {"text": [], "conf": [], "reason": []},
        "Vital": {"text": [], "conf": [], "reason": []}
    }

    for index, row in df.iterrows():
        print(f"Processing report {index + 1}/{len(df)}")
        text = row["Cleaned Data"]
        
        if pd.isna(text) or not str(text).strip():
            for cat in columns_data:
                columns_data[cat]["text"].append(f"No {cat.lower()} extracted")
                columns_data[cat]["conf"].append("")
                columns_data[cat]["reason"].append("")
        else:
            extracted = extract_all_categories(str(text))
            
            for cat in columns_data:
                txt_str, conf_str, reason_str = format_category_outputs(extracted[cat], cat)
                columns_data[cat]["text"].append(txt_str)
                columns_data[cat]["conf"].append(conf_str)
                columns_data[cat]["reason"].append(reason_str)

    # Assign columns to dataframe
    for cat in columns_data:
        df[f"{cat}_ClinicalBERT"] = columns_data[cat]["text"]
        df[f"{cat}_Confidence"] = columns_data[cat]["conf"]
        df[f"{cat}_Reasoning"] = columns_data[cat]["reason"]

    # -----------------------------
    # Save output (handles Windows file-lock case)
    # -----------------------------
    output_file = "clinicalbert_multi_category_output.xlsx"

    try:
        df.to_excel(output_file, index=False)
        print("\nCompleted successfully!")
        print(f"Saved to: {output_file}")
    except PermissionError:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_file = f"clinicalbert_multi_category_output_{timestamp}.xlsx"
        df.to_excel(fallback_file, index=False)
        print(f"\n'{output_file}' was locked (likely open in Excel or another program).")
        print(f"Saved instead to: {fallback_file}")
