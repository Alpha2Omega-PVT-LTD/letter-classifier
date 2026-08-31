"""
extract_clinical_info_qwen.py
------------------------------
Pipeline:
  1. OCR each PDF in sample_letters/ using your existing paddle_ocr.py
  2. Extract + classify + contextualise medical entities using a local
     Qwen model served by Ollama
  3. HARD anti-hallucination check (programmatic, not LLM-based): every
     entity's exact text must actually appear in the OCR'd letter, or it
     is rejected — regardless of what the model says
  4. A second Qwen pass re-validates each surviving entity + its context
     against the letter text and can mark it invalid / correct its context
  5. A final programmatic re-check re-verifies anything the validation
     pass touched, before writing output

Categories: diagnosis, symptom, procedure, medication, vitals

Run in your nhs_env (Ollama must be running: `ollama serve`, model pulled):
    python extract_clinical_info_qwen.py

Author note: written against the standard Ollama REST API
(POST /api/chat, format="json") and cannot be executed end-to-end here —
I don't have your PDFs, paddle_ocr.py, or a running Ollama instance in
this sandbox. Run it in nhs_env and send back any traceback.
"""

import copy
import csv
import difflib
import importlib.util
import inspect
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
import ollama

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("clinical_extract_qwen")

# ============================================================================
# CONFIG
# ============================================================================

SAMPLE_LETTERS_DIR = Path("sample_letters")
PADDLE_OCR_SCRIPT = Path("paddle_ocr.py")
OUTPUT_DIR = Path("output_qwen")

# Set this to exact pulled model from `ollama list`
OLLAMA_MODEL = "qwen2.5:7b-instruct"
MAX_RETRIES = 3

# Long letters are split into paragraph-based chunks so the model doesn't
# have to hold the whole document at once. Increase if your model/context
# window can comfortably handle more.
MAX_CHUNK_CHARS = 3000

# Allowed context values per category. If the model returns something
# outside this set, the entity is KEPT (never silently dropped) but
# flagged needs_review=true with the raw value preserved.
ALLOWED_CONTEXTS = {
    "diagnosis": {"current", "negated", "suspected"},
    "symptom": {"current", "worsening", "resolved", "side_effect", "negated"},
    "procedure": {"performed", "planned", "discussed_not_done"},
    "medication": {"current", "discontinued", "planned", "allergy"},
    "vitals": {"normal", "abnormal", "unspecified"},
}
CATEGORIES = list(ALLOWED_CONTEXTS.keys())

# Fuzzy verbatim-match threshold (0-1). Lower = more tolerant of OCR noise,
# higher = stricter. This is the core anti-hallucination gate.
VERBATIM_MATCH_THRESHOLD = 0.85

# ============================================================================
# STEP 1: OCR adapter (same approach as before)
# ============================================================================


def _load_paddle_ocr_module():
    if not PADDLE_OCR_SCRIPT.exists():
        raise FileNotFoundError(
            f"Can't find {PADDLE_OCR_SCRIPT}. Edit PADDLE_OCR_SCRIPT at the "
            f"top of this file."
        )
    spec = importlib.util.spec_from_file_location("paddle_ocr", PADDLE_OCR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalise_ocr_result(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        lines = []
        for item in result:
            if isinstance(item, str):
                lines.append(item)
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    if isinstance(sub, (list, tuple)) and len(sub) >= 1 \
                            and isinstance(sub[0], str):
                        lines.append(sub[0])
                    elif isinstance(sub, str):
                        lines.append(sub)
        return "\n".join(lines)
    return str(result)


def get_text_from_pdf(pdf_path: Path, ocr_module) -> str:
    """Uses paddle_ocr.py extract_text_from_pdf_paddle function."""
    if hasattr(ocr_module, "extract_text_from_pdf_paddle"):
        res = ocr_module.extract_text_from_pdf_paddle(str(pdf_path))
        if isinstance(res, dict) and "full_text" in res:
            return res["full_text"]
        return _normalise_ocr_result(res)

    candidate_fn_names = [
        "extract_text_from_pdf_paddle", "extract_text_from_pdf", "extract_text",
        "ocr_pdf", "pdf_to_text", "get_text_from_pdf", "get_text", "process_pdf", "run_ocr",
    ]
    for name in candidate_fn_names:
        fn = getattr(ocr_module, name, None)
        if fn is None or not callable(fn):
            continue
        try:
            result = fn(str(pdf_path))
            text = _normalise_ocr_result(result)
            if text and text.strip():
                log.info("OCR via %s() succeeded for %s (%d chars)", name, pdf_path.name, len(text))
                return text
        except Exception:
            continue

    raise RuntimeError(f"Could not extract text from {pdf_path.name} using {PADDLE_OCR_SCRIPT}")


def chunk_text(text: str, max_chars: int):
    paragraphs = re.split(r"\n\s*\n", text)
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chars:
                # paragraph itself too long, hard-split it
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i:i + max_chars])
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks or [text]


# ============================================================================
# STEP 2: Ollama / Qwen calls
# ============================================================================


def call_ollama(system_prompt: str, user_prompt: str, few_shot=None) -> dict:
    """Uses the native python `ollama` library directly."""
    messages = [{"role": "system", "content": system_prompt}]
    for ex_user, ex_assistant in (few_shot or []):
        messages.append({"role": "user", "content": ex_user})
        messages.append({"role": "assistant", "content": ex_assistant})
    messages.append({"role": "user", "content": user_prompt})

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                format="json",
                options={"temperature": 0.0}
            )
            content = res["message"]["content"]
            content = _strip_code_fences(content)
            return json.loads(content)
        except Exception as e:
            last_err = e
            log.warning("Ollama call failed (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Ollama call failed after {MAX_RETRIES} attempts: {last_err}")


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    return text


EXTRACTION_SYSTEM_PROMPT = """You are a clinical information extraction system.

You will be given a chunk of text OCR'd from a real clinical letter. Extract
every medical entity you find and classify each one.

CRITICAL RULES — you MUST follow these exactly:
1. Only extract entities whose exact wording appears in the text provided.
   Never infer, guess, add, or hallucinate anything that is not literally
   written in the text.
2. "text_in_letter" MUST be an exact, verbatim substring copied from the
   text (same words, same order, same spelling as in the source). Do not
   paraphrase or normalise it.
3. If you are not confident an entity is actually present and correctly
   classified, do NOT include it. It is far better to omit an uncertain
   entity than to include a wrong or invented one.
4. If the text contains no medical entities, return an empty list.

Categories and their allowed "context" values:
- diagnosis: current | negated | suspected
- symptom: current | worsening | resolved | side_effect | negated
- procedure: performed | planned | discussed_not_done
- medication: current | discontinued | planned | allergy
- vitals: (context is a status) normal | abnormal | unspecified
  For vitals also include "value" and "unit" fields if present in the text
  (e.g. text_in_letter="BP 150/95", value="150/95", unit="mmHg").

5. Do NOT extract conditions/symptoms/medications belonging to someone
   other than the patient (family history, other relatives). See the
   family-history example below — these are excluded entirely.
6. Do NOT extract vague, non-clinical, or purely administrative text
   (appointment dates alone, clinician names, letterhead, "thank you for
   referring", social history like occupation/smoking unless it's clearly
   a clinical finding, etc.) — only genuine diagnosis / symptom /
   procedure / medication / vitals entities.
7. Use abbreviation expansions you are confident about but keep
   "text_in_letter" as the exact original abbreviation as written (e.g.
   keep "SOB" as written, do not rewrite it as "shortness of breath").

Categories and their allowed "context" values:
- diagnosis: current | negated | suspected
- symptom: current | worsening | resolved | side_effect | negated
- procedure: performed | planned | discussed_not_done
- medication: current | discontinued | planned | allergy
- vitals: (context is a status) normal | abnormal | unspecified
  For vitals also include "value" and "unit" fields if present in the text
  (e.g. text_in_letter="BP 150/95", value="150/95", unit="mmHg").

Return ONLY valid JSON, no other text, in exactly this shape:
{
  "entities": [
    {
      "text_in_letter": "<exact verbatim text from the source>",
      "category": "diagnosis|symptom|procedure|medication|vitals",
      "context": "<one of the allowed values for that category>",
      "evidence_sentence": "<the exact sentence from the source the entity came from>",
      "value": "<only for vitals, else omit or empty string>",
      "unit": "<only for vitals, else omit or empty string>"
    }
  ]
}

The examples that follow (as prior turns in this conversation) show the
expected input/output behaviour, including negative examples of things
that should NOT be extracted. Follow that same pattern exactly.
"""


# ----------------------------------------------------------------------
# Few-shot examples for the extraction pass, presented as real user/
# assistant turns (this is far more effective for grounding a model's
# behaviour than prose instructions alone). Each covers a distinct
# category x context combination, plus negative examples.
# ----------------------------------------------------------------------

def _ex(entities):
    return json.dumps({"entities": entities}, ensure_ascii=False)


FEW_SHOT_EXTRACTION_EXAMPLES = [
    # --- diagnosis: current ---
    (
        "Mr Patel is a 62-year-old man with a background of Type 2 Diabetes "
        "Mellitus and hypertension, currently well controlled on oral therapy.",
        _ex([
            {"text_in_letter": "Type 2 Diabetes Mellitus", "category": "diagnosis",
             "context": "current",
             "evidence_sentence": "Mr Patel is a 62-year-old man with a background of Type 2 Diabetes Mellitus and hypertension, currently well controlled on oral therapy."},
            {"text_in_letter": "hypertension", "category": "diagnosis",
             "context": "current",
             "evidence_sentence": "Mr Patel is a 62-year-old man with a background of Type 2 Diabetes Mellitus and hypertension, currently well controlled on oral therapy."},
        ]),
    ),
    # --- diagnosis: negated ---
    (
        "Chest X-ray showed no evidence of malignancy. She denies any history "
        "of tuberculosis.",
        _ex([
            {"text_in_letter": "no evidence of malignancy", "category": "diagnosis",
             "context": "negated",
             "evidence_sentence": "Chest X-ray showed no evidence of malignancy."},
            {"text_in_letter": "denies any history of tuberculosis", "category": "diagnosis",
             "context": "negated",
             "evidence_sentence": "She denies any history of tuberculosis."},
        ]),
    ),
    # --- diagnosis: suspected (including abbreviations like ?/r/o) ---
    (
        "?Pneumonia - for review after course of antibiotics. Query UTI given "
        "raised inflammatory markers. R/O appendicitis, surgical review requested.",
        _ex([
            {"text_in_letter": "?Pneumonia", "category": "diagnosis", "context": "suspected",
             "evidence_sentence": "?Pneumonia - for review after course of antibiotics."},
            {"text_in_letter": "Query UTI", "category": "diagnosis", "context": "suspected",
             "evidence_sentence": "Query UTI given raised inflammatory markers."},
            {"text_in_letter": "R/O appendicitis", "category": "diagnosis", "context": "suspected",
             "evidence_sentence": "R/O appendicitis, surgical review requested."},
        ]),
    ),
    # --- diagnosis: family history exclusion (negative example) ---
    (
        "Family history: mother had breast cancer, father died of myocardial "
        "infarction aged 70. Patient herself has no significant past medical history.",
        _ex([]),
    ),
    # --- symptom: current ---
    (
        "The patient complains of shortness of breath on exertion and "
        "intermittent lower back pain over the past two weeks.",
        _ex([
            {"text_in_letter": "shortness of breath on exertion", "category": "symptom",
             "context": "current",
             "evidence_sentence": "The patient complains of shortness of breath on exertion and intermittent lower back pain over the past two weeks."},
            {"text_in_letter": "intermittent lower back pain", "category": "symptom",
             "context": "current",
             "evidence_sentence": "The patient complains of shortness of breath on exertion and intermittent lower back pain over the past two weeks."},
        ]),
    ),
    # --- symptom: worsening ---
    (
        "His chest pain has been worsening over the past week, now occurring "
        "at rest.",
        _ex([
            {"text_in_letter": "chest pain has been worsening", "category": "symptom",
             "context": "worsening",
             "evidence_sentence": "His chest pain has been worsening over the past week, now occurring at rest."},
        ]),
    ),
    # --- symptom: resolved ---
    (
        "Her nausea has completely resolved since the antiemetic was started.",
        _ex([
            {"text_in_letter": "nausea has completely resolved", "category": "symptom",
             "context": "resolved",
             "evidence_sentence": "Her nausea has completely resolved since the antiemetic was started."},
        ]),
    ),
    # --- symptom: negated ---
    (
        "On systems review, the patient denies any nausea, vomiting, or "
        "abdominal pain. No shortness of breath at rest.",
        _ex([
            {"text_in_letter": "denies any nausea, vomiting, or abdominal pain",
             "category": "symptom", "context": "negated",
             "evidence_sentence": "On systems review, the patient denies any nausea, vomiting, or abdominal pain."},
            {"text_in_letter": "No shortness of breath at rest", "category": "symptom",
             "context": "negated",
             "evidence_sentence": "No shortness of breath at rest."},
        ]),
    ),
    # --- symptom: side_effect ---
    (
        "Since starting amlodipine the patient has noted ankle swelling, "
        "thought to be a side effect of the medication.",
        _ex([
            {"text_in_letter": "ankle swelling", "category": "symptom",
             "context": "side_effect",
             "evidence_sentence": "Since starting amlodipine the patient has noted ankle swelling, thought to be a side effect of the medication."},
        ]),
    ),
    # --- procedure: performed ---
    (
        "The patient underwent coronary angiography on 12/03/2024, which "
        "showed no significant stenosis. Bloods were also taken.",
        _ex([
            {"text_in_letter": "underwent coronary angiography", "category": "procedure",
             "context": "performed",
             "evidence_sentence": "The patient underwent coronary angiography on 12/03/2024, which showed no significant stenosis."},
        ]),
    ),
    # --- procedure: planned ---
    (
        "An MRI of the brain has been scheduled for next week to further "
        "investigate the headaches.",
        _ex([
            {"text_in_letter": "MRI of the brain has been scheduled", "category": "procedure",
             "context": "planned",
             "evidence_sentence": "An MRI of the brain has been scheduled for next week to further investigate the headaches."},
        ]),
    ),
    # --- procedure: discussed_not_done ---
    (
        "We discussed the option of total knee replacement surgery in detail; "
        "the patient wishes to defer this for now and continue conservative "
        "management.",
        _ex([
            {"text_in_letter": "discussed the option of total knee replacement surgery",
             "category": "procedure", "context": "discussed_not_done",
             "evidence_sentence": "We discussed the option of total knee replacement surgery in detail; the patient wishes to defer this for now and continue conservative management."},
        ]),
    ),
    # --- medication: current ---
    (
        "She is currently taking Metformin 500mg twice daily and Ramipril "
        "5mg once daily.",
        _ex([
            {"text_in_letter": "Metformin 500mg twice daily", "category": "medication",
             "context": "current",
             "evidence_sentence": "She is currently taking Metformin 500mg twice daily and Ramipril 5mg once daily."},
            {"text_in_letter": "Ramipril 5mg once daily", "category": "medication",
             "context": "current",
             "evidence_sentence": "She is currently taking Metformin 500mg twice daily and Ramipril 5mg once daily."},
        ]),
    ),
    # --- medication: discontinued ---
    (
        "Atorvastatin was stopped last month due to myalgia.",
        _ex([
            {"text_in_letter": "Atorvastatin was stopped", "category": "medication",
             "context": "discontinued",
             "evidence_sentence": "Atorvastatin was stopped last month due to myalgia."},
        ]),
    ),
    # --- medication: planned ---
    (
        "Plan: will commence patient on Sertraline 50mg once daily and review "
        "in 4 weeks.",
        _ex([
            {"text_in_letter": "commence patient on Sertraline 50mg once daily",
             "category": "medication", "context": "planned",
             "evidence_sentence": "Plan: will commence patient on Sertraline 50mg once daily and review in 4 weeks."},
        ]),
    ),
    # --- medication: allergy ---
    (
        "Allergies: patient is allergic to Penicillin (rash) and has a known "
        "intolerance to Codeine.",
        _ex([
            {"text_in_letter": "allergic to Penicillin", "category": "medication",
             "context": "allergy",
             "evidence_sentence": "Allergies: patient is allergic to Penicillin (rash) and has a known intolerance to Codeine."},
            {"text_in_letter": "intolerance to Codeine", "category": "medication",
             "context": "allergy",
             "evidence_sentence": "Allergies: patient is allergic to Penicillin (rash) and has a known intolerance to Codeine."},
        ]),
    ),
    # --- vitals: normal / abnormal / unspecified, with value+unit ---
    (
        "Observations: BP 118/76 mmHg, HR 88 bpm, Temp 39.2\u00b0C, SpO2 96% "
        "on air.",
        _ex([
            {"text_in_letter": "BP 118/76 mmHg", "category": "vitals", "context": "normal",
             "value": "118/76", "unit": "mmHg",
             "evidence_sentence": "Observations: BP 118/76 mmHg, HR 88 bpm, Temp 39.2\u00b0C, SpO2 96% on air."},
            {"text_in_letter": "HR 88 bpm", "category": "vitals", "context": "unspecified",
             "value": "88", "unit": "bpm",
             "evidence_sentence": "Observations: BP 118/76 mmHg, HR 88 bpm, Temp 39.2\u00b0C, SpO2 96% on air."},
            {"text_in_letter": "Temp 39.2\u00b0C", "category": "vitals", "context": "abnormal",
             "value": "39.2", "unit": "\u00b0C",
             "evidence_sentence": "Observations: BP 118/76 mmHg, HR 88 bpm, Temp 39.2\u00b0C, SpO2 96% on air."},
            {"text_in_letter": "SpO2 96% on air", "category": "vitals", "context": "normal",
             "value": "96", "unit": "%",
             "evidence_sentence": "Observations: BP 118/76 mmHg, HR 88 bpm, Temp 39.2\u00b0C, SpO2 96% on air."},
        ]),
    ),
    # --- negative example: administrative / non-clinical text ---
    (
        "Thank you for referring this pleasant gentleman to the clinic. He "
        "works as an accountant and lives with his wife. I will see him again "
        "in the outpatient clinic in three months.",
        _ex([]),
    ),
    # --- mixed paragraph pulling several categories together ---
    (
        "Mrs Okafor presented with worsening dyspnoea and ankle oedema. "
        "Diagnosed with congestive heart failure. Started on Furosemide 40mg "
        "daily. Echocardiogram planned to assess ejection fraction. No chest "
        "pain reported. BP 145/92 mmHg on arrival.",
        _ex([
            {"text_in_letter": "worsening dyspnoea", "category": "symptom", "context": "worsening",
             "evidence_sentence": "Mrs Okafor presented with worsening dyspnoea and ankle oedema."},
            {"text_in_letter": "ankle oedema", "category": "symptom", "context": "current",
             "evidence_sentence": "Mrs Okafor presented with worsening dyspnoea and ankle oedema."},
            {"text_in_letter": "congestive heart failure", "category": "diagnosis", "context": "current",
             "evidence_sentence": "Diagnosed with congestive heart failure."},
            {"text_in_letter": "Started on Furosemide 40mg daily", "category": "medication", "context": "current",
             "evidence_sentence": "Started on Furosemide 40mg daily."},
            {"text_in_letter": "Echocardiogram planned", "category": "procedure", "context": "planned",
             "evidence_sentence": "Echocardiogram planned to assess ejection fraction."},
            {"text_in_letter": "No chest pain reported", "category": "symptom", "context": "negated",
             "evidence_sentence": "No chest pain reported."},
            {"text_in_letter": "BP 145/92 mmHg", "category": "vitals", "context": "abnormal",
             "value": "145/92", "unit": "mmHg",
             "evidence_sentence": "BP 145/92 mmHg on arrival."},
        ]),
    ),
]

VALIDATION_SYSTEM_PROMPT = """You are a strict clinical QA reviewer.

You will be given: (a) the original letter text, and (b) a list of
previously extracted entities with their category and context.

For EACH entity, check it against the letter text and decide:
- "valid": true if the entity genuinely appears in the text with that
  category/context correctly assigned, false otherwise
- if the context looks wrong given the text, provide "corrected_context"
  (must be one of the allowed values for that category); otherwise omit it
- "reason": a short explanation, especially when valid=false or the
  context was corrected

Do NOT invent new entities. Do NOT change "text_in_letter". You are only
confirming, rejecting, or correcting the context of what is given to you.
If an entity's text does not actually appear in the letter text, or its
category is clearly wrong, mark valid=false.

Return ONLY valid JSON, no other text, in exactly this shape:
{
  "reviewed": [
    {
      "text_in_letter": "<copied exactly from the input entity>",
      "valid": true,
      "corrected_context": "<optional>",
      "reason": "<short reason>"
    }
  ]
}

The examples that follow (as prior turns) show correct review behaviour.
"""


def _val_ex(letter_text, entities_in, reviewed_out):
    user = (
        f"LETTER TEXT:\n{letter_text}\n\n"
        f"PREVIOUSLY EXTRACTED ENTITIES TO REVIEW:\n"
        f"{json.dumps(entities_in, indent=2)}"
    )
    return user, json.dumps({"reviewed": reviewed_out}, ensure_ascii=False)


FEW_SHOT_VALIDATION_EXAMPLES = [
    # correct as-is
    _val_ex(
        "Mr Patel has a background of Type 2 Diabetes Mellitus, currently "
        "well controlled on Metformin 500mg twice daily.",
        [
            {"text_in_letter": "Type 2 Diabetes Mellitus", "category": "diagnosis", "context": "current"},
            {"text_in_letter": "Metformin 500mg twice daily", "category": "medication", "context": "current"},
        ],
        [
            {"text_in_letter": "Type 2 Diabetes Mellitus", "valid": True,
             "reason": "explicitly stated as a current background diagnosis"},
            {"text_in_letter": "Metformin 500mg twice daily", "valid": True,
             "reason": "explicitly stated as a current medication"},
        ],
    ),
    # context correction needed
    _val_ex(
        "Chest X-ray showed no evidence of malignancy.",
        [
            {"text_in_letter": "no evidence of malignancy", "category": "diagnosis", "context": "current"},
        ],
        [
            {"text_in_letter": "no evidence of malignancy", "valid": True,
             "corrected_context": "negated",
             "reason": "text explicitly negates malignancy; context 'current' was wrong, should be 'negated'"},
        ],
    ),
    # hallucinated entity not actually in text -> invalid
    _val_ex(
        "Patient reports mild headache, otherwise well.",
        [
            {"text_in_letter": "Patient reports mild headache", "category": "symptom", "context": "current"},
            {"text_in_letter": "hypertension", "category": "diagnosis", "context": "current"},
        ],
        [
            {"text_in_letter": "Patient reports mild headache", "valid": True,
             "reason": "matches text exactly"},
            {"text_in_letter": "hypertension", "valid": False,
             "reason": "the word 'hypertension' does not appear anywhere in the given letter text"},
        ],
    ),
    # wrong category -> invalid
    _val_ex(
        "The consultant reviewed the patient in outpatient clinic and "
        "arranged an MRI scan for next month.",
        [
            {"text_in_letter": "MRI scan", "category": "medication", "context": "planned"},
        ],
        [
            {"text_in_letter": "MRI scan", "valid": False,
             "reason": "an MRI scan is a procedure, not a medication; category is incorrect"},
        ],
    ),
]


# ============================================================================
# STEP 3: Anti-hallucination verbatim check (programmatic, not LLM)
# ============================================================================


def _normalise_for_match(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def verbatim_present(span: str, source_text: str, threshold: float = VERBATIM_MATCH_THRESHOLD) -> bool:
    """Returns True only if `span` genuinely appears (allowing minor OCR
    noise) inside `source_text`. This is the hard anti-hallucination gate
    and does not rely on the LLM at all."""
    if not span or not span.strip():
        return False

    span_norm = _normalise_for_match(span)
    source_norm = _normalise_for_match(source_text)

    if span_norm in source_norm:
        return True

    # sliding-window fuzzy match to tolerate OCR noise (extra/missing
    # spaces, minor char errors) without allowing genuinely absent text
    window = len(span_norm)
    if window == 0:
        return False
    step = max(1, window // 4)
    best_ratio = 0.0
    for i in range(0, max(1, len(source_norm) - window + 1), step):
        candidate = source_norm[i:i + window + 10]
        ratio = difflib.SequenceMatcher(None, span_norm, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
        if best_ratio >= threshold:
            return True
    return best_ratio >= threshold


# ============================================================================
# PIPELINE
# ============================================================================


def extract_chunk(chunk_text_: str) -> list:
    try:
        result = call_ollama(EXTRACTION_SYSTEM_PROMPT, chunk_text_,
                              few_shot=FEW_SHOT_EXTRACTION_EXAMPLES)
    except Exception as e:
        log.error("Extraction failed for chunk: %s", e)
        return []
    entities = result.get("entities", []) if isinstance(result, dict) else []
    return entities if isinstance(entities, list) else []


def hard_filter_entities(entities: list, source_text: str, source_file: str):
    """Programmatic verbatim + schema check. This is the non-negotiable
    anti-hallucination gate — nothing bypasses this regardless of LLM
    confidence."""
    kept, rejected = [], []
    for ent in entities:
        text_in_letter = (ent.get("text_in_letter") or "").strip()
        category = (ent.get("category") or "").strip().lower()
        context = (ent.get("context") or "").strip().lower()
        evidence = (ent.get("evidence_sentence") or "").strip()

        reasons = []
        if category not in CATEGORIES:
            reasons.append(f"unknown category '{category}'")
        if not verbatim_present(text_in_letter, source_text):
            reasons.append("text_in_letter not found verbatim in source "
                            "(possible hallucination)")
        if evidence and not verbatim_present(evidence, source_text):
            reasons.append("evidence_sentence not found verbatim in source "
                            "(possible hallucination)")

        record = {
            "source_file": source_file,
            "text_in_letter": text_in_letter,
            "category": category if category in CATEGORIES else "uncategorised",
            "context": context,
            "evidence_sentence": evidence,
            "value": ent.get("value", ""),
            "unit": ent.get("unit", ""),
            "needs_review": False,
        }

        if category in ALLOWED_CONTEXTS and context not in ALLOWED_CONTEXTS[category]:
            record["needs_review"] = True
            record["review_note"] = f"context '{context}' not in allowed set for {category}"

        if reasons:
            record["rejected_reason"] = "; ".join(reasons)
            rejected.append(record)
        else:
            kept.append(record)

    return kept, rejected


def validate_entities(entities: list, source_text: str):
    """Second LLM pass: re-checks each surviving entity against the
    letter text, can mark invalid or correct the context. Runs in
    batches to keep prompts a reasonable size."""
    if not entities:
        return [], []

    batch_size = 20
    validated = []
    rejected_by_validation = []

    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        payload_entities = [
            {"text_in_letter": e["text_in_letter"], "category": e["category"],
             "context": e["context"]}
            for e in batch
        ]
        user_prompt = (
            f"LETTER TEXT:\n{source_text}\n\n"
            f"PREVIOUSLY EXTRACTED ENTITIES TO REVIEW:\n"
            f"{json.dumps(payload_entities, indent=2)}"
        )
        try:
            result = call_ollama(VALIDATION_SYSTEM_PROMPT, user_prompt,
                                  few_shot=FEW_SHOT_VALIDATION_EXAMPLES)
        except Exception as e:
            log.error("Validation pass failed for a batch: %s — keeping "
                      "entities as unvalidated (flagged for manual review)", e)
            for e_rec in batch:
                rec = copy.deepcopy(e_rec)
                rec["needs_review"] = True
                rec["review_note"] = "validation pass failed to run"
                validated.append(rec)
            continue

        reviewed = result.get("reviewed", []) if isinstance(result, dict) else []
        reviewed_by_text = {}
        for r in reviewed:
            key = _normalise_for_match(r.get("text_in_letter", ""))
            reviewed_by_text[key] = r

        for e_rec in batch:
            key = _normalise_for_match(e_rec["text_in_letter"])
            review = reviewed_by_text.get(key)
            rec = copy.deepcopy(e_rec)
            if review is None:
                rec["needs_review"] = True
                rec["review_note"] = "not returned by validation pass"
                validated.append(rec)
                continue
            if not review.get("valid", False):
                rec["rejected_reason"] = (
                    f"failed LLM validation pass: {review.get('reason', '')}"
                )
                rejected_by_validation.append(rec)
                continue
            corrected = review.get("corrected_context")
            if corrected:
                corrected = corrected.strip().lower()
                if corrected in ALLOWED_CONTEXTS.get(rec["category"], set()):
                    rec["context"] = corrected
                    rec["review_note"] = (
                        rec.get("review_note", "") +
                        f" context corrected by validation pass ({review.get('reason', '')})"
                    ).strip()
                    rec["needs_review"] = True
            validated.append(rec)

    return validated, rejected_by_validation


def final_reverify(entities: list, source_text: str):
    """After validation may have edited context (never text), re-run the
    hard verbatim check on text_in_letter one more time before output —
    belt and braces."""
    kept, rejected = [], []
    for rec in entities:
        if verbatim_present(rec["text_in_letter"], source_text):
            kept.append(rec)
        else:
            rec["rejected_reason"] = "failed final re-verification"
            rejected.append(rec)
    return kept, rejected


def process_letter(pdf_path: Path, ocr_module):
    text = get_text_from_pdf(pdf_path, ocr_module)
    raw_text_path = OUTPUT_DIR / f"{pdf_path.stem}_ocr_text.txt"
    raw_text_path.write_text(text, encoding="utf-8")

    chunks = chunk_text(text, MAX_CHUNK_CHARS)
    log.info("  %s split into %d chunk(s) for extraction", pdf_path.name, len(chunks))

    raw_entities = []
    for chunk in chunks:
        raw_entities.extend(extract_chunk(chunk))
    log.info("  %s raw extraction: %d candidate entities", pdf_path.name, len(raw_entities))

    kept, hard_rejected = hard_filter_entities(raw_entities, text, pdf_path.name)
    log.info("  %s after verbatim anti-hallucination filter: %d kept, %d rejected",
              pdf_path.name, len(kept), len(hard_rejected))

    validated, validation_rejected = validate_entities(kept, text)
    log.info("  %s after LLM validation pass: %d kept, %d rejected",
              pdf_path.name, len(validated), len(validation_rejected))

    final_kept, final_rejected = final_reverify(validated, text)

    all_rejected = hard_rejected + validation_rejected + final_rejected

    # dedupe: same normalised text + category -> keep first
    seen = set()
    deduped = []
    for rec in final_kept:
        key = (_normalise_for_match(rec["text_in_letter"]), rec["category"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)

    return deduped, all_rejected


def main():
    ocr_module = _load_paddle_ocr_module()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(SAMPLE_LETTERS_DIR.glob("*.pdf"))
    if not pdfs:
        log.error("No PDFs found in %s", SAMPLE_LETTERS_DIR)
        return

    all_kept, all_rejected = [], []

    for pdf in pdfs:
        log.info("Processing %s ...", pdf.name)
        kept, rejected = process_letter(pdf, ocr_module)

        by_category = defaultdict(list)
        for rec in kept:
            by_category[rec["category"]].append(rec)

        letter_out = OUTPUT_DIR / f"{pdf.stem}_extracted.json"
        letter_out.write_text(
            json.dumps(by_category, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info(
            "  %s -> diagnosis=%d symptom=%d procedure=%d medication=%d vitals=%d "
            "(rejected=%d)",
            pdf.name,
            len(by_category.get("diagnosis", [])),
            len(by_category.get("symptom", [])),
            len(by_category.get("procedure", [])),
            len(by_category.get("medication", [])),
            len(by_category.get("vitals", [])),
            len(rejected),
        )

        all_kept.extend(kept)
        all_rejected.extend(rejected)

    combined_csv = OUTPUT_DIR / "all_letters_extracted.csv"
    fieldnames = ["source_file", "category", "text_in_letter", "context",
                  "value", "unit", "evidence_sentence", "needs_review",
                  "review_note"]
    with open(combined_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in all_kept:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})

    rejected_path = OUTPUT_DIR / "rejected_entities.json"
    rejected_path.write_text(
        json.dumps(all_rejected, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("\nDone. Outputs in %s/:", OUTPUT_DIR)
    log.info("  - <letter>_ocr_text.txt      raw OCR text per letter")
    log.info("  - <letter>_extracted.json    final validated entities per letter")
    log.info("  - all_letters_extracted.csv  combined, spreadsheet-friendly")
    log.info("  - rejected_entities.json     everything filtered out + why "
              "(failed verbatim check / failed validation pass) — review "
              "this if something expected is missing, and to audit that "
              "nothing hallucinated slipped through")


if __name__ == "__main__":
    sys.exit(main())