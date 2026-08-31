import os

# Ollama settings
MODEL_NAME = "qwen2.5:7b-instruct"
TEMPERATURE = 0.05
NUM_CTX = 4096

# File settings
INPUT_FILE = "Book2.xlsx"
OUTPUT_FILE = "qwennewpipe.xlsx"
TARGET_COLUMN = "Cleaned Data"

# Directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# SNOMED CT configuration
FHIR_BASE_URLS = [
    "https://r4.ontoserver.csiro.au/fhir",
    "https://snowstorm-training.snomedtools.org/fhir",
    "https://snowstorm.snomedtools.org/fhir",
    "https://snowstorm.ihtsdotools.org/fhir"
]
SNOMED_CT_SYSTEM_URI = "http://snomed.info/sct"
CACHE_PATH = os.path.join(BASE_DIR, "snomed_cache.json")
LOG_PATH = os.path.join(LOGS_DIR, "snomed_rejections.log")
MIN_ACCEPT_SCORE = 0.72

ECL_BY_CATEGORY = {
    "procedure": "<< 71388002",       # Procedure
    "medication": "<< 373873005",     # Pharmaceutical / biologic product
    "diagnosis": "<< 404684003",      # Clinical finding (covers disorder)
    "symptom": "<< 404684003",        # Clinical finding
}

ABBREVIATIONS = {
    "T2DM": "type 2 diabetes mellitus",
    "T1DM": "type 1 diabetes mellitus",
    "HTN": "essential hypertension",
    "CKD": "chronic kidney disease",
    "COPD": "chronic obstructive pulmonary disease",
    "IHD": "ischemic heart disease",
    "CHF": "congestive heart failure",
    "HF": "heart failure",
    "DVT": "deep vein thrombosis",
    "OGD": "oesophagogastroduodenoscopy",
    "ECG": "electrocardiogram",
    "MRI": "magnetic resonance imaging",
    "DEXA": "dual energy x ray absorptiometry",
    "U&E": "urea and electrolytes",
    "LFT": "liver function test",
    "LFTS": "liver function test",
    "TTE": "transthoracic echocardiography",
    "OCT": "optical coherence tomography",
    "PSA": "prostate specific antigen",
    "CLL": "chronic lymphocytic leukaemia",
    "AF": "atrial fibrillation",
    "MS":  "multiple sclerosis",
    "OSA":  "obstructive sleep apnoea",
    "CBT": "cognitive behavioral therapy"
}
