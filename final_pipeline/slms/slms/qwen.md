You are an expert Python software engineer, clinical NLP engineer, and LLM application architect.

Your task is to build a production-quality clinical information extraction system for NHS clinical letters.

The objective is NOT just to write code. The objective is to design a highly modular, maintainable, scalable pipeline that can be extended later with new models and entity types.

=========================================================
PROJECT OVERVIEW
=========================================================

I receive NHS clinical letters.

I need to automatically extract structured clinical information and save it into an Excel spreadsheet.

The final Excel columns are:

Letter_Type
Diagnosis
Diagnosis_SNOMED
Symptoms
Symptoms_SNOMED
Procedures
Procedures_SNOMED
Medications
Vitals_Labs

The system should be modular.

Each stage should have only ONE responsibility.

Avoid creating one giant script.

=========================================================
PIPELINE ARCHITECTURE
=========================================================

Clinical Letter

↓

STEP 1
Medical Entity Extraction

↓

STEP 2
Entity Classification

↓

STEP 3
Status Classification

↓

STEP 4
Validation

↓

STEP 5
SNOMED Mapping

↓

Excel Writer

=========================================================
STEP 1
Medical Entity Extraction
=========================================================

Use Qwen as the extraction model.

This stage should ONLY identify explicit medical entities.

DO NOT perform reasoning.

DO NOT classify historical.

DO NOT classify negation.

DO NOT perform SNOMED mapping.

DO NOT infer diseases.

Simply identify medical mentions.

For every entity return

{
    "id":"",
    "entity":"",
    "provisional_type":"",
    "evidence":"",
    "sentence":"",
    "confidence":""
}

The provisional_type should only be one of

Diagnosis
Symptom
Procedure
Medication
Vital
Unknown

Examples

COPD

Chest pain

Metformin

Colonoscopy

Gastroscopy

Barrett's oesophagus

HbA1c 7.4%

Weight 81 kg

Blood pressure 145/92

MRI brain

Minor reflux changes

The extractor should preserve the exact wording found in the document.

Do NOT normalise terminology.

Do NOT map SNOMED.

Do NOT perform any validation.

=========================================================
VITALS SPECIAL HANDLING
=========================================================

Vitals are different.

If provisional_type == Vital

they bypass every later classifier.

They go directly into the Vitals/Labs Excel column.

Examples

BP

Weight

Height

BMI

Heart rate

Respiratory rate

Temperature

HbA1c

CRP

ESR

Creatinine

eGFR

Platelets

WBC

Oxygen saturation

Peak flow

FEV1

Numeric laboratory values

Vitals should never be classified into Diagnosis, Symptom, Procedure or Medication.

=========================================================
STEP 2
Entity Classification
=========================================================

Only entities that are NOT Vital should enter this stage.

The classifier receives

entity

evidence

sentence

provisional type

Its task is ONLY to decide the correct entity category.

Possible outputs

Diagnosis

Symptom

Procedure

Medication

Reject

Unknown

Do not determine historical status.

Do not determine negation.

Do not perform SNOMED mapping.

One entity can belong to only one category.

=========================================================
STEP 3
Status Classification
=========================================================

Every classified entity receives a status.

Diagnosis

Current

Historical

Negated

Resolved

Suspected

Symptoms

Current

Historical

Negated

Resolved

Procedures

Performed

Planned

Recommended

Monitoring

Medication

Current

Started

Stopped

Changed

Recommended

Monitoring

This stage should ONLY determine status.

=========================================================
STEP 4
Validation
=========================================================

Validation is a completely independent LLM call.

The validator reviews every entity.

It should answer

Is this entity explicitly written?

Is the entity category correct?

Is the status correct?

Is there enough evidence?

Should this entity be rejected?

Reject anything inferred.

Examples

Asthma clinic

↓

Diagnosis = Asthma

↓

Reject

because the diagnosis is never explicitly written.

Another example

Metformin

↓

Diagnosis

↓

Reject

because it is medication.

=========================================================
STEP 5
SNOMED MAPPING
=========================================================

Only validated entities should reach this stage.

Map ONLY

Diagnosis

Symptoms

Procedures

Do NOT map

Medication

Vitals

Use the existing SNOMED API.

=========================================================
STEP 6
EXCEL WRITER
=========================================================

Write the validated output into Excel.

Columns

Letter_Type

Diagnosis

Diagnosis_SNOMED

Symptoms

Symptoms_SNOMED

Procedures

Procedures_SNOMED

Medications

Vitals_Labs

Multiple entities should be stored as numbered lists.

Example

Diagnosis

1. COPD

2. Hypertension

Symptoms

1. Chest pain

2. Breathlessness

=========================================================
PROJECT STRUCTURE
=========================================================

Create a proper Python project.

Example

project/

config/

prompts/

models/

pipeline/

utils/

output/

logs/

main.py

config.py

requirements.txt

=========================================================
CLASS DESIGN
=========================================================

MedicalEntityExtractor

EntityClassifier

StatusClassifier

Validator

SNOMEDMapper

ExcelWriter

ClinicalPipeline

Each class must have a single responsibility.

=========================================================
PROMPTS
=========================================================

Every LLM prompt should live in a separate file.

Do not hardcode prompts inside Python.

=========================================================
JSON OUTPUT
=========================================================

Every stage should return structured JSON.

Do not return plain text.

Validate every JSON using Pydantic models.

=========================================================
LOGGING
=========================================================

Use structured logging.

Log every pipeline stage.

Log rejected entities.

Log SNOMED failures.

Log validation failures.

=========================================================
ERROR HANDLING
=========================================================

Implement retry logic.

Gracefully recover from malformed JSON.

Gracefully recover from API failures.

Continue processing remaining entities.

=========================================================
BATCH PROCESSING
=========================================================

The pipeline should support processing hundreds of clinical letters.

Use progress bars.

Avoid loading the model repeatedly.

=========================================================
CODE QUALITY
=========================================================

Use

Python 3.12+

Type hints

Dataclasses or Pydantic

Docstrings

Modular functions

Clean architecture

SOLID principles

Avoid duplicated code.

=========================================================
IMPORTANT
=========================================================

Do NOT produce a prototype.

Do NOT generate a single Python file.

Generate a complete production-ready project with clean architecture.

Every pipeline stage must be independently testable.

The final code should be easy to extend with additional entity types, additional LLMs, and additional validation rules in the future.
input file : eleven.xslx column: Cleaned Data
output: qwennewpipe.xslx