import os
import datetime
import pandas as pd
from typing import List, Dict, Any
from utils.logger import logger

class ExcelWriter:
    """
    Step 6: Excel Writer.
    Formats the processed clinical extraction outputs and writes them into an Excel spreadsheet.
    """
    def __init__(self, output_path: str):
        self.output_path = output_path

    def format_numbered_list(self, items: List[str]) -> str:
        """Formats a list of strings into a 1-indexed numbered list."""
        if not items:
            return "1. None"
        return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))

    def format_procedures(self, procedures: List[Dict[str, Any]]) -> List[str]:
        """Formats procedures, showing their item and status."""
        return [f"{p.get('entity', p.get('item', ''))} ({p.get('status', '')})" for p in procedures]

    def format_medications(self, medications: List[Dict[str, Any]]) -> List[str]:
        """Formats medications, showing their item and status."""
        return [f"{m.get('entity', m.get('item', ''))} ({m.get('status', '')})" for m in medications]

    def format_diagnoses(self, diagnoses: List[Dict[str, Any]]) -> List[str]:
        """Formats diagnoses, showing their entity and status."""
        return [f"{d.get('entity', '')} ({d.get('status', '')})" for d in diagnoses]

    def format_symptoms(self, symptoms: List[Dict[str, Any]]) -> List[str]:
        """Formats symptoms, showing their entity and status."""
        return [f"{s.get('entity', '')} ({s.get('status', '')})" for s in symptoms]

    def format_vitals(self, vitals: List[Dict[str, Any]]) -> List[str]:
        """Formats vitals, showing their name and value."""
        return [f"{v.get('entity', v.get('name', ''))}: {v.get('value', '')}" for v in vitals]

    def write(self, df_input: pd.DataFrame, results: List[Dict[str, Any]]) -> bool:
        """
        Creates a copy of the input DataFrame, populates the clinical extraction columns,
        and saves it to the output Excel file.
        """
        logger.info(f"Step 6: Formatting extraction results and writing to Excel...")
        df = df_input.copy()

        letter_types = []
        diagnoses = []
        diagnoses_snomed = []
        symptoms = []
        symptoms_snomed = []
        procedures = []
        procedures_snomed = []
        medications = []
        vitals_labs = []

        for res in results:
            letter_types.append(res.get("letter_type", "Other"))
            
            # Format Diagnosis names and SNOMED codes
            diag_formatted = self.format_diagnoses(res.get("diagnoses", []))
            diagnoses.append(self.format_numbered_list(diag_formatted))
            diag_snomed_list = [d.get("snomed", "N/A") for d in res.get("diagnoses", [])]
            diagnoses_snomed.append(self.format_numbered_list(diag_snomed_list))

            # Format Symptom names and SNOMED codes
            symp_formatted = self.format_symptoms(res.get("symptoms", []))
            symptoms.append(self.format_numbered_list(symp_formatted))
            symp_snomed_list = [s.get("snomed", "N/A") for s in res.get("symptoms", [])]
            symptoms_snomed.append(self.format_numbered_list(symp_snomed_list))

            # Format Procedure names and SNOMED codes
            proc_formatted = self.format_procedures(res.get("procedures", []))
            procedures.append(self.format_numbered_list(proc_formatted))
            proc_snomed_list = [p.get("snomed", "N/A") for p in res.get("procedures", [])]
            procedures_snomed.append(self.format_numbered_list(proc_snomed_list))

            # Format Medications
            meds_formatted = self.format_medications(res.get("medications", []))
            medications.append(self.format_numbered_list(meds_formatted))

            # Format Vitals
            vitals_formatted = self.format_vitals(res.get("vitals", []))
            vitals_labs.append(self.format_numbered_list(vitals_formatted))

        # Assign columns to dataframe
        df["Letter_Type"] = letter_types
        df["Diagnosis"] = diagnoses
        df["Diagnosis_SNOMED"] = diagnoses_snomed
        df["Symptoms"] = symptoms
        df["Symptoms_SNOMED"] = symptoms_snomed
        df["Procedures"] = procedures
        df["Procedures_SNOMED"] = procedures_snomed
        df["Medications"] = medications
        df["Vitals_Labs"] = vitals_labs

        try:
            # Create directory if it does not exist
            out_dir = os.path.dirname(self.output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            df.to_excel(self.output_path, index=False)
            logger.info(f"Step 6: Excel report successfully written to '{self.output_path}'")
            return True
        except PermissionError:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(self.output_path)
            name, ext = os.path.splitext(filename)
            fallback_path = os.path.join(out_dir, f"{name}_locked_{timestamp}{ext}")
            df.to_excel(fallback_path, index=False)
            logger.error(
                f"Step 6: Permission denied writing to '{self.output_path}'. "
                f"Likely locked. Saved to alternative path: '{fallback_path}'"
            )
            return False
        except Exception as e:
            logger.error(f"Step 6: Failed to write Excel file: {e}")
            return False
