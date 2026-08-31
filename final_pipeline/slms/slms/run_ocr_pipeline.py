import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add current directory and parent directory to sys.path for clean imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SLMS_ROOT = CURRENT_DIR
FINAL_PIPELINE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))

if SLMS_ROOT not in sys.path:
    sys.path.insert(0, SLMS_ROOT)
if FINAL_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, FINAL_PIPELINE_DIR)

# Fix stdout/stderr encoding on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Import PaddleOCR from final_pipeline/paddle_ocr.py
try:
    from paddle_ocr import extract_text_from_pdf_paddle, initialize_paddle_ocr, PADDLE_AVAILABLE
except ImportError:
    print("❌ Failed to import paddle_ocr.py")
    PADDLE_AVAILABLE = False

# Import SLMS Clinical Pipeline
try:
    from pipeline.orchestrator import ClinicalPipeline
    from utils.logger import logger
except ImportError as e:
    print(f"❌ Failed to import SLMS ClinicalPipeline: {e}")
    sys.exit(1)


def run_pipeline_on_sample_letters():
    sample_letters_dir = os.path.join(FINAL_PIPELINE_DIR, "sample_letters")
    if not os.path.exists(sample_letters_dir):
        sample_letters_dir = os.path.join(os.path.dirname(FINAL_PIPELINE_DIR), "sample_letters")

    if not os.path.exists(sample_letters_dir):
        print(f"❌ Sample letters directory not found at: {sample_letters_dir}")
        return

    pdf_files = sorted([f for f in os.listdir(sample_letters_dir) if f.endswith(('.pdf', '.txt'))])
    if not pdf_files:
        print(f"❌ No .pdf or .txt sample letters found in {sample_letters_dir}")
        return

    print(f"\n================================================================================")
    print(f"🚀 SLMS QWEN CLINICAL PIPELINE WITH PADDLE OCR")
    print(f"================================================================================")
    print(f"📂 Found {len(pdf_files)} sample letters in '{sample_letters_dir}'")

    # Initialize PaddleOCR
    ocr_instance = None
    if PADDLE_AVAILABLE:
        print("📷 Initializing PaddleOCR engine...")
        ocr_instance = initialize_paddle_ocr()

    # Initialize SLMS Qwen Clinical Pipeline
    print("🧠 Initializing Qwen Clinical Pipeline (Ollama)...")
    pipeline = ClinicalPipeline()

    all_results = {}
    summary_rows = []

    for fname in pdf_files:
        fpath = os.path.join(sample_letters_dir, fname)
        print(f"\n--------------------------------------------------------------------------------")
        print(f"📄 Processing Document: {fname}")
        print(f"--------------------------------------------------------------------------------")

        # 1. OCR Extraction via PaddleOCR
        if fname.endswith('.pdf') and PADDLE_AVAILABLE:
            print(f"  [PaddleOCR] Extracting text from: {fname}...")
            ocr_res = extract_text_from_pdf_paddle(fpath, ocr_instance=ocr_instance)
            clinical_text = ocr_res.get("full_text", "")
        else:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                clinical_text = f.read()

        if not clinical_text.strip():
            print(f"  ⚠️ Warning: Extracted text is empty for {fname}")
            continue

        print(f"  [OCR Complete] Extracted {len(clinical_text)} characters.")
        print(f"  [Qwen Pipeline] Running 6-stage clinical extraction & SNOMED mapping...")

        # 2. SLMS Qwen Pipeline Extraction
        try:
            result = pipeline.process_letter(clinical_text)
            all_results[fname] = result
        except Exception as e:
            print(f"  ❌ Error processing letter with Qwen pipeline: {e}")
            result = {
                "letter_type": "Error",
                "diagnoses": [],
                "symptoms": [],
                "procedures": [],
                "medications": [],
                "vitals": []
            }
            all_results[fname] = result

        # Format summary for console table & Excel
        diag_str = "; ".join([f"{d['entity']} [{d.get('snomed', 'N/A')}]" for d in result.get("diagnoses", [])])
        symp_str = "; ".join([f"{s['entity']} [{s.get('snomed', 'N/A')}]" for s in result.get("symptoms", [])])
        proc_str = "; ".join([f"{p['entity']} [{p.get('snomed', 'N/A')}]" for p in result.get("procedures", [])])
        med_str = "; ".join([m['entity'] for m in result.get("medications", [])])
        vit_str = "; ".join([f"{v['entity']}: {v.get('value', '')}" for v in result.get("vitals", [])])

        summary_rows.append({
            "Filename": fname,
            "Letter Type": result.get("letter_type", "Unknown"),
            "Diagnoses": diag_str,
            "Symptoms": symp_str,
            "Procedures": proc_str,
            "Medications": med_str,
            "Vitals": vit_str
        })

    # Save JSON results
    json_out_path = os.path.join(CURRENT_DIR, "slms_extraction_results.json")
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved full JSON results to: {json_out_path}")

    # Save Excel summary
    excel_out_path = os.path.join(CURRENT_DIR, "slms_extraction_results.xlsx")
    pd.DataFrame(summary_rows).to_excel(excel_out_path, index=False)
    print(f"📊 Saved Excel summary table to: {excel_out_path}")

    # 3. Print Final Terminal Summary Table
    print("\n" + "=" * 80)
    print("SLMS QWEN CLINICAL EXTRACTION & SNOMED MAPPING SUMMARY")
    print("=" * 80)

    for fname, res in all_results.items():
        print(f"\n📄 [Letter] {fname}  (Type: {res.get('letter_type', 'Unknown')})")

        diag_list = [f"{d['entity']} (SNOMED: {d.get('snomed', 'N/A')})" for d in res.get("diagnoses", [])]
        print(f"  - Diagnoses  : {', '.join(diag_list) if diag_list else 'None'}")

        symp_list = [f"{s['entity']} (SNOMED: {s.get('snomed', 'N/A')})" for s in res.get("symptoms", [])]
        print(f"  - Symptoms   : {', '.join(symp_list) if symp_list else 'None'}")

        proc_list = [f"{p['entity']} (SNOMED: {p.get('snomed', 'N/A')})" for p in res.get("procedures", [])]
        print(f"  - Procedures : {', '.join(proc_list) if proc_list else 'None'}")

        med_list = [m['entity'] for m in res.get("medications", [])]
        if med_list:
            print(f"  - Medications: {', '.join(med_list)}")

        vit_list = [f"{v['entity']} ({v.get('value', '')})" for v in res.get("vitals", [])]
        if vit_list:
            print(f"  - Vitals     : {', '.join(vit_list)}")

    print("\n" + "=" * 80)
    print("🎉 Pipeline Execution Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline_on_sample_letters()
