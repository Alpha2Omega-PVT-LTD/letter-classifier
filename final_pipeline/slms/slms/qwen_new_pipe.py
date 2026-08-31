import os
import sys
import argparse
import pandas as pd
from typing import Optional

# Setup sys.path to resolve local imports cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure stdout/stderr encoding for Windows to prevent UnicodeEncodeErrors
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Import components
try:
    from config import INPUT_FILE, OUTPUT_FILE, TARGET_COLUMN
    from pipeline.orchestrator import ClinicalPipeline
    from pipeline.excel import ExcelWriter
    from utils.logger import logger, setup_logger
except ImportError as e:
    print(f"❌ Error importing pipeline modules: {e}")
    sys.exit(1)

def find_file(filename: str, search_dirs: list) -> Optional[str]:
    """Helper to locate a file in a list of directories."""
    for d in search_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None

def main():
    parser = argparse.ArgumentParser(description="Qwen Modular Clinical Information Extraction Pipeline")
    parser.add_argument("--input", type=str, default=INPUT_FILE, help="Path to input Excel spreadsheet")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Path to output Excel spreadsheet")
    parser.add_argument("--column", type=str, default=TARGET_COLUMN, help="Target column containing clinical text")
    args = parser.parse_args()

    # Search paths for the input file:
    # 1. Exact path specified
    # 2. Current working directory
    # 3. Parent directory (root of nhs-classifier)
    # 4. Same directory as this script
    search_dirs = [
        ".",
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ]
    input_path = args.input
    if not os.path.exists(input_path):
        found = find_file(args.input, search_dirs)
        if found:
            input_path = found
        else:
            logger.error(
                f"❌ Input file '{args.input}' not found. "
                f"Checked directories: {[os.path.abspath(d) for d in search_dirs]}"
            )
            sys.exit(1)

    logger.info(f"📂 Loading input spreadsheet: '{input_path}'...")
    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        logger.error(f"❌ Failed to read input Excel: {e}")
        sys.exit(1)

    if args.column not in df.columns:
        logger.error(
            f"❌ Target column '{args.column}' not found in the input spreadsheet. "
            f"Available columns are: {list(df.columns)}"
        )
        sys.exit(1)

    total_rows = len(df)
    logger.info(f"🚀 Initiating Clinical Extraction Pipeline for {total_rows} records...")

    pipeline = ClinicalPipeline()
    writer = ExcelWriter(args.output)
    results = []

    # Simple progress bar/indicator
    for index, row in df.iterrows():
        logger.info(f"\n==================================================")
        logger.info(f"Processing Record {index + 1} of {total_rows} (Row Index: {index})")
        logger.info(f"==================================================")
        
        raw_text = row[args.column]
        if pd.isna(raw_text) or str(raw_text).strip() == "":
            logger.info("⏹️ Record is empty. Skipping extraction stages.")
            result = {
                "letter_type": "Other",
                "diagnoses": [],
                "symptoms": [],
                "procedures": [],
                "medications": [],
                "vitals": []
            }
        else:
            try:
                result = pipeline.process_letter(str(raw_text))
            except Exception as e:
                logger.error(f"❌ Pipeline failed during processing of row {index}: {e}")
                result = {
                    "letter_type": "Other",
                    "diagnoses": [],
                    "symptoms": [],
                    "procedures": [],
                    "medications": [],
                    "vitals": []
                }
        results.append(result)

    # Write results to Excel
    success = writer.write(df, results)
    if success:
        logger.info("\n🎉 Pipeline Execution Completed Successfully!")
    else:
        logger.error("\n❌ Pipeline Execution Finished with Excel Writing Error.")

if __name__ == "__main__":
    main()
