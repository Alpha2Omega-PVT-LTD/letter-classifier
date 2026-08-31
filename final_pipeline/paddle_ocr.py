import os
import json
import tempfile
from typing import Dict, List, Any

# Import PyMuPDF (fitz) for PDF page to image rendering
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

# Disable PIR compiler & oneDNN executor flags for Paddle 3.x compatibility
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"

# Import PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

# Fallback text extractor (pypdf) if OCR dependencies unavailable
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


def initialize_paddle_ocr(lang: str = "en") -> Any:
    """Initializes and returns the PaddleOCR instance."""
    if not PADDLE_AVAILABLE:
        print("[Warning] paddleocr is not installed in the python environment.")
        return None
    
    print("Initializing PaddleOCR model...")
    try:
        ocr = PaddleOCR(lang=lang)
    except Exception as e:
        print(f"Error initializing PaddleOCR with lang={lang}: {e}")
        try:
            ocr = PaddleOCR()
        except Exception as e2:
            print(f"Error initializing PaddleOCR default: {e2}")
            return None
    print("PaddleOCR initialized successfully.")
    return ocr


def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[str]:
    """Converts pages of a PDF file into temporary PNG image files using PyMuPDF (fitz)."""
    image_paths = []
    if not FITZ_AVAILABLE:
        print(f"[Warning] PyMuPDF (fitz) not available to render PDF images for {pdf_path}")
        return image_paths

    doc = fitz.open(pdf_path)
    temp_dir = tempfile.mkdtemp(prefix="paddle_pdf_")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
        pix.save(img_path)
        image_paths.append(img_path)

    doc.close()
    return image_paths


def extract_text_from_pdf_paddle(pdf_path: str, ocr_instance: Any = None) -> Dict[str, Any]:
    """
    Extracts text contents from a PDF using PaddleOCR.
    Returns structured results containing full text, per-page text, and detected bounding boxes with confidence scores.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Fallback if PaddleOCR or fitz is not installed
    if not PADDLE_AVAILABLE or not FITZ_AVAILABLE:
        print(f"[Info] Using pypdf fallback extraction for {pdf_path} (PaddleOCR/PyMuPDF not available)")
        if PYPDF_AVAILABLE:
            reader = pypdf.PdfReader(pdf_path)
            extracted_pages = [page.extract_text() for page in reader.pages]
            full_text = "\n".join(extracted_pages)
            return {
                "pdf_path": pdf_path,
                "full_text": full_text,
                "pages": [{"page_num": i + 1, "text": text, "ocr_details": []} for i, text in enumerate(extracted_pages)],
                "method": "pypdf_fallback"
            }
        else:
            raise RuntimeError("Neither PaddleOCR nor pypdf is available for PDF text extraction.")

    if ocr_instance is None:
        ocr_instance = initialize_paddle_ocr()

    image_paths = pdf_to_images(pdf_path)
    pages_data = []
    full_text_lines = []

    for idx, img_path in enumerate(image_paths):
        page_num = idx + 1
        result = None
        if hasattr(ocr_instance, "predict"):
            try:
                result = ocr_instance.predict(img_path)
            except Exception:
                pass
        if result is None:
            try:
                result = ocr_instance.ocr(img_path)
            except Exception as e:
                print(f"[warn] PaddleOCR failed on image {img_path}: {e}")
                result = None

        page_lines = []
        ocr_details = []

        if result:
            res_item = result[0] if (isinstance(result, list) and len(result) > 0) else result
            
            # Extract dictionary representation if it's an OCRResult object
            res_dict = {}
            if hasattr(res_item, 'json') and isinstance(res_item.json, dict):
                res_dict = res_item.json.get('res', {})
            elif isinstance(res_item, dict):
                res_dict = res_item.get('res', res_item)
            elif hasattr(res_item, 'get'):
                res_dict = res_item

            rec_texts = res_dict.get('rec_texts', [])
            rec_scores = res_dict.get('rec_scores', [])
            rec_boxes = res_dict.get('rec_polys', res_dict.get('rec_boxes', []))

            if rec_texts:
                for i, text in enumerate(rec_texts):
                    score = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                    box = rec_boxes[i] if i < len(rec_boxes) else []
                    page_lines.append(str(text))
                    ocr_details.append({
                        "text": str(text),
                        "confidence": score,
                        "bbox": box
                    })
            elif isinstance(res_item, list):
                for line in res_item:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        bbox = line[0]
                        text_conf = line[1]
                        if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                            text, confidence = text_conf[0], text_conf[1]
                        else:
                            text, confidence = str(text_conf), 1.0
                        page_lines.append(str(text))
                        ocr_details.append({
                            "text": str(text),
                            "confidence": float(confidence),
                            "bbox": bbox
                        })

        page_text = "\n".join(page_lines)
        full_text_lines.append(page_text)

        pages_data.append({
            "page_num": page_num,
            "text": page_text,
            "ocr_details": ocr_details
        })

        # Cleanup temporary image file
        try:
            os.remove(img_path)
        except OSError:
            pass

    full_text = "\n\n".join(full_text_lines).strip()
    
    # Fallback to pypdf if PaddleOCR failed to extract text
    if not full_text and PYPDF_AVAILABLE:
        print(f"[Info] PaddleOCR output empty for {pdf_path}. Falling back to pypdf reader...")
        try:
            reader = pypdf.PdfReader(pdf_path)
            extracted_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            full_text = "\n".join(extracted_pages)
            pages_data = [{"page_num": i + 1, "text": text, "ocr_details": []} for i, text in enumerate(extracted_pages)]
            return {
                "pdf_path": pdf_path,
                "full_text": full_text,
                "pages": pages_data,
                "method": "pypdf_fallback"
            }
        except Exception as e_pdf:
            print(f"[warn] pypdf fallback also failed: {e_pdf}")

    return {
        "pdf_path": pdf_path,
        "full_text": full_text,
        "pages": pages_data,
        "method": "PaddleOCR"
    }


def process_sample_letters_ocr(sample_dir: str = "final_pipeline/sample_letters") -> Dict[str, Any]:
    """Processes all PDF sample letters in sample_dir using PaddleOCR."""
    if not os.path.exists(sample_dir):
        sample_dir = os.path.join(os.path.dirname(__file__), "sample_letters")

    files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".pdf")])
    print(f"Found {len(files)} PDF sample letters in '{sample_dir}' for PaddleOCR extraction.")

    ocr_instance = initialize_paddle_ocr() if PADDLE_AVAILABLE else None
    results = {}

    for fname in files:
        fpath = os.path.join(sample_dir, fname)
        print(f"Extracting PDF contents from: {fname}...")
        res = extract_text_from_pdf_paddle(fpath, ocr_instance=ocr_instance)
        results[fname] = res

    return results


if __name__ == "__main__":
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_letters")
    if not os.path.exists(sample_dir):
        sample_dir = "final_pipeline/sample_letters"

    ocr_results = process_sample_letters_ocr(sample_dir)

    output_json = os.path.join(os.path.dirname(__file__), "paddle_ocr_extracted_texts.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("PADDLE OCR PDF TEXT EXTRACTION SUMMARY")
    print("=" * 80)
    for fname, data in ocr_results.items():
        print(f"\n[Document] {fname} ({data['method']})")
        lines = [line for line in data['full_text'].splitlines() if line.strip()]
        preview = "\n  ".join(lines[:6])
        print(f"  {preview}")
        if len(lines) > 6:
            print(f"  ... ({len(lines) - 6} more lines)")
    print("=" * 80)
    print(f"\nSaved extracted OCR results to JSON: {output_json}")
