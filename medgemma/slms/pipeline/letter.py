import os
import ollama
from pydantic import BaseModel
from typing import Optional
from config import MODEL_NAME, TEMPERATURE, NUM_CTX, PROMPTS_DIR
from utils.json_parser import parse_and_validate_json
from utils.retry import retry_with_backoff
from utils.logger import logger

class LetterClassificationOutput(BaseModel):
    letter_type: str

class LetterClassifier:
    """
    Utility stage to classify the overall letter type of the document.
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.prompt_path = os.path.join(PROMPTS_DIR, "letter_classification.txt")
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def _call_llm(self, clinical_text: str) -> Optional[LetterClassificationOutput]:
        prompt = self.prompt_template.replace("{clinical_text}", clinical_text)
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={
                    'temperature': TEMPERATURE,
                    'num_ctx': NUM_CTX,
                }
            )
            raw_content = response['message']['content']
            return parse_and_validate_json(raw_content, LetterClassificationOutput)
        except Exception as e:
            logger.error(f"Error calling Ollama in LetterClassifier: {e}")
            return None

    def classify(self, clinical_text: str) -> str:
        """Classifies clinical letter and validates it is one of the allowed categories."""
        logger.info("Classifying clinical letter type...")
        result = retry_with_backoff(lambda: self._call_llm(clinical_text), retries=3)
        if result:
            lt = result.letter_type.strip()
            if lt in ("DNA", "Shared Care Prescribing", "A&E Discharge Summary", "Other"):
                logger.info(f"Letter type classified as: '{lt}'")
                return lt
            else:
                logger.warning(f"Returned invalid letter type '{lt}'. Defaulting to 'Other'.")
                return "Other"
        logger.warning("Letter classification failed. Defaulting to 'Other'.")
        return "Other"
