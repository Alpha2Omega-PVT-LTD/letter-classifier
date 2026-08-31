import os
import ollama
from pydantic import BaseModel, Field
from typing import Literal, Optional
from config import MODEL_NAME, TEMPERATURE, NUM_CTX, PROMPTS_DIR
from utils.json_parser import parse_and_validate_json
from utils.retry import retry_with_backoff
from utils.logger import logger
from pipeline.extractor import ExtractedEntity

class Step2Output(BaseModel):
    category: Literal["Diagnosis", "Symptom", "Procedure", "Medication", "Reject", "Unknown"]

class EntityClassifier:
    """
    Step 2: Entity Classifier.
    Receives an entity and its context, then classifies it into the correct category.
    Only non-Vital entities should enter this stage.
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.prompt_path = os.path.join(PROMPTS_DIR, "step2_classification.txt")
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def _call_llm(self, entity: ExtractedEntity) -> Optional[Step2Output]:
        prompt = (self.prompt_template
                  .replace("{entity}", entity.entity)
                  .replace("{evidence}", entity.evidence)
                  .replace("{sentence}", entity.sentence)
                  .replace("{provisional_type}", entity.provisional_type))
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
            return parse_and_validate_json(raw_content, Step2Output)
        except Exception as e:
            logger.error(f"Error calling Ollama in EntityClassifier for entity '{entity.entity}': {e}")
            return None

    def classify(self, entity: ExtractedEntity) -> str:
        """Classifies a single entity and returns the category string."""
        logger.debug(f"Step 2: Classifying category for entity '{entity.entity}'...")
        result = retry_with_backoff(lambda: self._call_llm(entity), retries=3)
        if result:
            logger.debug(f"Step 2: Classified '{entity.entity}' as '{result.category}'")
            return result.category
        logger.warning(f"Step 2: Failed to classify entity '{entity.entity}'. Defaulting to 'Unknown'.")
        return "Unknown"
