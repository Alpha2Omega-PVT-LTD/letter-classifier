import os
import ollama
from pydantic import BaseModel, Field
from typing import Optional
from config import MODEL_NAME, TEMPERATURE, NUM_CTX, PROMPTS_DIR
from utils.json_parser import parse_and_validate_json
from utils.retry import retry_with_backoff
from utils.logger import logger
from pipeline.extractor import ExtractedEntity

class Step3Output(BaseModel):
    status: str

# Valid statuses per category
VALID_STATUSES = {
    "Diagnosis": {"Current", "Historical", "Negated", "Resolved", "Suspected"},
    "Symptom": {"Current", "Historical", "Negated", "Resolved", "Warning", "Side Effects"},
    "Procedure": {"Performed", "Planned", "Recommended", "Monitoring"},
    "Medication": {"Current", "Started", "Stopped", "Changed", "Recommended", "Monitoring"}
}

DEFAULT_STATUS = {
    "Diagnosis": "Current",
    "Symptom": "Current",
    "Procedure": "Performed",
    "Medication": "Current"
}

class StatusClassifier:
    """
    Step 3: Status Classifier.
    Determines status for classified entities based on their category and context.
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.prompt_path = os.path.join(PROMPTS_DIR, "step3_status.txt")
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def _call_llm(self, entity: ExtractedEntity, category: str) -> Optional[Step3Output]:
        prompt = (self.prompt_template
                  .replace("{entity}", entity.entity)
                  .replace("{category}", category)
                  .replace("{evidence}", entity.evidence)
                  .replace("{sentence}", entity.sentence))
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
            return parse_and_validate_json(raw_content, Step3Output)
        except Exception as e:
            logger.error(f"Error calling Ollama in StatusClassifier for entity '{entity.entity}': {e}")
            return None

    def classify_status(self, entity: ExtractedEntity, category: str) -> str:
        """Classifies and validates status for a given category."""
        if category not in VALID_STATUSES:
            logger.warning(f"Step 3: Unknown category '{category}'. No status classified.")
            return "Unknown"
            
        logger.debug(f"Step 3: Classifying status for entity '{entity.entity}' (category: '{category}')...")
        result = retry_with_backoff(lambda: self._call_llm(entity, category), retries=3)
        
        if result:
            status = result.status.strip().title()
            # If Ollama gives lowercase or minor variation, check case-insensitively
            for valid in VALID_STATUSES[category]:
                if valid.lower() == status.lower():
                    logger.debug(f"Step 3: Status for '{entity.entity}' determined as '{valid}'")
                    return valid
            logger.warning(
                f"Step 3: Invalid status '{status}' returned for category '{category}'. "
                f"Defaulting to '{DEFAULT_STATUS[category]}'."
            )
            return DEFAULT_STATUS[category]
            
        logger.warning(f"Step 3: Failed to classify status for entity '{entity.entity}'. Defaulting to '{DEFAULT_STATUS[category]}'.")
        return DEFAULT_STATUS[category]
