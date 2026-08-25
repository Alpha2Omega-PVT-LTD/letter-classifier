import os
import ollama
from pydantic import BaseModel, Field
from typing import Optional
from config import MODEL_NAME, TEMPERATURE, NUM_CTX, PROMPTS_DIR
from utils.json_parser import parse_and_validate_json
from utils.retry import retry_with_backoff
from utils.logger import logger
from pipeline.extractor import ExtractedEntity

class Step4Output(BaseModel):
    explicitly_written: bool
    category_correct: bool
    status_correct: bool
    enough_evidence: bool
    should_reject: bool
    reason: str

class Validator:
    """
    Step 4: Validator.
    Reviews each entity, checking if it is explicitly written, has correct category/status,
    has enough evidence, and decides if it should be rejected.
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.prompt_path = os.path.join(PROMPTS_DIR, "step4_validation.txt")
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def _call_llm(self, entity: ExtractedEntity, category: str, status: str) -> Optional[Step4Output]:
        prompt = (self.prompt_template
                  .replace("{entity}", entity.entity)
                  .replace("{category}", category)
                  .replace("{status}", status)
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
            return parse_and_validate_json(raw_content, Step4Output)
        except Exception as e:
            logger.error(f"Error calling Ollama in Validator for entity '{entity.entity}': {e}")
            return None

    def validate(self, entity: ExtractedEntity, category: str, status: str) -> bool:
        """
        Validates an entity. Returns True if the entity is valid,
        and False if it is rejected.
        """
        logger.info(f"Step 4: Validating entity '{entity.entity}' (category: '{category}', status: '{status}')...")
        result = retry_with_backoff(lambda: self._call_llm(entity, category, status), retries=3)
        if result:
            if result.should_reject:
                logger.warning(
                    f"Step 4: Rejected entity '{entity.entity}'! "
                    f"Reason: {result.reason} | "
                    f"Checks -> explicit: {result.explicitly_written}, "
                    f"category: {result.category_correct}, status: {result.status_correct}, evidence: {result.enough_evidence}"
                )
                return False
            else:
                logger.info(f"Step 4: Validated entity '{entity.entity}' successfully. Reason: {result.reason}")
                return True
        logger.warning(f"Step 4: Validation call failed for entity '{entity.entity}'. Defaulting to valid.")
        return True
