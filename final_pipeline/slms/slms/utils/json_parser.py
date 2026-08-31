import json
import re
from typing import Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError
from utils.logger import logger

T = TypeVar('T', bound=BaseModel)

def _strip_code_fences(text: str) -> str:
    """Removes markdown code block markers from the LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def _extract_first_json_object(text: str) -> Optional[str]:
    """Extracts the first matching balanced JSON object {...} from the text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

def parse_and_validate_json(raw_text: str, schema_class: Type[T]) -> Optional[T]:
    """
    Cleans code fences, parses the JSON structure, and validates it using Pydantic.
    Gracefully falls back to search/extract balance of brackets if raw parse fails.
    """
    cleaned = _strip_code_fences(raw_text)
    parsed_dict = None
    
    # Try direct parse
    try:
        parsed_dict = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting first JSON object
        candidate = _extract_first_json_object(cleaned)
        if candidate:
            try:
                parsed_dict = json.loads(candidate)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse extracted JSON candidate: {e}")
        else:
            logger.error("No JSON-like structure found in the response.")
            
    if parsed_dict is None:
        return None
        
    # Validate with Pydantic
    try:
        validated = schema_class.model_validate(parsed_dict)
        return validated
    except ValidationError as ve:
        logger.error(f"Pydantic Validation Error for schema {schema_class.__name__}: {ve}")
        return None
