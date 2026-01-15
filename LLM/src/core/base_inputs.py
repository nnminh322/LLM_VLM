from dataclasses import dataclass
from typing import Dict, List, Any, Optional

class BaseInput:
    pass

@dataclass
class TextInput(BaseInput):
    messages: List[Dict[str, str]]
    max_length: Optional[int] = None
    
@dataclass
class TextAndImageInput(BaseInput):
    messages: List[Dict[str, str]]
    images: List[Any]
    max_length: Optional[int] = None