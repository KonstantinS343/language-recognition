from typing import NamedTuple, List
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel

class FileObject(NamedTuple):
    filename: str
    content: str
    
class RecognitionMethod(Enum):
    NGRAMM = 'ngramm'
    ALPHABET = 'alphabet'
    NEURO = 'neuro'
    
class Language(Enum):
    RUSSIAN = 'Russian'
    ENGLISH = 'English'
    
class LanguageResponse(BaseModel):
    doc: str
    language: str

class QueryRespose(BaseModel):
    response: List[LanguageResponse]
    precision: Decimal