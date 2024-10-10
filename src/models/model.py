from typing import NamedTuple
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
    

class QueryRespose(BaseModel):
    doc: str
    language: str