from typing import Sequence
from summarization.keywords.summarization import KeywordsSummarization
from recognition.neuro.neuro import neuro
from models.model import Language


class Summarization:
    
    @classmethod
    async def resolve(cls, text: str) -> Sequence[str]:
        lang = 'en' if await neuro.get_language(text) == Language.ENGLISH else 'ru'
        response = await KeywordsSummarization.keywords_method(text=text, language=lang)
        
        return response
    
    @staticmethod
    async def response(keywords: str) -> Sequence[str]:
        return [keywords]