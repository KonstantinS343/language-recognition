import json
from typing import Sequence, Mapping
from collections import defaultdict
import re

from parser.html_parser import Parser
from recognition.n_gramm.ngramm import Ngramm
from models.model import RecognitionMethod, FileObject, Language, QueryRespose


class Controller:
    
    IP = 'http://localhost:2000/static/'
    
    @classmethod
    async def init_resolver_mapping(cls):
        cls.resolver_mapping = {
            'ngramm': cls.ngramms,
            'alphabet': cls.alphabet,
            'neuro': cls.neuro,
        }
    
    @classmethod
    async def resolve(cls, texts: Sequence[FileObject], recognition_method: RecognitionMethod):
        if not hasattr(cls, 'resolver_mapping'):
            await cls.init_resolver_mapping()
        return await cls.resolver_mapping[recognition_method.value](texts)
    
    @classmethod
    async def ngramms(cls, texts: Sequence[FileObject]):
        language_recognition = []
        
        for text in texts:
            user_profile = await cls.preprocess_ngramms_data(text=text.content)
            
            language_recognition.append((text.filename, await Ngramm.ngramm_methods(user_profile)))
            
        
        return await cls.response(language_recognition)
    
    @classmethod  
    def alphabet(cls, texts: Sequence[str]):
        ...
    
    @classmethod   
    def neuro(cls, texts: Sequence[str]):
        ...
        
    @classmethod
    async def preprocess_ngramms_data(cls, text: str) -> Mapping[str, int]:
        content = await Parser.parse(text)
        new_text = re.sub(r'(\b\w*\d\w*\b|[^a-zA-Zа-яА-Я0-9\s])', ' ', content).strip()
            
        tokens = []
    
        for token in new_text.split(" "):
            if token != "":
                tokens.append(token.lower())
                    
        user_profile = defaultdict(int)
    
        for entry in tokens:
            user_profile[entry] += 1
            
        user_profile = {key: value for key, value in user_profile.items() if value > 1}
        
        return dict(sorted(user_profile.items(), key=lambda x:x[1], reverse=True))
    
    @classmethod
    async def response(cls, language_recognition: Sequence[Sequence[str|Language]]):
        reponse = []
        
        for i in language_recognition:
            reponse.append(
                QueryRespose(
                    doc=cls.IP + i[0],
                    language=i[1].value
                )
            )
        
        return reponse