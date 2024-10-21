import json
from typing import Sequence, Mapping
from collections import defaultdict
import re

from langdetect import detect
from decimal import Decimal

from parser.html_parser import Parser
from recognition.n_gramm.ngramm import Ngramm
from recognition.alphabet.alphabet import Alphabet
from recognition.neuro.neuro import neuro
from summarization.core import Summarization
from models.model import RecognitionMethod, FileObject, Language, QueryRespose, LanguageResponse


class Controller:
    
    IP = 'http://192.168.112.163/static/'
    
    @classmethod
    async def init_resolver_mapping(cls):
        cls.resolver_mapping = {
            'ngramm': Ngramm.ngramm_methods,
            'alphabet': Alphabet.alphabet_method,
            'neuro': neuro.get_language,
            'summarization': Summarization.resolve
        }
    
    @classmethod
    async def resolve(cls, texts: Sequence[FileObject], recognition_method: RecognitionMethod) -> Sequence[QueryRespose]:
        if not hasattr(cls, 'resolver_mapping'):
            await cls.init_resolver_mapping()
            
        responses = []
        
        for text in texts:
            if recognition_method == RecognitionMethod.ALPHABET:
                user_profile = await cls.preprocess_alphabet_data(text=text.content)
            elif recognition_method == RecognitionMethod.NGRAMM:
                user_profile = await cls.preprocess_ngramm_data(text=text.content)
            elif recognition_method == RecognitionMethod.NEURO:
                content = await Parser.parse(text.content)
                user_profile = re.sub(r'(\b\w*\d\w*\b|[^a-zA-Zа-яА-Я0-9\s])', ' ', content).strip()
            elif recognition_method == RecognitionMethod.SUMMARIZATION:
                content = await Parser.parse(text.content)
                user_profile = re.sub(r'(\b\w*\d\w*\b|[^a-zA-Zа-яА-Я0-9\s])', ' ', content).strip()
                
            with open('log.json', 'w') as f:
                json.dump(user_profile, f, indent=4, ensure_ascii=False)
            
            response = await cls.resolver_mapping[recognition_method.value](user_profile)
            
            responses.append((text.filename, response))
        precision = 1    
        
        if recognition_method != RecognitionMethod.SUMMARIZATION:
            precision = await cls.calculate_precision(responses, texts)
        
        return await cls.response(responses, precision)
        
    @classmethod
    async def preprocess_alphabet_data(cls, text: str) -> Mapping[str, int]:
        content = await Parser.parse(text)
        new_text = re.sub(r'(\b\w*\d\w*\b|[^a-zA-Zа-яА-Я0-9\s])', ' ', content).strip()
        
        letter_counts = defaultdict(int)
        for letter in new_text:
            if letter.isalpha():
                letter_counts[letter.lower()] += 1
                
        return letter_counts
        
    @classmethod
    async def preprocess_ngramm_data(cls, text: str) -> Mapping[str, int]:
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
    async def calculate_precision(cls, language_recognition: Sequence[Sequence[str|Language]], texts: Sequence[FileObject]):
        precision = len(texts)
        
        for text_id in range(len(texts)):
            content = await Parser.parse(texts[text_id].content)

            detected_lang = Language.RUSSIAN if detect(content) == 'ru' else Language.ENGLISH
            precision -= 0 if detected_lang == language_recognition[text_id][1] else 1
            
        return Decimal(precision / len(texts)).quantize(Decimal('1.000'))
    
    @classmethod
    async def response(cls, responses: Sequence[Sequence[str|Language]], precision: Decimal) -> QueryRespose:
        reponse = []
        
        for i in responses:
            
            try:
                item = LanguageResponse(doc=cls.IP + i[0], value=i[1].value)
            except AttributeError:
                item = LanguageResponse(doc=cls.IP + i[0], value=i[1])
            
            reponse.append(item)
        
        return QueryRespose(response=reponse, precision=precision)