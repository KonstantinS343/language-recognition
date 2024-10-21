import string
from typing import Sequence
import yake
import nltk
from nltk.corpus import stopwords


class KeywordsSummarization:
    
    @classmethod
    async def keywords_method(cls, 
                              text: str, 
                              language: str, 
                              max_ngram_size: int = 2, 
                              deduplication_threshold: float = 0.3,
                              deduplication_algo: str = 'seqm', 
                              window_size: int = 1) -> str:
        
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        cleaned_text = await cls.clean_text(text, language)
        num_of_keywords = await cls.determine_num_of_keywords(len(cleaned_text.split()))

        try:
            custom_kw_extractor = yake.KeywordExtractor(lan=language, 
                                                        n=max_ngram_size, 
                                                        dedupLim=deduplication_threshold, 
                                                        dedupFunc=deduplication_algo, 
                                                        windowsSize=window_size, 
                                                        top=num_of_keywords, 
                                                        features=None)
        except ModuleNotFoundError:
            raise ModuleNotFoundError("YAKE library not installed")

        keywords = custom_kw_extractor.extract_keywords(cleaned_text)

        top_keywords = [kw[0] for kw in keywords]

        return await cls.post_process(top_keywords)
    
    @classmethod
    async def clean_text(cls, text: str, language: str) -> str:

        stop_words = set(stopwords.words('russian' if language == 'ru' else 'english'))
        tokens = text.split()

        cleaned_tokens = [
            word for word in tokens 
            if word.lower() not in stop_words and word not in string.punctuation
        ]
        return ' '.join(cleaned_tokens)

    @classmethod
    async def determine_num_of_keywords(cls, text_length: int) -> int:
        if text_length < 50:
            return 5
        elif text_length < 200:
            return 10
        elif text_length < 500:
            return 15
        else:
            return 20

    @classmethod
    async def post_process(cls, words: Sequence[str]) -> str:
        print(words)
        return ', '.join(words)