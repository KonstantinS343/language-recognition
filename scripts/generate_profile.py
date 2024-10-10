import sys
import re
import json
from typing import Mapping, Sequence
from collections import defaultdict

from nltk.util import ngrams


re_mapping = {
    'en': r'(\b\w*\d\w*\b|[^a-zA-Z0-9\s])',
    'ru': r'(\b\w*\d\w*\b|[^а-яА-Я0-9\s])'
}


def preprocess_dataset(dataset: Sequence[Mapping[str, str]], lang: str) -> Sequence[str]:
    new_dataset = []
    
    for entry in dataset:
        text = re.sub(r'[\n\r\t]', ' ', entry.get('text', '')).strip()
        new_dataset.append(re.sub(re_mapping[lang], ' ', text))
        
    return new_dataset

def postprocess_ngramms_list(ngrams_list: Sequence[Sequence['str']]) -> Mapping[str, int]:
    ngramms_profile = defaultdict(int)
    
    for entry in ngrams_list:
        ngramms_profile[entry[0]] += 1
        
    ngramms_profile = {key: value for key, value in ngramms_profile.items() if value > 2}
        
    return dict(sorted(ngramms_profile.items(), key=lambda x:x[1], reverse=True))

def generate_ngramm_profile() -> None:
    try:
        gramms_amount = int(sys.argv[1])
        lang = sys.argv[2]
        
        if lang not in ['ru', 'en']:
            raise ValueError()
        
    except (ValueError, IndexError):
        exit('Error')
    
    with open(f'datasets/{lang}.json', 'r') as file:
        dataset = json.load(file)
        
   
    dataset = preprocess_dataset(dataset=dataset, lang=lang)

    tokens = []
    
    for text in dataset:
        for token in text.split(" "):
            if token != "":
                tokens.append(token.lower())
                
                
    ngrams_list = list(ngrams(tokens, gramms_amount))
    ngramms_profile = postprocess_ngramms_list(ngrams_list=ngrams_list)
    
    with open(f'src/recognition/n_gramm/datasets_profile/profile_{lang}.json', 'w') as file:
        json.dump(ngramms_profile, file, indent=4, ensure_ascii=False)
        
    
if __name__ == '__main__':
    generate_ngramm_profile()