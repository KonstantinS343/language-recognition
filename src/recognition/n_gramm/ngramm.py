import json
from typing import Mapping

from models.model import Language


class Ngramm:

    @classmethod
    async def ngramm_methods(cls, user_profile: Mapping[str, int]) -> Language:
        
        with open('src/recognition/n_gramm/datasets_profile/profile_en.json') as file:
            en_dataset_profile = json.load(file)
            
        with open('src/recognition/n_gramm/datasets_profile/profile_ru.json') as file:
            ru_dataset_profile = json.load(file)
            
        ru_dist = await cls.distance(user_profile, ru_dataset_profile)
        en_dist = await cls.distance(user_profile, en_dataset_profile)
        
        return Language.RUSSIAN if ru_dist < en_dist else Language.ENGLISH
            
    
    @classmethod
    async def distance(cls, user_profile: Mapping[str, int], dataset_profile: Mapping[str, int]) -> int:
        dist = 0
        
        max_dist = 100_000_000_000
        user_profile_list = list(user_profile.keys())
        dataset_profile_list = list(dataset_profile.keys())
        
        for index in range(len(user_profile_list)):
            try:
                dataset_index = abs(dataset_profile_list.index(user_profile_list[index]) - index)
            except ValueError:
                dataset_index = max_dist
            
            dist += dataset_index
            
        return dist