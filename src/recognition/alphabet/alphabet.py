from typing import Mapping

from models.model import Language


alphabet_frequencies = {
    Language.RUSSIAN: {
        "а": 0.0817,
        "б": 0.0159,
        "в": 0.0453,
        "г": 0.0170,
        "д": 0.0356,
        "е": 0.0843,
        "ё": 0.0020,
        "ж": 0.0054,
        "з": 0.0135,
        "и": 0.0709,
        "й": 0.0150,
        "к": 0.0350,
        "л": 0.0426,
        "м": 0.0294,
        "н": 0.0670,
        "о": 0.1095,
        "п": 0.0271,
        "р": 0.0422,
        "с": 0.0544,
        "т": 0.0657,
        "у": 0.0231,
        "ф": 0.0022,
        "х": 0.0060,
        "ц": 0.0047,
        "ч": 0.0156,
        "ш": 0.0069,
        "щ": 0.0016,
        "ъ": 0.0007,
        "ы": 0.0193,
        "ь": 0.0185,
        "э": 0.0123,
        "ю": 0.0077,
        "я": 0.0202,
    },
    Language.ENGLISH: {
        "a": 0.0817,
        "b": 0.0149,
        "c": 0.0278,
        "d": 0.0425,
        "e": 0.1270,
        "f": 0.0223,
        "g": 0.0202,
        "h": 0.0609,
        "i": 0.0697,
        "j": 0.0015,
        "k": 0.0077,
        "l": 0.0403,
        "m": 0.0241,
        "n": 0.0675,
        "o": 0.0751,
        "p": 0.0193,
        "q": 0.0010,
        "r": 0.0599,
        "s": 0.0633,
        "t": 0.0906,
        "u": 0.0276,
        "v": 0.0098,
        "w": 0.0236,
        "x": 0.0015,
        "y": 0.0197,
        "z": 0.0007,
    },
}

class Alphabet:       

    @classmethod
    async def alphabet_method(cls, user_profile: Mapping[str, int]):
        frequencies = {}
        text_length = len(user_profile)
        
        for letter, count in user_profile.items():
            frequencies[letter.lower()] = (count / text_length) / 100

        scores = {Language.RUSSIAN: 0, Language.ENGLISH: 0}

        for lang, freq_dict in alphabet_frequencies.items():
            for letter, expected_freq in freq_dict.items():
                scores[lang] += abs(
                    frequencies.get(letter, 0) - expected_freq
                )
        predicted_language = min(scores, key=scores.get)
        
        return predicted_language