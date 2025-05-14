from mecab import MeCab  # 모듈 import
from collections import Counter

def extract_pos_frequencies(text):
    mecab = MeCab()  # 모듈의 MeCab() 호출
    tokens = mecab.pos(text)

    nouns = [word for word, tag in tokens if tag.startswith('NN')]      # 명사
    verbs = [word for word, tag in tokens if tag.startswith('VV')]      # 동사
    adjectives = [word for word, tag in tokens if tag.startswith('VA')] # 형용사

    return {
        'nouns': Counter(nouns),
        'verbs': Counter(verbs),
        'adjectives': Counter(adjectives)
    }