from mecab import MeCab
from collections import Counter

# 붙일 수 있는 품사 조각들
ENDING_TAGS = {"EP", "EF", "EC", "ETM", "ETN", "VX"}

def extract_full_verbs(text):
    mecab = MeCab()
    tokens = mecab.pos(text)

    full_verbs = []
    i = 0

    while i < len(tokens):
        word, tag = tokens[i]

        # 동사(VV)가 시작되면
        if tag.startswith("VV"):
            components = [word]
            i += 1

            # 다음 토큰들을 병합
            while i < len(tokens):
                next_word, next_tag = tokens[i]

                # 다음이 VV로 시작하는 경우 = 새로운 동사이므로 끊기
                if next_tag.startswith("VV"):
                    break

                # 현재 구성 중인 동사에 붙일 수 있는 품사인지 확인
                if any(t in next_tag.split('+') for t in ENDING_TAGS):
                    components.append(next_word)
                    i += 1
                else:
                    break

            full_verbs.append("".join(components))
        else:
            i += 1

    return full_verbs

def extract_nouns_verbs_adjectives(text):
    mecab = MeCab()
    tokens = mecab.pos(text)

    nouns = [word for word, tag in tokens if tag.startswith('NN')]      # 명사
    verbs = extract_full_verbs(text)      # 동사
    adjectives = [word for word, tag in tokens if tag.startswith('VA')] # 형용사

    return {
        'nouns': Counter(nouns),
        'verbs': Counter(verbs),
        'adjectives': Counter(adjectives)
    }

# sample_text = "숙제를 끝내버렸다."
# print(extract_nouns_verbs_adjectives(sample_text))
#
# mecab = MeCab()
# print(mecab.pos(sample_text))