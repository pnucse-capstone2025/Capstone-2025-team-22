from mecab import MeCab
from collections import Counter
from collections import defaultdict
import re
import torch

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

def _estimate_token_offsets(text, tokens):
    """
    토큰 리스트와 원본 텍스트를 비교하여 각 토큰의 오프셋을 추정합니다.
    offset_mapping을 지원하지 않는 토크나이저를 위한 대안입니다.
    """
    offsets = []
    current_pos = 0

    for token in tokens:
        # [CLS], [SEP], [PAD] 등 스페셜 토큰 처리
        if token in ('[CLS]', '[SEP]', '[PAD]', '[UNK]'):
            offsets.append((0, 0)) # 실제 텍스트에 해당하지 않음
            continue
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분 수정 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 토크나이저가 추가하는 특수문자( )와 서브워드 접두사(##)를 모두 제거
        clean_token = token.lstrip('▁##')
        
        # 만약 clean_token이 비어있으면 (예: 토큰이 ' '였던 경우) 건너뜀
        if not clean_token:
            offsets.append((0, 0))
            continue
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        try:
            # 현재 위치부터 토큰 검색
            token_start = text.index(clean_token, current_pos)
            token_end = token_start + len(clean_token)
            offsets.append((token_start, token_end))
            # 다음 검색을 위해 현재 위치 업데이트
            current_pos = token_end
        except ValueError:
            # 텍스트에서 토큰을 찾지 못한 경우 (토크나이저 정규화 등으로 발생 가능)
            offsets.append((0, 0))
            # 위치를 업데이트하지 않고 다음 토큰으로 넘어감

    return offsets

# text_analysis.py 파일에 추가

def _filter_substrings(result_list):
    """
    결과 리스트에서 다른 단어에 포함되는 부분 문자열을 제거합니다.
    예: ['국회의원', '의원'] -> ['국회의원']
    """
    # 점수가 높아 순서가 빠른 단어부터 처리하므로, 긴 단어가 유지될 확률이 높음
    words_to_check = [item['keyword'] for item in result_list]
    filtered_results = []

    for item in result_list:
        current_word = item['keyword']
        # 현재 단어가 다른 단어의 부분 문자열인지 확인
        is_substring = any(current_word in other_word for other_word in words_to_check if current_word != other_word)
        
        if not is_substring:
            filtered_results.append(item)
            
    return filtered_results

def analyze_keyword_attention(text, keywords_info, attentions, tokenizer):
    """
    각 키워드가 주목한 명사/동사의 어텐션 스코어와 위치를 개별적으로 분석합니다.
    """
    # 0. 공통 데이터 준비
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    input_ids = inputs['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    estimated_offsets = _estimate_token_offsets(text, tokens)

    attention_tensor = torch.stack(attentions).squeeze(1)
    attention_matrix = attention_tensor.sum(dim=(0, 1))

    mecab = MeCab()
    pos_tokens = mecab.pos(text)

    # --- 명사와 동사구를 재구성하는 로직 ---
    reconstructed_phrases = []
    i = 0
    # 동사구를 만들기 위한 어미/접사 태그 집합
    VERB_COMPONENTS = {"VX", "EP", "EF", "EC", "ETN", "ETM", "XSV", "XSA"}
    
    while i < len(pos_tokens):
        word, tag = pos_tokens[i]
        
        # 명사인 경우
        if tag.startswith('NN'):
            reconstructed_phrases.append({'phrase': word, 'pos': 'Noun'})
            i += 1
        # 동사인 경우, 완전한 동사구로 재구성
        elif tag.startswith('VV'):
            components = [word]
            i += 1
            while i < len(pos_tokens):
                next_word, next_tag = pos_tokens[i]
                if any(comp_tag in next_tag for comp_tag in VERB_COMPONENTS):
                    components.append(next_word)
                    i += 1
                else:
                    break
            full_verb = "".join(components)
            reconstructed_phrases.append({'phrase': full_verb, 'pos': 'Verb'})
        else:
            i += 1
    
    word_spans = defaultdict(list)
    for phrase_info in reconstructed_phrases:
        phrase = phrase_info['phrase']
        for match in re.finditer(re.escape(phrase), text):
            word_spans[phrase].append(match.span())
    
    # 최종 결과를 담을 딕셔너리
    final_results = {}
    keyword_strings_set = {kw_info['keyword'] for kw_info in keywords_info}

    # 1. 키워드별로 분석 반복 (이하 로직은 거의 동일)
    for kw_info in keywords_info:
        keyword_str = kw_info['keyword']
        
        keyword_indices = set()
        start, end = kw_info['start'], kw_info['end']
        for i, (offset_start, offset_end) in enumerate(estimated_offsets):
            if offset_start < end and offset_end > start:
                keyword_indices.add(i)
        
        keyword_indices = list(keyword_indices)
        if not keyword_indices:
            final_results[keyword_str] = {'nouns': [], 'verbs': []} # 'adjectives' -> 'verbs'
            continue

        temp_attention_matrix = attention_matrix.clone()
        temp_attention_matrix[keyword_indices, keyword_indices] = 0
        keyword_attention_scores = temp_attention_matrix[keyword_indices, :].sum(dim=0)

        word_info = defaultdict(list)  # 각 위치별로 별도 항목 저장
        for i, score in enumerate(keyword_attention_scores):
            if score.item() == 0: continue
            token_start, token_end = estimated_offsets[i]
            if token_start == token_end: continue
            
            for word, spans in word_spans.items():
                for span_start, span_end in spans:
                    if token_start < span_end and token_end > span_start:
                        # 각 위치별로 별도의 항목으로 저장
                        word_info[word].append({
                            'score': score.item(),
                            'span': (span_start, span_end)
                        })
                        break
        
        keyword_result = {'nouns': [], 'verbs': []} # 'adjectives' -> 'verbs'
        # 재구성된 구문의 품사 정보를 담은 맵
        phrase_pos_map = {p['phrase']: p['pos'] for p in reconstructed_phrases}
        
        for word, info_list in word_info.items():
            # 필터 1: attended 단어가 전체 키워드 중 하나와 일치하면 제외
            if word in keyword_strings_set: 
                continue

            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분 추가 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            # 필터 2: attended 단어가 현재 분석 중인 키워드의 부분 문자열이면 제외
            if word in keyword_str:
                continue
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                
            pos = phrase_pos_map.get(word, "")
            
            # 각 위치별로 별도의 항목 생성
            for info in info_list:
                entry = {
                    'keyword': word, 
                    'score': info['score'], 
                    'start': info['span'][0],
                    'end': info['span'][1]
                }
                
                if pos == 'Noun':
                    keyword_result['nouns'].append(entry)
                elif pos == 'Verb':
                    keyword_result['verbs'].append(entry)

        # 점수 기준으로 내림차순 정렬
        nouns_sorted = sorted(keyword_result['nouns'], key=lambda x: x['score'], reverse=True)
        verbs_sorted = sorted(keyword_result['verbs'], key=lambda x: x['score'], reverse=True)

        # 부분 문자열 필터링 적용 후 상위 10개 선택
        filtered_nouns = _filter_substrings(nouns_sorted)[:10]
        filtered_verbs = _filter_substrings(verbs_sorted)[:10]
        
        # 리스트를 딕셔너리로 변환
        nouns_dict = {}
        for i, noun_item in enumerate(filtered_nouns):
            key = f"{noun_item['keyword']}_{noun_item['start']}_{noun_item['end']}"
            nouns_dict[key] = noun_item
        
        verbs_dict = {}
        for i, verb_item in enumerate(filtered_verbs):
            key = f"{verb_item['keyword']}_{verb_item['start']}_{verb_item['end']}"
            verbs_dict[key] = verb_item
        
        keyword_result['nouns'] = nouns_dict
        keyword_result['verbs'] = verbs_dict
        
        final_results[keyword_str] = keyword_result
        
    return final_results

# sample_text = "그런데 지금 이것에 목적 변경 절차를 두면 감독 당국에서 이것을 와치(watch)하는데 이게 시리어스 와치 리스트(serious watch list), 무슨 와치 리스트 그런 것 있잖아요. 그렇게 보는 거예요?"
# print(extract_pos_frequencies(sample_text))

# mecab = MeCab()
# print(mecab.pos(sample_text))