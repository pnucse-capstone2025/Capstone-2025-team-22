"""
키워드 추출 관련 유틸리티 (app.nlp.utils.extract 대체)
"""

def extract_keywords_from_bio_tags(tokens, bio_tags):
    """BIO 태그로부터 키워드 추출"""
    keywords = []
    current_keyword = []
    
    for i, (token, tag) in enumerate(zip(tokens, bio_tags)):
        if tag == 0:  # B 태그
            if current_keyword:  # 이전 키워드가 있으면 저장
                keywords.append(''.join(current_keyword))
            current_keyword = [token]
        elif tag == 1:  # I 태그
            if current_keyword:  # B 태그 다음에만 I 태그가 유효
                current_keyword.append(token)
        else:  # O 태그
            if current_keyword:  # 키워드 종료
                keywords.append(''.join(current_keyword))
                current_keyword = []
    
    # 마지막 키워드 처리
    if current_keyword:
        keywords.append(''.join(current_keyword))
    
    # 특수 토큰 정리
    cleaned_keywords = []
    for keyword in keywords:
        # ##로 시작하는 서브워드 토큰 처리
        cleaned = keyword.replace('##', '')
        if cleaned and len(cleaned) > 1:  # 의미있는 키워드만
            cleaned_keywords.append(cleaned)
    
    return cleaned_keywords
