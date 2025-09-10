import torch
from typing import List

def extract_keywords_from_bio_tags(tokens, bio_tags, attention_mask, tokenizer, device) -> List[str]:
    """
    BIO 태그를 기반으로 키워드를 추출합니다.
    
    Args:
        tokens: 토큰 ID 텐서
        bio_tags: BIO 태그 텐서 (0: B, 1: I, 2: O)
        attention_mask: 패딩 마스크 텐서
        tokenizer: 토크나이저
        device: 현재 사용 중인 디바이스
    
    Returns:
        list: 추출된 키워드 리스트
    """
    keywords = []
    current_keyword = []
    previous_tag = None
    
    # 토큰과 태그의 길이가 다르면 문제가 발생할 수 있음
    if len(tokens) != len(bio_tags) or len(tokens) != len(attention_mask):
        print("토큰, 태그, 마스크의 길이가 일치하지 않습니다.")
        return []
    
    # 텐서를 CPU에서 처리하기 위한 준비
    tokens_cpu = tokens.cpu() if torch.is_tensor(tokens) else tokens
    bio_tags_cpu = bio_tags.cpu() if torch.is_tensor(bio_tags) else bio_tags
    attention_mask_cpu = attention_mask.cpu() if torch.is_tensor(attention_mask) else attention_mask
    
    for i, (token, tag, mask) in enumerate(zip(tokens_cpu, bio_tags_cpu, attention_mask_cpu)):
        if not mask:  # 패딩된 토큰은 건너뛰기
            continue
            
        # 정수 값으로 변환
        token_val = token.item() if torch.is_tensor(token) else token
        tag_val = tag.item() if torch.is_tensor(tag) else tag
        
        if tag_val == 0:  # B 태그
            # 이전 키워드가 있었다면 저장
            if current_keyword:
                try:
                    keyword = tokenizer.decode(current_keyword).strip()
                    if keyword and keyword not in keywords:
                        keywords.append(keyword)
                except Exception as e:
                    print(f"토큰 디코딩 오류: {e}")
            
            # 새로운 키워드 시작
            current_keyword = [token_val]
            
            # 다음 토큰들이 I 태그인 동안 계속 추가
            j = i + 1
            while j < len(tokens_cpu) and attention_mask_cpu[j]:
                next_token = tokens_cpu[j]
                next_tag = bio_tags_cpu[j]
                next_tag_val = next_tag.item() if torch.is_tensor(next_tag) else next_tag
                
                if next_tag_val == 1:  # I 태그
                    current_keyword.append(next_token.item() if torch.is_tensor(next_token) else next_token)
                    j += 1
                else:
                    break
            
            # 현재 키워드 저장
            try:
                keyword = tokenizer.decode(current_keyword).strip()
                if keyword and keyword not in keywords:
                    keywords.append(keyword)
            except Exception as e:
                print(f"토큰 디코딩 오류: {e}")
            
            current_keyword = []
            
        elif tag_val == 2:  # O 태그
            # 이전 키워드가 있었다면 저장
            if current_keyword:
                try:
                    keyword = tokenizer.decode(current_keyword).strip()
                    if keyword and keyword not in keywords:
                        keywords.append(keyword)
                except Exception as e:
                    print(f"토큰 디코딩 오류: {e}")
                current_keyword = []
        else:
            continue
    
    # 마지막 키워드 처리
    if current_keyword:
        try:
            keyword = tokenizer.decode(current_keyword).strip()
            if keyword and keyword not in keywords:
                keywords.append(keyword)
        except Exception as e:
            print(f"토큰 디코딩 오류: {e}")
    
    return keywords