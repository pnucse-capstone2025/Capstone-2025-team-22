import torch
import json
import os
from typing import List, Dict, Any, Optional
from transformers import BertTokenizer

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """JSON 파일에서 데이터 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 딕셔너리 형태를 리스트로 변환
    if isinstance(data, dict):
        data = list(data.values())
    
    return data

def find_data_file(filename: str, search_dirs: List[str]) -> Optional[str]:
    """여러 경로에서 데이터 파일 찾기"""
    for dir_path in search_dirs:
        full_path = os.path.join(dir_path, filename)
        if os.path.exists(full_path):
            return full_path
    return None

def create_bio_tags_simple(text: str, keywords: List[str], tokenizer: BertTokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    """간단한 방식으로 BIO 태그 생성"""
    tags = torch.full((len(input_ids),), 2, dtype=torch.long)  # 모두 O로 초기화
    
    # 토큰을 텍스트로 변환
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # 각 키워드에 대해 BIO 태그 설정
    for keyword in keywords:
        keyword_tokens = tokenizer.tokenize(keyword)
        if not keyword_tokens:
            continue
            
        # 토큰 시퀀스에서 키워드 찾기
        for i in range(len(tokens) - len(keyword_tokens) + 1):
            match = True
            for j, kw_token in enumerate(keyword_tokens):
                if tokens[i + j] != kw_token:
                    match = False
                    break
            
            if match:
                tags[i] = 0  # B 태그
                for j in range(1, len(keyword_tokens)):
                    if i + j < len(tags):
                        tags[i + j] = 1  # I 태그
                break
    
    return tags

def calculate_model_size(model):
    """모델 크기 계산"""
    return sum(p.numel() for p in model.parameters())

def format_number(num):
    """숫자를 읽기 쉽게 포맷팅"""
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)
