#!/usr/bin/env python3
"""모델 예측 결과 디버깅"""

import torch
import sys
import os

# 부모 디렉토리 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from data import load_data, KeywordDataset, Collator
from model import KoKeyBERT
from real_distillation_gpu import extract_keywords_from_bio_tags, evaluate_keywords, create_real_models
from torch.utils.data import DataLoader

def debug_model_predictions():
    """모델 예측 결과 디버깅"""
    print("🔍 모델 예측 결과 디버깅")
    
    device = torch.device('cpu')
    
    # 토크나이저 로드
    try:
        sys.path.append('../../kobert_tokenizer')
        from kobert_tokenizer import KoBERTTokenizer
        tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        print("✅ KoBERT 토크나이저 로드 성공")
    except Exception as e:
        print(f"⚠️ KoBERT 토크나이저 로드 실패: {e}")
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("✅ 기본 BERT 토크나이저로 대체")
    
    # 데이터 로드
    print("\n📊 테스트 데이터 로딩 중...")
    test_data_path = "../../../src/data/test_clean.json"
    test_data = load_data(test_data_path)
    
    print(f"✅ 테스트 데이터: {len(test_data)} 항목")
    
    # Dataset과 DataLoader 생성
    test_dataset = KeywordDataset(test_data)
    collator = Collator(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collator)
    
    # 모델 생성
    print("\n🏗️ 모델 생성 중...")
    teacher_model, student_model = create_real_models(device)
    teacher_model.eval()
    student_model.eval()
    
    # 첫 번째 배치로 예측 테스트
    print("\n" + "="*50)
    print("📋 모델 예측 분석")
    print("="*50)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 1:  # 첫 번째 배치만
                break
                
            # test.py 방식의 batch 형식: (index, input_ids, attention_mask, tags)
            index, input_ids, attention_mask, tags = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            tags = tags.to(device)
            
            print(f"배치 크기: {input_ids.size(0)}")
            print(f"시퀀스 길이: {input_ids.size(1)}")
            
            # Teacher 모델 예측
            print("\n🏫 Teacher 모델 예측:")
            teacher_output = teacher_model(input_ids, attention_mask, tags)
            if isinstance(teacher_output, tuple):
                teacher_log_likelihood, teacher_sequence = teacher_output
                print(f"Teacher 출력 형태: (log_likelihood, sequence)")
                print(f"log_likelihood shape: {teacher_log_likelihood.shape if hasattr(teacher_log_likelihood, 'shape') else 'scalar'}")
                print(f"sequence length: {len(teacher_sequence)}")
                
                # Teacher 예측을 텐서로 변환
                teacher_predictions = torch.zeros_like(tags)
                for i, seq in enumerate(teacher_sequence):
                    if i < teacher_predictions.size(0) and len(seq) <= teacher_predictions.size(1):
                        teacher_predictions[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            else:
                teacher_predictions = teacher_output
                print(f"Teacher 출력 형태: direct predictions")
            
            # Student 모델 예측
            print("\n🎓 Student 모델 예측:")
            student_output = student_model(input_ids, attention_mask, tags, return_outputs=True)
            if len(student_output) == 3:
                student_loss, student_predictions, student_bert_outputs = student_output
                print(f"Student 출력 형태: (loss, predictions, bert_outputs)")
                print(f"Student loss: {student_loss.item():.4f}")
                print(f"Student predictions type: {type(student_predictions)}")
                
                # Student 예측을 텐서로 변환
                if isinstance(student_predictions, list):
                    student_pred_tensor = torch.zeros_like(tags)
                    for i, seq in enumerate(student_predictions):
                        if i < student_pred_tensor.size(0) and len(seq) <= student_pred_tensor.size(1):
                            student_pred_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                    student_predictions = student_pred_tensor
            
            # 각 샘플별 예측 결과 분석
            for i in range(min(2, input_ids.size(0))):
                print(f"\n--- 샘플 {i+1} ---")
                
                # 원본 데이터
                original_item = test_data[index[i].item()]
                actual_keywords = original_item['keyword']
                if isinstance(actual_keywords, str):
                    actual_keywords = [actual_keywords]
                
                print(f"실제 키워드: {actual_keywords}")
                
                # 정답 BIO 태그에서 키워드 추출
                true_keywords = extract_keywords_from_bio_tags(
                    input_ids[i], tags[i], attention_mask[i], tokenizer
                )
                print(f"정답 BIO → 키워드: {true_keywords}")
                
                # Teacher 예측에서 키워드 추출
                teacher_keywords = extract_keywords_from_bio_tags(
                    input_ids[i], teacher_predictions[i], attention_mask[i], tokenizer
                )
                print(f"Teacher 예측 → 키워드: {teacher_keywords}")
                
                # Student 예측에서 키워드 추출
                student_keywords = extract_keywords_from_bio_tags(
                    input_ids[i], student_predictions[i], attention_mask[i], tokenizer
                )
                print(f"Student 예측 → 키워드: {student_keywords}")
                
                # BIO 태그 비교
                print(f"정답 BIO: {tags[i][:10].tolist()}...")
                print(f"Teacher BIO: {teacher_predictions[i][:10].tolist()}...")
                print(f"Student BIO: {student_predictions[i][:10].tolist()}...")
                
                # 성능 계산
                teacher_tp, teacher_fp, teacher_fn = evaluate_keywords(teacher_keywords, actual_keywords)
                student_tp, student_fp, student_fn = evaluate_keywords(student_keywords, actual_keywords)
                
                print(f"Teacher: TP={teacher_tp}, FP={teacher_fp}, FN={teacher_fn}")
                print(f"Student: TP={student_tp}, FP={student_fp}, FN={student_fn}")
            
            break

if __name__ == "__main__":
    debug_model_predictions()
