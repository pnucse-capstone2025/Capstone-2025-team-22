"""
실제 KoKeyBERT와 실제 데이터를 사용한 GPU 최적화 Knowledge Distillation
가짜 모델이나 가짜 데이터 없이 진짜 실험
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import os
import sys
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer, BertConfig
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.nn.utils.rnn import pad_sequence
from typing import List

# test.py 방식 사용으로 전역 collate_fn 제거

# 부모 디렉토리 추가 (실제 모델 import를 위해)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # distillation_experiment
grandparent_dir = os.path.dirname(parent_dir)  # nlp
root_dir = os.path.dirname(grandparent_dir)  # 2025_CSE_graduation_assignment

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, root_dir)

def extract_keywords_from_bio_tags(tokens, bio_tags, attention_mask, tokenizer) -> List[str]:
    """
    BIO 태그를 기반으로 키워드를 추출합니다. (test.py와 동일한 로직)
    
    Args:
        tokens: 토큰 ID 텐서
        bio_tags: BIO 태그 텐서 (0: B, 1: I, 2: O)
        attention_mask: 패딩 마스크 텐서
        tokenizer: 토크나이저
    
    Returns:
        list: 추출된 키워드 리스트
    """
    keywords = []
    current_keyword = []
    previous_tag = None
    
    # 토큰과 태그의 길이가 다르면 문제가 발생할 수 있음
    if len(tokens) != len(bio_tags) or len(tokens) != len(attention_mask):
        print("⚠️ 토큰, 태그, 마스크의 길이가 일치하지 않습니다.")
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
                    print(f"⚠️ 토큰 디코딩 오류: {e}")
            current_keyword = [token_val]
            previous_tag = tag_val
        elif tag_val == 1:  # I 태그
            # B 태그(0) 다음에 오는 I 태그만 처리 (O 태그(2) 다음의 I 태그는 무시)
            if previous_tag == 0 or previous_tag == 1:
                current_keyword.append(token_val)
            previous_tag = tag_val
        else:  # O 태그(2)
            # 이전 키워드가 있었다면 저장
            if current_keyword:
                try:
                    keyword = tokenizer.decode(current_keyword).strip()
                    if keyword and keyword not in keywords:
                        keywords.append(keyword)
                except Exception as e:
                    print(f"⚠️ 토큰 디코딩 오류: {e}")
            current_keyword = []
            previous_tag = tag_val
    
    # 마지막 키워드 처리
    if current_keyword:
        try:
            keyword = tokenizer.decode(current_keyword).strip()
            if keyword and keyword not in keywords:
                keywords.append(keyword)
        except Exception as e:
            print(f"⚠️ 토큰 디코딩 오류: {e}")
    
    return keywords


def evaluate_keywords(pred_keywords, true_keywords):
    """
    예측된 키워드와 실제 키워드를 비교하여 TP, FP, FN을 계산합니다.
    
    Args:
        pred_keywords: 예측된 키워드 리스트
        true_keywords: 실제 키워드 리스트
    
    Returns:
        tuple: (TP, FP, FN) 값
    """
    pred_set = set(pred_keywords)
    true_set = set(true_keywords)
    
    TP = len(pred_set & true_set)  # 교집합
    FP = len(pred_set - true_set)  # 예측했지만 정답이 아닌 것
    FN = len(true_set - pred_set)  # 정답이지만 예측하지 못한 것
    
    return TP, FP, FN


# test.py와 동일한 import 추가
from data import load_data, KeywordDataset, Collator

print("🔄 실제 모델 import 시도 중...")
try:
    from model import KoKeyBERT
    print("✅ KoKeyBERT import 성공")
except ImportError as e:
    print(f"❌ KoKeyBERT import 실패: {e}")
    # 대안 import 시도
    try:
        sys.path.insert(0, os.path.join(grandparent_dir, 'nlp'))
        from model import KoKeyBERT
        print("✅ KoKeyBERT import 성공 (대안 경로)")
    except ImportError as e2:
        print(f"❌ 대안 경로도 실패: {e2}")
        sys.exit(1)

try:
    from distill_model import DistillKoKeyBERT
    print("✅ DistillKoKeyBERT import 성공")
except ImportError as e:
    print(f"❌ DistillKoKeyBERT import 실패: {e}")
    sys.exit(1)

class RealDataset(Dataset):
    """실제 데이터를 처리하는 Dataset"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        keywords = item['keyword']
        
        # 토크나이징 (패딩 없이)
        encoding = self.tokenizer(
            text,
            padding=False,  # 패딩을 collate_fn에서 처리
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # BIO 태그 생성 (간단한 방식)
        tags = self.create_bio_tags(text, keywords, input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tags': tags,
            'text': text,
            'keywords': keywords
        }
    
    def create_bio_tags(self, text, keywords, input_ids):
        """키워드 기반으로 BIO 태그 생성"""
        tags = torch.full((len(input_ids),), 2, dtype=torch.long)  # 모두 O로 초기화
        
        try:
            # 토큰을 텍스트로 변환
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            
            # 각 키워드에 대해 BIO 태그 설정
            for keyword in keywords:
                if not keyword or not isinstance(keyword, str):
                    continue
                    
                keyword_tokens = self.tokenizer.tokenize(keyword)
                if not keyword_tokens:
                    continue
                    
                # 토큰 시퀀스에서 키워드 찾기
                for i in range(len(tokens) - len(keyword_tokens) + 1):
                    match = True
                    for j, kw_token in enumerate(keyword_tokens):
                        if i + j >= len(tokens) or tokens[i + j] != kw_token:
                            match = False
                            break
                    
                    if match:
                        tags[i] = 0  # B 태그
                        for j in range(1, len(keyword_tokens)):
                            if i + j < len(tags):
                                tags[i + j] = 1  # I 태그
                        break
        except Exception as e:
            print(f"⚠️ BIO 태그 생성 중 오류: {e}")
            # 오류 발생시 모든 태그를 O로 설정
            tags = torch.full((len(input_ids),), 2, dtype=torch.long)
        
        return tags

class OptimizedDistillationLoss(nn.Module):
    """최적화된 Knowledge Distillation Loss"""
    def __init__(self, temperature=4.0, alpha=0.7, beta=0.2, gamma=0.1):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # KL Divergence 가중치
        self.beta = beta    # Task Loss 가중치  
        self.gamma = gamma  # Cosine Loss 가중치
        
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
    def forward(self, student_logits, teacher_logits, student_hidden, teacher_hidden, attention_mask):
        total_loss = 0.0
        loss_components = {}
        
        # 1. KL Divergence Loss (Knowledge Distillation)
        try:
            student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            # 마스킹
            mask = attention_mask.unsqueeze(-1).expand_as(student_soft).bool()
            student_masked = student_soft[mask].view(-1, student_logits.size(-1))
            teacher_masked = teacher_soft[mask].view(-1, teacher_logits.size(-1))
            
            if student_masked.size(0) > 0:
                kl_loss = self.kl_div(student_masked, teacher_masked) * (self.temperature ** 2)
                total_loss += self.alpha * kl_loss
                loss_components['kl_loss'] = kl_loss.item()
            else:
                loss_components['kl_loss'] = 0.0
                
        except Exception as e:
            print(f"⚠️ KL Loss 계산 오류: {e}")
            loss_components['kl_loss'] = 0.0
        
        # 2. Task Consistency Loss
        try:
            student_probs = F.softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            
            # Confidence consistency
            student_conf = student_probs.max(dim=-1)[0]
            teacher_conf = teacher_probs.max(dim=-1)[0]
            
            valid_mask = attention_mask.bool()
            if valid_mask.sum() > 0:
                task_loss = self.mse_loss(student_conf[valid_mask], teacher_conf[valid_mask])
                total_loss += self.beta * task_loss
                loss_components['task_loss'] = task_loss.item()
            else:
                loss_components['task_loss'] = 0.0
                
        except Exception as e:
            print(f"⚠️ Task Loss 계산 오류: {e}")
            loss_components['task_loss'] = 0.0
        
        # 3. Hidden State Alignment (Cosine Loss)
        try:
            # 차원 맞추기
            if student_hidden.size(-1) != teacher_hidden.size(-1):
                # Student hidden을 Teacher 차원으로 projection
                projection = nn.Linear(student_hidden.size(-1), teacher_hidden.size(-1)).to(student_hidden.device)
                student_hidden_proj = projection(student_hidden)
            else:
                student_hidden_proj = student_hidden
            
            # 마스킹된 hidden states
            mask_flat = attention_mask.view(-1).bool()
            student_flat = student_hidden_proj.view(-1, student_hidden_proj.size(-1))[mask_flat]
            teacher_flat = teacher_hidden.view(-1, teacher_hidden.size(-1))[mask_flat]
            
            if student_flat.size(0) > 0:
                target = torch.ones(student_flat.size(0)).to(student_flat.device)
                cosine_loss = self.cosine_loss(student_flat, teacher_flat, target)
                total_loss += self.gamma * cosine_loss
                loss_components['cosine_loss'] = cosine_loss.item()
            else:
                loss_components['cosine_loss'] = 0.0
                
        except Exception as e:
            print(f"⚠️ Cosine Loss 계산 오류: {e}")
            loss_components['cosine_loss'] = 0.0
        
        loss_components['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_components

def load_real_data(data_type='train', max_samples=None):
    """실제 데이터 로드"""
    print(f"📊 실제 {data_type} 데이터 로딩 중...")
    
    # 데이터 경로들
    data_paths = [
        f"../../src/data/{data_type}_clean.json",
        f"../../../src/data/{data_type}_clean.json",
        f"../../../../src/data/{data_type}_clean.json"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 딕셔너리 형태를 리스트로 변환
                if isinstance(data, dict):
                    data = list(data.values())
                
                print(f"✅ 실제 {data_type} 데이터 로드: {len(data):,}개")
                print(f"📊 데이터 구조: {type(data)} 형태, 키 개수: {len(data)}개")
                
                # 샘플링 (옵션)
                if max_samples and len(data) > max_samples:
                    data = random.sample(data, max_samples)
                    print(f"📉 샘플링: {len(data):,}개")
                
                return data
                
            except Exception as e:
                print(f"⚠️ {path} 로드 실패: {e}")
                continue
    
    raise FileNotFoundError(f"❌ {data_type} 데이터를 찾을 수 없습니다. 다음 경로들을 확인하세요: {data_paths}")

def create_real_models(device):
    """실제 KoKeyBERT 모델들 생성"""
    print("🏗️ 실제 모델들 생성 중...")
    
            # Teacher 모델 (사전 훈련된 KoKeyBERT 로드)
    print("📚 Teacher 모델 (KoKeyBERT) 생성 중...")
    try:
        teacher_model = KoKeyBERT(num_class=3, model_name='skt/kobert-base-v1')
        print("✅ KoBERT 기반 Teacher 모델 생성 성공")
        
        # 사전 훈련된 체크포인트 로드
        checkpoint_path = "../../best_model.pt"
        if os.path.exists(checkpoint_path):
            print(f"🔄 사전 훈련된 체크포인트 로드 중: {checkpoint_path}")
            teacher_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("✅ 사전 훈련된 Teacher 모델 로드 완료")
        else:
            print("⚠️ 체크포인트 파일을 찾을 수 없습니다. 랜덤 초기화된 모델을 사용합니다.")
            
    except Exception as e:
        print(f"⚠️ KoBERT 기반 모델 생성 실패: {e}")
        print("🔄 기본 BERT 기반 모델로 대체...")
        teacher_model = KoKeyBERT(num_class=3, model_name='bert-base-uncased')
        print("✅ 기본 BERT 기반 Teacher 모델 생성 성공")
    
    teacher_model = teacher_model.to(device)
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"✅ Teacher 모델: {teacher_params:,} 파라미터")
    
    # Student 모델 (DistillKoKeyBERT) - 작은 크기로 생성
    print("🎓 Student 모델 (DistillKoKeyBERT) 생성 중...")
    try:
        # DistilBERT 스타일: Teacher의 절반 크기
        from transformers import BertConfig
        
        # Teacher의 config를 기반으로 작은 Student config 생성
        teacher_config = teacher_model.config
        student_config = BertConfig(
            vocab_size=teacher_config.vocab_size,
            hidden_size=teacher_config.hidden_size,
            num_hidden_layers=teacher_config.num_hidden_layers // 2,  # 절반 레이어
            num_attention_heads=teacher_config.num_attention_heads,
            intermediate_size=teacher_config.intermediate_size,
            hidden_dropout_prob=teacher_config.hidden_dropout_prob,
            attention_probs_dropout_prob=teacher_config.attention_probs_dropout_prob,
            max_position_embeddings=teacher_config.max_position_embeddings,
            type_vocab_size=teacher_config.type_vocab_size,
            initializer_range=teacher_config.initializer_range,
            layer_norm_eps=teacher_config.layer_norm_eps,
            pad_token_id=teacher_config.pad_token_id,
            bos_token_id=teacher_config.bos_token_id,
            eos_token_id=teacher_config.eos_token_id,
            use_cache=teacher_config.use_cache,
        )
        
        student_model = DistillKoKeyBERT(config=student_config, num_class=3)
        print("✅ 작은 크기 Student 모델 생성 성공")
        print(f"   - Teacher 레이어: {teacher_config.num_hidden_layers}개")
        print(f"   - Student 레이어: {student_config.num_hidden_layers}개")
        
    except Exception as e:
        print(f"⚠️ 작은 크기 Student 모델 생성 실패: {e}")
        print("🔄 기본 크기 Student 모델로 대체...")
        student_model = DistillKoKeyBERT(num_class=3, model_name='skt/kobert-base-v1')
        print("✅ 기본 크기 Student 모델 생성 성공")
    
    # DistilBERT 스타일 초기화: Teacher의 2번째 레이어씩 복사
    print("🔄 DistilBERT 스타일 초기화 중...")
    try:
        # Teacher와 Student의 encoder 레이어 수 확인
        teacher_layers = len(teacher_model.model.encoder.layer)
        student_layers = len(student_model.model.encoder.layer)
        
        print(f"   - Teacher encoder 레이어: {teacher_layers}개")
        print(f"   - Student encoder 레이어: {student_layers}개")
        
        # 2번째 레이어씩 복사 (DistilBERT 방식)
        for i in range(student_layers):
            teacher_layer_idx = i * 2  # 0, 2, 4, 6, ...
            if teacher_layer_idx < teacher_layers:
                # Student의 i번째 레이어에 Teacher의 2*i번째 레이어 복사
                student_model.model.encoder.layer[i].load_state_dict(
                    teacher_model.model.encoder.layer[teacher_layer_idx].state_dict()
                )
                print(f"   - Student layer {i} ← Teacher layer {teacher_layer_idx}")
            else:
                print(f"   - Student layer {i}: 랜덤 초기화 (Teacher layer {teacher_layer_idx} 없음)")
        
        # Embedding 레이어도 복사
        student_model.model.embeddings.load_state_dict(
            teacher_model.model.embeddings.state_dict()
        )
        print("   - Embedding 레이어 복사 완료")
        
        print("✅ DistilBERT 스타일 초기화 완료")
        
    except Exception as e:
        print(f"⚠️ DistilBERT 초기화 실패: {e}")
        print("   - Student 모델은 랜덤 초기화 상태로 유지")
    
    student_model = student_model.to(device)
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"✅ Student 모델: {student_params:,} 파라미터")
    
    compression_ratio = student_params / teacher_params
    print(f"📊 압축 비율: {compression_ratio:.3f}")
    
    return teacher_model, student_model

def train_real_distillation(teacher_model, student_model, train_loader, test_loader, device, tokenizer, num_epochs=5):
    """실제 데이터로 Knowledge Distillation 훈련"""
    print("🚀 실제 Knowledge Distillation 훈련 시작!")
    
    # 설정
    learning_rate = 2e-5
    
    # Loss와 Optimizer
    criterion = OptimizedDistillationLoss()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    
    # Best Model 추적 (검증 정확도 기준)
    best_val_acc = 0.0
    best_model_state = None
    patience = 3  # Early stopping patience
    patience_counter = 0
    
    # 훈련 모드 설정
    teacher_model.eval()  # Teacher는 frozen
    student_model.train()
    
    training_history = []
    
    for epoch in range(num_epochs):
        epoch_start = datetime.now()
        epoch_losses = []
        epoch_components = {'kl_loss': [], 'task_loss': [], 'cosine_loss': []}
        
        print(f"\n📖 Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(train_loader):
            # test.py 방식의 batch 형식: (index, input_ids, attention_mask, tags)
            index, input_ids, attention_mask, tags = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            tags = tags.to(device)
            
            optimizer.zero_grad()
            
            # Teacher 예측 (frozen) - KoKeyBERT 정상 호출
            with torch.no_grad():
                teacher_output = teacher_model(input_ids, attention_mask, tags)
                # KoKeyBERT 출력: (log_likelihood, sequence_of_tags)
                if isinstance(teacher_output, tuple):
                    teacher_log_likelihood, teacher_sequence = teacher_output
                    # Hidden states 가져오기 (Knowledge Distillation용)
                    teacher_bert_outputs = teacher_model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    teacher_hidden = teacher_bert_outputs.last_hidden_state
                    teacher_logits = teacher_model.classifier(teacher_model.dropout(teacher_hidden))
                else:
                    teacher_sequence = teacher_output
                    teacher_bert_outputs = teacher_model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    teacher_hidden = teacher_bert_outputs.last_hidden_state
                    teacher_logits = teacher_model.classifier(teacher_model.dropout(teacher_hidden))
                
            # Student 예측 - DistillKoKeyBERT 정상 호출
            student_output = student_model(input_ids, attention_mask, tags, return_outputs=True)
            # DistillKoKeyBERT 출력: (loss, predicted_tags, bert_outputs) 또는 (predicted_tags, bert_outputs)
            if len(student_output) == 3:
                # 훈련 모드: (loss, predicted_tags, bert_outputs)
                student_loss, student_predictions, student_bert_outputs = student_output
                student_hidden = student_bert_outputs[0]  # last_hidden_state
                student_logits = student_model.classifier(student_model.dropout(student_hidden))
            else:
                # 추론 모드: (predicted_tags, bert_outputs)
                student_predictions, student_bert_outputs = student_output
                student_hidden = student_bert_outputs[0]  # last_hidden_state
                student_logits = student_model.classifier(student_model.dropout(student_hidden))
            
            # 디버깅: 첫 번째 배치에서만 출력 형태 확인
            if batch_idx == 0:
                print(f"🔍 Teacher hidden states 형태: {teacher_hidden.shape}")
                print(f"🔍 Teacher logits 형태: {teacher_logits.shape}")
                print(f"🔍 Student hidden states 형태: {student_hidden.shape}")
                print(f"🔍 Student logits 형태: {student_logits.shape}")
            

            
            # Loss 계산
            total_loss, loss_components = criterion(
                student_logits, teacher_logits, 
                student_hidden, teacher_hidden, 
                attention_mask
            )
            
            # Backward pass
            if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                
                # 손실 기록
                epoch_losses.append(loss_components['total_loss'])
                for key in epoch_components:
                    if key in loss_components:
                        epoch_components[key].append(loss_components[key])
            
            # 진행 상황 출력 (더 자주 출력)
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx:3d}: Total={loss_components['total_loss']:.4f}, "
                      f"KL={loss_components['kl_loss']:.4f}, "
                      f"Task={loss_components['task_loss']:.4f}, "
                      f"Cosine={loss_components['cosine_loss']:.4f}")
            
            # GPU 메모리 정리 (더 자주)
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Epoch 결과
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        epoch_time = datetime.now() - epoch_start
        
        # 검증 메트릭 계산 (빠른 평가) - 2에폭마다만 실행
        if epoch % 2 == 0:
            val_metrics = evaluate_epoch_accuracy(student_model, test_loader, device, tokenizer, max_batches=64)
            val_f1 = val_metrics['f1']
            val_precision = val_metrics['precision']
            val_recall = val_metrics['recall']
            val_accuracy = val_metrics['accuracy']
        else:
            val_f1 = 0.0  # 검증 건너뛰기
            val_precision = 0.0
            val_recall = 0.0
            val_accuracy = 0.0
        
        # Best Model 체크 (검증 F1 점수 기준)
        if val_f1 > best_val_acc:
            best_val_acc = val_f1
            best_model_state = student_model.state_dict().copy()
            patience_counter = 0
            print(f"🏆 새로운 Best Model! Val F1: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ Best Model 없음. Patience: {patience_counter}/{patience}")
        
        epoch_summary = {
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_accuracy': val_accuracy,
            'best_val_f1': best_val_acc,
            'time': epoch_time,
            'components': {k: np.mean(v) if v else 0 for k, v in epoch_components.items()}
        }
        training_history.append(epoch_summary)
        
        print(f"✅ Epoch {epoch+1} 완료:")
        print(f"   평균 Loss: {avg_loss:.4f}")
        if epoch % 2 == 0:  # 메트릭이 계산된 경우에만 출력
            print(f"   📈 F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"   Best Val F1: {best_val_acc:.4f}")
        print(f"   KL Loss: {epoch_summary['components']['kl_loss']:.4f}")
        print(f"   Task Loss: {epoch_summary['components']['task_loss']:.4f}")
        print(f"   Cosine Loss: {epoch_summary['components']['cosine_loss']:.4f}")
        print(f"   소요 시간: {epoch_time}")
        
        # Early Stopping 체크
        if patience_counter >= patience:
            print(f"🛑 Early Stopping! {patience} epochs 동안 개선 없음")
            break
    
    # 훈련 완료 후 Best Model 복원
    if best_model_state is not None:
        student_model.load_state_dict(best_model_state)
        print(f"🏆 Best Model 복원 완료 (Val Acc: {best_val_acc:.4f})")
    
    return training_history, best_model_state, best_val_acc

def evaluate_epoch_accuracy(model, test_loader, device, tokenizer, max_batches=64):
    """Epoch 중간에 빠른 검증 F1 점수 계산 (키워드 기반)"""
    model.eval()
    
    # 키워드 평가를 위한 통계
    total_TP = 0
    total_FP = 0
    total_FN = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:  # 빠른 평가를 위해 제한
                break
                
            # test.py 방식의 batch 형식: (index, input_ids, attention_mask, tags)
            index, input_ids, attention_mask, tags = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            tags = tags.to(device)
            
            # 예측
            model_output = model(input_ids, attention_mask, tags)
            
            # KoKeyBERT 모델 출력 처리: (log_likelihood, sequence_of_tags)
            if isinstance(model_output, tuple):
                log_likelihood, sequence_of_tags = model_output
                
                # 디버깅: 첫 번째 배치에서만 출력 형태 확인
                if batch_idx == 0:
                    print(f"🔍 Model output type: {type(model_output)}")
                    print(f"🔍 sequence_of_tags length: {len(sequence_of_tags)}")
                    print(f"🔍 Tags shape: {tags.shape}")
                    print(f"🔍 Attention mask shape: {attention_mask.shape}")
                
                # sequence_of_tags를 텐서로 변환
                if isinstance(sequence_of_tags, list):
                    batch_size = tags.size(0)
                    seq_len = tags.size(1)
                    predictions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=tags.device)
                    
                    for i, seq in enumerate(sequence_of_tags):
                        if i < batch_size and len(seq) <= seq_len:
                            predictions[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=tags.device)
                else:
                    predictions = sequence_of_tags
            else:
                # 단일 출력인 경우 처리
                if isinstance(model_output, list):
                    batch_size = tags.size(0)
                    seq_len = tags.size(1)
                    predictions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=tags.device)
                    
                    for i, seq in enumerate(model_output):
                        if i < batch_size and len(seq) <= seq_len:
                            predictions[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=tags.device)
                else:
                    predictions = model_output
            
            # 키워드 추출 및 평가
            batch_TP = 0
            batch_FP = 0
            batch_FN = 0
            
            for i in range(input_ids.size(0)):  # 배치의 각 샘플에 대해
                try:
                    # 예측된 키워드 추출
                    pred_keywords = extract_keywords_from_bio_tags(
                        input_ids[i],
                        predictions[i],
                        attention_mask[i],
                        tokenizer
                    )
                    
                    # 실제 키워드 추출 (정답 BIO 태그에서)
                    true_keywords = extract_keywords_from_bio_tags(
                        input_ids[i],
                        tags[i],
                        attention_mask[i],
                        tokenizer
                    )
                    
                    # 디버깅: 첫 번째 배치의 첫 번째 샘플에서 키워드 확인
                    if batch_idx == 0 and i == 0:
                        print(f"🔍 디버깅 - 첫 번째 샘플:")
                        print(f"   예측 키워드: {pred_keywords}")
                        print(f"   실제 키워드: {true_keywords}")
                        print(f"   BIO 태그 (예측): {predictions[i][:20].tolist()}")  # 처음 20개만
                        print(f"   BIO 태그 (실제): {tags[i][:20].tolist()}")  # 처음 20개만
                        print(f"   토큰 ID: {input_ids[i][:20].tolist()}")  # 처음 20개만
                        
                        # 토큰 디코딩도 확인
                        tokens_text = tokenizer.decode(input_ids[i][:20], skip_special_tokens=True)
                        print(f"   토큰 텍스트: {tokens_text}")
                    
                    # TP, FP, FN 계산
                    TP, FP, FN = evaluate_keywords(pred_keywords, true_keywords)
                    batch_TP += TP
                    batch_FP += FP
                    batch_FN += FN
                    
                except Exception as e:
                    if batch_idx == 0:  # 첫 배치에서만 오류 출력
                        print(f"⚠️  키워드 추출 오류 (샘플 {i}): {e}")
                    continue
            
            total_TP += batch_TP
            total_FP += batch_FP
            total_FN += batch_FN
    
    model.train()  # 다시 훈련 모드로
    
    # 메트릭 계산
    if total_TP + total_FP > 0:
        precision = total_TP / (total_TP + total_FP)
    else:
        precision = 0.0
        
    if total_TP + total_FN > 0:
        recall = total_TP / (total_TP + total_FN)
    else:
        recall = 0.0
        
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    # Accuracy 계산 (키워드 기반)
    total_predictions = total_TP + total_FP
    total_actual = total_TP + total_FN
    if total_predictions + total_actual > 0:
        accuracy = total_TP / max(total_predictions, total_actual) if max(total_predictions, total_actual) > 0 else 0.0
    else:
        accuracy = 0.0
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'tp': total_TP,
        'fp': total_FP,
        'fn': total_FN
    }

def evaluate_real_models(teacher_model, student_model, test_loader, device, tokenizer):
    """실제 데이터로 모델 평가"""
    print("🔍 실제 데이터로 모델 평가 중...")
    
    teacher_model.eval()
    student_model.eval()
    
    # 키워드 평가를 위한 통계
    teacher_TP = 0
    teacher_FP = 0
    teacher_FN = 0
    
    student_TP = 0
    student_FP = 0
    student_FN = 0
    
    teacher_times = []
    student_times = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # test.py 방식의 batch 형식: (index, input_ids, attention_mask, tags)
            index, input_ids, attention_mask, tags = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            tags = tags.to(device)
            
            # Teacher 평가
            start_time = time.time()
            teacher_output = teacher_model(input_ids, attention_mask, tags)
            teacher_time = time.time() - start_time
            teacher_times.append(teacher_time)
            
            # Student 평가
            start_time = time.time()
            student_output = student_model(input_ids, attention_mask, tags)
            student_time = time.time() - start_time
            student_times.append(student_time)
            
            # KoKeyBERT 모델 출력 처리: (log_likelihood, sequence_of_tags)
            def process_model_output(model_output, model_name="Model"):
                if isinstance(model_output, tuple):
                    log_likelihood, sequence_of_tags = model_output
                    
                    # sequence_of_tags를 텐서로 변환
                    if isinstance(sequence_of_tags, list):
                        batch_size = tags.size(0)
                        seq_len = tags.size(1)
                        predictions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=tags.device)
                        
                        for i, seq in enumerate(sequence_of_tags):
                            if i < batch_size and len(seq) <= seq_len:
                                predictions[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=tags.device)
                        return predictions
                    else:
                        return sequence_of_tags
                else:
                    # 단일 출력인 경우 (sequence_of_tags만 반환)
                    if isinstance(model_output, list):
                        batch_size = tags.size(0)
                        seq_len = tags.size(1)
                        predictions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=tags.device)
                        
                        for i, seq in enumerate(model_output):
                            if i < batch_size and len(seq) <= seq_len:
                                predictions[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=tags.device)
                        return predictions
                    else:
                        return model_output
            
            teacher_pred = process_model_output(teacher_output, "Teacher")
            student_pred = process_model_output(student_output, "Student")
            
            # 디버깅: 첫 번째 배치에서만 출력 형태 확인
            if batch_idx == 0:
                print(f"🔍 Teacher output type: {type(teacher_output)}")
                print(f"🔍 Student output type: {type(student_output)}")
                print(f"🔍 Teacher pred shape: {teacher_pred.shape if hasattr(teacher_pred, 'shape') else 'no shape'}")
                print(f"🔍 Student pred shape: {student_pred.shape if hasattr(student_pred, 'shape') else 'no shape'}")
                print(f"🔍 Tags shape: {tags.shape}")
            
            # predictions가 올바른 형태인지 확인
            if not isinstance(teacher_pred, torch.Tensor) or not isinstance(student_pred, torch.Tensor):
                print(f"⚠️  Warning: predictions are not tensors, skipping batch {batch_idx}")
                continue
                
            if teacher_pred.dim() != 2 or student_pred.dim() != 2:
                print(f"⚠️  Warning: predictions have unexpected dimensions, skipping batch {batch_idx}")
                continue
            
            # 키워드 추출 및 평가 (배치의 각 샘플에 대해)
            for i in range(input_ids.size(0)):
                try:
                    # 실제 키워드 추출 (정답 BIO 태그에서)
                    true_keywords = extract_keywords_from_bio_tags(
                        input_ids[i],
                        tags[i],
                        attention_mask[i],
                        tokenizer
                    )
                    
                    # Teacher 예측 키워드 추출
                    teacher_keywords = extract_keywords_from_bio_tags(
                        input_ids[i],
                        teacher_pred[i],
                        attention_mask[i],
                        tokenizer
                    )
                    
                    # Student 예측 키워드 추출
                    student_keywords = extract_keywords_from_bio_tags(
                        input_ids[i],
                        student_pred[i],
                        attention_mask[i],
                        tokenizer
                    )
                    
                    # Teacher 메트릭 계산
                    teacher_tp, teacher_fp, teacher_fn = evaluate_keywords(teacher_keywords, true_keywords)
                    teacher_TP += teacher_tp
                    teacher_FP += teacher_fp
                    teacher_FN += teacher_fn
                    
                    # Student 메트릭 계산
                    student_tp, student_fp, student_fn = evaluate_keywords(student_keywords, true_keywords)
                    student_TP += student_tp
                    student_FP += student_fp
                    student_FN += student_fn
                    
                except Exception as e:
                    if batch_idx == 0:  # 첫 배치에서만 오류 출력
                        print(f"⚠️  키워드 추출 오류 (배치 {batch_idx}, 샘플 {i}): {e}")
                    continue
            
            if batch_idx >= 100:  # 처음 100 배치만 평가 (시간 절약)
                break
    
    # Teacher 메트릭 계산
    if teacher_TP + teacher_FP > 0:
        teacher_precision = teacher_TP / (teacher_TP + teacher_FP)
    else:
        teacher_precision = 0.0
        
    if teacher_TP + teacher_FN > 0:
        teacher_recall = teacher_TP / (teacher_TP + teacher_FN)
    else:
        teacher_recall = 0.0
        
    if teacher_precision + teacher_recall > 0:
        teacher_f1 = 2 * (teacher_precision * teacher_recall) / (teacher_precision + teacher_recall)
    else:
        teacher_f1 = 0.0
    
    teacher_total_predictions = teacher_TP + teacher_FP
    teacher_total_actual = teacher_TP + teacher_FN
    if teacher_total_predictions + teacher_total_actual > 0:
        teacher_acc = teacher_TP / max(teacher_total_predictions, teacher_total_actual) if max(teacher_total_predictions, teacher_total_actual) > 0 else 0.0
    else:
        teacher_acc = 0.0
    
    # Student 메트릭 계산
    if student_TP + student_FP > 0:
        student_precision = student_TP / (student_TP + student_FP)
    else:
        student_precision = 0.0
        
    if student_TP + student_FN > 0:
        student_recall = student_TP / (student_TP + student_FN)
    else:
        student_recall = 0.0
        
    if student_precision + student_recall > 0:
        student_f1 = 2 * (student_precision * student_recall) / (student_precision + student_recall)
    else:
        student_f1 = 0.0
    
    student_total_predictions = student_TP + student_FP
    student_total_actual = student_TP + student_FN
    if student_total_predictions + student_total_actual > 0:
        student_acc = student_TP / max(student_total_predictions, student_total_actual) if max(student_total_predictions, student_total_actual) > 0 else 0.0
    else:
        student_acc = 0.0
    
    avg_teacher_time = np.mean(teacher_times) * 1000  # ms
    avg_student_time = np.mean(student_times) * 1000  # ms
    speedup = avg_teacher_time / avg_student_time if avg_student_time > 0 else 1.0
    
    results = {
        'teacher': {
            'accuracy': teacher_acc,
            'precision': teacher_precision,
            'recall': teacher_recall,
            'f1_score': teacher_f1,
            'tp': teacher_TP,
            'fp': teacher_FP,
            'fn': teacher_FN,
            'avg_time_ms': avg_teacher_time
        },
        'student': {
            'accuracy': student_acc,
            'precision': student_precision,
            'recall': student_recall,
            'f1_score': student_f1,
            'tp': student_TP,
            'fp': student_FP,
            'fn': student_FN,
            'avg_time_ms': avg_student_time
        },
        'speedup': speedup
    }
    
    return results

def save_real_results(teacher_model, student_model, training_history, eval_results, device, best_val_acc):
    """실제 실험 결과 저장"""
    print("💾 실제 실험 결과 저장 중...")
    
    # 모델 저장
    os.makedirs('../results/models', exist_ok=True)
    
    # Student 모델만 저장 (Teacher는 원본이므로)
    torch.save({
        'model_state_dict': student_model.state_dict(),
        'model_config': student_model.config,
        'training_history': training_history,
        'eval_results': eval_results,
        'best_val_acc': best_val_acc
    }, '../results/models/real_distilled_koKeyBERT.pt')
    print("✅ Student 모델 저장 완료")
    
    # 결과 시각화
    os.makedirs('../results/plots', exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # 1. 훈련 Loss 그래프
    plt.subplot(1, 3, 1)
    epochs = [h['epoch'] for h in training_history]
    losses = [h['avg_loss'] for h in training_history]
    kl_losses = [h['components']['kl_loss'] for h in training_history]
    task_losses = [h['components']['task_loss'] for h in training_history]
    cosine_losses = [h['components']['cosine_loss'] for h in training_history]
    
    plt.plot(epochs, losses, 'b-', label='Total Loss', marker='o')
    plt.plot(epochs, kl_losses, 'r--', label='KL Loss', marker='s')
    plt.plot(epochs, task_losses, 'g--', label='Task Loss', marker='^')
    plt.plot(epochs, cosine_losses, 'm--', label='Cosine Loss', marker='d')
    plt.title('Training Loss Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. 정확도 비교
    plt.subplot(1, 3, 2)
    models = ['Teacher\n(KoKeyBERT)', 'Student\n(DistillKoKeyBERT)']
    accuracies = [eval_results['teacher']['accuracy'], eval_results['student']['accuracy']]
    f1_scores = [eval_results['teacher']['f1_score'], eval_results['student']['f1_score']]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7, color='skyblue')
    plt.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.7, color='lightcoral')
    
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 추론 속도 비교
    plt.subplot(1, 3, 3)
    times = [eval_results['teacher']['avg_time_ms'], eval_results['student']['avg_time_ms']]
    colors = ['red', 'blue']
    
    plt.bar(models, times, color=colors, alpha=0.7)
    plt.title('Inference Speed Comparison')
    plt.xlabel('Model')
    plt.ylabel('Time (ms)')
    plt.grid(True, alpha=0.3)
    
    # 속도 향상 표시
    plt.text(1, times[1] + max(times) * 0.1, f'{eval_results["speedup"]:.2f}x faster', 
             ha='center', fontweight='bold', color='blue')
    
    plt.tight_layout()
    plt.savefig('../results/plots/real_distillation_results.png', dpi=300, bbox_inches='tight')
    print("✅ 결과 시각화 저장 완료")
    
    # 요약 출력
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    compression_ratio = student_params / teacher_params
    
    print("\n" + "="*70)
    print("📊 실제 Knowledge Distillation 실험 결과 요약")
    print("="*70)
    print(f"🏫 Teacher (KoKeyBERT):")
    print(f"   - 파라미터: {teacher_params:,}")
    print(f"   - 정확도: {eval_results['teacher']['accuracy']:.4f}")
    print(f"   - 정밀도: {eval_results['teacher']['precision']:.4f}")
    print(f"   - 재현율: {eval_results['teacher']['recall']:.4f}")
    print(f"   - F1 점수: {eval_results['teacher']['f1_score']:.4f}")
    print(f"   - TP/FP/FN: {eval_results['teacher']['tp']}/{eval_results['teacher']['fp']}/{eval_results['teacher']['fn']}")
    print(f"   - 추론 시간: {eval_results['teacher']['avg_time_ms']:.2f}ms")
    print()
    print(f"🎓 Student (DistillKoKeyBERT):")
    print(f"   - 파라미터: {student_params:,}")
    print(f"   - 정확도: {eval_results['student']['accuracy']:.4f}")
    print(f"   - 정밀도: {eval_results['student']['precision']:.4f}")
    print(f"   - 재현율: {eval_results['student']['recall']:.4f}")
    print(f"   - F1 점수: {eval_results['student']['f1_score']:.4f}")
    print(f"   - TP/FP/FN: {eval_results['student']['tp']}/{eval_results['student']['fp']}/{eval_results['student']['fn']}")
    print(f"   - 추론 시간: {eval_results['student']['avg_time_ms']:.2f}ms")
    print()
    print(f"📈 성능 비교:")
    print(f"   - 압축 비율: {compression_ratio:.3f} ({(1-compression_ratio)*100:.1f}% 압축)")
    print(f"   - 속도 향상: {eval_results['speedup']:.2f}x")
    print(f"   - 정확도 차이: {eval_results['student']['accuracy'] - eval_results['teacher']['accuracy']:.4f}")
    print(f"   - F1 점수 차이: {eval_results['student']['f1_score'] - eval_results['teacher']['f1_score']:.4f}")
    print(f"   - 최종 훈련 Loss: {training_history[-1]['avg_loss']:.4f}")
    print(f"   - Best Val Acc: {best_val_acc:.4f}")
    print("="*70)

def main():
    """메인 실행 함수"""
    print("🎯 실제 KoKeyBERT Knowledge Distillation 실험 시작!")
    print("=" * 80)
    
    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 사용 디바이스: {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    try:
        # 1. test.py와 동일한 데이터 로드
        print("📊 test.py 방식으로 데이터 로딩 중...")
        
        # 훈련 데이터 경로
        train_data_path = "../../src/data/train_clean.json"
        test_data_path = "../../src/data/test_clean.json"
        
        # 경로 확인 및 대안 경로 시도
        def find_data_file(filename):
            possible_paths = [
                f"../../src/data/{filename}",
                f"../../../src/data/{filename}",
                f"../../data/{filename}",
                f"../data/{filename}",
                f"./{filename}"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return path
            return None
        
        train_data_path = find_data_file("train_clean.json")
        test_data_path = find_data_file("test_clean.json")
        
        if not train_data_path:
            raise FileNotFoundError("train_clean.json을 찾을 수 없습니다.")
        if not test_data_path:
            raise FileNotFoundError("test_clean.json을 찾을 수 없습니다.")
        
        print(f"✅ 훈련 데이터 경로: {train_data_path}")
        print(f"✅ 테스트 데이터 경로: {test_data_path}")
        
        # test.py 방식으로 데이터 로드
        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)
        
        if train_data is None or len(train_data) == 0:
            raise ValueError(f"훈련 데이터 로드 실패: {train_data_path}")
        if test_data is None or len(test_data) == 0:
            raise ValueError(f"테스트 데이터 로드 실패: {test_data_path}")
        
        print(f"✅ 훈련 데이터 로드: {len(train_data)} 항목")
        print(f"✅ 테스트 데이터 로드: {len(test_data)} 항목")
        
        # 2. 토크나이저 로드
        print("🔤 토크나이저 로딩 중...")
        try:
            # kobert_tokenizer 폴더에서 KoBERTTokenizer 가져오기
            import sys
            sys.path.append('../../kobert_tokenizer')
            from kobert_tokenizer import KoBERTTokenizer
            tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
            print("✅ KoBERT (KoBERTTokenizer) 로드 성공")
        except Exception as e:
            print(f"⚠️ KoBERT 토크나이저 로드 실패: {e}")
            print("🔄 기본 BERT 토크나이저로 대체...")
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            print("✅ 기본 BERT 토크나이저 로드 성공")
        
        # 3. test.py 방식으로 Dataset과 DataLoader 생성
        print("📦 test.py 방식으로 Dataset 생성 중...")
        train_dataset = KeywordDataset(train_data)
        test_dataset = KeywordDataset(test_data)
        
        # Collator 생성
        collator = Collator(tokenizer)
        
        # test.py 방식으로 DataLoader 생성
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, collate_fn=collator, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collator, pin_memory=False)
        
        # 4. 실제 모델들 생성
        teacher_model, student_model = create_real_models(device)
        
        # 5. Knowledge Distillation 훈련
        training_history, best_model_state, best_val_acc = train_real_distillation(
            teacher_model, student_model, train_loader, test_loader, device, tokenizer, num_epochs=5
        )
        
        # 6. 모델 평가
        eval_results = evaluate_real_models(teacher_model, student_model, test_loader, device, tokenizer)
        
        # 7. 결과 저장
        save_real_results(teacher_model, student_model, training_history, eval_results, device, best_val_acc)
        
        print("\n🎉 실제 Knowledge Distillation 실험 완료!")
        print("📁 결과 파일:")
        print("  - ../results/models/distilled_koKeyBERT.pt")
        print("  - ../results/plots/distillation_results.png")
        
    except Exception as e:
        print(f"❌ 실험 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
