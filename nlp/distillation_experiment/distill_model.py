"""
CRF 없이 선형 변환과 규칙 기반 후처리를 사용하는 간단한 Student 모델
"""
import torch
from torch import nn
from transformers import BertModel, BertConfig
from typing import Optional
import torch.nn.functional as F


class DistillKoKeyBERT(nn.Module):
    """CRF 없는 Distilled Korean KeyBERT Model (Student용)"""
    
    def __init__(self, 
                 config: BertConfig = None, 
                 num_class: int = 3, 
                 model_name: str = 'skt/kobert-base-v1') -> None:
        
        super().__init__()
        
        if config is None:
            self.model = BertModel.from_pretrained(model_name)
            self.config = self.model.config
        else:
            self.config = config
            # 'model_name'을 기준으로 모델을 불러오지만, 'config'에 명시된 구조(예: 6개 레이어)로 생성합니다.
            self.model = BertModel.from_pretrained(model_name, config=self.config)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_class)
        
        # BIO 태그: 0=B, 1=I, 2=O
        self.num_class = num_class
        
    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                tags: Optional[torch.LongTensor] = None,
                return_outputs: Optional[bool] = False):
        """
        Args:
            input_ids: (B, L)
            attention_mask: (B, L)
            tags: (B, L) - 정답 태그 (훈련시에만)
            return_outputs: BERT 출력도 함께 반환할지 여부
        """
        
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).float()

        # BERT 인코딩
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # (B, L, H)
        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)
        
        # (B, L, num_class)
        logits = self.classifier(last_hidden_state)
        
        if tags is not None:
            # 훈련 모드: loss 계산
            loss = self.compute_loss(logits, tags, attention_mask)
            predicted_tags = self.predict_tags(logits, attention_mask)
            
            if return_outputs:
                return loss, predicted_tags, outputs
            return loss, predicted_tags
        else:
            # 추론 모드: 예측만
            predicted_tags = self.predict_tags(logits, attention_mask)
            
            if return_outputs:
                return predicted_tags, outputs
            return predicted_tags
    
    def compute_loss(self, logits, tags, attention_mask):
        """Cross-entropy loss 계산"""
        # Flatten for loss calculation
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_class)[active_loss]
        active_labels = tags.view(-1)[active_loss]
        
        loss = F.cross_entropy(active_logits, active_labels)
        return loss
    
    def predict_tags(self, logits, attention_mask):
        """규칙 기반 BIO 태그 예측 및 후처리"""
        batch_size, seq_len = logits.shape[:2]
        
        # Softmax로 확률 계산
        probs = F.softmax(logits, dim=-1)  # (B, L, 3)
        
        # 가장 높은 확률의 태그 선택
        raw_predictions = torch.argmax(probs, dim=-1)  # (B, L)
        
        # 규칙 기반 후처리
        refined_predictions = []
        
        for batch_idx in range(batch_size):
            seq_preds = raw_predictions[batch_idx]
            mask = attention_mask[batch_idx].bool()
            
            # 유효한 토큰만 처리
            valid_preds = seq_preds[mask].tolist()
            refined_seq = self.apply_bio_rules(valid_preds)
            
            # 패딩 부분은 원래 예측값 유지
            full_seq = seq_preds.tolist()
            valid_len = len(refined_seq)
            full_seq[:valid_len] = refined_seq
            
            refined_predictions.append(full_seq)
        
        return refined_predictions
    
    def apply_bio_rules(self, predictions):
        """BIO 태그 규칙 적용"""
        if not predictions:
            return predictions
            
        refined = []
        prev_tag = 2  # O 태그로 시작
        
        for i, pred in enumerate(predictions):
            current_tag = pred
            
            # 규칙 1: I 태그는 B 태그 다음에만 올 수 있음
            if current_tag == 1:  # I 태그
                if prev_tag == 0 or prev_tag == 1:  # 이전이 B 또는 I
                    refined.append(1)  # I 유지
                else:  # 이전이 O
                    refined.append(0)  # I를 B로 변경
                    current_tag = 0
            else:
                refined.append(current_tag)
            
            prev_tag = current_tag
        
        return refined
    
    def extract_keywords(self, text, tokenizer):
        """텍스트에서 키워드 추출"""
        # 토크나이징
        inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                          padding=True, max_length=512)
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # 예측
        with torch.no_grad():
            predicted_tags = self.forward(input_ids, attention_mask)
        
        # predicted_tags는 리스트 형태로 반환되므로 첫 번째 배치의 결과 사용
        if isinstance(predicted_tags, list):
            predictions = predicted_tags[0]
        else:
            predictions = predicted_tags
        
        # 키워드 추출
        keywords = self.extract_keywords_from_predictions(
            input_ids[0], predictions, attention_mask[0], tokenizer
        )
        
        return keywords
    
    def extract_keywords_from_predictions(self, input_ids, predictions, attention_mask, tokenizer):
        """예측 결과에서 키워드 추출"""
        keywords = []
        current_keyword_tokens = []
        
        # 유효한 토큰만 처리 (패딩 제외)
        valid_length = attention_mask.sum().item()
        
        for i in range(1, valid_length - 1):  # [CLS], [SEP] 제외
            token_id = input_ids[i].item()
            tag = predictions[i]
            
            if tag == 0:  # B 태그
                # 이전 키워드가 있으면 저장
                if current_keyword_tokens:
                    keyword = tokenizer.decode(current_keyword_tokens).strip()
                    if keyword and keyword not in keywords:
                        keywords.append(keyword)
                
                # 새 키워드 시작
                current_keyword_tokens = [token_id]
                
            elif tag == 1:  # I 태그
                # 현재 키워드에 추가
                if current_keyword_tokens:  # B 태그가 먼저 있어야 함
                    current_keyword_tokens.append(token_id)
                    
            else:  # O 태그
                # 현재 키워드 종료
                if current_keyword_tokens:
                    keyword = tokenizer.decode(current_keyword_tokens).strip()
                    if keyword and keyword not in keywords:
                        keywords.append(keyword)
                    current_keyword_tokens = []
        
        # 마지막 키워드 처리
        if current_keyword_tokens:
            keyword = tokenizer.decode(current_keyword_tokens).strip()
            if keyword and keyword not in keywords:
                keywords.append(keyword)
        
        return keywords
    
    @classmethod
    def from_original_model(cls, original_model):
        """
        원본 KoKeyBERT 모델을 DistillKoKeyBERT로 변환합니다.
        
        Args:
            original_model: 원본 KoKeyBERT 모델
        
        Returns:
            DistillKoKeyBERT: 변환된 모델
        """
        # 원본 모델의 설정을 가져와서 Student 모델 설정 생성
        teacher_config = original_model.config
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
        
        # Student 모델 생성
        student_model = cls(config=student_config, num_class=3)
        
        # 원본 모델의 가중치를 Student 모델에 복사 (가능한 부분만)
        student_state_dict = student_model.state_dict()
        original_state_dict = original_model.state_dict()
        
        # 공통 키만 복사
        for key in student_state_dict.keys():
            if key in original_state_dict:
                if 'encoder.layer' in key:
                    # 레이어 인덱스 조정 (0-5만 사용)
                    layer_idx = int(key.split('.')[2])
                    if layer_idx < 6:  # Student 모델의 레이어 수만큼만
                        student_state_dict[key] = original_state_dict[key]
                else:
                    # encoder.layer가 아닌 키는 그대로 복사
                    student_state_dict[key] = original_state_dict[key]
        
        student_model.load_state_dict(student_state_dict)
        return student_model


if __name__ == '__main__':
    from transformers import BertConfig
    from kobert_tokenizer import KoBERTTokenizer
    
    # 모델 테스트
    config = BertConfig.from_pretrained('skt/kobert-base-v1')
    config.num_hidden_layers = 6  # 더 작은 모델
    config.hidden_size = 384
    config.intermediate_size = 1536
    config.num_attention_heads = 6
    
    model = DistillKoKeyBERT(config=config)
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    
    # 테스트 입력
    text = "인공지능과 머신러닝 기술이 발전하고 있습니다."
    keywords = model.extract_keywords(text, tokenizer)
    print(f"추출된 키워드: {keywords}")
