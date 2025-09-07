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
            self.model = BertModel.from_pretrained(model_name, config=self.config, attn_implementation="eager")
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
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        
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
        """텍스트에서 키워드와 위치 인덱스를 추출합니다."""
        inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                          padding=True, max_length=512)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            predicted_tags, outputs = self.forward(input_ids, attention_mask, return_outputs=True)
        if isinstance(predicted_tags, list):
            predictions = predicted_tags[0]
        else:
            predictions = predicted_tags

        # 원본 텍스트(text)를 전달하여 키워드 위치를 찾도록 함
        keywords_with_indices = self.extract_keywords_from_predictions(
            input_ids[0], predictions, attention_mask[0], tokenizer, text
        )
        
        # 이하 로직은 이전 제안과 동일
        if not keywords_with_indices:
            return [], None
        final_keywords = []
        keyword_strings = {kw[0] for kw in keywords_with_indices}
        for keyword, start, end in keywords_with_indices:
            if not any(keyword in other for other in keyword_strings if keyword != other):
                final_keywords.append({'keyword': keyword, 'start': start, 'end': end})
        
        return final_keywords, outputs

    def extract_keywords_from_predictions(self, input_ids, predictions, attention_mask, tokenizer, text):
        """
        예측 결과에서 키워드를 추출하고, 원본 text.find()로 위치를 찾습니다.
        """
        keywords = []
        current_keyword_tokens = []
        last_keyword_end_pos = 0 # 중복 키워드 검색을 위한 포인터
        
        valid_length = attention_mask.sum().item()
        
        for i in range(1, valid_length): # [CLS]부터 [SEP]까지
            token_id = input_ids[i].item()
            tag = predictions[i]
            
            # B 태그에서 새 키워드 시작
            if tag == 0:
                if current_keyword_tokens: # 이전 키워드 저장
                    keyword = tokenizer.decode(current_keyword_tokens).strip()
                    if keyword:
                        try:
                            start = text.index(keyword, last_keyword_end_pos)
                            end = start + len(keyword)
                            keywords.append((keyword, start, end))
                            last_keyword_end_pos = end
                        except ValueError:
                            pass # 텍스트에서 못찾으면 무시
                current_keyword_tokens = [token_id]
            # I 태그는 키워드에 추가
            elif tag == 1 and current_keyword_tokens:
                current_keyword_tokens.append(token_id)
            # O 태그이거나 I로 시작하면 키워드 종료
            else:
                if current_keyword_tokens:
                    keyword = tokenizer.decode(current_keyword_tokens).strip()
                    if keyword:
                        try:
                            start = text.index(keyword, last_keyword_end_pos)
                            end = start + len(keyword)
                            keywords.append((keyword, start, end))
                            last_keyword_end_pos = end
                        except ValueError:
                            pass
                    current_keyword_tokens = []
        
        # 루프 종료 후 마지막 키워드 처리
        if current_keyword_tokens:
            keyword = tokenizer.decode(current_keyword_tokens).strip()
            if keyword:
                try:
                    start = text.index(keyword, last_keyword_end_pos)
                    end = start + len(keyword)
                    keywords.append((keyword, start, end))
                except ValueError:
                    pass

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
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from transformers import BertConfig
    from kobert_tokenizer import KoBERTTokenizer 
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
    from back_end.app.text_analysis import analyze_keyword_attention
    import pprint

    checkpoint = torch.load('distill_KoKeyBERT.pt', map_location='cpu', weights_only=False)
    config = checkpoint['model_config']
    model = DistillKoKeyBERT(config=config)
    model.load_state_dict(checkpoint['model_state_dict'])

    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

    # 테스트 입력
    text = "글쎄요, 어제 이것을 기여금으로 하든 또는 기금으로 하든 기금으로 해도 역시 부담금을 통해 재원을 충당하는 것이기 때문에…… 기여금이라고 하든, 기금이라고 하든 기금이라고 할 경우에도 역시 부담금을 통한 재원 마련이기 때문에 이것의 위헌 여부에 대해 우리 최재천 위원님께서 지적이 있으셨습니다. 거기에 대해서 정부 측에서 먼저 의견을 좀 말씀해 주시기 바랍니다."
    keywords_info, outputs = model.extract_keywords(text, tokenizer)
    pprint.pprint(keywords_info)

    if keywords_info and outputs and outputs.attentions:
        attention_analysis_result = analyze_keyword_attention(
            text=text,
            keywords_info=keywords_info,
            attentions=outputs.attentions,
            tokenizer=tokenizer
        )
        print("\n--- attention_analysis_result 출력 ---")
        pprint.pprint(attention_analysis_result)
        

        # --- 위치 인덱스 검증 코드 (수정됨) ---
        print("\n--- 위치 인덱스 검증 시작 ---")
        verification_passed = True
        # 키워드별로 결과를 순회
        for keyword, results_dict in attention_analysis_result.items():
            for pos_tag, results_dict_inside in results_dict.items():
                for item in results_dict_inside.values():
                    word = item['keyword']
                    start, end = item['start'], item['end']
                    sliced_text = text[start:end]
                    if word != sliced_text:
                            print(f"[검증 실패] 키워드: '{keyword}', 단어: '{word}', 위치: ({start},{end}), 추출된 텍스트: '{sliced_text}'")
                            verification_passed = False
        
        if verification_passed:
            print("✅ 모든 위치 인덱스가 원본 텍스트와 일치합니다.")
        else:
            print("❗️ 일부 위치 인덱스가 원본 텍스트와 일치하지 않습니다.")