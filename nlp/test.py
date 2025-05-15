from data import KeywordDataset, Collator, load_data
from model import KoKeyBERT
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertConfig
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np

import argparse
import logging
from torch.utils.data import DataLoader, SequentialSampler
import torch
import os
from torch.nn.utils.rnn import pad_sequence
import json
from datetime import datetime
# 전역 로거 설정
logger = logging.getLogger(__name__)

def extract_keywords_from_bio_tags(tokens, bio_tags, attention_mask, tokenizer, device):
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
        logger.warning("토큰, 태그, 마스크의 길이가 일치하지 않습니다.")
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
                    logger.warning(f"토큰 디코딩 오류: {e}")
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
                    logger.warning(f"토큰 디코딩 오류: {e}")
            current_keyword = []
            previous_tag = tag_val
    
    # 마지막 키워드 처리
    if current_keyword:
        try:
            keyword = tokenizer.decode(current_keyword).strip()
            if keyword and keyword not in keywords:
                keywords.append(keyword)
        except Exception as e:
            logger.warning(f"토큰 디코딩 오류: {e}")
    
    return keywords

def evaluate_keywords(pred_keywords, true_keywords):
    """
    예측된 키워드와 실제 키워드를 비교하여 confusion matrix를 계산합니다.
    
    Args:
        pred_keywords: 예측된 키워드 리스트
        true_keywords: 실제 키워드 리스트
    
    Returns:
        tuple: (TP, FP, FN) 값
    """
    # 키워드 정규화: 공백 처리
    pred_set = {keyword.strip() for keyword in pred_keywords if keyword}
    true_set = {keyword.strip() for keyword in true_keywords if keyword}
    
    TP = len(pred_set.intersection(true_set))
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set)
    
    return TP, FP, FN

def test(model: KoKeyBERT,
         test_dataset: KeywordDataset,
         collator: Collator = None,
         args: argparse.Namespace = None,
         logger: logging.Logger = None,
         device: torch.device = None,
         ):

    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info("KoKeyBERT 테스트 시작")
    logger.info("배치 크기: %d", args.batch_size if hasattr(args, 'batch_size') else 1)
    start_time = datetime.now()
    logger.info("start time: %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=args.batch_size, 
                                collate_fn=collator, 
                                sampler=sampler,
                                num_workers=args.num_workers,
                                drop_last=False)
    
    model.eval()
    total_test_loss = 0.0
    total_test_acc = 0.0
    test_step = 0
    
    # 키워드 평가를 위한 통계
    total_TP = 0
    total_FP = 0
    total_FN = 0
    
    # 모든 예측 및 실제 키워드 저장 (결과 분석용)
    all_pred_keywords = []
    all_true_keywords = []
    with torch.no_grad():
        for batch in test_dataloader:
            test_step += 1
            
            # 배치 데이터 가져오기
            try:
                index, text_ids, text_attention_mask, bio_tags = batch
            except ValueError as e:
                logger.error(f"배치 데이터 파싱 오류: {e}")
                continue

            # 모델 추론
            try:
                log_likelihood, sequence_of_tags = model(input_ids=text_ids.to(device), 
                                                        attention_mask=text_attention_mask.to(device), 
                                                        tags=bio_tags.to(device), 
                                                        return_outputs=False)
                log_likelihood = -log_likelihood.mean()
                total_test_loss += log_likelihood.item()
            except Exception as e:
                logger.error(f"모델 추론 오류: {e}")
                continue
            
            # BIO 태그 시퀀스 처리
            tag_seqs = [torch.tensor(s, dtype=torch.long, device=device) for s in sequence_of_tags]
            padded = pad_sequence(tag_seqs, batch_first=True, padding_value=model.config.pad_token_id).to(device)
            
            # 정확도 계산
            mb_acc = (padded == bio_tags.to(device)).float()[text_attention_mask.to(device).bool()].mean()
            total_test_acc += mb_acc.item()
            
            # 키워드 추출 및 평가
            batch_TP = 0
            batch_FP = 0
            batch_FN = 0
            
            for i in range(len(text_ids)):
                try:
                    # 예측된 키워드 추출
                    pred_keywords = extract_keywords_from_bio_tags(
                        text_ids[i],
                        padded[i],
                        text_attention_mask[i],
                        collator.tokenizer,
                        device
                    )
                    
                    # 실제 키워드 추출
                    try:
                        # index 구조에 따라 적절히 접근
                        idx = index[i][0] if isinstance(index[i], (list, tuple)) else index[i]
                        if isinstance(test_dataset.data[idx], dict) and "keyword" in test_dataset.data[idx]:
                            true_keywords = test_dataset.data[idx]["keyword"]
                            # 리스트가 아닌 경우 리스트로 변환
                            if not isinstance(true_keywords, list):
                                true_keywords = [true_keywords]
                        else:
                            logger.warning(f"데이터셋에 'keyword' 키가 없습니다: {idx}")
                            true_keywords = []
                    except (IndexError, KeyError) as e:
                        logger.warning(f"데이터셋 인덱싱 오류: {e}")
                        true_keywords = []
                    
                    # 모든 예측 및 실제 키워드 저장
                    all_pred_keywords.append(pred_keywords)
                    all_true_keywords.append(true_keywords)
                    
                    # 키워드 평가
                    TP, FP, FN = evaluate_keywords(pred_keywords, true_keywords)
                    batch_TP += TP
                    batch_FP += FP
                    batch_FN += FN
                except Exception as e:
                    logger.warning(f"키워드 추출 및 평가 오류: {e}")
                    continue
            
            total_TP += batch_TP
            total_FP += batch_FP
            total_FN += batch_FN
            
            # 현재 배치의 성능 지표 계산
            precision = batch_TP / (batch_TP + batch_FP) if (batch_TP + batch_FP) > 0 else 0
            recall = batch_TP / (batch_TP + batch_FN) if (batch_TP + batch_FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 로깅 (자세한 배치별 로깅은 디버그 레벨로)
            if hasattr(args, 'log_freq') and (test_step % args.log_freq == 0 or test_step == 1):
                logger.info("Step: %d/%d, Loss: %.4f, Acc: %.4f, P: %.4f, R: %.4f, F1: %.4f", 
                           test_step, len(test_dataloader), log_likelihood.item(), mb_acc.item(), 
                           precision, recall, f1)
            else:
                logger.debug("Step: %d/%d, Loss: %.4f, Acc: %.4f, P: %.4f, R: %.4f, F1: %.4f", 
                           test_step, len(test_dataloader), log_likelihood.item(), mb_acc.item(), 
                           precision, recall, f1)
    
    # 테스트 데이터가 없는 경우 처리
    if test_step == 0:
        logger.warning("테스트 데이터가 없거나 모든 배치에서 오류가 발생했습니다.")
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # 평균 손실 및 정확도 계산
    total_test_loss /= test_step
    total_test_acc /= test_step
    
    # 전체 성능 지표 계산
    total_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    total_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
    
    logger.info("테스트 완료 - 총 %d 배치", test_step)
    logger.info("최종 손실: %.4f, 정확도: %.4f", total_test_loss, total_test_acc)
    logger.info("키워드 성능 - Precision: %.4f, Recall: %.4f, F1: %.4f", total_precision, total_recall, total_f1)
    logger.info("Confusion Matrix - TP: %d, FP: %d, FN: %d", total_TP, total_FP, total_FN)
    logger.info("end time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("total time: %s", datetime.now() - start_time)

    # 상세 분석을 위한 결과 저장 (옵션)
    if hasattr(args, 'save_results') and args.save_results:
        try:
            # 결과를 파일로 저장
            results = {
                'metrics': {
                    'loss': total_test_loss,
                    'accuracy': total_test_acc,
                    'precision': total_precision,
                    'recall': total_recall,
                    'f1': total_f1,
                    'TP': total_TP,
                    'FP': total_FP,
                    'FN': total_FN
                },
                'predictions': []
            }
            
            # 각 샘플의 예측 결과 저장
            for i in range(len(all_pred_keywords)):
                results['predictions'].append({
                    'predicted': all_pred_keywords[i],
                    'true': all_true_keywords[i] if i < len(all_true_keywords) else []
                })
            
            with open(os.path.join('./results', f'test_results_{args.test_logger_name}.json'), 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"상세 결과가 './results/test_results_{args.test_logger_name}.json'에 저장되었습니다.")
        except Exception as e:
            logger.warning(f"결과 저장 중 오류 발생: {e}")
    
    return total_test_loss, total_test_acc, total_precision, total_recall, total_f1

def test_with_keybert(test_dataset, args, logger, device=None):
    """
    KeyBERT 라이브러리를 사용하여 키워드 추출 테스트를 수행합니다.
    
    Args:
        test_dataset: 테스트 데이터셋
        args: 파라미터 (model_name 등)
        logger: 로거
        device: 사용할 디바이스
    
    Returns:
        tuple: (정밀도, 재현율, F1 점수)
    """
    logger.info("KeyBERT 라이브러리로 테스트 시작")
    logger.info("배치 크기: %d", args.batch_size if hasattr(args, 'batch_size') else 1)
    start_time = datetime.now()
    logger.info("start time: %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        from keybert import KeyBERT
        from transformers import BertModel
        import torch
        
        # KeyBERT 모델 초기화
        model = BertModel.from_pretrained(args.model_name)
        model.to(device)
        kw_model = KeyBERT(model=model)
        logger.info(f"KeyBERT 모델 초기화 완료: {args.model_name}")
        logger.info(f"device: {model.device}")
        
        # 평가 지표 초기화
        total_TP = 0
        total_FP = 0
        total_FN = 0
        
        # 모든 예측 및 실제 키워드 저장 (결과 분석용)
        all_pred_keywords = []
        all_true_keywords = []
        
        # 메모리 관리를 위해 GPU 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 배치 처리를 위한 준비
        batch_size = args.batch_size if hasattr(args, 'batch_size') else 1
        num_samples = len(test_dataset.data)
        num_batches = (num_samples + batch_size - 1) // batch_size  # 올림 나눗셈
        test_step = 0
        
        for batch_idx in range(0, num_samples, batch_size):
            test_step += 1
            batch_end = min(batch_idx + batch_size, num_samples)
            batch_data = test_dataset.data[batch_idx:batch_end]
            
            # 배치별 성능 지표
            batch_TP = 0
            batch_FP = 0
            batch_FN = 0
            
            for idx, data in enumerate(batch_data):
                # 텍스트와 실제 키워드 추출
                if isinstance(data, dict) and "text" in data and "keyword" in data:
                    text = data["text"]
                    true_keywords = data["keyword"]
                    
                    # 리스트가 아닌 경우 리스트로 변환
                    if not isinstance(true_keywords, list):
                        true_keywords = [true_keywords]
                    
                    # KeyBERT를 사용하여 키워드 추출
                    try:
                        keywords = kw_model.extract_keywords(
                            docs=text,
                            keyphrase_ngram_range=(1, 1),
                            top_n=args.num_keywords,
                        )
                        
                        # 키워드만 추출 (점수 제외)
                        pred_keywords = [kw[0] for kw in keywords]
                        
                        # 모든 예측 및 실제 키워드 저장
                        all_pred_keywords.append(pred_keywords)
                        all_true_keywords.append(true_keywords)
                        
                        # 키워드 평가
                        TP, FP, FN = evaluate_keywords(pred_keywords, true_keywords)
                        batch_TP += TP
                        batch_FP += FP
                        batch_FN += FN
                        
                    except Exception as e:
                        logger.warning(f"KeyBERT 키워드 추출 오류: {e}")
                        import traceback
                        logger.debug(f"상세 오류: {traceback.format_exc()}")
                        continue
                else:
                    logger.warning(f"데이터 형식 오류: {batch_idx + idx}번째 데이터")
            
            # 배치 성능 지표 계산
            precision = batch_TP / (batch_TP + batch_FP) if (batch_TP + batch_FP) > 0 else 0
            recall = batch_TP / (batch_TP + batch_FN) if (batch_TP + batch_FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 로깅 (자세한 배치별 로깅은 디버그 레벨로)
            if hasattr(args, 'log_freq') and (test_step % args.log_freq == 0 or test_step == 1):
                logger.info("Step: %d/%d, P: %.4f, R: %.4f, F1: %.4f", 
                           test_step, num_batches, precision, recall, f1)
            else:
                logger.debug("Step: %d/%d, P: %.4f, R: %.4f, F1: %.4f", 
                           test_step, num_batches, precision, recall, f1)
            
            # 전체 통계에 추가
            total_TP += batch_TP
            total_FP += batch_FP
            total_FN += batch_FN
        
        # 테스트 데이터가 없는 경우 처리
        if test_step == 0:
            logger.warning("테스트 데이터가 없거나 모든 배치에서 오류가 발생했습니다.")
            return 0.0, 0.0, 0.0
        
        # 전체 성능 지표 계산
        total_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        total_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
        
        logger.info("KeyBERT 테스트 완료 - 총 %d 배치", test_step)
        logger.info("키워드 성능 - Precision: %.4f, Recall: %.4f, F1: %.4f", total_precision, total_recall, total_f1)
        logger.info("Confusion Matrix - TP: %d, FP: %d, FN: %d", total_TP, total_FP, total_FN)
        logger.info("end time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("total time: %s", datetime.now() - start_time)
        # 상세 분석을 위한 결과 저장 (옵션)
        if hasattr(args, 'save_results') and args.save_results:
            try:
                # 결과를 파일로 저장
                results = {
                    'metrics': {
                        'precision': total_precision,
                        'recall': total_recall,
                        'f1': total_f1,
                        'TP': total_TP,
                        'FP': total_FP,
                        'FN': total_FN
                    },
                    'predictions': []
                }
                
                # 각 샘플의 예측 결과 저장
                for i in range(len(all_pred_keywords)):
                    results['predictions'].append({
                        'predicted': all_pred_keywords[i],
                        'true': all_true_keywords[i] if i < len(all_true_keywords) else []
                    })
                
                with open(os.path.join('./results', f'keybert_results_{args.test_logger_name}.json'), 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                logger.info(f"상세 결과가 './results/keybert_results_{args.test_logger_name}.json'에 저장되었습니다.")
            except Exception as e:
                logger.warning(f"결과 저장 중 오류 발생: {e}")
        
        return total_precision, total_recall, total_f1
        
    except ImportError:
        logger.error("KeyBERT 라이브러리를 설치해야 합니다. 'pip install keybert'를 실행하세요.")
        return 0.0, 0.0, 0.0

def main():
    parser = argparse.ArgumentParser(description='KoKeyBERT Testing')
    parser.add_argument("--test_data_path", type=str, default="./src/data/test_clean.json")
    parser.add_argument("--model_name", type=str, default="skt/kobert-base-v1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda for gpu, cpu for cpu")
    parser.add_argument("--test_logger_name", type=str, default="test")
    parser.add_argument("--num_workers", type=int, default=8, help="A100: 12, 8 recommanded")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_model.pt", help="Path to checkpoint to load model from")
    parser.add_argument("--log_freq", type=int, default=5, help="로깅 빈도 (몇 배치마다 로그를 출력할지)")
    parser.add_argument("--save_results", action="store_true", help="테스트 결과를 JSON 파일로 저장")
    parser.add_argument("--use_keybert", action="store_true", help="KeyBERT 라이브러리를 사용하여 테스트")
    parser.add_argument("--num_keywords", type=int, default=3, help="키워드 개수")
    args = parser.parse_args()

    # Create necessary directories
    os.makedirs("./log", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    # Initialize logger
    logger = logging.getLogger(args.test_logger_name)
    logger.setLevel(logging.DEBUG)
    
    # 이미 핸들러가 있는 경우 제거
    if logger.handlers:
        logger.handlers.clear()

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # file handler
    file_handler = logging.FileHandler("./log/" + args.test_logger_name + ".log")
    file_handler.setLevel(logging.DEBUG)

    # format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # Set device
    try:
        if args.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"GPU 사용: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("CPU 사용")
    except Exception as e:
        logger.warning(f"디바이스 설정 오류: {e}, CPU 사용")
        device = torch.device("cpu")

    # Load test data
    try:
        test_data = load_data(args.test_data_path)
        if test_data is None or len(test_data) == 0:
            raise ValueError(f"데이터 로드 실패 또는 비어있음: {args.test_data_path}")
        logger.info(f"테스트 데이터 로드 완료: {len(test_data)} 항목")
        test_dataset = KeywordDataset(test_data)
    except Exception as e:
        logger.error(f"데이터 로드 오류: {e}")
        return

    # KeyBERT 라이브러리 테스트
    if args.use_keybert:
        logger.info("********** KeyBERT 라이브러리 테스트 시작 **********")
        keybert_precision, keybert_recall, keybert_f1 = test_with_keybert(
            test_dataset=test_dataset,
            args=args,
            logger=logger,
            device=device
        )
        logger.info("********** KeyBERT 라이브러리 테스트 완료 **********")
        logger.info("KeyBERT 성능 - Precision: %.4f, Recall: %.4f, F1: %.4f", 
                   keybert_precision, keybert_recall, keybert_f1)
        return

    # KoKeyBERT 모델 테스트
    else:
        # Initialize tokenizer and collator
        try:
            tokenizer = KoBERTTokenizer.from_pretrained(args.model_name)
            collator = Collator(tokenizer)
            logger.info(f"토크나이저 초기화 완료")
        except Exception as e:
            logger.error(f"토크나이저 초기화 오류: {e}")
            return

        # Load model from checkpoint
        try:
            logger.info(f"모델 로드 중: {args.checkpoint_path}")
            config = BertConfig.from_pretrained(args.model_name)
            model = KoKeyBERT(config=config)
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
            model.to(device)
            logger.info("모델 로드 완료")
        except Exception as e:
            logger.error(f"모델 로드 오류: {e}")
            return

        # Run test
        logger.info("********** KoKeyBERT 테스트 시작 **********")
        logger.info("배치 크기: %d", args.batch_size)
        
        try:
            kokeybert_loss, kokeybert_acc, kokeybert_precision, kokeybert_recall, kokeybert_f1 = test(
                model=model,
                test_dataset=test_dataset,
                collator=collator,
                args=args,
                logger=logger,
                device=device
            )
            
            logger.info("********** KoKeyBERT 테스트 완료 **********")
            logger.info("KoKeyBERT 최종 결과 - 손실: %.4f, 정확도: %.4f", kokeybert_loss, kokeybert_acc)
            logger.info("KoKeyBERT 성능 - Precision: %.4f, Recall: %.4f, F1: %.4f", 
                       kokeybert_precision, kokeybert_recall, kokeybert_f1)
        except Exception as e:
            logger.error(f"테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
