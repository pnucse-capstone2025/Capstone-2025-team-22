import torch
import sys
import os
import logging
from torch.utils.data import DataLoader

# 부모 디렉토리 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

# test.py에서 import
from ...src.data.dataset import load_data, KeywordDataset, Collator
from ...src.models.kokeybert import KoKeyBERT
from test import test, extract_keywords_from_bio_tags, evaluate_keywords
from transformers import BertConfig

def test_kokeybert_performance():
    """test.py와 동일한 방식으로 KoKeyBERT 성능 측정"""
    print("🎯 KoKeyBERT 성능 측정 (test.py 방식)")
    
    # 로거 설정
    logger = logging.getLogger("kokeybert_test")
    logger.setLevel(logging.INFO)
    
    # 핸들러 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # 스트림 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 사용 디바이스: {device}")
    
    try:
        # 1. test.py와 동일한 데이터 로드
        print("📊 테스트 데이터 로딩 중...")
        test_data_path = "../../src/data/test_clean.json"
        
        if not os.path.exists(test_data_path):
            print(f"❌ 데이터 파일 없음: {test_data_path}")
            # 대안 경로들 시도
            alternative_paths = [
                "../../../src/data/test_clean.json",
                "../../data/test_clean.json", 
                "../data/test_clean.json",
                "./test_clean.json"
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    test_data_path = alt_path
                    print(f"✅ 대안 경로 발견: {test_data_path}")
                    break
            else:
                print("❌ 모든 대안 경로에서 데이터 파일을 찾을 수 없습니다.")
                return
        
        test_data = load_data(test_data_path)
        if test_data is None or len(test_data) == 0:
            raise ValueError(f"데이터 로드 실패 또는 비어있음: {test_data_path}")
        
        print(f"✅ 테스트 데이터 로드: {len(test_data)} 항목")
        test_dataset = KeywordDataset(test_data)
        
        # 2. 토크나이저 초기화
        print("🔤 토크나이저 로딩 중...")
        try:
            sys.path.append('../../kobert_tokenizer')
            from ....tokenizer.kobert_tokenizer import KoBERTTokenizer
            tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
            collator = Collator(tokenizer)
            print("✅ KoBERT 토크나이저 로드 성공")
        except Exception as e:
            print(f"⚠️ KoBERT 토크나이저 로드 실패: {e}")
            return
        
        # 3. 모델 로드
        print("🏗️ KoKeyBERT 모델 로딩 중...")
        model_name = "skt/kobert-base-v1"
        checkpoint_path = "../../checkpoints/best_model.pt"
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ 체크포인트 파일 없음: {checkpoint_path}")
            # 대안 경로들 시도
            alternative_checkpoints = [
                "../../../checkpoints/best_model.pt",
                "../../best_model.pt",
                "../best_model.pt",
                "./best_model.pt"
            ]
            
            for alt_path in alternative_checkpoints:
                if os.path.exists(alt_path):
                    checkpoint_path = alt_path
                    print(f"✅ 대안 체크포인트 발견: {checkpoint_path}")
                    break
            else:
                print("❌ 체크포인트 파일을 찾을 수 없습니다.")
                return
        
        try:
            config = BertConfig.from_pretrained(model_name)
            model = KoKeyBERT(config=config)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)
            print("✅ KoKeyBERT 모델 로드 완료")
        except Exception as e:
            print(f"❌ 모델 로드 오류: {e}")
            return
        
        # 4. 테스트 실행 (test.py와 동일한 방식)
        print("\n🚀 KoKeyBERT 테스트 시작!")
        print("="*50)
        
        # 가짜 args 객체 생성
        class Args:
            def __init__(self):
                self.batch_size = 8
                self.num_workers = 0  # multiprocessing 문제 방지
                self.log_freq = 10
                self.save_results = False
        
        args = Args()
        
        # test 함수 실행
        kokeybert_loss, kokeybert_acc, kokeybert_precision, kokeybert_recall, kokeybert_f1 = test(
            model=model,
            test_dataset=test_dataset,
            collator=collator,
            args=args,
            logger=logger,
            device=device
        )
        
        print("\n" + "="*50)
        print("🎉 KoKeyBERT 테스트 완료!")
        print("="*50)
        print(f"📊 최종 결과:")
        print(f"   손실: {kokeybert_loss:.4f}")
        print(f"   정확도: {kokeybert_acc:.4f}")
        print(f"   정밀도: {kokeybert_precision:.4f}")
        print(f"   재현율: {kokeybert_recall:.4f}")
        print(f"   F1 점수: {kokeybert_f1:.4f}")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_kokeybert_performance()
