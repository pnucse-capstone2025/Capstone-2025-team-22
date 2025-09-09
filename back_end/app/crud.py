# DB 관련 CRUD 함수들 (데이터 생성, 조회, 삭제 기능 담당)
from sqlalchemy.orm import Session
from app import models, schemas

# 입력값을 데이터베이스에 저장하는 함수
def create_user_input(db: Session, user_input: schemas.UserInputCreate):
    db_user_input = models.UserInput(text=user_input.text)
    db.add(db_user_input)
    db.commit()
    db.refresh(db_user_input)  # 갱신
    return db_user_input

#형태소별로 정제된 단어들을 저장하는 함수
def create_pos_result(db: Session, result: schemas.POSResultCreate):
    pos_result = models.PosResult(
        user_input_id=result.user_input_id,
        noun=result.noun,
        verb=result.verb,
        adjective=result.adjective,
        keyword=result.keyword
    )
    db.add(pos_result)
    db.commit()
    db.refresh(pos_result)
    return pos_result

# crud.py
def get_recent_pos_results(db: Session, limit: int = 15):
    return db.query(models.PosResult).order_by(models.PosResult.index.desc()).limit(limit).all()

def create_attention_results(db: Session, results: list[schemas.AttentionResultCreate]):
    db_attention_results = []
    for result in results:
        db_attention_result = models.AttentionResult(
            user_input_id=result.user_input_id,
            keyword=result.keyword,
            attention_type=result.attention_type,
            attended_word=result.attended_word,
            score=result.score,
            start_offset=result.start_offset,
            end_offset=result.end_offset
        )
        db.add(db_attention_result)
        db_attention_results.append(db_attention_result)
    db.commit()
    for result in db_attention_results:
        db.refresh(result)
    return db_attention_results

def get_attention_results_by_user_input_id(db: Session, user_input_id: int):
    return db.query(models.AttentionResult).filter(models.AttentionResult.user_input_id == user_input_id).all()
