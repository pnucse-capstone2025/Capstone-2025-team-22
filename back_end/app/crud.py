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
        noun=result.noun,
        verb=result.verb,
        adjective=result.adjective
    )
    db.add(pos_result)
    db.commit()
    db.refresh(pos_result)
    return pos_result

# 저장된 입력값을 가져오는 함수
def get_user_inputs(db: Session, skip: int = 0, limit: int = 10):
    return db.query(models.UserInput).offset(skip).limit(limit).all()