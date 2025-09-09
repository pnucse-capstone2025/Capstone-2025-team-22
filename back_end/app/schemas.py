# Pydantic 모델 정의 (요청/응답 데이터 검증용)
from pydantic import BaseModel
from datetime import date

class UserInputCreate(BaseModel):
    text: str

class POSResultCreate(BaseModel):
    user_input_id: int
    noun: str
    verb: str
    adjective: str
    keyword: str

class POSResult(POSResultCreate):
    id: int
    created_at: date

    class Config:
        from_attributes = True  # Pydantic v2에서 권장

class AttentionResultCreate(BaseModel):
    user_input_id: int
    keyword: str
    attention_type: str
    attended_word: str
    score: float
    start_offset: int
    end_offset: int

class AttentionResult(AttentionResultCreate):
    id: int

    class Config:
        from_attributes = True

class TextInput(BaseModel):
    text: str

# class Config:
#     from_attributes = True  # Pydantic v2에서 권장