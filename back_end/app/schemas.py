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

class TextInput(BaseModel):
    text: str

# class Config:
#     from_attributes = True  # Pydantic v2에서 권장