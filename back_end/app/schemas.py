# Pydantic 모델 정의 (요청/응답 데이터 검증용)
from pydantic import BaseModel

class UserInputCreate(BaseModel):
    text: str

class POSResultCreate(BaseModel):
    noun: str
    verb: str
    adjective: str

class POSResult(POSResultCreate):
    id: int

    class Config:
        orm_mode = True

class TextInput(BaseModel):
    text: str

class Config:
    from_attributes = True  # Pydantic v2에서 권장