# Pydantic 모델 정의 (요청/응답 데이터 검증용)
from pydantic import BaseModel

class UserInputCreate(BaseModel):
    text: str