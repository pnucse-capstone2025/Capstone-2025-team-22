# SQLAlchemy ORM 모델 정의
from sqlalchemy import Column, Integer, String
from .database import Base

class UserInput(Base):
    __tablename__ = "user_input"
    index = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)

class PosResult(Base):
    __tablename__ = "pos_result"
    index = Column(Integer, primary_key=True, index=True)
    noun = Column(String)
    verb = Column(String)
    adjective = Column(String)