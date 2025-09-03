# SQLAlchemy ORM 모델 정의
from sqlalchemy import Column, Integer, String, ForeignKey, Date
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class UserInput(Base):
    __tablename__ = "user_input"
    index = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)

    pos_results = relationship("PosResult", back_populates="user_input")

class PosResult(Base):
    __tablename__ = "pos_result"
    index = Column(Integer, primary_key=True, index=True)
    user_input_id = Column(Integer, ForeignKey("user_input.index"))
    created_at = Column(Date, default=func.current_date())
    noun = Column(String)
    verb = Column(String)
    adjective = Column(String)
    keyword = Column(String)

    user_input = relationship("UserInput", back_populates="pos_results")