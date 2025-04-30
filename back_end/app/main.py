# app/main.py

from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app import models, schemas, crud, database

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# POST: 입력값 처리
@app.post("/input")
def handle_input(
    request: Request,
    text: str = Form(...),
    db: Session = Depends(database.get_db)
):
    user_input = schemas.UserInputCreate(text=text)  # 입력값을 Pydantic 모델로 감싸기
    crud.create_user_input(db=db, user_input=user_input)  # DB에 저장
    return templates.TemplateResponse("index.html", {"request": request, "user_input": text})