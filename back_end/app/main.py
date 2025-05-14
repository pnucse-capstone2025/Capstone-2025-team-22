from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app import models, schemas, crud, database
from app.text_analysis import extract_pos_frequencies

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# 기본 폼 페이지 렌더링
@app.get("/")
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# POST: 입력값 처리 및 형태소 분석
@app.post("/input")
def handle_input(
    request: Request,
    text: str = Form(...),
    db: Session = Depends(database.get_db)
):
    # DB 저장
    user_input = schemas.UserInputCreate(text=text)
    crud.create_user_input(db=db, user_input=user_input)

    # 형태소 분석
    result = extract_pos_frequencies(text)
    nouns = list(result['nouns'].keys())
    verbs = list(result['verbs'].keys())
    adjectives = list(result['adjectives'].keys())

    # 분석 결과 출력
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user_input": text,
            "nouns": nouns,
            "verbs": verbs,
            "adjectives": adjectives
        }
    )