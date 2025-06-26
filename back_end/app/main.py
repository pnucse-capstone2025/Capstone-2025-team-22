from fastapi import FastAPI, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import torch

from app import models, schemas, crud, database
from app.text_analysis import extract_pos_frequencies
from app.database import Base, engine
from app.nlp.model import KoKeyBERT
from app.nlp.kobert_tokenizer.kobert_tokenizer import KoBERTTokenizer
from app.nlp.utils.extract import extract_keywords_from_bio_tags
from transformers import BertConfig

app = FastAPI()
Base.metadata.create_all(bind=engine)

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 초기화
config = BertConfig.from_pretrained('skt/kobert-base-v1')
model = KoKeyBERT(config=config)
model.load_state_dict(torch.load('app/nlp/best_model.pt', map_location=torch.device('cpu'), weights_only=True))
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model.eval()

def extract_keywords_from_text(text: str):
    return model.extract_keywords(text, tokenizer)

@app.post("/extract_keywords")
def extract_keywords(text: str = Form(...), db: Session = Depends(database.get_db)):
    user_input = schemas.UserInputCreate(text=text)
    crud.create_user_input(db=db, user_input=user_input)

    pos_result = extract_pos_frequencies(text)
    nouns = list(pos_result['nouns'].keys())
    verbs = list(pos_result['verbs'].keys())
    adjectives = list(pos_result['adjectives'].keys())
    keywords = extract_keywords_from_text(text)
    print("추출된 키워드:", keywords)

    pos_result_db = schemas.POSResultCreate(
        noun=", ".join(nouns),
        verb=", ".join(verbs),
        adjective=", ".join(adjectives)
    )
    crud.create_pos_result(db=db, result=pos_result_db)

    return JSONResponse(content={
        "text": text,
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjectives,
        "keywords": list(keywords)
    })

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")