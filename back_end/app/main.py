from fastapi import FastAPI, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import torch

from app import models, schemas, crud, database
from app.text_analysis import extract_pos_frequencies, analyze_keyword_attention
from app.database import Base, engine
from app.nlp.distillation_experiment.distill_model import DistillKoKeyBERT
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
checkpoint = torch.load('app/nlp/distill_KoKeyBERT.pt', map_location=torch.device('cpu'), weights_only=False)
config = checkpoint['model_config']
model = DistillKoKeyBERT(config=config)
model.load_state_dict(checkpoint['model_state_dict'])
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
    keywords, outputs = extract_keywords_from_text(text)
    attention_analysis_result = analyze_keyword_attention(
        text=text,
        keywords_info=keywords,
        attentions=outputs.attentions,
        tokenizer=tokenizer
    )
    logger.info(f"추출된 키워드: {keywords}")
    logger.info(f"키워드별 어텐션 분석 결과: {attention_analysis_result}")

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