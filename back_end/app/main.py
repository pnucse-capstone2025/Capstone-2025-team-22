from fastapi import FastAPI, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import torch, json

from app import models, schemas, crud, database
from app.text_analysis import extract_nouns_verbs_adjectives, analyze_keyword_attention
from app.database import Base, engine
from app.nlp.experiments.distillation.distill_model import DistillKoKeyBERT
from app.nlp.tokenizer.kobert_tokenizer import KoBERTTokenizer
from app.nlp.src.utils.extract import extract_keywords_from_bio_tags
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
checkpoint = torch.load('app/nlp/models/kokeybert_distilled.pt', map_location=torch.device('cpu'), weights_only=False)
config = checkpoint['model_config']
model = DistillKoKeyBERT(config=config)
model.load_state_dict(checkpoint['model_state_dict'])
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model.eval()

def extract_keywords_from_text(text: str):
    return model.extract_keywords(text, tokenizer)

def extract_nva(text: str):
    pos_result = extract_nouns_verbs_adjectives(text)
    nouns = list(pos_result['nouns'].keys())
    verbs = list(pos_result['verbs'].keys())
    adjectives = list(pos_result['adjectives'].keys())
    
    keywords, outputs = extract_keywords_from_text(text)
    if keywords: 
        attention_analysis_result = analyze_keyword_attention(
            text=text,
            keywords_info=keywords,
            attentions=outputs.attentions,
            tokenizer=tokenizer
        )
        return nouns, verbs, adjectives, attention_analysis_result, keywords
    else:
        return nouns, verbs, adjectives, [], []

def save_pos_result_to_db(nouns, verbs, adjectives, keyword, user_input_id: int, db: Session):
    # keyword 처리: 중복 제거 및 문자열 변환
    if isinstance(keyword, list) and keyword and isinstance(keyword[0], dict):
        # 딕셔너리 리스트에서 'keyword' 값들 추출 후 중복 제거
        keywords = list(dict.fromkeys([item.get('keyword', '') for item in keyword if item.get('keyword')]))
    elif isinstance(keyword, (list, tuple, set)):
        # 일반 리스트/튜플/셋에서 중복 제거
        keywords = list(dict.fromkeys([str(k) for k in keyword if k]))
    else:
        # 단일 값인 경우
        keywords = [str(keyword)] if keyword else []
    
    pos_result_db = schemas.POSResultCreate(
        user_input_id=user_input_id,
        noun=", ".join(nouns) if nouns else "",
        verb=", ".join(verbs) if verbs else "",
        adjective=", ".join(adjectives) if adjectives else "",
        keyword=", ".join(keywords)
    )
    crud.create_pos_result(db=db, result=pos_result_db)

@app.get("/recent_results")
def recent_results(db: Session = Depends(database.get_db)):
    results = crud.get_recent_pos_results(db)
    response = []

    for result in results:
        user_input_text = db.query(models.UserInput).filter(models.UserInput.index == result.user_input_id).first()
        response.append({
            "id": result.user_input_id, # Frontend에서 `id`로 참조할 수 있도록 `user_input_id`를 반환
            "text": user_input_text.text if user_input_text else "",
            "created_at": result.created_at.isoformat(), # 저장 날짜 추가
            "nouns": result.noun.split(", ") if result.noun else [],
            "verbs": result.verb.split(", ") if result.verb else [],
            "adjectives": result.adjective.split(", ") if result.adjective else [],
            "keywords": result.keyword.split(", ") if result.keyword and result.keyword.strip() else [],
        })

    return JSONResponse(content={"results": response})

@app.get("/analysis_results/{user_input_id}")
def get_analysis_results(user_input_id: int, db: Session = Depends(database.get_db)):
    user_input_db = db.query(models.UserInput).filter(models.UserInput.index == user_input_id).first()
    pos_result_db = db.query(models.PosResult).filter(models.PosResult.user_input_id == user_input_id).first()

    if not user_input_db or not pos_result_db:
        return JSONResponse(content={"detail": "Analysis result not found"}, status_code=404)

    attention_results_db = crud.get_attention_results_by_user_input_id(db, user_input_id)
    formatted_attention_results = {}
    for ar in attention_results_db:
        if ar.keyword not in formatted_attention_results:
            formatted_attention_results[ar.keyword] = {"nouns": {}, "verbs": {}}
        
        item = {
            'keyword': ar.attended_word,
            'score': ar.score,
            'start': ar.start_offset,
            'end': ar.end_offset
        }
        
        key = f"{ar.attended_word}_{ar.start_offset}_{ar.end_offset}"
        if ar.attention_type == "noun":
            formatted_attention_results[ar.keyword]["nouns"][key] = item
        elif ar.attention_type == "verb":
            formatted_attention_results[ar.keyword]["verbs"][key] = item

    return JSONResponse(content={
        "id": user_input_db.index,
        "text": user_input_db.text,
        "created_at": pos_result_db.created_at.isoformat(),
        "nouns": pos_result_db.noun.split(", ") if pos_result_db.noun else [],
        "verbs": pos_result_db.verb.split(", ") if pos_result_db.verb else [],
        "adjectives": pos_result_db.adjective.split(", ") if pos_result_db.adjective else [],
        "keywords": pos_result_db.keyword.split(", ") if pos_result_db.keyword and pos_result_db.keyword.strip() else [],
        "noun_count": len(pos_result_db.noun.split(", ") if pos_result_db.noun else []),
        "verb_count": len(pos_result_db.verb.split(", ") if pos_result_db.verb else []),
        "adjective_count": len(pos_result_db.adjective.split(", ") if pos_result_db.adjective else []),
        "attention_result": formatted_attention_results
    })

@app.post("/extract_keywords")
def extract_keywords(text: str = Form(...), db: Session = Depends(database.get_db)):
    user_input = schemas.UserInputCreate(text=text)
    db_user_input = crud.create_user_input(db=db, user_input=user_input)

    nouns, verbs, adjectives, attention_analysis_result, keyword = extract_nva(text)
    save_pos_result_to_db(nouns, verbs, adjectives, keyword, db_user_input.index, db)

    # Attention 분석 결과 저장
    attention_results_to_create = []
    for kw_str, analysis_data in attention_analysis_result.items():
        for noun_item in analysis_data['nouns'].values():
            attention_results_to_create.append(schemas.AttentionResultCreate(
                user_input_id=db_user_input.index,
                keyword=kw_str,
                attention_type="noun",
                attended_word=noun_item['keyword'],
                score=noun_item['score'],
                start_offset=noun_item['start'],
                end_offset=noun_item['end']
            ))
        for verb_item in analysis_data['verbs'].values():
            attention_results_to_create.append(schemas.AttentionResultCreate(
                user_input_id=db_user_input.index,
                keyword=kw_str,
                attention_type="verb",
                attended_word=verb_item['keyword'],
                score=verb_item['score'],
                start_offset=verb_item['start'],
                end_offset=verb_item['end']
            ))
    crud.create_attention_results(db=db, results=attention_results_to_create)

    return JSONResponse(content={
        "text": text,
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjectives,
        "keywords": list(keyword) if keyword else [], # 빈 set일 경우 빈 리스트 반환
        "noun_count": len(nouns),
        "verb_count": len(verbs),
        "adjective_count": len(adjectives),
        "attention_result": attention_analysis_result
    })

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")