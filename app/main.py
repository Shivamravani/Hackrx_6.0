from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from app.utils import download_and_parse_pdf
from app.retrieval import answer_questions
from typing import Optional

app = FastAPI()

VALID_TOKEN = "c61acf6dfe00a39f662ac0e4c9dbebf0700f169710c2e07dd95e56636418ab65"

class RequestData(BaseModel):
    documents: str
    questions: list[str]

@app.post("/api/v1/hackrx/run")
def run_submission(data: RequestData, Authorization: Optional[str] = Header(None)):
    if Authorization is not None:
        if Authorization.split()[1] != VALID_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid token")
    text = download_and_parse_pdf(data.documents)
    result = answer_questions(text, data.questions)
    return result