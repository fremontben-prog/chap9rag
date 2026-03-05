from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from rag.chatbot import ask_chatbot
from scripts.main import build


app = FastAPI(
    title="Event Chatbot API",
    description="Chatbot RAG basé sur FAISS + Mistral",
    version="1.0"
)

    
class QuestionRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("La question ne peut pas être vide.")
        if len(v.strip()) < 3:
            raise ValueError("La question est trop courte.")
        return v.strip()


@app.post("/chat")
def chat(request: QuestionRequest):
    try:
        response = ask_chatbot(question=request.question)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")


@app.post("/rebuild")
def rebuild():
    vector = build()
    if  vector.index.ntotal == 0 :
        return "Pas de vecteur"
    else :
        return "✅ Rebuid OK"