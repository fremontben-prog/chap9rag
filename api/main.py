from fastapi import FastAPI, HTTPException, Header, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator
from rag.chatbot import ask_chatbot
from scripts.main import build
import os
from dotenv import load_dotenv


VERSION = "1.0"

load_dotenv()
ADMIN_KEY = os.getenv("ADMIN_KEY", "changeme")
api_key_header = APIKeyHeader(name="x-admin-key", auto_error=False)

app = FastAPI(
    title="Event Chatbot API",
    description="Chatbot RAG basé sur FAISS + Mistral",
    version=VERSION

)

    
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5  #  valeur par défaut cohérente avec top_k_default dans /metadata

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("La question ne peut pas être vide.")
        if len(v.strip()) < 3:
            raise ValueError("La question est trop courte.")
        if len(v.strip()) > 500:  # déplacé avant le return
            raise ValueError("La question ne peut pas dépasser 500 caractères.")
        return v.strip()
    
    
class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[str] = []
    
    
@app.get("/health")
def health():
    """Vérifie que l'API est opérationnelle."""
    return {"status": "ok", "version": VERSION}


@app.get("/metadata")
def metadata():
    """Informations sur le système RAG."""
    return {
        "model": "mistral",
        "embedding_model": "mistral-embed",
        "index": "mistral_index",
        "chunk_size": 800,
        "top_k_default": 5, # top_k correspond au nombre de documents récupérés dans l'index FAISS avant de les envoyer au LLM.
        "description": "RAG sur corpus d'événements culturels"
    }


@app.post("/chat")
def chat(request: QuestionRequest):
    try:
        response = ask_chatbot(question=request.question, top_k=request.top_k)
        return ChatResponse(
            question=request.question,
            answer=response["answer"],
            sources=response.get("sources", [])
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")


@app.post("/rebuild")
def rebuild(x_admin_key: str = Security(api_key_header)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Accès non autorisé.")
    try:
        vector = build()
        if vector.index.ntotal == 0:
            return {"status": "warning", "message": "Pas de vecteur"}
        return {"status": "ok", "message": f"Rebuild OK — {vector.index.ntotal} vecteurs"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du rebuild : {str(e)}")