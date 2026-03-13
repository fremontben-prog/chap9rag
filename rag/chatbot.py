import os
from datetime import datetime

from langchain_mistralai.chat_models import ChatMistralAI
from rag.retriever import load_vectorstore

vectorstore = load_vectorstore()

# Supprimer la ligne : retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

llm = ChatMistralAI(
    model="mistral-small",
    mistral_api_key=os.environ["MISTRAL_API_KEY"]
)


def filter_results(docs, start_date=None, end_date=None):
    filtered = []
    for doc in docs:
        meta = doc.metadata
        if start_date and end_date:
            date_str = meta.get("date_begin")
            if not date_str:
                continue
            event_date = datetime.fromisoformat(date_str)
            if not (start_date <= event_date <= end_date):
                continue
        filtered.append(doc)
    return filtered


def ask_chatbot(question, start_date=None, end_date=None, top_k=10):
    if not question or not question.strip():
        raise ValueError("La question est vide.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})  # à la volée

    try:
        docs = retriever.invoke(question)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la recherche vectorielle : {str(e)}")

    docs = filter_results(docs, start_date, end_date)

    if not docs:
        return {
            "answer": "Je n'ai trouvé aucun événement correspondant à votre demande.",
            "sources": []
        }

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
            Tu es un assistant qui recommande des événements.
            Voici les SEULS événements disponibles :

            {context}

            Question utilisateur : {question}

            IMPORTANT : Réponds UNIQUEMENT en utilisant les événements listés ci-dessus.
            Ne mentionne aucun événement qui n'est pas dans la liste.
            Mentionne le titre, la date et le lieu si possible.
            """

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de l'appel au LLM : {str(e)}")

    return {
        "answer": response.content,
        "sources": [
            {**doc.metadata, "page_content": doc.page_content}
            for doc in docs
        ],
        "prompt": prompt
    }