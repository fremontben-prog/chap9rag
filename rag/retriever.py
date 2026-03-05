import os
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
index_path = Path(os.environ["INDEX_PATH"])
index_path = index_path / "mistral_index"

def load_vectorstore():

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.environ["MISTRAL_API_KEY"]
    )

    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore