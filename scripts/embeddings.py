from langchain_mistralai import MistralAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from pathlib import Path

# Chargement des variables environnement - MISTRAL_API_KEY
load_dotenv()

def build_embeddings(documents, index_path=Path(os.environ["INDEX_PATH"])):
    # Initialisation du modèle d'embeddings
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.environ["MISTRAL_API_KEY"]
    )

    # Création de la base vectorielle (FAISS génère les embeddings automatiquement)
    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    # Sauvegarde de l’index
    index_path = index_path / "mistral_index"
    vectorstore.save_local(index_path)

    print(f"✅ Index FAISS créé et sauvegardé dans '{index_path}'")
    print(f"Nombre de documents indexés : {len(documents)}")
    print("Nombre de vecteurs dans FAISS :", vectorstore.index.ntotal)
    
    assert vectorstore.index.ntotal == len(documents), f"❌ KO Erreur d'indexation : {vectorstore.index.ntotal} vecteurs créés pour {len(documents)} docs."

    return vectorstore

# Test de recherche
def test_similarity_search(vectorstore):
    query = "concert gratuit à Rennes"
    results = vectorstore.similarity_search(query, k=20)

    print("\n🔎 Résultats pour :", query)
    for i, doc in enumerate(results):
        print(f"\nRésultat {i+1}")
        print("Titre :", doc.metadata["title"])
        print("Ville :", doc.metadata["city"])