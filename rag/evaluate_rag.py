import os
import pandas as pd
from datasets import Dataset
import asyncio
import nest_asyncio
import traceback
from typing import List, Dict
import requests
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from dotenv import load_dotenv

nest_asyncio.apply()

BASE_URL = "http://localhost:8000"

test_data = [
    {
        "question": "Y a-t-il des événements liés à l'industrie ?",
        "ground_truth": "Il existe un atelier de découverte du secteur de l'industrie." 
    },
    {
        "question": "Quels événements ont lieu à Caen ?",
        "ground_truth": "Il y a un événement 1001 sports et animations à Caen en avril 2025."
    },
    {
        "question": "Y a-t-il des événements culturels à Rennes ?",
        "ground_truth": "Le Diwali - Fête de la lumière se tient à Rennes en octobre 2025."
    }
]

def call_api(question):
    response = requests.post(f"{BASE_URL}/chat", json={"question": question})
    data = response.json()
    contexts = [s.get("page_content", "") for s in data["sources"]]
    return data["answer"], contexts

# Construction du dataset
print("=== Évaluation RAG avec Ragas ===\n")
questions_test, answers, placeholder_contexts, ground_truths = [], [], [], []

for item in test_data:
    print(f"📨 Question : {item['question']}")
    answer, context = call_api(item["question"])
    questions_test.append(item["question"])
    answers.append(answer)
    placeholder_contexts.append(context)
    ground_truths.append(item["ground_truth"])

evaluation_data = {
    "question": questions_test,
    "answer": answers,
    "contexts": placeholder_contexts,
    "ground_truth": ground_truths
}
evaluation_dataset = Dataset.from_dict(evaluation_data)
print("Dataset d'évaluation prêt.")

load_dotenv(override=False)
# --- Configuration et Exécution de l'Évaluation
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "VOTRE_CLE_API_MISTRAL_ICI")
if MISTRAL_API_KEY == "VOTRE_CLE_API_MISTRAL_ICI" or not MISTRAL_API_KEY:
    print("⚠️ AVERTISSEMENT : Clé API Mistral non trouvée ou non définie.")

try:
    # 1. Initialisation du LLM et des Embeddings (via Langchain)
    print("Initialisation LLM et Embeddings Mistral...")
    mistral_llm = ChatMistralAI(mistral_api_key=MISTRAL_API_KEY, model="mistral-large-latest", temperature=0.1)
    mistral_embeddings = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
    print("LLM et Embeddings initialisés.")

    # 2. Définition des métriques à calculer
    metrics_to_evaluate = [
        faithfulness,       # Génération: fidèle au contexte ?
        #answer_relevancy,   # Génération: réponse pertinente à la question ?# # ❌ Bug ragas 0.4.3 - TypeError += dict
        context_precision,  # Récupération: contexte précis (peu de bruit) ?
        context_recall,     # Récupération: infos clés récupérées (nécessite ground_truth) ?
    ]
    print(f"Métriques sélectionnées: {[m.name for m in metrics_to_evaluate]}")

    # 3. Lancement de l'évaluation Ragas
    print("\nLancement de l'évaluation Ragas (peut prendre du temps)...")
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics_to_evaluate,
        llm=mistral_llm,
        embeddings=mistral_embeddings
    )
    print("\n--- Évaluation Ragas terminée ---")

    # 4. Affichage des résultats sous forme de DataFrame
    print("\n--- Résultats de l'évaluation (DataFrame) ---")
    results_df = results.to_pandas()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 150)
    print(results_df)

    # 5. Calcul et affichage des scores moyens
    print("\n--- Scores Moyens (sur tout le dataset) ---")
    average_scores = results_df.mean(numeric_only=True)
    print(average_scores)

    # 6. Vérification des seuils pour la CI
    print("\n--- Vérification des seuils CI ---")
    if average_scores.get("faithfulness", 0) < 0.7:
        raise AssertionError("❌ Fidélité trop faible")
    #if average_scores.get("answer_relevancy", 0) < 0.7:
    #    raise AssertionError("❌ Pertinence trop faible")
    if average_scores.get("context_precision", 0) < 0.7:
        raise AssertionError("❌ Précision du contexte trop faible")
    if average_scores.get("context_recall", 0) < 0.7:
        raise AssertionError("❌ Rappel du contexte trop faible")
    print("✅ Tous les seuils sont respectés")

except AssertionError as e:
    print(f"\n{e}")
    raise  # Fait échouer la CI

except Exception as e:
    print(f"\n❌ ERREUR lors de l'initialisation ou de l'évaluation Ragas : {e}")
    print("\nTraceback:")
    traceback.print_exc()
    raise  # Fait échouer la CI