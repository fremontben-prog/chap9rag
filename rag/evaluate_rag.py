from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
import requests

BASE_URL = "http://localhost:8000"

# Jeu de données de test avec réponses humaines annotées
test_data = [
    {
        "question": "Quels événements ont lieu à Paris ce week-end ?",
        "ground_truth": "Il y a un concert de jazz au Parc de la Villette samedi soir."
    },
    {
        "question": "Y a-t-il des expositions en ce moment ?",
        "ground_truth": "Une exposition de peinture contemporaine est visible au Centre Pompidou."
    },
    {
        "question": "Quels sont les événements gratuits à Lyon ?",
        "ground_truth": "Un marché artisanal gratuit se tient place Bellecour dimanche."
    }
]

def call_api(question):
    response = requests.post(f"{BASE_URL}/chat", json={"question": question})
    data = response.json()
    return data["answer"], [s.get("description", "") for s in data["sources"]]


def build_dataset():
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in test_data:
        print(f"📨 Question : {item['question']}")
        answer, context = call_api(item["question"])
        questions.append(item["question"])
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })


if __name__ == "__main__":
    print("=== Évaluation RAG avec Ragas ===\n")

    dataset = build_dataset()

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )

    print("\n=== Résultats ===")
    print(results)

    # Seuils minimaux pour la CI
    assert results["faithfulness"] >= 0.7, "❌ Fidélité trop faible"
    assert results["answer_relevancy"] >= 0.7, "❌ Pertinence trop faible"
    print("\n✅ Évaluation passée avec succès")