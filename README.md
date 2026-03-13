# README — Projet **RAG Événements**

## 1. Présentation du projet

Ce projet implémente un **chatbot intelligent basé sur une architecture RAG (Retrieval-Augmented Generation)** capable de recommander des événements à partir de données indexées.

Le système combine :

* un moteur de recherche sémantique basé sur **FAISS**
* un modèle de langage de **Mistral AI**
* une orchestration via **LangChain**
* une API REST développée avec **FastAPI**

L’objectif est de démontrer la capacité d’un système RAG à :

* rechercher des événements pertinents dans une base vectorielle
* générer des réponses naturelles
* recommander des événements personnalisés
* mesurer la qualité des réponses générées

---

# 2. Architecture du système

Le pipeline du chatbot fonctionne selon les étapes suivantes :

1. L’utilisateur pose une question via l’API.
2. La question est transformée en vecteur d’embedding.
3. Une recherche sémantique est effectuée dans l’index FAISS.
4. Les événements les plus pertinents sont récupérés.
5. Ces événements sont transmis au modèle Mistral.
6. Le LLM génère une réponse augmentée.

Schéma simplifié :

Utilisateur
↓
API FastAPI
↓
Retriever LangChain
↓
Index vectoriel FAISS
↓
Documents pertinents
↓
LLM Mistral
↓
Réponse générée

---

# 3. Structure du projet

```
project/
│
├── api/
│   └── main.py              # API FastAPI
│
├── rag/
│   ├── retriever.py         # Chargement FAISS
│   ├── chatbot.py           # Pipeline RAG
│   └── evaluate_rag.py      # Évaluation qualité
│
├── test/
│   └── api_tests.py         # Tests API
│
├── vectorstore/
│   └── mistral_index/       # Index FAISS
│
├── environment.yml         # Environnement conda
├── README.txt
└── .github/workflows/ci.yml
```

---

# 4. Installation

## 4.1 Prérequis

* Python 3.10
* Conda ou Miniconda
* Clé API Mistral

---

## 4.2 Création de l’environnement

Créer l’environnement avec :

```
conda env create -f environment.yml
conda activate chap9rag
```

### Contenu de l’environnement

```
name: chap9rag
channels:
  - conda-forge
dependencies:
  - python=3.10
  - faiss-cpu=1.7.4
  - pandas
  - fastapi
  - pip
  - pip:
      - langchain>=0.2.0
      - langchain-community>=0.2.0
      - langchain-mistralai
      - langchain-text-splitters
      - sentence-transformers
      - mistralai
      - python-dotenv
      - ragas
      - datasets
      - uvicorn
```


# 5. Configuration

Créer un fichier `.env` contenant :

```
MISTRAL_API_KEY=your_api_key_here
```


# 6. Lancer l’API

Démarrer le serveur :

```
uvicorn api.main:app --reload
```

Interface Swagger disponible sur :

```
http://localhost:8000/docs
```

---

# 7. Utilisation de l’API

Endpoint principal :

```
POST /chat
```

Exemple de requête :

```
{
  "question": "Quels événements à Paris ce week-end ?"
}
```

Réponse attendue :

```
{
  "answer": "... réponse générée ...",
  "sources": [...]
}
```

Les sources contiennent les métadonnées des événements récupérés.

---

# 8. Tests de l’API

Les tests sont définis dans :

```
tests/api_tests.py
```

Ils vérifient notamment :

* question valide
* question vide
* question trop courte
* champ manquant
* mauvais type
* reconstruction de l’index

Lancer les tests :

```
python tests/api_tests.py
```

---

# 9. Évaluation de la qualité du RAG

Le projet utilise **Ragas** pour mesurer la qualité du système.

Script :

```
rag/evaluate_rag.py
```

Métriques utilisées :

* **Faithfulness** : fidélité de la réponse au contexte
* **Context Precision** : pertinence des documents récupérés
* **Context Recall** : capacité à récupérer les informations clés

Exécution :

```
python rag/evaluate_rag.py
```

Seuils de validation CI :

* Faithfulness ≥ 0.7
* Context Precision ≥ 0.7
* Context Recall ≥ 0.7

Si un seuil n’est pas respecté, la CI échoue.

---

# 10. Intégration continue

Le projet utilise **GitHub Actions** pour automatiser :

* l’installation de l’environnement
* le démarrage de l’API
* l’exécution des tests
* l’évaluation RAG

Workflow CI :

```
name: RAG evenement

on:
  push:
    branches: [master, develop]
    tags:
      - "v*.*"
  pull_request:
    branches: [master, develop]
```

Étapes principales :

1. Checkout du code
2. Installation Python
3. Création environnement Conda
4. Démarrage de l’API
5. Tests fonctionnels
6. Évaluation RAG avec Ragas

---

# 11. Rebuild de l’index

L’API propose un endpoint permettant de reconstruire l’index vectoriel.

```
POST /rebuild
```

Ce endpoint :

1. recharge les données source
2. régénère les embeddings
3. recrée l’index FAISS

Ce test est ignoré en CI car les données source ne sont pas disponibles.

---

# 12. Jeux de données d’évaluation

Les questions utilisées pour l’évaluation :

* événements liés à l’industrie
* événements à Caen
* événements culturels à Rennes

Chaque question possède une **réponse de référence humaine** permettant de mesurer la qualité du RAG.

---

# 13. Points forts du projet

Ce POC démontre :

* un système RAG complet
* une API REST exploitable
* un moteur de recherche sémantique
* une génération de réponses naturelles
* un pipeline CI/CD
* une évaluation automatique des performances

---

# 14. Perspectives d’amélioration

Améliorations possibles :

* ajout de mémoire conversationnelle
* automatisation du chargement des évènements
* amélioration du ranking des événements
* hybrid search (BM25 + vecteurs)
* déploiement cloud

---

# 15. Auteur

Projet réalisé dans le cadre du parcours **Data Scientist**.

Chapitre : **Créez un assistant intelligent pour recommander des événements culturels**.
