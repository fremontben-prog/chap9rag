import requests

BASE_URL = "http://localhost:8000"


def test_question_normale():
    print("🧪 Test 1 : Question normale")
    response = requests.post(f"{BASE_URL}/chat", json={"question": "Quels événements à Paris ce week-end ?"})
    print(f"{BASE_URL}/chat")
    print(f"Réponse : {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    print("✅ OK")
    print("Réponse :", data["answer"][:200])


def test_question_vide():
    print("\n🧪 Test 2 : Question vide")
    response = requests.post(f"{BASE_URL}/chat", json={"question": ""})
    assert response.status_code == 422, f"Attendu 422, reçu {response.status_code}"
    print("✅ OK - erreur 422 bien retournée")


def test_question_trop_courte():
    print("\n🧪 Test 3 : Question trop courte")
    response = requests.post(f"{BASE_URL}/chat", json={"question": "ab"})
    assert response.status_code == 422, f"Attendu 422, reçu {response.status_code}"
    print("✅ OK - erreur 422 bien retournée")


def test_champ_manquant():
    print("\n🧪 Test 4 : Champ 'question' manquant")
    response = requests.post(f"{BASE_URL}/chat", json={})
    assert response.status_code == 422, f"Attendu 422, reçu {response.status_code}"
    print("✅ OK - erreur 422 bien retournée")


def test_mauvais_type():
    print("\n🧪 Test 5 : Mauvais type (nombre au lieu de texte)")
    response = requests.post(f"{BASE_URL}/chat", json={"question": 12345})
    assert response.status_code == 422, f"Attendu 422, reçu {response.status_code}"
    print("✅ OK - Pydantic refuse le mauvais type")


def test_rebuild():
    print("\n🧪 Test 6 : Rebuild de l'index")
    response = requests.post(f"{BASE_URL}/rebuild")
    assert response.status_code == 200
    print("✅ OK - Réponse :", response.json())


if __name__ == "__main__":
    print("=== Lancement des tests API ===\n")
    test_question_normale()
    test_question_vide()
    test_question_trop_courte()
    test_champ_manquant()
    test_mauvais_type()
    test_rebuild()
    print("\n=== Tous les tests sont passés ✅ ===")