from scripts.events import fetch_events
from scripts.clean_events import clean_events
from scripts.metadata import build_documents
from scripts.embeddings import build_embeddings, test_similarity_search
from dotenv import load_dotenv

from pathlib import Path
import os

load_dotenv()

def build():
    # Récupération des évènements sur le site externe
    df = fetch_events()
    
    data_path = Path(os.environ["DATA_PATH"])
    input_path = data_path / "raw_events.json"
    output_path = data_path / "cleaned_events.json"
    
    
    print("Avant nettoyage :")
    print(df[["description_fr", "longdescription_fr"]].head(3))

    # Nettoyage des évènements
    df = clean_events(input_path,output_path)

    print("Après nettoyage :")
    print(df[["description_fr", "longdescription_fr"]].head(3))
    
    
    print(df.columns.tolist())
    print(df[["description_fr", "longdescription_fr"]].head(3))
    
    # Construction des documents, metadata et chunk
    documents = build_documents(df)
    
    # Construction des embeddings
    vector = build_embeddings(documents)
    
    return vector
    
        
    

if __name__ == "__main__":
    vector = build()
    test_similarity_search(vector)