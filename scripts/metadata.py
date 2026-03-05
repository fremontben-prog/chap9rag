from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pandas as pd

def has_description(row):
    desc = row.get("description_fr", "")
    long_desc = row.get("longdescription_fr", "")
    
    # Gère None, NaN, chaîne vide
    desc_ok = desc and not (isinstance(desc, float) and pd.isna(desc))
    long_ok = long_desc and not (isinstance(long_desc, float) and pd.isna(long_desc))
    
    return desc_ok or long_ok

def build_content(row):
    return f"""
        Titre : {row.get('title_fr', '')}

        Lieu : {row.get('location_name', '')}, {row.get('location_city', '')}
        Date : {row.get('firstdate_begin', '')} - {row.get('firstdate_end', '')}

        Description courte :
        {row.get('description_fr', '')}

        Description détaillée :
        {row.get('longdescription_fr', '')}

        Conditions :
        {row.get('conditions_fr', '')}
"""

def create_text_splitter():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    return text_splitter
    
    
def build_metadata(row):
    metadata = {
        "title": row.get("title_fr", ""),
        "date_begin": row.get("firstdate_begin", ""),
        "date_end": row.get("firstdate_end", ""),
        "location_name":row.get("location_name", ""),
        "city": row.get("location_city", ""),
        "postal_code": row.get("location_postalcode", ""),
        "department": row.get("location_department", ""),
        "region": row.get("location_region", ""),
        "url": row.get("canonicalurl", ""),
        "accessibility_label_fr": row.get("accessibility_label_fr", "")            
    }
    return metadata

def build_documents(df):
    documents = []
    text_splitter = create_text_splitter()
    skipped = 0
    
    for _, row in df.iterrows():
        if not has_description(row):
            skipped += 1
            continue
            
        content = build_content(row)
        chunks = text_splitter.split_text(content)
        metadata = build_metadata(row)
        
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={**metadata, "chunk_index": i}
            ))
    
    print(f"⚠️ Événements ignorés (sans description) : {skipped}")
    print(f"✅ Documents préparés : {len(documents)}")
    
    # Sécurité : évite le crash FAISS
    if not documents:
        raise ValueError("Aucun document généré — vérifie les noms de colonnes et les données.")
    
    return documents