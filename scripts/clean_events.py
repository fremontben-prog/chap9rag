import pandas as pd
import re

# Fonction permettantde le nettoyage des caractères propre HTML
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)  # supprimer HTML
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Nettoyage du dataframe
def clean_events(input_path, output_path):
    df = pd.read_json(input_path)
    
    print("Colonnes disponibles :", df.columns.tolist())  # debug
    
    cols_to_fill = [
        "title_fr", "description_fr", "longdescription_fr",
        "conditions_fr", "location_name", "location_city",
        "location_postalcode", "location_department",
        "location_region", "canonicalurl", "accessibility_label_fr",
        "firstdate_begin", "firstdate_end"
    ]
    
    # Seulement les colonnes qui existent dans le df
    existing_cols = [col for col in cols_to_fill if col in df.columns]
    missing_cols = [col for col in cols_to_fill if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️ Colonnes absentes ignorées : {missing_cols}")
    
    df[existing_cols] = df[existing_cols].fillna("")
    df[existing_cols] = df[existing_cols].map(clean_text)
    
    df.to_csv(output_path, index=False)
    print(f"✅ Données nettoyées : {len(df)}")
    return df

    