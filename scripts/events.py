import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path


BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

def fetch_events(city=None, event_type=None):
    today = datetime.today() 
    one_year_more = today + timedelta(days=365)
    one_year_more = one_year_more.strftime("%Y-%m-%dT00:00:00Z")
    
    one_year_ago = today - timedelta(days=365)
    one_year_ago = one_year_ago.strftime("%Y-%m-%dT00:00:00Z")
   

    
    periode = f"firstdate_begin >= '{one_year_ago}' AND firstdate_begin <= '{one_year_more}'"
    
    # Paramètre transmis à l'URL API 
    params = {
        "limit": 100,
        "lang": "fr",
        "where": periode
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # lève une exception si status HTTP erreur
        print(response.url)
        
        data = response.json()
        events = data.get("results", [])

        if not events:
            print("⚠️ Aucun événement retourné par l'API")
            return pd.DataFrame()

        df = pd.json_normalize(events)

        # Filtrage par ville
        if city:
            if "location_city" not in df.columns:
                print("⚠️ Colonne 'location_city' absente, filtrage ville ignoré")
            else:
                df = df[df["location_city"].str.contains(city, case=False, na=False)]

        # Filtrage par type
        if event_type:
            if "keywords_fr" not in df.columns:
                print("⚠️ Colonne 'keywords_fr' absente, filtrage type ignoré")
            else:
                df = df[df["keywords_fr"].apply(
                    lambda x: event_type in x if isinstance(x, list) else False
                )]

        print(f"✅ Evènements récupérés : {len(df)}")
        # Sauvegarde json
        data_path = Path(os.environ["DATA_PATH"])
        input_path = data_path / "raw_events.json"
        df.to_json(input_path, orient="records", force_ascii=False)
        return df

    except requests.exceptions.ConnectionError:
        print("❌ Erreur de connexion : vérifiez votre accès réseau")
    except requests.exceptions.Timeout:
        print("❌ Timeout : l'API ne répond pas")
    except requests.exceptions.HTTPError as e:
        print(f"❌ Erreur HTTP {response.status_code} : {e}")
    except ValueError as e:
        print(f"❌ Erreur de parsing JSON : {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")

    return pd.DataFrame()  # retourne un df vide en cas d'erreur







