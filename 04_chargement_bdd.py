import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.types import Date, DateTime, String, Integer, JSON

# --- HISTORIQUE ---
# V1.1 : Ajout sys.exit(1) sur échec connexion BDD.
# V1.2 : Ajout chargement fichiers Rech_Nom et Rech_Middle.

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR if (SCRIPT_DIR / "Input_Data").exists() else SCRIPT_DIR.parent
INPUT_DIR = BASE_DIR / "Input_Data"
OUTPUT_DIR = BASE_DIR / "Output"

# CONFIGURATION BDD
PG_HOST = "bdd-T0XX0052.alias"
PG_PORT = "5577"
PG_DB = "supervisionpsc_db"
PG_USER = "rptpsc"
PG_PASSWORD = "rptpsc_xx"
PG_SCHEMA = "rptpsc"

# --- UTILITAIRES ---

def get_engine():
    # Construction URL
    url = f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    return create_engine(url)

def clean_col_name(col):
    """Nettoie les noms de colonnes pour PostgreSQL."""
    if not isinstance(col, str): return str(col)
    col = col.replace('\ufeff', '').replace('\u200b', '')
    col = col.lower().strip()
    col = col.replace(" ", "_").replace("-", "_").replace(".", "")
    col = col.replace("é", "e").replace("è", "e").replace("'", "")
    col = col.replace("__", "_")
    return col

def load_csv_robust(path):
    if not path.exists(): return None
    read_params = {'sep': None, 'engine': 'python', 'dtype': str}
    attempts = [
        {'encoding': 'utf-8-sig', 'skiprows': 0},
        {'encoding': 'utf-8', 'skiprows': 0},
        {'encoding': 'latin1', 'skiprows': 0},
        {'encoding': 'utf-8-sig', 'skiprows': 2},
        {'encoding': 'latin1', 'skiprows': 2}
    ]
    for params in attempts:
        current_params = read_params.copy()
        current_params.update(params)
        try:
            df = pd.read_csv(path, **current_params)
            if len(df.columns) > 1:
                # Nettoyage immédiat des colonnes
                df.columns = [clean_col_name(c) for c in df.columns]
                df = df.loc[:, ~df.columns.duplicated()]
                return df
        except: continue
    return None

def find_latest_prefix(directory):
    if not directory.exists(): return None
    for f in directory.glob("*_New_S.csv"):
        if f.name[:8].isdigit(): return f.name[:8]
    return None

# --- FONCTIONS CHARGEMENT ---

def upload_dataframe(df, table_name, engine, flux_id):
    if df is None or df.empty:
        print(f"   ⚠️ {table_name} : Fichier vide ou manquant. (Ignoré)")
        return

    df = df.copy()
    
    # Gestion ID conflictuel
    if 'id' in df.columns:
        df.rename(columns={'id': 'id_csv'}, inplace=True)

    # Métadonnées
    df['flux_id'] = flux_id
    df['date_import'] = datetime.now()
    
    # Typage
    dtype_map = {}
    for col in df.columns:
        if "date" in col and col != "date_import":
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            dtype_map[col] = Date()
    
    df = df.where(pd.notnull(df), None)

    # Nettoyage pré-insertion (Idempotence)
    with engine.connect() as conn:
        try:
            conn.execute(text(f"DELETE FROM {PG_SCHEMA}.{table_name} WHERE flux_id = :fid"), {"fid": flux_id})
            conn.commit()
        except Exception as e:
            # Si la table n'existe pas encore, ce n'est pas grave
            pass

    # Insertion
    try:
        df.to_sql(
            table_name, 
            engine, 
            schema=PG_SCHEMA, 
            if_exists='append', 
            index=False,
            dtype=dtype_map
        )
        print(f"   ✅ {table_name} : {len(df)} lignes insérées.")
    except Exception as e:
        print(f"   ❌ ERREUR insertion {table_name} : {e}")
        print(f"      Conseil : Si vous avez ajouté des colonnes (ex: Email_CIAM), la table doit être recréée ou modifiée.")

def upload_json(path, table_name, engine, flux_id):
    if not path.exists(): return
    try:
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
    except Exception as e:
        print(f"   ❌ Erreur lecture JSON : {e}")
        return
    
    df = pd.DataFrame([{
        "flux_id": flux_id,
        "date_import": datetime.now(),
        "kpi_data": data
    }])
    
    with engine.connect() as conn:
        try:
            conn.execute(text(f"DELETE FROM {PG_SCHEMA}.{table_name} WHERE flux_id = :fid"), {"fid": flux_id})
            conn.commit()
        except: pass
        
    try:
        df.to_sql(
            table_name, 
            engine, 
            schema=PG_SCHEMA, 
            if_exists='append', 
            index=False, 
            dtype={"kpi_data": JSON}
        )
        print(f"   ✅ {table_name} : KPIs sauvegardés.")
    except Exception as e:
        print(f"   ❌ Erreur insertion JSON : {e}")

# --- MAIN ---

def main():
    prefix = find_latest_prefix(INPUT_DIR)
    if not prefix:
        print("❌ Préfixe introuvable dans Input_Data.")
        sys.exit(1)

    print(f"🚀 Chargement BDD [{PG_HOST}] pour : {prefix}")
    
    try:
        engine = get_engine()
        with engine.connect() as conn: pass
    except Exception as e:
        print(f"❌ ECHEC CONNEXION BDD : {e}")
        print("   Vérifiez le VPN, le mot de passe ou l'adresse.")
        sys.exit(1)

    print("\n--- INPUTS ---")
    upload_dataframe(load_csv_robust(INPUT_DIR / f"{prefix}_New_S.csv"), "input_new_s", engine, prefix)
    upload_dataframe(load_csv_robust(INPUT_DIR / f"{prefix}_IEHE.csv"),  "input_iehe", engine, prefix)
    upload_dataframe(load_csv_robust(INPUT_DIR / f"{prefix}_CK.csv"),    "input_ck", engine, prefix)
    upload_dataframe(load_csv_robust(INPUT_DIR / f"{prefix}_CM.csv"),    "input_cm", engine, prefix)
    
    # Ajout des fichiers de recherche manuelle
    upload_dataframe(load_csv_robust(INPUT_DIR / f"{prefix}_Rech_Nom.csv"), "input_rech_nom", engine, prefix)
    upload_dataframe(load_csv_robust(INPUT_DIR / f"{prefix}_Rech_Middle.csv"), "input_rech_middle", engine, prefix)

    print("\n--- OUTPUTS CSV ---")
    upload_dataframe(load_csv_robust(OUTPUT_DIR / f"{prefix}_NS_CIAM.csv"), "output_new_s_ciam", engine, prefix)
    upload_dataframe(load_csv_robust(OUTPUT_DIR / f"{prefix}_NS_IEHE.csv"), "output_new_s_iehe", engine, prefix)

    print("\n--- OUTPUT JSON ---")
    upload_json(OUTPUT_DIR / f"{prefix}_KPI_Resultats.json", "output_json", engine, prefix)

    print("\n🏁 Chargement terminé.")

if __name__ == "__main__":
    main()
