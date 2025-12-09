import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.types import Date, DateTime, String, Integer, JSON

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR if (SCRIPT_DIR / "Input_Data").exists() else SCRIPT_DIR.parent
INPUT_DIR = BASE_DIR / "Input_Data"
OUTPUT_DIR = BASE_DIR / "Output"

PG_HOST = "bdd-T0XX0052.alias"
PG_PORT = "5577"
PG_DB = "supervisionpsc_db"
PG_USER = "rptpsc"
PG_PASSWORD = "rptpsc_xx"
PG_SCHEMA = "rptpsc"

def get_engine():
    url = f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    return create_engine(url)

def clean_col_name(col):
    if not isinstance(col, str): return str(col)
    col = col.lower().strip().replace(" ", "_").replace("-", "_").replace(".", "")
    col = col.replace("é", "e").replace("è", "e").replace("'", "")
    return col

def load_csv_robust(path):
    if not path.exists(): return None
    try:
        df = pd.read_csv(path, sep=None, engine='python', dtype=str)
        df.columns = [clean_col_name(c) for c in df.columns]
        # Nettoyage doublons colonnes
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except: return None

def upload_dataframe(df, table_name, engine, flux_id):
    if df is None or df.empty: return
    df = df.copy()
    if 'id' in df.columns: df.rename(columns={'id': 'id_csv'}, inplace=True)
    df['flux_id'] = flux_id
    df['date_import'] = datetime.now()
    
    # Nettoyage préalable (Delete by flux_id)
    with engine.connect() as conn:
        try:
            conn.execute(text(f"DELETE FROM {PG_SCHEMA}.{table_name} WHERE flux_id = :fid"), {"fid": flux_id})
            conn.commit()
        except: pass

    # Insertion (append avec création auto des colonnes si nouvelles)
    try:
        df.to_sql(table_name, engine, schema=PG_SCHEMA, if_exists='append', index=False)
        print(f"   ✅ {table_name} : Chargé ({len(df)} lignes).")
    except Exception as e:
        print(f"   ❌ Erreur {table_name} : {e}")

def main():
    # Détection prefixe
    prefix = None
    for f in INPUT_DIR.glob("*_New_S.csv"):
        if f.name[:8].isdigit(): prefix = f.name[:8]; break
    
    if not prefix: 
        print("❌ Préfixe introuvable.")
        return

    print(f"🚀 Chargement BDD pour : {prefix}")
    try: engine = get_engine()
    except: return

    # Inputs
    upload_dataframe(load_csv_robust(INPUT_DIR / f"{prefix}_New_S.csv"), "input_new_s", engine, prefix)
    upload_dataframe(load_csv_robust(INPUT_DIR / f"{prefix}_CM.csv"), "input_cm", engine, prefix)
    upload_dataframe(load_csv_robust(INPUT_DIR / f"{prefix}_CK.csv"), "input_ck", engine, prefix)
    upload_dataframe(load_csv_robust(INPUT_DIR / f"{prefix}_Rech_Nom.csv"), "input_rech_nom", engine, prefix)
    upload_dataframe(load_csv_robust(INPUT_DIR / f"{prefix}_Rech_Middle.csv"), "input_rech_middle", engine, prefix)
    
    # Outputs
    upload_dataframe(load_csv_robust(OUTPUT_DIR / f"{prefix}_NS_CIAM.csv"), "output_new_s_ciam", engine, prefix)
    
    print("🏁 Chargement terminé.")

if __name__ == "__main__":
    main()
