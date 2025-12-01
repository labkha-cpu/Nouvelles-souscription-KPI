import pandas as pd
import os
import sys
import csv
import re
from pathlib import Path
from datetime import datetime
import numpy as np

# Pour la connexion BDD
try:
    import psycopg
except ImportError:
    pass

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR if (SCRIPT_DIR / "Input_Data").exists() else SCRIPT_DIR.parent
INPUT_DIR = BASE_DIR / "Input_Data"
OUTPUT_DIR = BASE_DIR / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Config BDD (IEHE)
PG_HOST = "bdd-X0ED0550.alias" # Ou l'IP 100.54.41.6
PG_PORT = 5559
PG_DB = "choregie_db"
PG_USER = "u_lpillon"
PG_PASSWORD = "T_Run_Asc_2025#"
IEHE_SCHEMA = "iehe"
IEHE_TABLE  = "refkpep"
IEHE_COL_ID_TABLE = "refperboccn"

# --- UTILITAIRES ---

def connect_pg(host, port, db):
    return psycopg.connect(host=host, port=port, dbname=db, user=PG_USER, password=PG_PASSWORD, connect_timeout=3)

def connect_iehe_auto():
    # Liste de tentatives de connexion (Failover)
    hosts = ["bdd-X0ED0550.alias", "100.54.41.6"]
    ports = [5559, 5432]
    dbs = ["choregie_db", "postgres"]
    
    for h in hosts:
        for p in ports:
            for d in dbs:
                try: 
                    conn = connect_pg(h, p, d)
                    if conn: return conn
                except: continue
    return None

def clean_col_name(col):
    """Nettoie les noms de colonnes (suppression BOM, accents, espaces)."""
    if not isinstance(col, str): return str(col)
    col = col.replace('\ufeff', '').replace('\u200b', '') # Suppression BOM
    col = col.lower().strip()
    return col

def load_csv_robust(path):
    if not path.exists(): return None
    read_params = {'sep': None, 'engine': 'python', 'dtype': str}
    attempts = [
        {'encoding': 'utf-8-sig', 'skiprows': 0}, 
        {'encoding': 'utf-8', 'skiprows': 0},
        {'encoding': 'latin1', 'skiprows': 0},
        {'encoding': 'utf-8-sig', 'skiprows': 2}, # Format Accolade
        {'encoding': 'latin1', 'skiprows': 2}
    ]
    for params in attempts:
        current_params = read_params.copy()
        current_params.update(params)
        try:
            df = pd.read_csv(path, **current_params)
            # Vérification basique
            cols = [clean_col_name(c) for c in df.columns]
            keywords = ['adhesion', 'assure', 'email', 'personne', 'kpep']
            if any(k in c for c in cols for k in keywords):
                # On applique le nettoyage des colonnes
                df.columns = [clean_col_name(c) for c in df.columns]
                return df
        except: continue
    return None

def get_col_name(df, candidates):
    """Trouve la première colonne qui matche."""
    if df is None: return None
    for col in candidates:
        if col in df.columns: return col
    return None

def find_latest_new_s(directory):
    if not directory.exists(): return None, None
    candidates = []
    excluded_suffixes = ["_IEHE", "_CM", "_CK", "_REQ", "Resultats", "KPI"]
    
    for f in directory.glob("*.csv"):
        if any(kw in f.name for kw in excluded_suffixes): continue
        match = re.search(r"(\d{8})", f.name)
        if match:
            try: candidates.append({"date": datetime.strptime(match.group(1), "%d%m%Y"), "prefix": match.group(1), "path": f})
            except: continue
        else:
            dt_mtime = datetime.fromtimestamp(f.stat().st_mtime)
            candidates.append({"date": dt_mtime, "prefix": dt_mtime.strftime("%d%m%Y"), "path": f})

    if not candidates: return None, None
    best = sorted(candidates, key=lambda x: x["date"], reverse=True)[0]
    return best["path"], best["prefix"]

# --- ETAPE 1 : GENERATION IEHE ---

def run_iehe_step(df_ns, output_iehe_path):
    print(f"   🔨 Génération IEHE ({output_iehe_path.name})...")
    
    if output_iehe_path.exists():
        print(f"      ✅ Fichier déjà présent.")
        return

    if 'psycopg' not in sys.modules:
        print("      ⚠️  Module 'psycopg' manquant. Saut de l'étape IEHE.")
        return
    
    # Recherche intelligente de la colonne ID
    col_target = get_col_name(df_ns, ['num_personne', 'numpersonne', 'num_pers'])
    
    if not col_target:
        print(f"      ❌ Colonne ID introuvable dans le fichier source.")
        return

    ids = df_ns[col_target].dropna().unique().tolist()
    if not ids:
        print("      ⚠️  Aucun ID trouvé.")
        return
    
    conn = connect_iehe_auto()
    if not conn:
        print("      ❌ Echec Connexion BDD IEHE.")
        return
    
    sql = f"WITH ids AS (SELECT unnest(%(vals)s::text[]) AS v) SELECT r.* FROM {IEHE_SCHEMA}.{IEHE_TABLE} r JOIN ids ON ids.v = r.{IEHE_COL_ID_TABLE}"
    try:
        with conn.cursor() as cur:
            cur.execute(sql, {"vals": ids})
            rows, cols = cur.fetchall(), [d.name for d in cur.description]
        conn.close()
        with open(output_iehe_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
        print(f"      ✅ Généré ({len(rows)} lignes).")
    except Exception as e:
        print(f"      ❌ Erreur SQL : {e}")

# --- ETAPE 2 : GENERATION SQL ---

def run_sql_step(df, input_dir, output_dir, prefix):
    print(f"   🔨 Génération Requêtes SQL (Date d'exécution dynamique)...")
    
    # Recherche intelligente des colonnes
    col_mail = get_col_name(df, ['mailciam', 'mail_ciam', 'mail ciam', 'email_ciam'])
    col_val = get_col_name(df, ['valeur_coordonnee', 'valeur coordonnee', 'mail', 'email'])
    col_kpep = get_col_name(df, ['idkpep', 'kpep', 'id_kpep'])

    # 1. Liste Emails UNIFIÉE (Mail CIAM + Valeur Coordonnée)
    sources_emails = []
    if col_mail:
        sources_emails.append(df[col_mail])
    if col_val:
        sources_emails.append(df[col_val])
    
    email_list = []
    if sources_emails:
        combined = pd.concat(sources_emails)
        email_list = combined.replace('', np.nan).dropna().astype(str).str.strip().unique().tolist()
        email_list = [e for e in email_list if '@' in e]

    # 2. Liste KPEP
    kpep_list = []
    if col_kpep:
        if col_mail:
            mask_no_mail = (df[col_mail].astype(str).str.strip() == '') | (df[col_mail].isna())
            kpep_list = df.loc[mask_no_mail, col_kpep].replace('', np.nan).dropna().str.strip().unique().tolist()
        else:
            kpep_list = df[col_kpep].replace('', np.nan).dropna().str.strip().unique().tolist()

    # Configuration des tâches
    tasks = [
        ("00-Export_CIAM_EMAIL_With_Distinct.sql", "00-Export_CIAM_EMAIL_Global", email_list),
        ("00-Export_CIAM_KPEP_With_Distinct.sql", "00-Export_CIAM_KPEP_Global", kpep_list)
    ]
    
    # Date du jour pour le filtre SQL
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    generated_count = 0
    for tpl_name, output_suffix, data_list in tasks:
        tpl_path = input_dir / tpl_name
        if not tpl_path.exists():
            print(f"      ⚠️  Template absent : {tpl_name}")
            continue
        
        if not data_list:
            print(f"      ℹ️  Aucune donnée pour {output_suffix}")
            continue

        try:
            with open(tpl_path, 'r', encoding='utf-8') as f: base_sql = f.read()
            
            sanitized_list = [x.replace("'", "''") for x in data_list]
            values_str = "'" + "','".join(sanitized_list) + "'"
            
            # 1. Injection des IDs
            final_sql = base_sql.replace("__LISTE_IDS__", values_str)
            final_sql = final_sql.replace("'KPEP001', 'KPEP002', 'KPEP003'", values_str)
            
            # 2. Mise à jour de la date de fin (Remplacement de la date template par la date du jour)
            # On cible spécifiquement la date '2025-11-30' présente dans le template
            final_sql = final_sql.replace("2025-11-30", today_str)
            
            header = f"/* GENERATED {datetime.now()} | SOURCE: {prefix} | TYPE: {output_suffix} | NB: {len(data_list)} */\n"
            out_name = f"{prefix}_REQ_{output_suffix}.sql"
            
            with open(output_dir / out_name, 'w', encoding='utf-8') as f_out: 
                f_out.write(header + final_sql)
            print(f"      ✅ Requête générée : Output/{out_name} (Date fin: {today_str})")
            generated_count += 1
        except Exception as e:
            print(f"      ❌ Erreur sur {output_suffix} : {e}")

# --- MAIN ---

def main():
    if not INPUT_DIR.exists(): 
        print(f"❌ Dossier {INPUT_DIR} introuvable.")
        return
        
    ns_path, prefix = find_latest_new_s(INPUT_DIR)
    if not ns_path:
        print("❌ Aucun fichier New_S trouvé.")
        return

    print(f"📂 Préparation pour le flux : {prefix}")
    
    df_ns = load_csv_robust(ns_path)
    if df_ns is None:
        print("❌ Erreur critique : Chargement New_S impossible.")
        return

    # 1. IEHE
    iehe_path = INPUT_DIR / f"{prefix}_IEHE.csv"
    run_iehe_step(df_ns, iehe_path)

    # 2. SQL
    run_sql_step(df_ns, INPUT_DIR, OUTPUT_DIR, prefix)
    
    print("\n🏁 Etape 1 terminée.")

if __name__ == "__main__":
    main()