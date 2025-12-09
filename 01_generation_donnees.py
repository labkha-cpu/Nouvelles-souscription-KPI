import pandas as pd
import os
import sys
import csv
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import warnings

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
PG_HOST = "bdd-X0ED0550.alias" 
PG_PORT = 5559
PG_DB = "choregie_db"
PG_USER = "u_lpillon"
PG_PASSWORD = "T_Run_Asc_2025#"
IEHE_SCHEMA = "iehe"
IEHE_TABLE  = "refkpep"
IEHE_COL_ID_TABLE = "refperboccn"

# --- UTILITAIRES ---

def connect_pg(host, port, db):
    try:
        return psycopg.connect(host=host, port=port, dbname=db, user=PG_USER, password=PG_PASSWORD, connect_timeout=3)
    except:
        return None

def connect_iehe_auto():
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
        {'encoding': 'utf-8-sig', 'skiprows': 2},
        {'encoding': 'latin1', 'skiprows': 2}
    ]
    for params in attempts:
        current_params = read_params.copy()
        current_params.update(params)
        try:
            df = pd.read_csv(path, **current_params)
            cols = [clean_col_name(c) for c in df.columns]
            keywords = ['adhesion', 'assure', 'email', 'personne', 'kpep', 'realm']
            if any(k in c for c in cols for k in keywords):
                df.columns = [clean_col_name(c) for c in df.columns]
                return df
        except: continue
    return None

def get_col_name(df, candidates):
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
    
    safe_ids = [str(i).replace("'", "") for i in ids]
    
    sql = f"""
    WITH input_ids AS (
        SELECT unnest(%(vals)s::text[]) AS v
    )
    SELECT DISTINCT r1.*
    FROM {IEHE_SCHEMA}.{IEHE_TABLE} r1
    JOIN {IEHE_SCHEMA}.{IEHE_TABLE} r2 ON r2.idrpp = r1.idrpp
    JOIN input_ids ON input_ids.v = r1.{IEHE_COL_ID_TABLE}
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(sql, {"vals": safe_ids})
            rows = cur.fetchall()
            cols = [d.name for d in cur.description]
        conn.close()
        with open(output_iehe_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
        print(f"      ✅ Généré ({len(rows)} lignes).")
    except Exception as e:
        print(f"      ❌ Erreur SQL : {e}")

# --- ETAPE 2 : GENERATION SQL (CASCADE) ---

def run_sql_step(df, input_dir, output_dir, prefix):
    print(f"   🔨 Génération Requêtes SQL (Mode Cascade + LowerCase)...")
    
    # 1. Filtre CIAM
    df_ciam = df.copy()
    col_type = get_col_name(df_ciam, ['type_assure', 'typeassure', 'code_role_personne', 'role'])
    if col_type:
        mask_conjoi = df_ciam[col_type].astype(str).str.upper().str.strip() == 'CONJOI'
        df_ciam = df_ciam[~mask_conjoi]

    # 2. Identification Colonnes
    col_mail = get_col_name(df_ciam, ['mailciam', 'mail_ciam', 'mail ciam', 'email_ciam'])
    col_val = get_col_name(df_ciam, ['valeur_coordonnee', 'valeur coordonnee', 'mail', 'email'])
    col_kpep = get_col_name(df_ciam, ['idkpep', 'kpep', 'id_kpep'])
    col_nom = get_col_name(df_ciam, ['nom_long', 'nom', 'lastname'])
    col_dnaiss = get_col_name(df_ciam, ['date_naissance', 'datenaissance', 'birthdate'])

    # 3. Chargement Résultats Existants
    already_found_emails = set()
    already_found_kpeps = set()
    
    cm_path = input_dir / f"{prefix}_CM.csv"
    if cm_path.exists():
        try:
            df_cm = pd.read_csv(cm_path, engine='python', dtype=str)
            col_email_cm = get_col_name(df_cm, ['email', 'cm_email'])
            if col_email_cm:
                clean_mails = df_cm[col_email_cm].str.replace('"', '', regex=False).str.lower().str.strip().dropna()
                already_found_emails = set(clean_mails)
            print(f"      ℹ️  CM.csv : {len(already_found_emails)} emails exclus.")
        except: pass

    ck_path = input_dir / f"{prefix}_CK.csv"
    if ck_path.exists():
        try:
            df_ck = pd.read_csv(ck_path, engine='python', dtype=str)
            col_kpep_ck = get_col_name(df_ck, ['idkpep', 'ck_kpep', 'kpep'])
            if col_kpep_ck:
                clean_kpeps = df_ck[col_kpep_ck].str.replace('"', '', regex=False).str.strip().dropna()
                already_found_kpeps = set(clean_kpeps)
            print(f"      ℹ️  CK.csv : {len(already_found_kpeps)} KPEPs exclus.")
        except: pass

    # 4. Création Masques d'Exclusion
    key_mail = df_ciam[col_mail].astype(str).str.replace('"', '', regex=False).str.lower().str.strip() if col_mail else pd.Series()
    key_val = df_ciam[col_val].astype(str).str.replace('"', '', regex=False).str.lower().str.strip() if col_val else pd.Series()
    mask_found_by_email = (key_mail.isin(already_found_emails)) | (key_val.isin(already_found_emails))
    
    key_kpep_src = df_ciam[col_kpep].astype(str).str.replace('"', '', regex=False).str.strip() if col_kpep else pd.Series()
    mask_found_by_kpep = key_kpep_src.isin(already_found_kpeps)

    # 5. Listes de Recherche
    
    # LISTE 1 : EMAIL
    email_list = []
    sources_emails = []
    if col_mail: sources_emails.append(df_ciam[col_mail])
    if col_val: sources_emails.append(df_ciam[col_val])
    if sources_emails:
        combined = pd.concat(sources_emails)
        email_list = combined.replace('', np.nan).dropna().astype(str).str.strip().str.lower().unique().tolist()
        email_list = [e for e in email_list if '@' in e]

    # LISTE 2 : KPEP
    kpep_list = []
    excluded_count_kpep = 0
    if col_kpep:
        df_kpep_target = df_ciam[~mask_found_by_email] 
        excluded_count_kpep = len(df_ciam) - len(df_kpep_target)
        kpep_list = df_kpep_target[col_kpep].replace('', np.nan).dropna().str.strip().unique().tolist()

    # LISTE 3 : NOM + DATE (CORRECTION DATE ICI)
    nom_date_list = []
    excluded_count_nom = 0
    df_reliquat = df_ciam[~(mask_found_by_email | mask_found_by_kpep)] 
    excluded_count_nom = len(df_ciam) - len(df_reliquat)
    
    if col_nom and col_dnaiss and not df_reliquat.empty:
        temp_df = df_reliquat[[col_nom, col_dnaiss]].dropna().copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # --- FIX DATE : Force dayfirst=True pour le format Français (JJ/MM/AAAA) ---
            # On convertit d'abord en objet Date (en forçant l'interprétation Jour/Mois)
            temp_df['dt_obj'] = pd.to_datetime(temp_df[col_dnaiss], dayfirst=True, errors='coerce')
            
            # Ensuite on formate strictement en YYYY-MM-DD pour le SQL (standard ISO DB)
            temp_df['dt_fmt'] = temp_df['dt_obj'].dt.strftime('%Y-%m-%d')
        
        temp_df = temp_df.dropna(subset=['dt_fmt'])
        temp_df['nom_fmt'] = temp_df[col_nom].astype(str).str.strip().str.replace("'", "''") 
        
        raw_tuples = list(zip(temp_df['nom_fmt'], temp_df['dt_fmt']))
        nom_date_list = sorted(list(set(raw_tuples)))

    # --- LOGS ---
    print("\n      📊 [VOLUMETRIE REQUETES]")
    print(f"      1. Requête EMAIL générée sur : {len(email_list)} adresses (Lower)")
    print(f"      2. Requête KPEP générée sur  : {len(kpep_list)} IDs")
    print(f"         └-> Exclus (déjà trouvés Email) : {excluded_count_kpep}")
    print(f"      3. Requête LARGE (Nom/Middle) sur : {len(nom_date_list)} Couples (Nom, Date)")
    print(f"         └-> Exclus (déjà trouvés Email/KPEP) : {excluded_count_nom}")
    print("      ------------------------------------------------")

    # 6. Écriture
    tasks = [
        ("00-Export_CIAM_EMAIL_With_Distinct.sql", "00-Export_CIAM_EMAIL_Global", email_list, "simple_email"),
        ("00-Export_CIAM_KPEP_With_Distinct.sql", "00-Export_CIAM_KPEP_Global", kpep_list, "simple"),
        ("00-Export_CIAM_LAST_NAME_With_Distinct.sql", "01-Rech_Manuelle_LastName_Date", nom_date_list, "complex_lastname"),
        ("00-Export_CIAM_MiddleName_With_Distinct.sql", "01-Rech_Manuelle_MiddleName_Date", nom_date_list, "complex_middlename")
    ]
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    for tpl_name, output_suffix, data_list, data_type in tasks:
        tpl_path = input_dir / tpl_name
        
        if not data_list:
            continue
            
        if not tpl_path.exists():
            print(f"      ⚠️  Template absent : {tpl_name}")
            continue

        try:
            with open(tpl_path, 'r', encoding='utf-8') as f: base_sql = f.read()
            
            if "simple_email" == data_type:
                base_sql = base_sql.replace("usr.email IN", "LOWER(usr.email) IN")
                base_sql = base_sql.replace("usr.email =", "LOWER(usr.email) =")
            
            values_str = ""
            if "simple" in data_type:
                sanitized_list = [x.replace("'", "''") for x in data_list]
                values_str = "'" + "','".join(sanitized_list) + "'"
            
            elif "complex" in data_type:
                target_col = "usr.last_name" if data_type == "complex_lastname" else "attmiddle.value"
                conditions = []
                for nom, dt in data_list:
                    # Ici dt est déjà garanti au format YYYY-MM-DD
                    cond = f"({target_col} ILIKE '{nom}' AND att2.value = '{dt}')"
                    conditions.append(cond)
                values_str = "\n      OR ".join(conditions)
            
            if "__LISTE_IDS__" in base_sql:
                final_sql = base_sql.replace("__LISTE_IDS__", values_str)
                final_sql = final_sql.replace("2025-11-30", today_str) 
                
                header = f"/* GENERATED {datetime.now()} | SOURCE: {prefix} | NB: {len(data_list)} */\n"
                out_name = f"{prefix}_REQ_{output_suffix}.sql"
                
                with open(output_dir / out_name, 'w', encoding='utf-8') as f_out: 
                    f_out.write(header + final_sql)
            else:
                print(f"      ❌ Erreur Template {tpl_name} : Balise __LISTE_IDS__ introuvable.")

        except Exception as e:
            print(f"      ❌ Erreur sur {output_suffix} : {e}")
    
    print(f"      ✅ Génération SQL terminée (Vérifiez le dossier Output).")

# --- MAIN ---

def main():
    if not INPUT_DIR.exists(): return
    ns_path, prefix = find_latest_new_s(INPUT_DIR)
    if not ns_path: return
    
    print(f"📂 Préparation pour le flux : {prefix}")
    df_ns = load_csv_robust(ns_path)
    if df_ns is None: return

    # 1. IEHE
    run_iehe_step(df_ns, INPUT_DIR / f"{prefix}_IEHE.csv")

    # 2. SQL
    run_sql_step(df_ns, INPUT_DIR, OUTPUT_DIR, prefix)
    
    print("\n🏁 Etape 1 terminée.")

if __name__ == "__main__":
    main()
