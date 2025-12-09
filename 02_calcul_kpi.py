import pandas as pd
import json
import re
import sys
import warnings
from pathlib import Path
from datetime import datetime

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR if (SCRIPT_DIR / "Input_Data").exists() else SCRIPT_DIR.parent
INPUT_DIR = BASE_DIR / "Input_Data"
OUTPUT_DIR = BASE_DIR / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- UTILITAIRES SIMPLIFIES ---
def fmt_kpi(value, total):
    safe_total = total if total > 0 else 1
    rate = round((value / safe_total * 100), 2)
    return {"Nombre": int(value), "Taux": rate}

def load_csv(path):
    if not path or not path.exists(): return None
    try:
        return pd.read_csv(path, engine='python', dtype=str, on_bad_lines='skip')
    except: return None

def get_col_name(df, candidates):
    if df is None: return None
    for col in candidates:
        if col in df.columns: return col
    return None

def normalize_cols(df):
    if df is not None:
        df.columns = [c.lower().strip().replace('"', '') for c in df.columns]
    return df

def find_latest_file(directory, pattern):
    if not directory.exists(): return None
    candidates = []
    for f in directory.glob(pattern):
        match = re.search(r"(\d{8})", f.name)
        if match: candidates.append({"date": match.group(1), "path": f})
    if not candidates: return None
    best = sorted(candidates, key=lambda x: x["date"], reverse=True)[0]
    return best["path"], best["date"]

# --- MAIN ---
def main():
    # On cherche le fichier résultat généré par le script 03 (NS_CIAM)
    res = find_latest_file(OUTPUT_DIR, "*_NS_CIAM.csv")
    if not res: 
        print("❌ Fichier NS_CIAM introuvable. Lancez le script 03 d'abord.")
        return
    
    ciam_path, prefix = res["path"], res["date"]
    print(f"🚀 Calcul KPI sur : {ciam_path.name}")
    
    df = normalize_cols(load_csv(ciam_path))
    if df is None: return

    # --- 1. FILTRAGE POPULATION (REQ 4) ---
    col_type = get_col_name(df, ['type_assure', 'typeassure', 'code_role_personne', 'role'])
    
    # Total Brut
    vol_brut = len(df)
    
    # Filtre Conjoints
    df_eligible = df.copy()
    if col_type:
        # Exclusion stricte CONJOI
        mask_conjo = df_eligible[col_type].astype(str).str.upper().str.contains('CONJOI')
        df_eligible = df_eligible[~mask_conjo]
        
        # Inclusion Types Éligibles (Si liste fournie, sinon on prend tout sauf conjoints)
        # eligible_types = ['ASSPRI', 'MPRETR', 'ASSAYD']
        # mask_eligible = df_eligible[col_type].astype(str).str.upper().isin(eligible_types)
        # df_eligible = df_eligible[mask_eligible]
    
    vol_eligible = len(df_eligible)
    
    # --- 2. KPI RAPPROCHEMENT ---
    
    col_statut = get_col_name(df_eligible, ['statut_rapprochement', 'indicateur_rapprochement'])
    col_method = get_col_name(df_eligible, ['methode_retenue'])
    
    if col_statut:
        # Rapprochés
        mask_ok = df_eligible[col_statut].astype(str).str.contains('Rapproché', case=False)
        nb_ok = mask_ok.sum()
        nb_ko = vol_eligible - nb_ok
        
        # Détail par méthode
        detail_methodes = df_eligible[mask_ok][col_method].value_counts().to_dict() if col_method else {}
    else:
        nb_ok = 0
        nb_ko = vol_eligible
        detail_methodes = {}

    # --- 3. ANALYSE QUALITE EMAILS ---
    # Comparaison Email Source vs Email CIAM Retrouvé
    col_src = get_col_name(df_eligible, ['valeur_coordonnee', 'email', 'mail'])
    col_res = 'email_ciam'
    
    nb_identique = 0
    if col_src and col_res in df_eligible.columns:
        s_src = df_eligible[col_src].astype(str).str.lower().str.strip()
        s_res = df_eligible[col_res].astype(str).str.lower().str.strip()
        nb_identique = ((s_src == s_res) & (s_src != '') & (s_src != 'nan')).sum()

    # --- SORTIE JSON ---
    kpi_data = {
        "Meta": {"Fichier": str(ciam_path.name), "Date": prefix},
        "Volumetrie": {
            "Total_Lignes_Fichier": vol_brut,
            "Total_Eligibles_Analyses": vol_eligible,
            "Exclus_Conjoints_Autres": vol_brut - vol_eligible
        },
        "Rapprochement_CIAM": {
            "Global": fmt_kpi(nb_ok, vol_eligible),
            "Non_Rapproches": fmt_kpi(nb_ko, vol_eligible),
            "Detail_Par_Methode": detail_methodes
        },
        "Qualite_Donnees": {
            "Email_Identique_Source_Ciam": fmt_kpi(nb_identique, nb_ok)
        }
    }
    
    out_file = OUTPUT_DIR / f"{prefix}_KPI_Resultats.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(kpi_data, f, indent=4, ensure_ascii=False)
    
    print(json.dumps(kpi_data, indent=4, ensure_ascii=False))
    print(f"✅ KPI sauvegardés : {out_file}")

if __name__ == "__main__":
    main()
