import pandas as pd
import numpy as np
import re
import sys
import unicodedata
from pathlib import Path
from datetime import datetime
import warnings

"""
================================================================================
SCRIPT : 03_generation_fichiers_detail.py
DESCRIPTION : Consolidation des retours CIAM et application de la cascade de règles.
              Integre la recherche elargie (Last Name seul / Middle Name seul).

--- NOUSVEAUTES ---
- Normalisation systématique des emails (LOWER)
- Ajout des étapes de matching "Sans Prénom"
- Ajout des colonnes de tracabilité (Email_CM, Email_CK, Email_Rech...)
================================================================================
"""

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR if (SCRIPT_DIR / "Input_Data").exists() else SCRIPT_DIR.parent
INPUT_DIR = BASE_DIR / "Input_Data"
OUTPUT_DIR = BASE_DIR / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- UTILITAIRES ---

def clean_text(text):
    """Nettoie un texte pour le matching (Majuscules, sans accents, sans tirets)."""
    if pd.isna(text) or text == '': return ""
    text = str(text).upper()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^A-Z]', '', text)
    return text

def load_csv(path):
    if not path or not path.exists(): return None
    read_params = {'sep': None, 'engine': 'python', 'dtype': str}
    attempts = [
        {'encoding': 'utf-8', 'skiprows': 0},
        {'encoding': 'latin1', 'skiprows': 0},
        {'encoding': 'utf-8', 'skiprows': 2},
        {'encoding': 'latin1', 'skiprows': 2}
    ]
    for params in attempts:
        current_params = read_params.copy()
        current_params.update(params)
        try:
            df = pd.read_csv(path, **current_params)
            cols = [c.lower() for c in df.columns]
            keywords = ['adhesion', 'assure', 'email', 'personne', 'kpep', 'realm', 'date', 'id', 'refper', 'last_name', 'nom', 'middlename']
            if any(k in c for c in cols for k in keywords):
                return df
        except: continue
    return None

def normalize_cols(df):
    if df is not None:
        df.columns = [c.lower().strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({'nan': '', 'None': ''})
    return df

def get_col_name(df, candidates):
    if df is None: return None
    for col in candidates:
        if col in df.columns: return col
    return None

def find_latest_new_s(directory):
    if not directory.exists(): return None, None
    candidates = []
    excluded_suffixes = ["_IEHE", "_CM", "_CK", "_REQ", "Resultats", "KPI", "NS_CIAM", "NS_IEHE", "Rech_Nom", "Rech_Middle"]
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

# --- MAIN ---

def main():
    if not INPUT_DIR.exists(): return
    ns_path, prefix = find_latest_new_s(INPUT_DIR)
    if not ns_path: 
        print("❌ Aucun fichier source trouvé.")
        return

    print(f"📂 Génération des fichiers détails enrichis : {prefix}")
    
    # 1. Chargement des fichiers
    print("   📥 Chargement des données...")
    df_ns = normalize_cols(load_csv(ns_path))
    
    df_iehe = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_IEHE.csv"))
    df_cm = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_CM.csv"))
    df_ck = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_CK.csv"))
    df_nom = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_Rech_Nom.csv"))
    df_middle = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_Rech_Middle.csv"))

    if df_ns is None: 
        print("   ❌ Fichier New_S invalide ou vide.")
        return

    # --- 2. PREPARATION DONNEES POUR MATCHING ---
    print("   ⚙️  Normalisation et Création des Clés...")
    
    # Identification Colonnes NS
    col_mail_ciam = get_col_name(df_ns, ['mailciam', 'mail ciam', 'mail_ciam'])
    col_val_coord = get_col_name(df_ns, ['valeur_coordonnee', 'valeur coordonnee', 'mail'])
    col_kpep = get_col_name(df_ns, ['idkpep', 'kpep'])
    col_nom = get_col_name(df_ns, ['nom_long', 'nom', 'lastname'])
    col_prenom = get_col_name(df_ns, ['prenom', 'firstname'])
    col_dnaiss = get_col_name(df_ns, ['date_naissance', 'datenaissance', 'birthdate'])

    df_work = df_ns.copy()
    
    # Clés Techniques NS (Lower Case)
    df_work['key_mail'] = df_work[col_mail_ciam].str.lower().str.strip().replace('', np.nan) if col_mail_ciam else np.nan
    df_work['key_val'] = df_work[col_val_coord].str.lower().str.strip().replace('', np.nan) if col_val_coord else np.nan
    df_work['key_kpep'] = df_work[col_kpep].replace('', np.nan) if col_kpep else np.nan
    
    # Clés Identité NS
    df_work['norm_nom'] = df_work[col_nom].apply(clean_text) if col_nom else ""
    df_work['norm_prenom'] = df_work[col_prenom].apply(clean_text) if col_prenom else ""
    
    # Clé Nom+Date (Pour Recherche Élargie)
    if col_dnaiss:
        # On ne garde que les 10 premiers chars pour YYYY-MM-DD
        df_work['dt_fmt'] = pd.to_datetime(df_work[col_dnaiss], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
        df_work['key_nom_date'] = (df_work['norm_nom'] + "|" + df_work['dt_fmt']).replace(r'^\|.*$', np.nan, regex=True).replace(r'.*\|$', np.nan, regex=True)
    else:
        df_work['key_nom_date'] = np.nan

    # Clé Identité Complète (Nom+Prenom+Date)
    df_work['key_identite_full'] = (df_work['norm_nom'] + "|" + df_work['norm_prenom'] + "|" + df_work['dt_fmt']).replace(r'\|\|', np.nan, regex=True)
    # Clé Identité Simple (Nom+Prenom)
    df_work['key_identite_simple'] = (df_work['norm_nom'] + "|" + df_work['norm_prenom']).replace('|', np.nan)

    # --- PREPARATION DES REFERENTIELS (LOOKUP TABLES) ---
    
    def prep_ref(df_source, source_label):
        if df_source is None or df_source.empty: return pd.DataFrame()
        temp = pd.DataFrame()
        temp['realm_id'] = df_source['realm_id'] if 'realm_id' in df_source.columns else np.nan
        temp['email'] = df_source['email'] if 'email' in df_source.columns else ""
        temp['idkpep'] = df_source['idkpep'] if 'idkpep' in df_source.columns else ""
        
        # Clés
        c_nom = get_col_name(df_source, ['last_name', 'lastname', 'nom', 'middlename']) # Middlename treated as Lastname for matching logic
        c_pnom = get_col_name(df_source, ['first_name', 'firstname', 'prenom'])
        c_dn = get_col_name(df_source, ['birthdate', 'date_naissance'])
        
        temp['norm_nom'] = df_source[c_nom].apply(clean_text) if c_nom else ""
        temp['norm_prenom'] = df_source[c_pnom].apply(clean_text) if c_pnom else ""
        
        temp['key_email'] = temp['email'].str.lower().str.strip().replace('', np.nan)
        temp['key_kpep'] = temp['idkpep'].replace('', np.nan)
        
        if c_dn:
             dt_str = pd.to_datetime(df_source[c_dn], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
             temp['key_identite_full'] = (temp['norm_nom'] + "|" + temp['norm_prenom'] + "|" + dt_str)
             temp['key_nom_date'] = (temp['norm_nom'] + "|" + dt_str) # Clé sans Prénom
        else:
             temp['key_identite_full'] = np.nan
             temp['key_nom_date'] = np.nan
             
        temp['key_identite_simple'] = (temp['norm_nom'] + "|" + temp['norm_prenom'])
        
        # Ajout Label Source
        temp['source_file'] = source_label
        return temp

    # On prépare chaque source indépendamment
    ref_cm = prep_ref(df_cm, 'CM')
    ref_ck = prep_ref(df_ck, 'CK')
    ref_nom = prep_ref(df_nom, 'Rech_Nom')
    ref_middle = prep_ref(df_middle, 'Rech_Middle')

    # Création des index de matching
    # Note : drop_duplicates pour éviter l'explosion du nombre de lignes
    
    # 1. Email (CM est la référence prioritaire)
    idx_email = ref_cm.dropna(subset=['key_email']).drop_duplicates('key_email').set_index('key_email')
    
    # 2. KPEP (CK est la référence)
    idx_kpep = ref_ck.dropna(subset=['key_kpep']).drop_duplicates('key_kpep').set_index('key_kpep')
    
    # 3. Identité Full (Concaténation de tout pour max chance)
    all_refs = pd.concat([ref_cm, ref_ck, ref_nom, ref_middle], ignore_index=True)
    idx_identite_full = all_refs.dropna(subset=['key_identite_full']).drop_duplicates('key_identite_full').set_index('key_identite_full')
    idx_identite_simple = all_refs.dropna(subset=['key_identite_simple']).drop_duplicates('key_identite_simple').set_index('key_identite_simple')

    # 4. Indexes Spécifiques "Sans Prénom"
    idx_rech_nom = ref_nom.dropna(subset=['key_nom_date']).drop_duplicates('key_nom_date').set_index('key_nom_date')
    idx_rech_middle = ref_middle.dropna(subset=['key_nom_date']).drop_duplicates('key_nom_date').set_index('key_nom_date')

    # --- 3. EXECUTION DES MATCHINGS ---
    print("   🔍 Exécution des rapprochements...")

    # A. Récupération des données brutes pour les colonnes de contrôle (REQ 3)
    # On fait des merges simples pour récupérer l'email de chaque source si dispo
    m_cm = df_work.merge(idx_email[['email']], left_on='key_val', right_index=True, how='left').rename(columns={'email': 'Email_CM'})
    m_ck = df_work.merge(idx_kpep[['email']], left_on='key_kpep', right_index=True, how='left').rename(columns={'email': 'Email_CK'})
    m_rn = df_work.merge(idx_rech_nom[['email']], left_on='key_nom_date', right_index=True, how='left').rename(columns={'email': 'Email_Rech_LastName'})
    m_rm = df_work.merge(idx_rech_middle[['email']], left_on='key_nom_date', right_index=True, how='left').rename(columns={'email': 'Email_Rech_Middle'})

    # Intégration des colonnes de contrôle dans df_work
    df_work['Email_CM'] = m_cm['Email_CM']
    df_work['Email_CK'] = m_ck['Email_CK']
    df_work['Email_Rech_LastName'] = m_rn['Email_Rech_LastName']
    df_work['Email_Rech_Middle'] = m_rm['Email_Rech_Middle']
    
    # Colonne aggrégée pour les recherches "Sans Prénom"
    df_work['Email_Rech_SansPrenom'] = df_work['Email_Rech_LastName'].combine_first(df_work['Email_Rech_Middle'])

    # B. Waterfall de Décision (Priorités)
    
    # Initialisation
    df_work['Indicateur_Rapprochement'] = 'CIAM_NON_TROUVE'
    df_work['Methode_Retenue'] = 'AUCUNE'
    df_work['CIAM_Email_Cible'] = np.nan
    df_work['CIAM_Societe'] = np.nan

    # 1. Matching Identité Simple (Priorité Basse - Écrasé par la suite si mieux)
    match = df_work.merge(idx_identite_simple, left_on='key_identite_simple', right_index=True, how='left', suffixes=('', '_m'))
    mask = match['realm_id'].notna()
    df_work.loc[mask, 'Indicateur_Rapprochement'] = 'CIAM_TROUVE_IDENTITE_FAIBLE'
    df_work.loc[mask, 'Methode_Retenue'] = 'Nom+Prenom'
    df_work.loc[mask, 'CIAM_Email_Cible'] = match.loc[mask, 'email']
    df_work.loc[mask, 'CIAM_Societe'] = match.loc[mask, 'realm_id']

    # 2. Matching Identité Complète (Moyenne)
    match = df_work.merge(idx_identite_full, left_on='key_identite_full', right_index=True, how='left', suffixes=('', '_m'))
    mask = match['realm_id'].notna()
    df_work.loc[mask, 'Indicateur_Rapprochement'] = 'CIAM_TROUVE_IDENTITE'
    df_work.loc[mask, 'Methode_Retenue'] = 'Nom+Prenom+Date'
    df_work.loc[mask, 'CIAM_Email_Cible'] = match.loc[mask, 'email']
    df_work.loc[mask, 'CIAM_Societe'] = match.loc[mask, 'realm_id']
    
    # 3. KPEP (Forte)
    match = df_work.merge(idx_kpep, left_on='key_kpep', right_index=True, how='left', suffixes=('', '_m'))
    mask = match['realm_id'].notna()
    df_work.loc[mask, 'Indicateur_Rapprochement'] = 'CIAM_TROUVE_KPEP'
    df_work.loc[mask, 'Methode_Retenue'] = 'KPEP'
    df_work.loc[mask, 'CIAM_Email_Cible'] = match.loc[mask, 'email']
    df_work.loc[mask, 'CIAM_Societe'] = match.loc[mask, 'realm_id']

    # 4. Email Val Coord (Très Forte)
    match = df_work.merge(idx_email, left_on='key_val', right_index=True, how='left', suffixes=('', '_m'))
    mask = match['realm_id'].notna()
    df_work.loc[mask, 'Indicateur_Rapprochement'] = 'CIAM_TROUVE_EMAIL'
    df_work.loc[mask, 'Methode_Retenue'] = 'Val_Coord'
    df_work.loc[mask, 'CIAM_Email_Cible'] = match.loc[mask, 'email']
    df_work.loc[mask, 'CIAM_Societe'] = match.loc[mask, 'realm_id']

    # 5. Mail CIAM (Maximale)
    match = df_work.merge(idx_email, left_on='key_mail', right_index=True, how='left', suffixes=('', '_m'))
    mask = match['realm_id'].notna()
    df_work.loc[mask, 'Indicateur_Rapprochement'] = 'CIAM_TROUVE_EMAIL'
    df_work.loc[mask, 'Methode_Retenue'] = 'Mail_CIAM'
    df_work.loc[mask, 'CIAM_Email_Cible'] = match.loc[mask, 'email']
    df_work.loc[mask, 'CIAM_Societe'] = match.loc[mask, 'realm_id']

    # --- ETAPE SPECIALE : RECHERCHE ELARGIE (Sur le reliquat uniquement) ---
    # On applique ces règles SEULEMENT si non rapproché par les étapes standards (1-5)
    
    mask_reliquat = (df_work['Indicateur_Rapprochement'] == 'CIAM_NON_TROUVE')
    
    # 6. Recherche Last Name sans Prénom
    # Merge sur key_nom_date (Nom + Date) avec ref_nom
    match_ln = df_work[mask_reliquat].merge(idx_rech_nom, left_on='key_nom_date', right_index=True, how='left')
    mask_found_ln = match_ln['realm_id'].notna()
    
    # Application sur le DataFrame principal via l'index
    idx_to_update = df_work[mask_reliquat][mask_found_ln].index
    df_work.loc[idx_to_update, 'Indicateur_Rapprochement'] = 'Rapproché_LastName_SansPrenom'
    df_work.loc[idx_to_update, 'Methode_Retenue'] = 'LastName_SansPrenom'
    df_work.loc[idx_to_update, 'CIAM_Email_Cible'] = match_ln.loc[mask_found_ln, 'email']
    df_work.loc[idx_to_update, 'CIAM_Societe'] = match_ln.loc[mask_found_ln, 'realm_id']

    # Mise à jour du masque reliquat (ceux qui restent encore non trouvés)
    mask_reliquat = (df_work['Indicateur_Rapprochement'] == 'CIAM_NON_TROUVE')

    # 7. Recherche Middle Name sans Prénom
    match_mn = df_work[mask_reliquat].merge(idx_rech_middle, left_on='key_nom_date', right_index=True, how='left')
    mask_found_mn = match_mn['realm_id'].notna()
    
    idx_to_update_mn = df_work[mask_reliquat][mask_found_mn].index
    df_work.loc[idx_to_update_mn, 'Indicateur_Rapprochement'] = 'Rapproché_MiddleName_SansPrenom'
    df_work.loc[idx_to_update_mn, 'Methode_Retenue'] = 'MiddleName_SansPrenom'
    df_work.loc[idx_to_update_mn, 'CIAM_Email_Cible'] = match_mn.loc[mask_found_mn, 'email']
    df_work.loc[idx_to_update_mn, 'CIAM_Societe'] = match_mn.loc[mask_found_mn, 'realm_id']

    # --- FINALISATION ---
    print("   📝 Finalisation du fichier NS_CIAM...")
    
    df_work['Email_CIAM'] = df_work['CIAM_Email_Cible']
    df_work['Statut_Rapprochement'] = df_work['Indicateur_Rapprochement'].apply(
        lambda x: 'Non Rapproché' if x == 'CIAM_NON_TROUVE' else 'Rapproché'
    )

    # Nettoyage
    cols_drop = [c for c in df_work.columns if c.startswith('key_') or c.startswith('norm_')]
    df_work.drop(columns=cols_drop, inplace=True, errors='ignore')

    f_ciam = OUTPUT_DIR / f"{prefix}_NS_CIAM.csv"
    df_work.to_csv(f_ciam, index=False, sep=',', encoding='utf-8-sig')
    print(f"   ✅ Fichier généré : {f_ciam.name}")

    # NS_IEHE (inchangé, juste pour complétude)
    col_id_ns = get_col_name(df_ns, ['num_personne', 'numpersonne'])
    if df_iehe is not None and col_id_ns and 'refperboccn' in df_iehe.columns:
        df_iehe_ref = df_iehe.drop_duplicates(subset=['refperboccn']).set_index('refperboccn')
        df_merged = df_ns.merge(df_iehe_ref, left_on=col_id_ns, right_index=True, how='left', suffixes=('', '_iehe'))
        df_merged['IEHE_Present'] = df_merged.get('idrpp', pd.Series()).notna().map({True: 'OUI', False: 'NON'})
        df_merged.to_csv(OUTPUT_DIR / f"{prefix}_NS_IEHE.csv", index=False, sep=',', encoding='utf-8-sig')

if __name__ == "__main__":
    main()
