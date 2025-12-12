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

--- CORRECTIFS & NOUVEAUTES ---
- Exclusion stricte des Conjoints (type_assure = CONJOI) du fichier de sortie CIAM.
- Maintien des Conjoints dans le fichier de sortie IEHE.
- Ajout de la colonne 'Email_Other' récupérée depuis les référentiels (CK, CM...).
- ENRICHISSEMENT MAXIMAL : Ajout Nom, Prenom, Date, Origine, ID Technique, Telephone, Type Event.
- STRUCTURE COLONNES AUDIT : Ajout des blocs CM_, CK_, RechNom_, RechMiddle_ pour traçabilité complète.
- BONUS : Ajout Source_Match_Email et Match_Status.
- Gestion des types (Object) pour éviter les warnings.
- [FIX] Gestion robuste des fichiers vides pour éviter les KeyError sur key_nom_date.
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
            # Petit nettoyage preventif si le fichier est vide ou presque
            if df.empty:
                return df
            
            cols = [c.lower() for c in df.columns]
            keywords = ['adhesion', 'assure', 'email', 'personne', 'kpep', 'realm', 'date', 'id', 'refper', 'last_name', 'nom', 'middlename']
            if any(k in c for c in cols for k in keywords):
                return df
        except: continue
    return None

def normalize_cols(df):
    if df is not None and not df.empty:
        df.columns = [c.lower().strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({'nan': '', 'None': ''})
    return df

def get_col_name(df, candidates):
    if df is None or df.empty: return None
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

def get_merge_col(df, base_col, suffix='_m'):
    """Helper pour récupérer la colonne fusionnée."""
    if f"{base_col}{suffix}" in df.columns:
        return f"{base_col}{suffix}"
    elif base_col in df.columns:
        return base_col
    return None

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
    
    # Chargement avec tolerance aux fichiers manquants (renvoie None ou Empty DF)
    df_iehe = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_IEHE.csv"))
    df_cm = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_CM.csv"))
    df_ck = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_CK.csv"))
    df_nom = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_Rech_Nom.csv"))
    df_middle = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_Rech_Middle.csv"))

    if df_ns is None or df_ns.empty: 
        print("   ❌ Fichier New_S invalide ou vide.")
        return

    # --- 1.0 SAUVEGARDE DATAFRAME COMPLET POUR IEHE ---
    # On garde une copie intacte (avec conjoints) pour la génération IEHE à la fin
    df_ns_full = df_ns.copy()

    # --- 1.1 FILTRAGE POPULATION (CONJOI) POUR CIAM ---
    col_type = get_col_name(df_ns, ['type_assure', 'typeassure', 'code_role_personne', 'role'])
    if col_type:
        mask_conjoi = df_ns[col_type].astype(str).str.upper().str.strip() == 'CONJOI'
        nb_excluded = mask_conjoi.sum()
        if nb_excluded > 0:
            print(f"   🚫 Exclusion de {nb_excluded} ligne(s) 'CONJOI' du traitement CIAM.")
            df_ns = df_ns[~mask_conjoi].copy()
        else:
            print("   ℹ️  Aucun conjoint détecté.")
    
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_work['dt_fmt'] = pd.to_datetime(df_work[col_dnaiss], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
        
        df_work['key_nom_date'] = (df_work['norm_nom'] + "|" + df_work['dt_fmt']).replace(r'^\|.*$', np.nan, regex=True).replace(r'.*\|$', np.nan, regex=True)
        df_work['key_identite_full'] = (df_work['norm_nom'] + "|" + df_work['norm_prenom'] + "|" + df_work['dt_fmt']).replace(r'\|\|', np.nan, regex=True)
    else:
        df_work['dt_fmt'] = ""
        df_work['key_nom_date'] = np.nan
        df_work['key_identite_full'] = np.nan

    df_work['key_identite_simple'] = (df_work['norm_nom'] + "|" + df_work['norm_prenom']).replace('|', np.nan)

    # --- PREPARATION DES REFERENTIELS (LOOKUP TABLES) ---
    
    def prep_ref(df_source, source_label):
        if df_source is None or df_source.empty: return pd.DataFrame()
        temp = pd.DataFrame()
        
        # --- Mapping des colonnes CIAM (Enrichissement) ---
        temp['realm_id'] = df_source.get('realm_id', np.nan)
        temp['email'] = df_source.get('email', "")
        temp['idkpep'] = df_source.get('idkpep', "")
        temp['email_other'] = df_source.get('email_other', "")
        
        # Données de Fiabilisation / KPI
        temp['first_name'] = df_source.get('first_name', "")
        temp['last_name'] = df_source.get('last_name', "")
        temp['date_evt'] = df_source.get('date_evt', "") # Date dernière action
        temp['origincreation'] = df_source.get('origincreation', "") # Canal
        
        # === NOUVEAUX CHAMPS ===
        temp['id'] = df_source.get('id', "") # CIAM ID Technique
        temp['phonenumber'] = df_source.get('phonenumber', "") # Telephone
        temp['type'] = df_source.get('type', "") # Type Event (Login/Creation...)
        # =======================

        # Clés de Matching
        c_nom = get_col_name(df_source, ['last_name', 'lastname', 'nom', 'middlename'])
        c_pnom = get_col_name(df_source, ['first_name', 'firstname', 'prenom'])
        c_dn = get_col_name(df_source, ['birthdate', 'date_naissance'])
        
        temp['norm_nom'] = df_source[c_nom].apply(clean_text) if c_nom else ""
        temp['norm_prenom'] = df_source[c_pnom].apply(clean_text) if c_pnom else ""
        
        temp['key_email'] = temp['email'].str.lower().str.strip().replace('', np.nan)
        temp['key_kpep'] = temp['idkpep'].replace('', np.nan)
        
        if c_dn:
             with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt_str = pd.to_datetime(df_source[c_dn], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
             temp['birthdate'] = dt_str # Stockage au format propre
             temp['key_identite_full'] = (temp['norm_nom'] + "|" + temp['norm_prenom'] + "|" + dt_str)
             temp['key_nom_date'] = (temp['norm_nom'] + "|" + dt_str) 
        else:
             temp['birthdate'] = ""
             temp['key_identite_full'] = np.nan
             temp['key_nom_date'] = np.nan
             
        temp['key_identite_simple'] = (temp['norm_nom'] + "|" + temp['norm_prenom'])
        temp['source_file'] = source_label
        
        return temp

    # On prépare chaque source
    ref_cm = prep_ref(df_cm, 'CM')
    ref_ck = prep_ref(df_ck, 'CK')
    ref_nom = prep_ref(df_nom, 'Rech_Nom')
    ref_middle = prep_ref(df_middle, 'Rech_Middle')

    # Création des index de matching (pour le waterfall)
    idx_email = pd.DataFrame()
    if not ref_cm.empty and 'key_email' in ref_cm.columns:
        idx_email = ref_cm.dropna(subset=['key_email']).drop_duplicates('key_email').set_index('key_email')
        
    idx_kpep = pd.DataFrame()
    if not ref_ck.empty and 'key_kpep' in ref_ck.columns:
        idx_kpep = ref_ck.dropna(subset=['key_kpep']).drop_duplicates('key_kpep').set_index('key_kpep')
    
    all_refs = pd.concat([ref_cm, ref_ck, ref_nom, ref_middle], ignore_index=True)
    
    idx_identite_full = pd.DataFrame()
    if not all_refs.empty and 'key_identite_full' in all_refs.columns:
        idx_identite_full = all_refs.dropna(subset=['key_identite_full']).drop_duplicates('key_identite_full').set_index('key_identite_full')

    idx_identite_simple = pd.DataFrame()
    if not all_refs.empty and 'key_identite_simple' in all_refs.columns:
        idx_identite_simple = all_refs.dropna(subset=['key_identite_simple']).drop_duplicates('key_identite_simple').set_index('key_identite_simple')

    idx_rech_nom = pd.DataFrame()
    if not ref_nom.empty and 'key_nom_date' in ref_nom.columns:
        idx_rech_nom = ref_nom.dropna(subset=['key_nom_date']).drop_duplicates('key_nom_date').set_index('key_nom_date')

    idx_rech_middle = pd.DataFrame()
    if not ref_middle.empty and 'key_nom_date' in ref_middle.columns:
        idx_rech_middle = ref_middle.dropna(subset=['key_nom_date']).drop_duplicates('key_nom_date').set_index('key_nom_date')

    # --- 3. EXECUTION DES MATCHINGS ---
    print("   🔍 Exécution des rapprochements...")

    # =========================================================================
    # A. ENRICHISSEMENT SPECIFIQUE (TRAÇABILITÉ)
    # =========================================================================
    
    # Initialisation des colonnes de sortie
    audit_cols = [
        'CM_Email', 'CM_KPEP',
        'CK_Email', 'CK_KPEP',
        'RechMiddle_Email', 'RechMiddle_Nom', 'RechMiddle_Prenom', 'RechMiddle_DateNaissance', 'RechMiddle_KPEP',
        'RechNom_Email', 'RechNom_Nom', 'RechNom_Prenom', 'RechNom_DateNaissance', 'RechNom_KPEP',
        'Source_Match_Email', 'Match_Status', 'Email_CIAM_Cible'
    ]
    for ac in audit_cols:
        df_work[ac] = np.nan
        df_work[ac] = df_work[ac].astype(object)

    # 1) BLOC CM (Matching par Email)
    # On utilise key_val (valeur_coordonnee) comme clé primaire, sinon key_mail (mail_ciam)
    # Dans ref_cm, la clé unique fiable est key_email.
    if not ref_cm.empty:
        # On tente le merge sur key_val (prioritaire)
        temp_cm = df_work[['key_val']].merge(
            ref_cm[['key_email', 'email', 'idkpep']].drop_duplicates('key_email'), 
            left_on='key_val', right_on='key_email', how='left'
        )
        df_work['CM_Email'] = temp_cm['email']
        df_work['CM_KPEP'] = temp_cm['idkpep']

    # 2) BLOC CK (Matching par KPEP)
    if not ref_ck.empty:
        temp_ck = df_work[['key_kpep']].merge(
            ref_ck[['key_kpep', 'email', 'idkpep']].drop_duplicates('key_kpep'), 
            left_on='key_kpep', right_on='key_kpep', how='left'
        )
        df_work['CK_Email'] = temp_ck['email']
        df_work['CK_KPEP'] = temp_ck['idkpep']

    # 3) BLOC Rech_Middle (Matching Identité)
    if not ref_middle.empty:
        # On ne garde que ceux qui ont une clé identité valide
        rm_clean = ref_middle.dropna(subset=['key_nom_date']).drop_duplicates('key_nom_date')
        temp_rm = df_work[['key_nom_date']].merge(
            rm_clean[['key_nom_date', 'email', 'last_name', 'first_name', 'birthdate', 'idkpep']],
            left_on='key_nom_date', right_on='key_nom_date', how='left'
        )
        df_work['RechMiddle_Email'] = temp_rm['email']
        df_work['RechMiddle_Nom'] = temp_rm['last_name']
        df_work['RechMiddle_Prenom'] = temp_rm['first_name']
        df_work['RechMiddle_DateNaissance'] = temp_rm['birthdate']
        df_work['RechMiddle_KPEP'] = temp_rm['idkpep']

    # 4) BLOC Rech_Nom (Matching Identité)
    if not ref_nom.empty:
        rn_clean = ref_nom.dropna(subset=['key_nom_date']).drop_duplicates('key_nom_date')
        temp_rn = df_work[['key_nom_date']].merge(
            rn_clean[['key_nom_date', 'email', 'last_name', 'first_name', 'birthdate', 'idkpep']],
            left_on='key_nom_date', right_on='key_nom_date', how='left'
        )
        df_work['RechNom_Email'] = temp_rn['email']
        df_work['RechNom_Nom'] = temp_rn['last_name']
        df_work['RechNom_Prenom'] = temp_rn['first_name']
        df_work['RechNom_DateNaissance'] = temp_rn['birthdate']
        df_work['RechNom_KPEP'] = temp_rn['idkpep']

    # 5) CALCUL BONUS (Source Match & Status)
    def compute_source_list(row):
        srcs = []
        if pd.notna(row['CM_Email']) and row['CM_Email'] != '': srcs.append('CM')
        if pd.notna(row['CK_Email']) and row['CK_Email'] != '': srcs.append('CK')
        if pd.notna(row['RechNom_Email']) and row['RechNom_Email'] != '': srcs.append('Rech_Nom')
        if pd.notna(row['RechMiddle_Email']) and row['RechMiddle_Email'] != '': srcs.append('Rech_Middle')
        return ', '.join(srcs) if srcs else 'NULL'

    df_work['Source_Match_Email'] = df_work.apply(compute_source_list, axis=1)

    def compute_status(row):
        src = row['Source_Match_Email']
        if src == 'NULL': return 'NO_MATCH'
        
        # Vérification Cohérence KPEP (Si dispo dans New_S et dans CK)
        if pd.notna(row['key_kpep']) and pd.notna(row['CK_KPEP']):
            if str(row['key_kpep']) != str(row['CK_KPEP']):
                return 'INCOHERENT_KPEP'
        
        return 'MATCH_OK'

    df_work['Match_Status'] = df_work.apply(compute_status, axis=1)

    # =========================================================================
    # B. WATERFALL DE DECISION (Priorités existantes)
    # =========================================================================
    
    df_work['Indicateur_Rapprochement'] = 'CIAM_NON_TROUVE'
    df_work['Methode_Retenue'] = 'AUCUNE'
    
    # Init colonnes cibles (Object)
    target_cols = [
        'CIAM_Societe', 'CIAM_Email_Other', 
        'CIAM_Nom', 'CIAM_Prenom', 'CIAM_Date_Evt', 'CIAM_Origine', 
        'CIAM_KPEP_Trouve', 'CIAM_ID_Technique', 'CIAM_Telephone', 'CIAM_Type_Event'
    ]
    # Note: CIAM_Email_Cible est déjà dans audit_cols
    
    for col in target_cols:
        df_work[col] = np.nan
        df_work[col] = df_work[col].astype(object)

    # Fonction locale pour appliquer le match et enrichir
    def apply_match(match_df, method_name, status):
        col_realm = get_merge_col(match_df, 'realm_id', '_m')
        
        # Colonnes additionnelles à récupérer
        c_email = get_merge_col(match_df, 'email', '_m')
        c_other = get_merge_col(match_df, 'email_other', '_m')
        c_nom = get_merge_col(match_df, 'last_name', '_m')
        c_prenom = get_merge_col(match_df, 'first_name', '_m')
        c_evt = get_merge_col(match_df, 'date_evt', '_m')
        c_orig = get_merge_col(match_df, 'origincreation', '_m')
        c_kpep = get_merge_col(match_df, 'idkpep', '_m')
        c_id = get_merge_col(match_df, 'id', '_m')
        c_phone = get_merge_col(match_df, 'phonenumber', '_m')
        c_type = get_merge_col(match_df, 'type', '_m')
        
        if col_realm:
            mask = match_df[col_realm].notna()
            if mask.any():
                df_work.loc[mask, 'Indicateur_Rapprochement'] = status
                df_work.loc[mask, 'Methode_Retenue'] = method_name
                df_work.loc[mask, 'CIAM_Societe'] = match_df.loc[mask, col_realm]
                
                # Enrichissement
                if c_email: df_work.loc[mask, 'CIAM_Email_Cible'] = match_df.loc[mask, c_email]
                if c_other: df_work.loc[mask, 'CIAM_Email_Other'] = match_df.loc[mask, c_other]
                if c_nom: df_work.loc[mask, 'CIAM_Nom'] = match_df.loc[mask, c_nom]
                if c_prenom: df_work.loc[mask, 'CIAM_Prenom'] = match_df.loc[mask, c_prenom]
                if c_evt: df_work.loc[mask, 'CIAM_Date_Evt'] = match_df.loc[mask, c_evt]
                if c_orig: df_work.loc[mask, 'CIAM_Origine'] = match_df.loc[mask, c_orig]
                if c_kpep: df_work.loc[mask, 'CIAM_KPEP_Trouve'] = match_df.loc[mask, c_kpep]
                if c_id: df_work.loc[mask, 'CIAM_ID_Technique'] = match_df.loc[mask, c_id]
                if c_phone: df_work.loc[mask, 'CIAM_Telephone'] = match_df.loc[mask, c_phone]
                if c_type: df_work.loc[mask, 'CIAM_Type_Event'] = match_df.loc[mask, c_type]

    # 1. Matching Identité Simple (Faible)
    if not idx_identite_simple.empty:
        m = df_work.merge(idx_identite_simple, left_on='key_identite_simple', right_index=True, how='left', suffixes=('', '_m'))
        apply_match(m, 'Nom+Prenom', 'CIAM_TROUVE_IDENTITE_FAIBLE')

    # 2. Matching Identité Complète (Moyenne)
    if not idx_identite_full.empty:
        m = df_work.merge(idx_identite_full, left_on='key_identite_full', right_index=True, how='left', suffixes=('', '_m'))
        apply_match(m, 'Nom+Prenom+Date', 'CIAM_TROUVE_IDENTITE')
    
    # 3. KPEP (Forte)
    if not idx_kpep.empty:
        m = df_work.merge(idx_kpep, left_on='key_kpep', right_index=True, how='left', suffixes=('', '_m'))
        apply_match(m, 'KPEP', 'CIAM_TROUVE_KPEP')

    # 4. Email Val Coord (Très Forte)
    if not idx_email.empty:
        m = df_work.merge(idx_email, left_on='key_val', right_index=True, how='left', suffixes=('', '_m'))
        apply_match(m, 'Val_Coord', 'CIAM_TROUVE_EMAIL')

    # 5. Mail CIAM (Maximale)
    if not idx_email.empty:
        m = df_work.merge(idx_email, left_on='key_mail', right_index=True, how='left', suffixes=('', '_m'))
        apply_match(m, 'Mail_CIAM', 'CIAM_TROUVE_EMAIL')

    # --- ETAPE SPECIALE : RECHERCHE ELARGIE (Sur le reliquat) ---
    
    mask_reliquat = (df_work['Indicateur_Rapprochement'] == 'CIAM_NON_TROUVE')
    
    # 6. Recherche Last Name sans Prénom
    if mask_reliquat.any() and not idx_rech_nom.empty:
        m_ln = df_work[mask_reliquat].merge(idx_rech_nom, left_on='key_nom_date', right_index=True, how='left', suffixes=('', '_m'))
        col_realm = get_merge_col(m_ln, 'realm_id', '_m')
        
        # Helper manuel pour reliquat
        def apply_reliquat(match_df, mask_f, method_n, status_n):
            idx_found = df_work[mask_reliquat][mask_f].index
            df_work.loc[idx_found, 'Indicateur_Rapprochement'] = status_n
            df_work.loc[idx_found, 'Methode_Retenue'] = method_n
            df_work.loc[idx_found, 'CIAM_Societe'] = match_df.loc[mask_f, col_realm]
            
            c_list = [('email', 'CIAM_Email_Cible'), ('email_other', 'CIAM_Email_Other'),
                      ('last_name', 'CIAM_Nom'), ('first_name', 'CIAM_Prenom'),
                      ('date_evt', 'CIAM_Date_Evt'), ('origincreation', 'CIAM_Origine'),
                      ('idkpep', 'CIAM_KPEP_Trouve'), ('id', 'CIAM_ID_Technique'),
                      ('phonenumber', 'CIAM_Telephone'), ('type', 'CIAM_Type_Event')]
            
            for c_src, c_dst in c_list:
                c_m = get_merge_col(match_df, c_src, '_m')
                if c_m: df_work.loc[idx_found, c_dst] = match_df.loc[mask_f, c_m]

        if col_realm:
            mask_found = m_ln[col_realm].notna()
            if mask_found.any():
                apply_reliquat(m_ln, mask_found, 'LastName_SansPrenom', 'Rapproché_LastName_SansPrenom')

    # Mise à jour reliquat
    mask_reliquat = (df_work['Indicateur_Rapprochement'] == 'CIAM_NON_TROUVE')

    # 7. Recherche Middle Name sans Prénom
    if mask_reliquat.any() and not idx_rech_middle.empty:
        m_mn = df_work[mask_reliquat].merge(idx_rech_middle, left_on='key_nom_date', right_index=True, how='left', suffixes=('', '_m'))
        col_realm = get_merge_col(m_mn, 'realm_id', '_m')
        
        if col_realm:
            mask_found = m_mn[col_realm].notna()
            if mask_found.any():
                idx_found = df_work[mask_reliquat][mask_found].index
                df_work.loc[idx_found, 'Indicateur_Rapprochement'] = 'Rapproché_MiddleName_SansPrenom'
                df_work.loc[idx_found, 'Methode_Retenue'] = 'MiddleName_SansPrenom'
                df_work.loc[idx_found, 'CIAM_Societe'] = m_mn.loc[mask_found, col_realm]
                
                c_list = [('email', 'CIAM_Email_Cible'), ('email_other', 'CIAM_Email_Other'),
                      ('last_name', 'CIAM_Nom'), ('first_name', 'CIAM_Prenom'),
                      ('date_evt', 'CIAM_Date_Evt'), ('origincreation', 'CIAM_Origine'),
                      ('idkpep', 'CIAM_KPEP_Trouve'), ('id', 'CIAM_ID_Technique'),
                      ('phonenumber', 'CIAM_Telephone'), ('type', 'CIAM_Type_Event')]
            
                for c_src, c_dst in c_list:
                    c_m = get_merge_col(m_mn, c_src, '_m')
                    if c_m: df_work.loc[idx_found, c_dst] = m_mn.loc[mask_found, c_m]

    # --- FINALISATION ---
    print("   📝 Finalisation du fichier NS_CIAM...")
    
    # On popule Email_CIAM_Cible si la waterfall l'a trouvé mais pas le reliquat (redondance sécu)
    # Dans la logique ci-dessus, CIAM_Email_Cible est rempli par apply_match.
    
    df_work['Email_CIAM'] = df_work['CIAM_Email_Cible']
    df_work['Email_Other'] = df_work['CIAM_Email_Other']
    
    df_work['Statut_Rapprochement'] = df_work['Indicateur_Rapprochement'].apply(
        lambda x: 'Non Rapproché' if x == 'CIAM_NON_TROUVE' else 'Rapproché'
    )

    # Nettoyage colonnes techniques
    cols_drop = [c for c in df_work.columns if c.startswith('key_') or c.startswith('norm_') or c == 'dt_fmt']
    df_work.drop(columns=cols_drop, inplace=True, errors='ignore')

    f_ciam = OUTPUT_DIR / f"{prefix}_NS_CIAM.csv"
    df_work.to_csv(f_ciam, index=False, sep=',', encoding='utf-8-sig')
    print(f"   ✅ Fichier généré : {f_ciam.name}")

    # NS_IEHE
    col_id_ns = get_col_name(df_ns_full, ['num_personne', 'numpersonne'])
    if df_iehe is not None and not df_iehe.empty and col_id_ns and 'refperboccn' in df_iehe.columns:
        df_iehe_ref = df_iehe.drop_duplicates(subset=['refperboccn']).set_index('refperboccn')
        df_merged = df_ns_full.merge(df_iehe_ref, left_on=col_id_ns, right_index=True, how='left', suffixes=('', '_iehe'))
        df_merged['IEHE_Present'] = df_merged.get('idrpp', pd.Series()).notna().map({True: 'OUI', False: 'NON'})
        df_merged.to_csv(OUTPUT_DIR / f"{prefix}_NS_IEHE.csv", index=False, sep=',', encoding='utf-8-sig')

if __name__ == "__main__":
    main()
