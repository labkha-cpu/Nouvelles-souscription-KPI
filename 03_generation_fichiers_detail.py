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
DESCRIPTION : Génération des fichiers de détail enrichis (NS_CIAM, NS_IEHE)
              avec consolidation des résultats de matching.

--- HISTORIQUE DES VERSIONS ---
VERSION | DATE       | DESCRIPTION
--------------------------------------------------------------------------------
1.0     | Initial    | Création du script
1.1     | 2025-10-XX | Correction accent colonne CIAM_Societe
1.2     | 2025-12-08 | Ajout fichiers Rech_Nom/Rech_Middle dans la consolidation
                     | Ajout colonnes Email_CIAM et Statut_Rapprochement (NS_CIAM uniquement)
1.3     | 2025-12-08 | FIX BUG : Exclusion des clés vides lors du matching (replace '' par NaN)
1.4     | 2025-12-08 | FIX : Normalisation Lowercase systématique des emails pour matching
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
    # Normalisation Unicode (enlève les accents)
    text = str(text).upper()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    # Garde uniquement les lettres A-Z
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
            # Mots-clés pour valider que c'est bien un fichier de données attendu
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
    # On exclut les fichiers générés pour ne garder que la source New_S
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
    
    # Fichiers Automatiques
    df_iehe = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_IEHE.csv"))
    df_cm = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_CM.csv"))
    df_ck = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_CK.csv"))
    
    # Fichiers Manuels (Rech Nom / Middle)
    df_nom = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_Rech_Nom.csv"))
    df_middle = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_Rech_Middle.csv"))

    if df_ns is None: 
        print("   ❌ Fichier New_S invalide ou vide.")
        return

    # --- 2. PREPARATION DONNEES POUR MATCHING ---
    print("   ⚙️  Normalisation des identités...")
    
    # Identification Colonnes NS
    col_mail_ciam = get_col_name(df_ns, ['mailciam', 'mail ciam', 'mail_ciam'])
    col_val_coord = get_col_name(df_ns, ['valeur_coordonnee', 'valeur coordonnee', 'mail'])
    col_kpep = get_col_name(df_ns, ['idkpep', 'kpep'])
    col_nom = get_col_name(df_ns, ['nom_long', 'nom', 'lastname'])
    col_prenom = get_col_name(df_ns, ['prenom', 'firstname'])
    col_dnaiss = get_col_name(df_ns, ['date_naissance', 'datenaissance', 'birthdate'])

    df_work = df_ns.copy()
    
    # Création Clés Techniques NS
    # FIX: On force .str.lower() pour normaliser et on remplace les chaines vides par NaN
    df_work['key_mail'] = df_work[col_mail_ciam].str.lower().str.strip().replace('', np.nan) if col_mail_ciam else np.nan
    df_work['key_val'] = df_work[col_val_coord].str.lower().str.strip().replace('', np.nan) if col_val_coord else np.nan
    df_work['key_kpep'] = df_work[col_kpep].replace('', np.nan) if col_kpep else np.nan
    
    # Création Clés Identité NS
    if col_nom and col_prenom:
        df_work['norm_nom'] = df_work[col_nom].apply(clean_text)
        df_work['norm_prenom'] = df_work[col_prenom].apply(clean_text)
        
        # Clé Nom+Prenom
        df_work['key_identite_simple'] = (df_work['norm_nom'] + "|" + df_work['norm_prenom']).replace('|', np.nan)
        
        # Clé Nom+Prenom+Date (si date dispo)
        if col_dnaiss:
            full_key = df_work['norm_nom'] + "|" + df_work['norm_prenom'] + "|" + df_work[col_dnaiss].str[:10]
            df_work['key_identite_full'] = full_key.replace(r'^\|\|.*$', np.nan, regex=True).replace(r'.*\|$', np.nan, regex=True)
        else:
            df_work['key_identite_full'] = np.nan
    else:
        df_work['key_identite_simple'] = np.nan
        df_work['key_identite_full'] = np.nan

    # Préparation Référentiel GLOBAL (CM + CK + Nom + Middle concaténés)
    refs_config = []
    if df_cm is not None: refs_config.append((df_cm, ['last_name', 'lastname', 'nom']))
    if df_ck is not None: refs_config.append((df_ck, ['last_name', 'lastname', 'nom']))
    if df_nom is not None: refs_config.append((df_nom, ['last_name', 'lastname', 'nom']))
    if df_middle is not None: refs_config.append((df_middle, ['middlename', 'middle_name', 'last_name', 'nom']))

    df_ref_all = pd.DataFrame()
    
    if refs_config:
        normalized_refs = []
        for df_source, nom_candidates in refs_config:
            temp = pd.DataFrame()
            # Mapping
            temp['realm_id'] = df_source['realm_id'] if 'realm_id' in df_source.columns else np.nan
            temp['email'] = df_source['email'] if 'email' in df_source.columns else ""
            temp['idkpep'] = df_source['idkpep'] if 'idkpep' in df_source.columns else ""
            
            # Identité
            c_nom = get_col_name(df_source, nom_candidates)
            c_pnom = get_col_name(df_source, ['first_name', 'firstname', 'prenom'])
            c_dn = get_col_name(df_source, ['birthdate', 'date_naissance'])
            
            if c_nom and c_pnom:
                temp['norm_nom'] = df_source[c_nom].apply(clean_text)
                temp['norm_prenom'] = df_source[c_pnom].apply(clean_text)
                temp['key_identite_simple'] = (temp['norm_nom'] + "|" + temp['norm_prenom']).replace('|', np.nan)
                if c_dn:
                    temp['key_identite_full'] = (temp['norm_nom'] + "|" + temp['norm_prenom'] + "|" + df_source[c_dn].str[:10])
                else:
                    temp['key_identite_full'] = np.nan
            else:
                temp['key_identite_simple'] = np.nan
                temp['key_identite_full'] = np.nan
            
            # Clés techniques (Normalisation Lowercase + NaN)
            temp['key_email'] = temp['email'].str.lower().str.strip().replace('', np.nan)
            temp['key_kpep'] = temp['idkpep'].replace('', np.nan)
            
            normalized_refs.append(temp)
        
        if normalized_refs:
            df_ref_all = pd.concat(normalized_refs, ignore_index=True)

    # Création des lookup tables (dropna pour éviter les index NaN)
    ref_email = df_ref_all.dropna(subset=['key_email']).drop_duplicates('key_email').set_index('key_email') if not df_ref_all.empty else pd.DataFrame()
    ref_kpep = df_ref_all.dropna(subset=['key_kpep']).drop_duplicates('key_kpep').set_index('key_kpep') if not df_ref_all.empty else pd.DataFrame()
    ref_identite_full = df_ref_all.dropna(subset=['key_identite_full']).drop_duplicates('key_identite_full').set_index('key_identite_full') if not df_ref_all.empty else pd.DataFrame()
    ref_identite_simple = df_ref_all.dropna(subset=['key_identite_simple']).drop_duplicates('key_identite_simple').set_index('key_identite_simple') if not df_ref_all.empty else pd.DataFrame()

    # --- 3. EXECUTION DES MATCHINGS ---
    print("   🔍 Exécution des méthodes de rapprochement (Email > KPEP > Identité)...")

    # 1. Mail CIAM
    m1 = df_work.merge(ref_email, left_on='key_mail', right_index=True, how='left', suffixes=('', '_m1'))
    # 2. Val Coord
    m2 = df_work.merge(ref_email, left_on='key_val', right_index=True, how='left', suffixes=('', '_m2'))
    # 3. KPEP
    m3 = df_work.merge(ref_kpep, left_on='key_kpep', right_index=True, how='left', suffixes=('', '_m3'))
    # 4. Identité Complète (Nom + Prenom + Date)
    m4 = df_work.merge(ref_identite_full, left_on='key_identite_full', right_index=True, how='left', suffixes=('', '_m4'))
    # 5. Identité Simple (Nom + Prenom) - Match faible
    m5 = df_work.merge(ref_identite_simple, left_on='key_identite_simple', right_index=True, how='left', suffixes=('', '_m5'))

    # --- 4. CONSOLIDATION ---
    
    # Indicateurs de match par méthode
    df_work['Match_MailCIAM'] = m1['realm_id'].notna().map({True: 'OUI', False: 'NON'})
    df_work['Match_ValCoord'] = m2['realm_id'].notna().map({True: 'OUI', False: 'NON'})
    df_work['Match_KPEP'] = m3['realm_id'].notna().map({True: 'OUI', False: 'NON'})
    df_work['Match_Identite_Complete'] = m4['realm_id'].notna().map({True: 'OUI', False: 'NON'})
    df_work['Match_Identite_NomPrenom'] = m5['realm_id'].notna().map({True: 'OUI', False: 'NON'})

    # Décision finale (Waterfall)
    df_work['Indicateur_Rapprochement'] = 'CIAM_NON_TROUVE'
    df_work['Methode_Retenue'] = 'AUCUNE'
    
    # Initialisation colonnes cibles
    target_cols = ['CIAM_Source', 'CIAM_Email_Cible', 'CIAM_KPEP_Cible', 'CIAM_Societe']
    for col in target_cols: 
        df_work[col] = np.nan
        df_work[col] = df_work[col].astype('object')

    # Priorité 5 : Identité Simple (Faible)
    mask = m5['realm_id'].notna()
    df_work.loc[mask, 'Indicateur_Rapprochement'] = 'CIAM_TROUVE_IDENTITE_FAIBLE'
    df_work.loc[mask, 'Methode_Retenue'] = 'Nom+Prenom'
    df_work.loc[mask, 'CIAM_Societe'] = m5.loc[mask, 'realm_id']
    df_work.loc[mask, 'CIAM_Email_Cible'] = m5.loc[mask, 'email']

    # Priorité 4 : Identité Complète (Forte)
    mask = m4['realm_id'].notna()
    df_work.loc[mask, 'Indicateur_Rapprochement'] = 'CIAM_TROUVE_IDENTITE'
    df_work.loc[mask, 'Methode_Retenue'] = 'Nom+Prenom+Date'
    df_work.loc[mask, 'CIAM_Societe'] = m4.loc[mask, 'realm_id']
    df_work.loc[mask, 'CIAM_Email_Cible'] = m4.loc[mask, 'email']

    # Priorité 3 : KPEP
    mask = m3['realm_id'].notna()
    df_work.loc[mask, 'Indicateur_Rapprochement'] = 'CIAM_TROUVE_KPEP'
    df_work.loc[mask, 'Methode_Retenue'] = 'KPEP'
    df_work.loc[mask, 'CIAM_Source'] = 'CK'
    df_work.loc[mask, 'CIAM_Societe'] = m3.loc[mask, 'realm_id']
    df_work.loc[mask, 'CIAM_KPEP_Cible'] = m3.loc[mask, 'key_kpep']
    # KPEP ramène aussi l'email si présent dans le ref
    df_work.loc[mask, 'CIAM_Email_Cible'] = m3.loc[mask, 'email']

    # Priorité 2 : Val Coord (Email)
    mask = m2['realm_id'].notna()
    df_work.loc[mask, 'Indicateur_Rapprochement'] = 'CIAM_TROUVE_EMAIL'
    df_work.loc[mask, 'Methode_Retenue'] = 'Val_Coord'
    df_work.loc[mask, 'CIAM_Source'] = 'CM'
    df_work.loc[mask, 'CIAM_Societe'] = m2.loc[mask, 'realm_id']
    df_work.loc[mask, 'CIAM_Email_Cible'] = m2.loc[mask, 'email']

    # Priorité 1 : Mail CIAM (Priorité Absolue)
    mask = m1['realm_id'].notna()
    df_work.loc[mask, 'Indicateur_Rapprochement'] = 'CIAM_TROUVE_EMAIL'
    df_work.loc[mask, 'Methode_Retenue'] = 'Mail_CIAM'
    df_work.loc[mask, 'CIAM_Source'] = 'CM'
    df_work.loc[mask, 'CIAM_Societe'] = m1.loc[mask, 'realm_id']
    df_work.loc[mask, 'CIAM_Email_Cible'] = m1.loc[mask, 'email']

    # --- AJOUTS DEMANDÉS (Email_CIAM & Statut_Rapprochement) ---
    print("   📝 Ajout des colonnes finales (Email_CIAM, Statut_Rapprochement) au fichier NS_CIAM...")
    
    # 1. Email_CIAM : Copie explicite de l'email trouvé (quelle que soit la méthode)
    df_work['Email_CIAM'] = df_work['CIAM_Email_Cible']
    
    # 2. Statut_Rapprochement : Statut explicite Rapproché / Non Rapproché
    df_work['Statut_Rapprochement'] = df_work['Indicateur_Rapprochement'].apply(
        lambda x: 'Non Rapproché' if x == 'CIAM_NON_TROUVE' else 'Rapproché'
    )

    # Nettoyage colonnes techniques
    cols_to_drop = ['key_mail', 'key_val', 'key_kpep', 'norm_nom', 'norm_prenom', 'key_identite_simple', 'key_identite_full']
    df_work.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Sauvegarde NS_CIAM
    f_ciam = OUTPUT_DIR / f"{prefix}_NS_CIAM.csv"
    df_work.to_csv(f_ciam, index=False, sep=',', encoding='utf-8-sig')
    print(f"   ✅ Fichier NS_CIAM généré : {f_ciam.name}")

    # --- 5. GENERATION NS_IEHE ---
    # (Pas de changement majeur ici, juste génération classique)
    print("   🔨 Construction NS_IEHE...")
    
    col_id_ns = get_col_name(df_ns, ['num_personne', 'numpersonne'])
    col_id_iehe = 'refperboccn'
    
    if df_iehe is not None and col_id_ns and col_id_iehe in df_iehe.columns:
        df_iehe_ref = df_iehe.drop_duplicates(subset=[col_id_iehe]).set_index(col_id_iehe)
        df_merged = df_ns.copy().merge(df_iehe_ref, left_on=col_id_ns, right_index=True, how='left', suffixes=('', '_iehe'))
        
        col_temoin = get_col_name(df_iehe, ['idrpp', 'adrmailctc', 'telmbictc'])
        if col_temoin:
            df_merged['IEHE_Present'] = df_merged[col_temoin].notna().map({True: 'OUI', False: 'NON'})
        else:
            df_merged['IEHE_Present'] = 'INCONNU'
            
        f_iehe = OUTPUT_DIR / f"{prefix}_NS_IEHE.csv"
        df_merged.to_csv(f_iehe, index=False, sep=',', encoding='utf-8-sig')
        print(f"   ✅ Fichier NS_IEHE généré : {f_iehe.name}")
    else:
        print("   ⚠️ Impossible de générer NS_IEHE (Fichier IEHE manquant ou colonne ID introuvable)")

if __name__ == "__main__":
    main()
