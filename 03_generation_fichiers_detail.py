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
1.1     | 2025-10-XX | - Correction nom colonne 'CIAM_Société' -> 'CIAM_Societe'
                     |   (suppression accent pour compatibilité BDD/SQL)
                     | - Sécurisation génération NS_IEHE (vérification colonnes)
                     |   et conservation logique LEFT JOIN.
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
            keywords = ['adhesion', 'assure', 'email', 'personne', 'kpep', 'realm', 'date', 'id', 'refper', 'last_name', 'nom']
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
    excluded_suffixes = ["_IEHE", "_CM", "_CK", "_REQ", "Resultats", "KPI", "NS_CIAM", "NS_IEHE"]
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
    
    # Chargement
    df_ns = normalize_cols(load_csv(ns_path))
    df_iehe = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_IEHE.csv"))
    df_cm = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_CM.csv"))
    df_ck = normalize_cols(load_csv(INPUT_DIR / f"{prefix}_CK.csv"))

    if df_ns is None: return

    # --- 1. PREPARATION DONNEES POUR MATCHING ---
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
    df_work['key_mail'] = df_work[col_mail_ciam].str.lower() if col_mail_ciam else ""
    df_work['key_val'] = df_work[col_val_coord].str.lower() if col_val_coord else ""
    df_work['key_kpep'] = df_work[col_kpep] if col_kpep else ""
    
    # Création Clés Identité NS
    if col_nom and col_prenom:
        df_work['norm_nom'] = df_work[col_nom].apply(clean_text)
        df_work['norm_prenom'] = df_work[col_prenom].apply(clean_text)
        
        # Clé Nom+Prenom
        df_work['key_identite_simple'] = df_work['norm_nom'] + "|" + df_work['norm_prenom']
        
        # Clé Nom+Prenom+Date (si date dispo)
        if col_dnaiss:
            # On garde la date brute (string) si le format est ISO YYYY-MM-DD dans les deux fichiers
            # Pour être sûr, on prend les 10 premiers caractères
            df_work['key_identite_full'] = df_work['norm_nom'] + "|" + df_work['norm_prenom'] + "|" + df_work[col_dnaiss].str[:10]
        else:
            df_work['key_identite_full'] = ""
    else:
        df_work['key_identite_simple'] = ""
        df_work['key_identite_full'] = ""

    # Préparation Référentiel GLOBAL (CM + CK concaténés pour la recherche identité)
    # On a besoin d'un référentiel unique d'identité
    cols_ref = ['realm_id', 'email', 'idkpep', 'first_name', 'last_name', 'birthdate']
    
    refs_list = []
    if df_cm is not None: refs_list.append(df_cm)
    if df_ck is not None: refs_list.append(df_ck)
    
    df_ref_all = pd.DataFrame()
    if refs_list:
        # On harmonise les colonnes
        normalized_refs = []
        for df in refs_list:
            temp = pd.DataFrame()
            # Mapping
            temp['realm_id'] = df['realm_id'] if 'realm_id' in df.columns else np.nan
            temp['email'] = df['email'] if 'email' in df.columns else ""
            temp['idkpep'] = df['idkpep'] if 'idkpep' in df.columns else ""
            
            # Identité
            c_nom = get_col_name(df, ['last_name', 'lastname', 'nom'])
            c_pnom = get_col_name(df, ['first_name', 'firstname', 'prenom'])
            c_dn = get_col_name(df, ['birthdate', 'date_naissance'])
            
            if c_nom and c_pnom:
                temp['norm_nom'] = df[c_nom].apply(clean_text)
                temp['norm_prenom'] = df[c_pnom].apply(clean_text)
                temp['key_identite_simple'] = temp['norm_nom'] + "|" + temp['norm_prenom']
                if c_dn:
                    temp['key_identite_full'] = temp['norm_nom'] + "|" + temp['norm_prenom'] + "|" + df[c_dn].str[:10]
                else:
                    temp['key_identite_full'] = ""
            
            # Clés techniques
            temp['key_email'] = temp['email'].str.lower()
            temp['key_kpep'] = temp['idkpep']
            
            normalized_refs.append(temp)
        
        df_ref_all = pd.concat(normalized_refs, ignore_index=True)

    # Création des lookup tables optimisées
    ref_email = df_ref_all.drop_duplicates('key_email').set_index('key_email') if not df_ref_all.empty else pd.DataFrame()
    ref_kpep = df_ref_all.drop_duplicates('key_kpep').set_index('key_kpep') if not df_ref_all.empty else pd.DataFrame()
    ref_identite_full = df_ref_all.drop_duplicates('key_identite_full').set_index('key_identite_full') if not df_ref_all.empty else pd.DataFrame()
    ref_identite_simple = df_ref_all.drop_duplicates('key_identite_simple').set_index('key_identite_simple') if not df_ref_all.empty else pd.DataFrame()

    # --- 2. EXECUTION DES MATCHINGS ---
    print("   🔍 Exécution des 5 méthodes de rapprochement...")

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

    # --- 3. CONSOLIDATION ---
    
    # Ajout des colonnes "Top_Match_..." (OUI/NON)
    df_work['Match_MailCIAM'] = m1['realm_id'].notna().map({True: 'OUI', False: 'NON'})
    df_work['Match_ValCoord'] = m2['realm_id'].notna().map({True: 'OUI', False: 'NON'})
    df_work['Match_KPEP'] = m3['realm_id'].notna().map({True: 'OUI', False: 'NON'})
    df_work['Match_Identite_Complete'] = m4['realm_id'].notna().map({True: 'OUI', False: 'NON'})
    df_work['Match_Identite_NomPrenom'] = m5['realm_id'].notna().map({True: 'OUI', False: 'NON'})

    # Décision finale (Indicateur Rapprochement)
    df_work['Indicateur_Rapprochement'] = 'CIAM_NON_TROUVE'
    df_work['Methode_Retenue'] = 'AUCUNE'
    
    # Colonnes cibles à remplir (initialisation object pour éviter warning)
    # Correction: 'CIAM_Societe' sans accent pour éviter problèmes SQL
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

    # Nettoyage
    cols_to_drop = ['key_mail', 'key_val', 'key_kpep', 'norm_nom', 'norm_prenom', 'key_identite_simple', 'key_identite_full']
    df_work.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Sauvegarde NS_CIAM
    f_ciam = OUTPUT_DIR / f"{prefix}_NS_CIAM.csv"
    df_work.to_csv(f_ciam, index=False, sep=',', encoding='utf-8-sig')
    print(f"   ✅ {f_ciam.name}")

    # --- 4. GENERATION NS_IEHE ---
    print("   🔨 Construction NS_IEHE...")
    
    # On repart du fichier NS propre
    df_iehe_out = df_ns.copy()
    col_id_ns = get_col_name(df_ns, ['num_personne', 'numpersonne'])
    col_id_iehe = 'refperboccn'
    
    if df_iehe is not None and col_id_ns and col_id_iehe in df_iehe.columns:
        # Dédoublonnage sur la clé IEHE pour éviter l'explosion du nombre de lignes
        df_iehe_ref = df_iehe.drop_duplicates(subset=[col_id_iehe]).set_index(col_id_iehe)
        
        # Merge LEFT pour conserver tous les enregistrements NS
        df_merged = df_iehe_out.merge(df_iehe_ref, left_on=col_id_ns, right_index=True, how='left', suffixes=('', '_iehe'))
        
        # Calcul de l'indicateur de présence
        col_temoin = get_col_name(df_iehe, ['idrpp', 'adrmailctc', 'telmbictc'])
        if col_temoin:
            df_merged['IEHE_Present'] = df_merged[col_temoin].notna().map({True: 'OUI', False: 'NON'})
        else:
            df_merged['IEHE_Present'] = 'INCONNU'
            
        f_iehe = OUTPUT_DIR / f"{prefix}_NS_IEHE.csv"
        df_merged.to_csv(f_iehe, index=False, sep=',', encoding='utf-8-sig')
        print(f"   ✅ {f_iehe.name}")
    else:
        print("   ⚠️ Impossible de générer NS_IEHE (Fichier IEHE manquant ou colonne ID introuvable)")

if __name__ == "__main__":
    main()
