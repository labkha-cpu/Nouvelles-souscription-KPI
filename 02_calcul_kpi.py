import pandas as pd
import json
import re
import sys
import warnings
import unicodedata
from pathlib import Path
from datetime import datetime
import numpy as np

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR if (SCRIPT_DIR / "Input_Data").exists() else SCRIPT_DIR.parent
INPUT_DIR = BASE_DIR / "Input_Data"
OUTPUT_DIR = BASE_DIR / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- UTILITAIRES ---

def clean_text(text):
    """Nettoie un texte (Majuscules, sans accents, sans tirets)."""
    if pd.isna(text) or text == '': return ""
    text = str(text).upper()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^A-Z]', '', text)
    return text

def fmt_kpi(value, total, label_val="Nombre", label_taux="Taux"):
    """Formate un KPI avec sa valeur absolue et son pourcentage."""
    safe_total = total if total > 0 else 1
    rate = round((value / safe_total * 100), 2)
    return {label_val: int(value), label_taux: rate}

def calculate_stats(series):
    """Calcule Moyenne/Mediane/Min/Max sur une série numérique."""
    if series.empty: return {"Moyenne": 0, "Mediane": 0, "Min": 0, "Max": 0}
    return {
        "Moyenne": round(float(series.mean()), 2),
        "Mediane": round(float(series.median()), 2),
        "Min": int(series.min()),
        "Max": int(series.max())
    }

def load_csv(path):
    """Charge un CSV de manière robuste."""
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
            keywords = ['adhesion', 'assure', 'email', 'personne', 'kpep', 'realm', 'id', 'date', 'nom', 'last_name']
            if any(k in c for c in cols for k in keywords):
                return df
        except: continue
    return None

def normalize_cols(df):
    """Nettoie les noms de colonnes et SUPPRIME LES DOUBLONS + GESTION QUOTES."""
    if df is not None:
        # 1. Nettoyage basique noms colonnes + suppression quotes dans headers
        df.columns = [c.lower().strip().replace('"', '') for c in df.columns]
        
        # 2. Suppression des colonnes parasites
        cols_to_keep = [c for c in df.columns if not c.startswith('unnamed')]
        df = df[cols_to_keep]

        # 3. Dédoublonnage des colonnes
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 4. Nettoyage des valeurs (strip espaces ET quotes)
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace('"', '', regex=False).replace({'nan': '', 'None': ''})
            
    return df

def parse_date(ds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(ds, dayfirst=True, errors='coerce')

def find_latest_new_s(directory):
    if not directory.exists(): return None, None
    candidates = []
    excluded_suffixes = ["_IEHE", "_CM", "_CK", "_REQ", "Resultats", "KPI", "_Rech_"]
    
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

def get_col_name(df, candidates):
    """Cherche la première colonne existante parmi une liste de candidats."""
    if df is None: return None
    for col in candidates:
        if col in df.columns:
            return col
    return None

# --- FONCTIONS QUALITÉ ---

def check_email_quality(df, col_name, label):
    if df is None or col_name is None or col_name not in df.columns: 
        return {label: "Non disponible"}
    series = df[col_name].astype(str).str.lower().str.strip().replace({'nan': '', 'none': ''})
    total = len(series)
    vides = (series == '').sum()
    regex = r"[^@\s]+@[^@\s]+\.[^@\s]+"
    disp_domains = ['yopmail', 'mailinator', 'tempmail', '10minutemail', 'trashmail']
    def analyze(val):
        if not val: return (False, False)
        return (re.match(regex, val) is None, any(d in val for d in disp_domains))
    results = series.apply(analyze)
    non_conformes = results.apply(lambda x: x[0]).sum()
    jetables = results.apply(lambda x: x[1]).sum()
    remplis = total - vides
    taux = round(((remplis - non_conformes - jetables) / remplis * 100), 2) if remplis > 0 else 0
    return {label: {"Total": int(total), "Vides": int(vides), "Remplis": int(remplis), 
                    "Non_Conformes": int(non_conformes), "Jetables": int(jetables), "Taux_Conformite": taux}}

def check_phone_quality(df, col_name, label):
    if df is None or col_name is None or col_name not in df.columns: 
        return {label: "Non disponible"}
    series = df[col_name].astype(str).str.replace(r"[^\d+]", "", regex=True).str.strip()
    total = len(series)
    vides = (series == '').sum()
    regex_phone = r"^\+?\d{6,15}$"
    nb_ok = series.str.match(regex_phone).sum()
    remplis = total - vides
    taux = round((nb_ok / remplis * 100), 2) if remplis > 0 else 0
    return {label: {"Total": int(total), "Vides": int(vides), "Remplis": int(remplis), 
                    "Valides_Format": int(nb_ok), "Invalides_Format": int(remplis - nb_ok), "Taux_Qualite": taux}}

# --- MAIN ---

def main():
    if not INPUT_DIR.exists(): 
        print(f"❌ Dossier {INPUT_DIR} introuvable.")
        return

    ns_path, prefix = find_latest_new_s(INPUT_DIR)
    if not ns_path: 
        print("❌ ECHEC : Aucun fichier source (New_S) trouvé.")
        return

    print(f"🚀 Analyse KPI pour le flux : {prefix}")
    print(f"📂 Fichier SOURCE : {ns_path.name}")
    
    # Chemins des fichiers potentiels
    iehe_path = INPUT_DIR / f"{prefix}_IEHE.csv"
    cm_path = INPUT_DIR / f"{prefix}_CM.csv"
    ck_path = INPUT_DIR / f"{prefix}_CK.csv"
    
    # Nouveaux fichiers de recherche manuelle
    nom_path = INPUT_DIR / f"{prefix}_Rech_Nom.csv"
    middle_path = INPUT_DIR / f"{prefix}_Rech_Middle.csv"

    # Chargement
    df_ns = normalize_cols(load_csv(ns_path))
    if df_ns is None:
        print("❌ Erreur critique : Fichier New_S illisible.")
        return

    df_iehe = normalize_cols(load_csv(iehe_path))
    df_cm = normalize_cols(load_csv(cm_path))
    df_ck = normalize_cols(load_csv(ck_path))
    df_nom = normalize_cols(load_csv(nom_path))
    df_middle = normalize_cols(load_csv(middle_path))

    files_used = {
        "New_S": str(ns_path.name),
        "IEHE": str(iehe_path.name) if iehe_path.exists() else "⚠️ ABSENT",
        "CM": str(cm_path.name) if cm_path.exists() else "⚠️ ABSENT",
        "CK": str(ck_path.name) if ck_path.exists() else "⚠️ ABSENT",
        "Rech_Nom": str(nom_path.name) if nom_path.exists() else "⚠️ ABSENT",
        "Rech_Middle": str(middle_path.name) if middle_path.exists() else "⚠️ ABSENT"
    }
    
    # === 1. ACTIVITÉ (PERIMETRE GLOBAL) ===
    vol = len(df_ns)
    pres = [c for c in ['code_soc_appart', 'date_effet_adhesion', 'type_assure'] if c in df_ns.columns]
    det_act = df_ns.groupby(pres).size().reset_index(name='count').to_dict('records') if pres else []
    for d in det_act: d['count'] = int(d['count'])
    
    repart_type = {}
    if 'type_assure' in df_ns.columns:
        raw_repart = df_ns['type_assure'].value_counts().to_dict()
        repart_type = {k: fmt_kpi(v, vol) for k, v in raw_repart.items()}

    rad_cols = [c for c in df_ns.columns if 'rad' in c and 'date' in c]
    vol_rad = int(df_ns[rad_cols[0]].replace('', pd.NA).dropna().count()) if rad_cols else 0

    age_stats = {}
    if 'date_naissance' in df_ns.columns and 'date_adhesion' in df_ns.columns:
        dt_naiss = parse_date(df_ns['date_naissance'])
        dt_adh = parse_date(df_ns['date_adhesion'])
        age_series = (dt_adh - dt_naiss).dt.days / 365.25
        age_stats = calculate_stats(age_series.dropna())

    activite_data = {
        "Global": vol,
        "Volumetrie_Radiation": fmt_kpi(vol_rad, vol),
        "Demographie_Age": age_stats,
        "Repartition_Par_Type_Assure": repart_type,
        "Details": det_act
    }

    # === 2. OPS & TP ===
    op_stats = {}
    if 'date_adhesion' in df_ns.columns:
        try: dt_file = datetime.strptime(prefix, "%d%m%Y")
        except: dt_file = datetime.now()
        dt_adh = parse_date(df_ns['date_adhesion'])
        op_stats = calculate_stats((dt_file - dt_adh).dt.days.dropna())

    tp_met = {'tp_ok_total': fmt_kpi(0,0)}
    ged_met = {'Taux_Presence_GED': 0}

    if 'date_adhesion' in df_ns.columns and 'date_effet_adhesion' in df_ns.columns:
        d = (parse_date(df_ns['date_effet_adhesion']) - parse_date(df_ns['date_adhesion'])).dt.days
        neg = int((d < 0).sum())
        ok_pos = int(((d >= 0) & (d <= 21)).sum())
        ko = int((d > 21).sum())
        total_tp = neg + ok_pos + ko
        
        tp_met = {
            'delai_negatif': fmt_kpi(neg, total_tp),
            'delai_ok_0_21': fmt_kpi(ok_pos, total_tp),
            'delai_ko_sup_21': fmt_kpi(ko, total_tp),
            'tp_total_ok': fmt_kpi(neg + ok_pos, total_tp)
        }
        
        refs = set(df_iehe['refperboccn']) if df_iehe is not None and 'refperboccn' in df_iehe.columns else set()
        mask_eligible = d < 22
        df_eligible = df_ns[mask_eligible]
        vol_elig = len(df_eligible)
        ged_ok = int(df_eligible['num_personne'].isin(refs).sum()) if vol_elig > 0 else 0
        ged_met = {
            "Eligibles": fmt_kpi(vol_elig, vol),
            "GED_OK_sur_Eligibles": fmt_kpi(ged_ok, vol_elig),
            "GED_KO_sur_Eligibles": fmt_kpi(vol_elig - ged_ok, vol_elig)
        }

    tp_ops_data = {"Delai_Saisie_Stats": op_stats, "Conformite_Delai_Effet": tp_met}

    # === 3. DOUBLONS ===
    print("⏳ Calcul des Doublons...")
    dup_raw = {}
    cols_dup_cands = {
        'num_personne': ['num_personne', 'numpersonne'],
        'num_ctr_indiv': ['num_ctr_indiv', 'numcontrat', 'contrat'],
        'valeur_coordonnee': ['valeur_coordonnee', 'valeur coordonnee', 'mail', 'email'],
        'idkpep': ['idkpep', 'kpep', 'id_kpep'],
        'mail_ciam': ['mailciam', 'mail ciam', 'mail_ciam', 'email_ciam']
    }
    
    for key, cands in cols_dup_cands.items():
        found = get_col_name(df_ns, cands)
        if found:
            temp_series = df_ns[found].astype(str).str.strip()
            mask_not_empty = (temp_series != '') & (temp_series.str.lower() != 'nan')
            series_clean = temp_series[mask_not_empty].str.lower()
            dup_raw[key] = int(series_clean.duplicated(keep='first').sum())
        else:
            dup_raw[key] = 0

    comp = ['nom_long', 'prenom', 'date_naissance']
    real_comp = [get_col_name(df_ns, [c]) for c in comp]
    if all(real_comp):
        col_nom, col_prenom, col_date = real_comp[0], real_comp[1], real_comp[2]
        s_nom = df_ns[col_nom].astype(str).str.strip().str.lower()
        s_prenom = df_ns[col_prenom].astype(str).str.strip().str.lower()
        s_date = df_ns[col_date].astype(str).str.strip() 
        composite_series = s_nom + '|' + s_prenom + '|' + s_date
        mask_valid = (s_nom != '') & (s_prenom != '') & (s_nom != 'nan')
        dup_raw['nom+prenom+datenaissance'] = int(composite_series[mask_valid].duplicated(keep='first').sum())
    else: 
        dup_raw['nom+prenom+datenaissance'] = 0
        
    dup_data = {"Indicateurs": {k: fmt_kpi(v, vol) for k, v in dup_raw.items()}}

    # === 4. MATCHING (PERIMETRE RESTREINT: HORS CONJOINT) ===
    print("⏳ Calcul du Matching (Fusion CM + CK + Recherche Nom/Date)...")
    
    # Filtre CIAM pour les KPIs Matching
    col_type = get_col_name(df_ns, ['type_assure', 'typeassure', 'code_role_personne', 'role'])
    if col_type:
        mask_conjoi = df_ns[col_type].astype(str).str.upper().str.strip() == 'CONJOI'
        df_c = df_ns[~mask_conjoi].copy()
        df_c.reset_index(drop=True, inplace=True)
    else:
        df_c = df_ns.copy()
        
    vol_c = len(df_c)

    # Identification colonnes NS
    col_mail_ciam = get_col_name(df_c, ['mailciam', 'mail ciam', 'mail_ciam', 'email_ciam'])
    col_val_coord = get_col_name(df_c, ['valeur_coordonnee', 'valeur coordonnee', 'mail', 'email'])
    col_kpep = get_col_name(df_c, ['idkpep', 'id kpep', 'kpep', 'id_kpep'])
    col_nom = get_col_name(df_c, ['nom_long', 'nom', 'lastname'])
    col_prenom = get_col_name(df_c, ['prenom', 'firstname'])
    col_dnaiss = get_col_name(df_c, ['date_naissance', 'datenaissance', 'birthdate'])

    # Clés NS
    # FIX: On utilise .replace('', np.nan) pour invalider les clés vides
    df_c['key_mail_ciam'] = df_c[col_mail_ciam].astype(str).str.lower().str.strip().replace('', np.nan) if col_mail_ciam else np.nan
    df_c['key_val_coord'] = df_c[col_val_coord].astype(str).str.lower().str.strip().replace('', np.nan) if col_val_coord else np.nan
    df_c['key_kpep'] = df_c[col_kpep].astype(str).str.strip().replace('', np.nan) if col_kpep else np.nan
    
    # Clé Identité (Complex & Simple)
    # FIX: On force à NaN si l'identité est incomplète ou vide
    df_c['key_identite'] = np.nan
    df_c['key_identite_simple'] = np.nan
    
    if col_nom and col_prenom:
        clean_nom = df_c[col_nom].apply(clean_text)
        clean_prenom = df_c[col_prenom].apply(clean_text)
        
        # Simple : Nom | Prenom
        # Si nom ou prenom vide, on a '|' ou 'nom|' ou '|prenom', on replace tout cela par NaN
        df_c['key_identite_simple'] = (clean_nom + "|" + clean_prenom).replace('|', np.nan)
        
        # Full : Nom | Prenom | Date (si dispo)
        if col_dnaiss:
            dt_str = pd.to_datetime(df_c[col_dnaiss], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d').fillna('')
            full_key = clean_nom + "|" + clean_prenom + "|" + dt_str
            # Invalidation si la date est manquante ou si le nom/prenom est manquant
            df_c['key_identite'] = full_key.replace(r'^\|\|.*$', np.nan, regex=True).replace(r'.*\|$', np.nan, regex=True)

    # Préparation Référentiels
    def prep_ref(df, prefix_col):
        if df is None or df.empty: return pd.DataFrame()
        
        rename_dict = {'email': f'{prefix_col}_email', 'idkpep': f'{prefix_col}_kpep', 'realm_id': f'{prefix_col}_realm', 
                       'birthdate': f'{prefix_col}_birthdate', 'last_name': f'{prefix_col}_lastname', 'middleName': f'{prefix_col}_middle', 
                       'first_name': f'{prefix_col}_firstname', # Ajout pour clé simple
                       'origincreation': f'{prefix_col}_origin', 'date_evt': f'{prefix_col}_date_evt'}
        
        df = df.rename(columns={k:v for k,v in rename_dict.items() if k in df.columns})
        
        # Clés Techniques (replace vide par NaN)
        if f'{prefix_col}_email' in df.columns:
            df['key_email'] = df[f'{prefix_col}_email'].astype(str).str.lower().str.strip().replace('', np.nan)
        
        if f'{prefix_col}_kpep' in df.columns:
            df['key_kpep'] = df[f'{prefix_col}_kpep'].astype(str).str.strip().replace('', np.nan)
            
        # Clés Identité (Complex & Simple)
        c_nom = f'{prefix_col}_lastname'
        c_bd = f'{prefix_col}_birthdate'
        c_prenom = f'{prefix_col}_firstname'
        
        # FIX: On utilise le préfixe 'middle' pour détecter le fichier Middle et utiliser la bonne colonne
        if 'middle' in prefix_col and f'{prefix_col}_middle' in df.columns:
            c_nom = f'{prefix_col}_middle'
            
        if c_nom in df.columns and c_prenom in df.columns:
            cl_nm = df[c_nom].apply(clean_text)
            cl_pnm = df[c_prenom].apply(clean_text)
            
            # Gestion des clés vides
            df['key_identite_simple'] = (cl_nm + "|" + cl_pnm).replace('|', np.nan)
            
            if c_bd in df.columns:
                dt_st = pd.to_datetime(df[c_bd], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d').fillna('')
                df['key_identite'] = (cl_nm + "|" + cl_pnm + "|" + dt_st).replace(r'^\|\|.*$', np.nan, regex=True).replace(r'.*\|$', np.nan, regex=True)
            
        return df

    # Construction des tables de référence avec dropna sur les clés
    df_ref_email = prep_ref(df_cm, 'cm').dropna(subset=['key_email']).drop_duplicates('key_email').set_index('key_email') if df_cm is not None else pd.DataFrame()
    df_ref_kpep = prep_ref(df_ck, 'ck').dropna(subset=['key_kpep']).drop_duplicates('key_kpep').set_index('key_kpep') if df_ck is not None else pd.DataFrame()
    
    # Consolidation Nom/Middle
    refs_identite = []
    if df_nom is not None: refs_identite.append(prep_ref(df_nom, 'nom'))
    if df_middle is not None: refs_identite.append(prep_ref(df_middle, 'middle'))
    
    df_ref_identite = pd.DataFrame()
    df_ref_identite_simple = pd.DataFrame()
    
    if refs_identite:
        df_concat = pd.concat(refs_identite, ignore_index=True)
        # Coalesce des colonnes critiques pour avoir un référentiel unique
        df_concat['final_realm'] = df_concat.get('nom_realm', pd.Series()).combine_first(df_concat.get('middle_realm', pd.Series()))
        df_concat['final_email'] = df_concat.get('nom_email', pd.Series()).combine_first(df_concat.get('middle_email', pd.Series()))

        if 'key_identite' in df_concat.columns:
            df_ref_identite = df_concat.dropna(subset=['key_identite', 'final_realm']).drop_duplicates('key_identite').set_index('key_identite')
        
        if 'key_identite_simple' in df_concat.columns:
            df_ref_identite_simple = df_concat.dropna(subset=['key_identite_simple', 'final_realm']).drop_duplicates('key_identite_simple').set_index('key_identite_simple')

    # Matching
    m1 = df_c.merge(df_ref_email, left_on='key_mail_ciam', right_index=True, how='left') if not df_ref_email.empty else pd.DataFrame()
    m2 = df_c.merge(df_ref_email, left_on='key_val_coord', right_index=True, how='left', suffixes=('', '_m2')) if not df_ref_email.empty else pd.DataFrame()
    m3 = df_c.merge(df_ref_kpep, left_on='key_kpep', right_index=True, how='left', suffixes=('', '_kpep')) if not df_ref_kpep.empty else pd.DataFrame()
    m4 = df_c.merge(df_ref_identite, left_on='key_identite', right_index=True, how='left', suffixes=('', '_ident')) if not df_ref_identite.empty else pd.DataFrame()
    m5 = df_c.merge(df_ref_identite_simple, left_on='key_identite_simple', right_index=True, how='left', suffixes=('', '_simple')) if not df_ref_identite_simple.empty else pd.DataFrame()

    # --- WATERFALL (CONSOLIDATION) ---
    # On reconstruit la logique de priorité pour déterminer l'Email final et le statut
    
    df_c['Statut_Match'] = 'Non Rapproché'
    df_c['Email_CIAM_Found'] = np.nan
    
    # Priorité 5 : Identité Simple
    if not m5.empty and 'final_realm' in m5.columns:
        mask = m5['final_realm'].notna()
        df_c.loc[mask, 'Statut_Match'] = 'Rapproché'
        df_c.loc[mask, 'Email_CIAM_Found'] = m5.loc[mask, 'final_email']

    # Priorité 4 : Identité Complète
    if not m4.empty and 'final_realm' in m4.columns:
        mask = m4['final_realm'].notna()
        df_c.loc[mask, 'Statut_Match'] = 'Rapproché'
        df_c.loc[mask, 'Email_CIAM_Found'] = m4.loc[mask, 'final_email']

    # Priorité 3 : KPEP
    if not m3.empty and 'ck_realm' in m3.columns:
        mask = m3['ck_realm'].notna()
        df_c.loc[mask, 'Statut_Match'] = 'Rapproché'
        df_c.loc[mask, 'Email_CIAM_Found'] = m3.loc[mask, 'ck_email']

    # Priorité 2 : Val Coord
    if not m2.empty and 'cm_realm' in m2.columns:
        mask = m2['cm_realm'].notna()
        df_c.loc[mask, 'Statut_Match'] = 'Rapproché'
        df_c.loc[mask, 'Email_CIAM_Found'] = m2.loc[mask, 'cm_email']

    # Priorité 1 : Mail CIAM
    if not m1.empty and 'cm_realm' in m1.columns:
        mask = m1['cm_realm'].notna()
        df_c.loc[mask, 'Statut_Match'] = 'Rapproché'
        df_c.loc[mask, 'Email_CIAM_Found'] = m1.loc[mask, 'cm_email']

    # --- CALCUL DES KPIS SPECIFIQUES ---
    
    # 1. Qualité Email (Strictement Identique)
    # Comparaison case-insensitive standard (lower vs lower)
    mask_qualite = (df_c['Email_CIAM_Found'].str.lower().str.strip() == df_c['key_val_coord'])
    count_qualite_ok = mask_qualite.sum()
    
    # 2. Emails Vides (Rapprochés mais email vide/null)
    mask_rapproche = (df_c['Statut_Match'] == 'Rapproché')
    mask_vide = df_c['Email_CIAM_Found'].isna() | (df_c['Email_CIAM_Found'] == '')
    count_email_vide = (mask_rapproche & mask_vide).sum()
    
    # 3. Non Rapprochés
    count_non_rapproche = (df_c['Statut_Match'] == 'Non Rapproché').sum()
    
    # Total Rapprochés pour calculs
    count_rapproche_total = mask_rapproche.sum()

    # Match IEHE
    refs = set(df_iehe['refperboccn']) if df_iehe is not None and 'refperboccn' in df_iehe.columns else set()
    match_iehe = df_ns['num_personne'].isin(refs).sum() if 'num_personne' in df_ns.columns else 0

    matching_data = {
        "Indicateurs_Clefs": {
            "Total_Contrats_Analyses": vol_c,
            "Total_Rapproches_CIAM": fmt_kpi(count_rapproche_total, vol_c),
            "Total_Non_Rapproches_CIAM": fmt_kpi(count_non_rapproche, vol_c),
            "NS_vers_IEHE": fmt_kpi(match_iehe, vol),
            "Qualite_Donnees_CIAM": {
                "Email_Identique_ValCoord": fmt_kpi(count_qualite_ok, count_rapproche_total),
                "Email_CIAM_Vide_ou_Null": fmt_kpi(count_email_vide, count_rapproche_total)
            }
        }
    }

    # === 5. QUALITÉ CONTACT ===
    ct_met = {"_Glossaire": "Syntaxe emails/téléphones + Profils orphelins."}
    
    tgts = [
        (df_ns, col_val_coord, "NS_Valeur"), 
        (df_ns, col_mail_ciam, "NS_MailCIAM"),
        (df_cm, "email", "CM_Email"), 
        (df_ck, "email", "CK_Email")
    ]
    for df_t, c_t, l in tgts:
        if df_t is not None:
            if c_t is None: c_t = get_col_name(df_t, ['valeur_coordonnee', 'mail', 'email'])
            ct_met.update(check_email_quality(df_t, c_t, l))
    
    ph_tgts = [(df_cm, "phonenumber", "CM_Phone"), (df_iehe, "telmbictc", "IEHE_Phone")]
    for df_t, c_t, l in ph_tgts:
        if df_t is not None:
             if c_t is None or c_t not in df_t.columns: c_t = get_col_name(df_t, ['tel', 'phone', 'telephone', 'mobile', 'telmbictc', 'phonenumber'])
             ct_met.update(check_phone_quality(df_t, c_t, l))

    out = {
        "Meta": {"Fichier": str(ns_path.name), "Prefixe": prefix, "Fichiers_Utilises": files_used},
        "Activite": activite_data, "Doublons": dup_data, "Matching_Synthese": matching_data,
        "Service_Client_TP": tp_ops_data, "GED": ged_met, "Qualite_Contact": ct_met
    }

    f_out = OUTPUT_DIR / f"{prefix}_KPI_Resultats.json"
    with open(f_out, 'w', encoding='utf-8') as f: json.dump(out, f, indent=4, ensure_ascii=False)
    print(json.dumps(out, indent=4, ensure_ascii=False))
    print(f"✅ Terminé. JSON sauvegardé : {f_out}")

if __name__ == "__main__":
    main()

# --- VERSION DU SCRIPT ---
# Version: 3.4
# Date: 08/12/2025
# Modifications :
# - FIX BUG : Exclusion des clés vides lors du matching KPI (replace '' par NaN)
# -------------------------
