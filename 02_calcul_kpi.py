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
    """Nettoie un texte (Majuscules, sans accents, sans tirets, lettres uniquement)."""
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
        df.columns = [c.lower().strip().replace('"', '') for c in df.columns]
        cols_to_keep = [c for c in df.columns if not c.startswith('unnamed')]
        df = df[cols_to_keep]
        df = df.loc[:, ~df.columns.duplicated()]
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
    # On cherche le fichier source New_S
    excluded_suffixes = ["_IEHE", "_CM", "_CK", "_REQ", "Resultats", "KPI", "_Rech_", "NS_CIAM", "NS_IEHE"]
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
    if df is None: return None
    for col in candidates:
        if col in df.columns: return col
    return None

def get_merge_col(df, base_col, suffix='_m'):
    """Retourne le nom de colonne correct après un merge (avec ou sans suffixe)."""
    if f"{base_col}{suffix}" in df.columns:
        return f"{base_col}{suffix}"
    elif base_col in df.columns:
        return base_col
    return None

# --- FONCTIONS QUALITÉ ---

def check_email_quality(df, col_name, label):
    if df is None or col_name is None or col_name not in df.columns: 
        return {label: "Non disponible"}
    # Normalisation Lowercase explicite
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
    if not INPUT_DIR.exists(): return
    ns_path, prefix = find_latest_new_s(INPUT_DIR)
    if not ns_path: return

    print(f"🚀 Analyse KPI (Complet + Matching Élargi) pour le flux : {prefix}")
    
    # Chemins
    iehe_path = INPUT_DIR / f"{prefix}_IEHE.csv"
    cm_path = INPUT_DIR / f"{prefix}_CM.csv"
    ck_path = INPUT_DIR / f"{prefix}_CK.csv"
    nom_path = INPUT_DIR / f"{prefix}_Rech_Nom.csv"
    middle_path = INPUT_DIR / f"{prefix}_Rech_Middle.csv"

    # Chargement
    df_ns = normalize_cols(load_csv(ns_path))
    if df_ns is None: return

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
    
    # === 1. ACTIVITÉ (KPI INITIAUX CONSERVÉS) ===
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

    # === 2. OPS & TP (KPI INITIAUX CONSERVÉS) ===
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

    # === 3. DOUBLONS (KPI INITIAUX CONSERVÉS) ===
    dup_raw = {}
    cols_dup_cands = {
        'num_personne': ['num_personne', 'numpersonne'],
        'num_ctr_indiv': ['num_ctr_indiv', 'numcontrat', 'contrat'],
        'valeur_coordonnee': ['valeur_coordonnee', 'valeur coordonnee', 'mail', 'email'],
        'idkpep': ['idkpep', 'kpep', 'id_kpep'],
        'mail_ciam': ['mail_ciam', 'mail ciam', 'email_ciam']
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
        col_nm, col_pnm, col_dt = real_comp[0], real_comp[1], real_comp[2]
        s_nom = df_ns[col_nm].astype(str).str.strip().str.lower()
        s_prenom = df_ns[col_pnm].astype(str).str.strip().str.lower()
        s_date = df_ns[col_dt].astype(str).str.strip() 
        composite_series = s_nom + '|' + s_prenom + '|' + s_date
        mask_valid = (s_nom != '') & (s_prenom != '') & (s_nom != 'nan')
        dup_raw['nom+prenom+datenaissance'] = int(composite_series[mask_valid].duplicated(keep='first').sum())
    else: 
        dup_raw['nom+prenom+datenaissance'] = 0
        
    dup_data = {"Indicateurs": {k: fmt_kpi(v, vol) for k, v in dup_raw.items()}}

    # === 4. MATCHING (MISE A JOUR MAJEURE + FILTRAGE) ===
    
    # 4.1. Filtrage Population (Exclure Conjoints)
    col_type = get_col_name(df_ns, ['type_assure', 'typeassure', 'code_role_personne', 'role'])
    
    if col_type:
        mask_conjoi = df_ns[col_type].astype(str).str.upper().str.contains('CONJOI')
        # On ne garde que les NON conjoints pour le calcul KPI matching
        df_c = df_ns[~mask_conjoi].copy()
        df_c.reset_index(drop=True, inplace=True)
    else:
        df_c = df_ns.copy()
        
    vol_c = len(df_c) # Volume Eligibles

    # 4.2. Identification Colonnes
    col_mail_ciam = get_col_name(df_c, ['mailciam', 'mail ciam', 'mail_ciam', 'email_ciam'])
    col_val_coord = get_col_name(df_c, ['valeur_coordonnee', 'valeur coordonnee', 'mail', 'email'])
    col_kpep = get_col_name(df_c, ['idkpep', 'id kpep', 'kpep', 'id_kpep'])
    col_nom = get_col_name(df_c, ['nom_long', 'nom', 'lastname'])
    col_prenom = get_col_name(df_c, ['prenom', 'firstname'])
    col_dnaiss = get_col_name(df_c, ['date_naissance', 'datenaissance', 'birthdate'])

    # 4.3. Création Clés NS (Lower Case systématique)
    df_c['key_mail'] = df_c[col_mail_ciam].str.lower().str.strip().replace('', np.nan) if col_mail_ciam else np.nan
    df_c['key_val'] = df_c[col_val_coord].str.lower().str.strip().replace('', np.nan) if col_val_coord else np.nan
    df_c['key_kpep'] = df_c[col_kpep].replace('', np.nan) if col_kpep else np.nan
    
    df_c['norm_nom'] = df_c[col_nom].apply(clean_text) if col_nom else ""
    df_c['norm_prenom'] = df_c[col_prenom].apply(clean_text) if col_prenom else ""
    
    if col_dnaiss:
        # FIX: Gestion du format date pour éviter UserWarning
        # On tente de parser en forçant dayfirst, et on ignore le warning si le format est ambigu
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_c['dt_fmt'] = pd.to_datetime(df_c[col_dnaiss], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
            
        df_c['key_nom_date'] = (df_c['norm_nom'] + "|" + df_c['dt_fmt']).replace(r'^\|.*$', np.nan, regex=True).replace(r'.*\|$', np.nan, regex=True)
        df_c['key_identite_full'] = (df_c['norm_nom'] + "|" + df_c['norm_prenom'] + "|" + df_c['dt_fmt']).replace(r'\|\|', np.nan, regex=True)
    else:
        df_c['key_nom_date'] = np.nan
        df_c['key_identite_full'] = np.nan

    df_c['key_identite_simple'] = (df_c['norm_nom'] + "|" + df_c['norm_prenom']).replace('|', np.nan)

    # 4.4. Préparation Référentiels
    def prep_ref(df_in):
        if df_in is None or df_in.empty: return pd.DataFrame()
        temp = pd.DataFrame()
        # On ne mappe que les champs utiles pour le match
        cols_map = {'realm_id': 'realm_id', 'email': 'email', 'idkpep': 'idkpep'}
        for c_out, c_in in cols_map.items():
            if c_in in df_in.columns: temp[c_out] = df_in[c_in]
        
        # Clés Techniques Ref
        if 'email' in temp.columns: 
            temp['key_email'] = temp['email'].str.lower().str.strip().replace('', np.nan)
        if 'idkpep' in temp.columns: 
            temp['key_kpep'] = temp['idkpep'].replace('', np.nan)

        # Clés Identité Ref
        c_r_nom = get_col_name(df_in, ['last_name', 'lastname', 'nom', 'middlename'])
        c_r_pnom = get_col_name(df_in, ['first_name', 'firstname', 'prenom'])
        c_r_date = get_col_name(df_in, ['birthdate', 'date_naissance'])
        
        temp['norm_nom'] = df_in[c_r_nom].apply(clean_text) if c_r_nom else ""
        temp['norm_prenom'] = df_in[c_r_pnom].apply(clean_text) if c_r_pnom else ""
        
        if c_r_date:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt_r = pd.to_datetime(df_in[c_r_date], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
            temp['key_nom_date'] = (temp['norm_nom'] + "|" + dt_r)
            temp['key_identite_full'] = (temp['norm_nom'] + "|" + temp['norm_prenom'] + "|" + dt_r)
        
        temp['key_identite_simple'] = (temp['norm_nom'] + "|" + temp['norm_prenom'])
        return temp

    # Prépa Sources
    ref_cm = prep_ref(df_cm)
    ref_ck = prep_ref(df_ck)
    ref_nom = prep_ref(df_nom)
    ref_middle = prep_ref(df_middle)
    
    # Agrégation des refs identité (CM + CK + Nom + Middle)
    all_refs = pd.concat([ref_cm, ref_ck, ref_nom, ref_middle], ignore_index=True)

    # Indexes
    idx_email = ref_cm.dropna(subset=['key_email']).drop_duplicates('key_email').set_index('key_email') if not ref_cm.empty else pd.DataFrame()
    idx_kpep = ref_ck.dropna(subset=['key_kpep']).drop_duplicates('key_kpep').set_index('key_kpep') if not ref_ck.empty else pd.DataFrame()
    idx_identite = all_refs.dropna(subset=['key_identite_full']).drop_duplicates('key_identite_full').set_index('key_identite_full') if not all_refs.empty else pd.DataFrame()
    
    # Indexes Recherche Élargie (Sans Prénom)
    idx_nom = ref_nom.dropna(subset=['key_nom_date']).drop_duplicates('key_nom_date').set_index('key_nom_date') if not ref_nom.empty else pd.DataFrame()
    idx_middle = ref_middle.dropna(subset=['key_nom_date']).drop_duplicates('key_nom_date').set_index('key_nom_date') if not ref_middle.empty else pd.DataFrame()

    # 4.5. WATERFALL MATCHING
    # Initialisation
    df_c['Found'] = False
    df_c['Methode'] = 'Aucune'

    # Etape 1: Mail CIAM (Priorité Absolue)
    if not idx_email.empty and 'realm_id' in idx_email.columns:
        m = df_c.merge(idx_email, left_on='key_mail', right_index=True, how='left', suffixes=('', '_m'))
        # FIX: Check dynamique du nom de colonne après merge (car _m n'est pas garanti si pas de collision)
        col_res = get_merge_col(m, 'realm_id', '_m')
        if col_res:
            mask = m[col_res].notna()
            df_c.loc[mask, 'Found'] = True
            df_c.loc[mask, 'Methode'] = 'Mail_CIAM'

    # Etape 2: Val Coord (Sur non trouvé)
    if not idx_email.empty and 'realm_id' in idx_email.columns:
        mask_remain = ~df_c['Found']
        m = df_c[mask_remain].merge(idx_email, left_on='key_val', right_index=True, how='left', suffixes=('', '_m'))
        col_res = get_merge_col(m, 'realm_id', '_m')
        if col_res:
            mask = m[col_res].notna()
            found_idx = df_c[mask_remain][mask].index
            df_c.loc[found_idx, 'Found'] = True
            df_c.loc[found_idx, 'Methode'] = 'Val_Coord'

    # Etape 3: KPEP
    if not idx_kpep.empty and 'realm_id' in idx_kpep.columns:
        mask_remain = ~df_c['Found']
        m = df_c[mask_remain].merge(idx_kpep, left_on='key_kpep', right_index=True, how='left', suffixes=('', '_m'))
        col_res = get_merge_col(m, 'realm_id', '_m')
        if col_res:
            mask = m[col_res].notna()
            found_idx = df_c[mask_remain][mask].index
            df_c.loc[found_idx, 'Found'] = True
            df_c.loc[found_idx, 'Methode'] = 'KPEP'

    # Etape 4: Identité Complète (Nom+Prenom+Date)
    if not idx_identite.empty and 'realm_id' in idx_identite.columns:
        mask_remain = ~df_c['Found']
        m = df_c[mask_remain].merge(idx_identite, left_on='key_identite_full', right_index=True, how='left', suffixes=('', '_m'))
        col_res = get_merge_col(m, 'realm_id', '_m')
        if col_res:
            mask = m[col_res].notna()
            found_idx = df_c[mask_remain][mask].index
            df_c.loc[found_idx, 'Found'] = True
            df_c.loc[found_idx, 'Methode'] = 'Identite_Full'

    # Etape 5: Recherche Élargie (Last Name sans Prénom) - NOUVEAU
    if not idx_nom.empty and 'realm_id' in idx_nom.columns:
        mask_remain = ~df_c['Found']
        m = df_c[mask_remain].merge(idx_nom, left_on='key_nom_date', right_index=True, how='left', suffixes=('', '_m'))
        col_res = get_merge_col(m, 'realm_id', '_m')
        if col_res:
            mask = m[col_res].notna()
            found_idx = df_c[mask_remain][mask].index
            df_c.loc[found_idx, 'Found'] = True
            df_c.loc[found_idx, 'Methode'] = 'Recherche_Large_Nom'

    # Etape 6: Recherche Élargie (Middle Name sans Prénom) - NOUVEAU
    if not idx_middle.empty and 'realm_id' in idx_middle.columns:
        mask_remain = ~df_c['Found']
        m = df_c[mask_remain].merge(idx_middle, left_on='key_nom_date', right_index=True, how='left', suffixes=('', '_m'))
        col_res = get_merge_col(m, 'realm_id', '_m')
        if col_res:
            mask = m[col_res].notna()
            found_idx = df_c[mask_remain][mask].index
            df_c.loc[found_idx, 'Found'] = True
            df_c.loc[found_idx, 'Methode'] = 'Recherche_Large_Middle'

    # KPIs Matching
    count_rapproche = df_c['Found'].sum()
    count_non_rapproche = vol_c - count_rapproche
    
    # Détail par méthode
    raw_methods = df_c[df_c['Found']]['Methode'].value_counts().to_dict()
    # On convertit les numpy int en int natifs pour JSON
    detail_methodes = {k: int(v) for k, v in raw_methods.items()}

    # Match IEHE (hors population ciblée car IEHE global)
    match_iehe = df_ns['num_personne'].isin(set(df_iehe['refperboccn'])).sum() if df_iehe is not None and 'num_personne' in df_ns.columns and 'refperboccn' in df_iehe.columns else 0

    # -----------------------------------------------------------------------
    # NOUVEAU BLOC : QUALITE DONNEES CIAM
    # -----------------------------------------------------------------------
    nb_identique = 0
    nb_vide_ciam = 0
    
    # On travaille sur la population éligible (df_c)
    if col_mail_ciam:
        # Nettoyage
        s_ciam = df_c[col_mail_ciam].astype(str).str.lower().str.strip().replace({'nan': '', 'none': ''})
        nb_vide_ciam = int((s_ciam == '').sum())
        
        if col_val_coord:
            s_val = df_c[col_val_coord].astype(str).str.lower().str.strip().replace({'nan': '', 'none': ''})
            # On considère identique si CIAM n'est pas vide ET strictement égal à ValCoord
            mask_ident = (s_ciam != '') & (s_ciam == s_val)
            nb_identique = int(mask_ident.sum())
            
    qualite_ciam_data = {
        "Email_Identique_ValCoord": fmt_kpi(nb_identique, vol_c),
        "Email_CIAM_Vide_ou_Null": fmt_kpi(nb_vide_ciam, vol_c)
    }

    matching_data = {
        "Indicateurs_Clefs": {
            "Population_Analysee_Eligible": vol_c,
            "Population_Exclue_Conjoints": vol - vol_c,
            "Total_Rapproches_CIAM": fmt_kpi(count_rapproche, vol_c),
            "Total_Non_Rapproches_CIAM": fmt_kpi(count_non_rapproche, vol_c),
            "Detail_Rapprochement_Methode": detail_methodes,
            "NS_vers_IEHE_Global": fmt_kpi(match_iehe, vol)
        },
        "Qualite_Donnees_CIAM": qualite_ciam_data
    }

    # === 5. QUALITÉ CONTACT (KPI INITIAUX CONSERVÉS) ===
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
