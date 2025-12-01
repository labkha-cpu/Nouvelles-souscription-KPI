import pandas as pd
import json
import re
import sys
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
# Ajustement si exécution depuis racine ou sous-dossier
BASE_DIR = SCRIPT_DIR if (SCRIPT_DIR / "Input_Data").exists() else SCRIPT_DIR.parent
INPUT_DIR = BASE_DIR / "Input_Data"
OUTPUT_DIR = BASE_DIR / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- UTILITAIRES ---

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
        {'encoding': 'utf-8', 'skiprows': 0}, # Standard UTF-8
        {'encoding': 'latin1', 'skiprows': 0}, # Standard Latin
        {'encoding': 'utf-8', 'skiprows': 2}, # Format Accolade
        {'encoding': 'latin1', 'skiprows': 2} # Format Accolade Latin
    ]
    
    for params in attempts:
        current_params = read_params.copy()
        current_params.update(params)
        try:
            df = pd.read_csv(path, **current_params)
            # Validation basique sur le contenu des colonnes
            cols = [c.lower() for c in df.columns]
            keywords = ['adhesion', 'assure', 'email', 'personne', 'kpep', 'realm', 'id', 'date']
            if any(k in c for c in cols for k in keywords):
                return df
        except: continue
    return None

def normalize_cols(df):
    """Nettoie les noms de colonnes et SUPPRIME LES DOUBLONS."""
    if df is not None:
        # 1. Nettoyage (minuscule + strip)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # 2. Dédoublonnage des colonnes (Garde la première occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 3. Nettoyage des valeurs
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({'nan': '', 'None': ''})
            
    return df

def parse_date(ds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(ds, dayfirst=True, errors='coerce')

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
    
    iehe_path = INPUT_DIR / f"{prefix}_IEHE.csv"
    cm_path = INPUT_DIR / f"{prefix}_CM.csv"
    ck_path = INPUT_DIR / f"{prefix}_CK.csv"

    # Chargement & Dédoublonnage colonnes
    df_ns = normalize_cols(load_csv(ns_path))
    if df_ns is None:
        print("❌ Erreur critique : Fichier New_S illisible.")
        return

    df_iehe = normalize_cols(load_csv(iehe_path))
    df_cm = normalize_cols(load_csv(cm_path))
    df_ck = normalize_cols(load_csv(ck_path))

    files_used = {
        "New_S": str(ns_path.name),
        "IEHE": str(iehe_path.name) if iehe_path.exists() else "⚠️ ABSENT",
        "CM": str(cm_path.name) if cm_path.exists() else "⚠️ ABSENT",
        "CK": str(ck_path.name) if ck_path.exists() else "⚠️ ABSENT"
    }
    
    # === 1. ACTIVITÉ ===
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

    # === 3. DOUBLONS (SECTION MISE À JOUR & VALIDÉE) ===
    # Règle : Insensible à la casse, insensible aux espaces, compte les lignes excédentaires.
    print("⏳ Calcul des Doublons (Nouvelle logique)...")
    dup_raw = {}
    cols_dup_cands = {
        'num_personne': ['num_personne', 'numpersonne'],
        'num_ctr_indiv': ['num_ctr_indiv', 'numcontrat', 'contrat'],
        'valeur_coordonnee': ['valeur_coordonnee', 'valeur coordonnee', 'mail', 'email'],
        'idkpep': ['idkpep', 'kpep', 'id_kpep'],
        'mail_ciam': ['mailciam', 'mail ciam', 'mail_ciam', 'email_ciam']
    }
    
    # A. Colonnes simples
    for key, cands in cols_dup_cands.items():
        found = get_col_name(df_ns, cands)
        if found:
            # 1. Préparation (nettoyage)
            # On remplace les NaN par vide pour traiter tout en string
            temp_series = df_ns[found].astype(str).str.strip()
            
            # 2. Filtrage : on ignore les vides et les 'nan' littéraux
            mask_not_empty = (temp_series != '') & (temp_series.str.lower() != 'nan')
            
            # 3. Normalisation (minuscule)
            series_clean = temp_series[mask_not_empty].str.lower()
            
            # 4. Calcul (lignes excédentaires)
            dup_raw[key] = int(series_clean.duplicated(keep='first').sum())
        else:
            dup_raw[key] = 0

    # B. Clé composite
    comp = ['nom_long', 'prenom', 'date_naissance']
    real_comp = [get_col_name(df_ns, [c]) for c in comp]
    
    if all(real_comp):
        # Récupération des colonnes exactes
        col_nom, col_prenom, col_date = real_comp[0], real_comp[1], real_comp[2]
        
        # Nettoyage et normalisation
        s_nom = df_ns[col_nom].astype(str).str.strip().str.lower()
        s_prenom = df_ns[col_prenom].astype(str).str.strip().str.lower()
        s_date = df_ns[col_date].astype(str).str.strip() # Pas de lower() sur date
        
        # Création de la clé sécurisée avec séparateur '|'
        composite_series = s_nom + '|' + s_prenom + '|' + s_date
        
        # Filtre : On ne compte les doublons que si Nom ET Prénom sont remplis
        mask_valid = (s_nom != '') & (s_prenom != '') & (s_nom != 'nan')
        
        # Calcul
        dup_raw['nom+prenom+datenaissance'] = int(composite_series[mask_valid].duplicated(keep='first').sum())
    else: 
        dup_raw['nom+prenom+datenaissance'] = 0
        
    dup_data = {"Indicateurs": {k: fmt_kpi(v, vol) for k, v in dup_raw.items()}}

    # === 4. MATCHING ===
    print("⏳ Calcul du Matching (Fusion CM + CK)...")
    
    if 'type_assure' in df_ns.columns:
        df_c = df_ns[df_ns['type_assure'] != 'CONJOI'].copy()
    else: df_c = df_ns.copy()
    vol_c = len(df_c)

    col_mail_ciam = get_col_name(df_c, ['mailciam', 'mail ciam', 'mail_ciam', 'email_ciam'])
    col_val_coord = get_col_name(df_c, ['valeur_coordonnee', 'valeur coordonnee', 'mail', 'email'])
    col_kpep = get_col_name(df_c, ['idkpep', 'id kpep', 'kpep', 'id_kpep'])

    df_c['key_mail_ciam'] = df_c[col_mail_ciam].astype(str).str.lower().str.strip() if col_mail_ciam else ""
    df_c['key_val_coord'] = df_c[col_val_coord].astype(str).str.lower().str.strip() if col_val_coord else ""
    df_c['key_kpep'] = df_c[col_kpep].astype(str).str.strip() if col_kpep else ""

    df_ref_email = pd.DataFrame()
    df_ref_kpep = pd.DataFrame()

    if df_cm is not None and not df_cm.empty:
        rename_dict = {'email': 'cm_email', 'idkpep': 'cm_kpep', 'birthdate': 'cm_birthdate', 'date_evt': 'cm_date_evt', 'realm_id': 'cm_realm', 'origincreation': 'cm_origin', 'channel': 'cm_origin'}
        df_cm_clean = df_cm.rename(columns={k:v for k,v in rename_dict.items() if k in df_cm.columns})
        if 'cm_email' in df_cm_clean.columns:
            df_cm_clean['key_email'] = df_cm_clean['cm_email'].astype(str).str.lower().str.strip()
            df_ref_email = df_cm_clean.drop_duplicates(subset=['key_email']).set_index('key_email')

    if df_ck is not None and not df_ck.empty:
        rename_dict = {'email': 'ck_email', 'idkpep': 'ck_kpep', 'birthdate': 'ck_birthdate', 'date_evt': 'ck_date_evt', 'realm_id': 'ck_realm', 'origincreation': 'ck_origin', 'channel': 'ck_origin'}
        df_ck_clean = df_ck.rename(columns={k:v for k,v in rename_dict.items() if k in df_ck.columns})
        if 'ck_kpep' in df_ck_clean.columns:
            df_ck_clean['key_kpep'] = df_ck_clean['ck_kpep'].astype(str).str.strip()
            df_ref_kpep = df_ck_clean.drop_duplicates(subset=['key_kpep']).set_index('key_kpep')

    m1 = df_c.merge(df_ref_email, left_on='key_mail_ciam', right_index=True, how='left') if not df_ref_email.empty else pd.DataFrame()
    m2 = df_c.merge(df_ref_email, left_on='key_val_coord', right_index=True, how='left', suffixes=('', '_m2')) if not df_ref_email.empty else pd.DataFrame()
    m3 = df_c.merge(df_ref_kpep, left_on='key_kpep', right_index=True, how='left', suffixes=('', '_kpep')) if not df_ref_kpep.empty else pd.DataFrame()

    df_final = df_c.copy()
    has_match_mailciam = pd.Series(False, index=df_c.index)
    has_match_valcoord = pd.Series(False, index=df_c.index)
    has_match_kpep = pd.Series(False, index=df_c.index)

    if not m1.empty and 'cm_realm' in m1.columns:
        has_match_mailciam = m1['cm_realm'].notna()
        has_match_mailciam.index = df_c.index 
    if not m2.empty and 'cm_realm' in m2.columns:
        has_match_valcoord = m2['cm_realm'].notna()
        has_match_valcoord.index = df_c.index
    if not m3.empty and 'ck_realm' in m3.columns:
        has_match_kpep = m3['ck_realm'].notna()
        has_match_kpep.index = df_c.index

    cols_map = [('cm_birthdate', 'ck_birthdate'), ('cm_date_evt', 'ck_date_evt'), ('cm_realm', 'ck_realm'), ('cm_origin', 'ck_origin')]
    for col_cm, col_ck in cols_map:
        target_col = col_cm.replace('cm_', 'final_')
        df_final[target_col] = np.nan
        
        if not m1.empty and col_cm in m1.columns: 
            s = m1[col_cm]
            if isinstance(s, pd.DataFrame): s = s.iloc[:, 0] 
            s.index = df_c.index
            df_final[target_col] = s
            
        if not m2.empty and col_cm in m2.columns: 
            s = m2[col_cm]
            if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
            s.index = df_c.index
            df_final[target_col] = df_final[target_col].combine_first(s)
            
        if not m3.empty and col_ck in m3.columns: 
            s = m3[col_ck]
            if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
            s.index = df_c.index
            df_final[target_col] = df_final[target_col].combine_first(s)

    is_matched_global = has_match_mailciam | has_match_valcoord | has_match_kpep
    match_global_count = int(is_matched_global.sum())
    
    df_matched = df_final[is_matched_global].copy()
    coherence_identity_ko = 0
    digi_delay_stats = calculate_stats(pd.Series())
    origin_distrib, realm_map_distrib = {}, {}

    if not df_matched.empty:
        if 'final_origin' in df_matched.columns:
            origin_distrib = {k: fmt_kpi(v, match_global_count) for k, v in df_matched['final_origin'].fillna('Inconnu').value_counts().to_dict().items()}
        date_ns = parse_date(df_matched['date_naissance'])
        date_cm = parse_date(df_matched['final_birthdate'])
        mask_valid = (date_ns.notna()) & (date_cm.notna())
        coherence_identity_ko = int((date_ns[mask_valid] != date_cm[mask_valid]).sum())
        digi_delay_stats = calculate_stats((parse_date(df_matched['final_date_evt']) - parse_date(df_matched['date_adhesion'])).dt.days.dropna())
        if 'code_soc_appart' in df_matched.columns and 'final_realm' in df_matched.columns:
            mapping = df_matched['code_soc_appart'].astype(str) + " -> " + df_matched['final_realm'].astype(str)
            realm_map_distrib = {k: fmt_kpi(v, match_global_count) for k, v in mapping.value_counts().head(20).to_dict().items()}

    refs = set(df_iehe['refperboccn']) if df_iehe is not None and 'refperboccn' in df_iehe.columns else set()
    match_iehe = df_ns['num_personne'].isin(refs).sum() if 'num_personne' in df_ns.columns else 0

    matching_data = {
        "Indicateurs_Clefs": {
            "NS_vers_CM_Global": fmt_kpi(match_global_count, vol_c),
            "NS_vers_CIAM_via_Email": fmt_kpi((has_match_mailciam | has_match_valcoord).sum(), vol_c),
            "NS_vers_CIAM_via_KPEP": fmt_kpi(has_match_kpep.sum(), vol_c),
            "NS_vers_IEHE": fmt_kpi(match_iehe, vol),
        },
        "Coherence_Etendue": {
            "Anomalies_Date_Naissance": fmt_kpi(coherence_identity_ko, match_global_count),
            "Delai_Creation_Compte_Stats": digi_delay_stats,
            "Mapping_CodeSoc_Realm_Top20": realm_map_distrib,
            "Repartition_Origine_Compte": origin_distrib
        }
    }

    # === 5. QUALITÉ CONTACT ===
    ct_met = {"_Glossaire": "Syntaxe emails/téléphones + Profils orphelins (vides intégralement)."}
    
    tgts = [
        (df_ns, col_val_coord, "NS_Valeur"), 
        (df_ns, col_mail_ciam, "NS_MailCIAM"),
        (df_cm, "email", "CM_Email"), 
        (df_ck, "email", "CK_Email"), 
        (df_iehe, "adrmailctc", "IEHE_Email")
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

    sm = df_ns[col_mail_ciam].astype(str).str.lower().str.strip() if col_mail_ciam else pd.Series()
    sv = df_ns[col_val_coord].astype(str).str.lower().str.strip() if col_val_coord else pd.Series()
    
    if not sm.empty and not sv.empty:
        ct_met["Coherence_NS_mail"] = {
            "Identiques": fmt_kpi((sm == sv).sum(), len(sm)),
            "Differents": fmt_kpi((sm != sv).sum(), len(sm)),
            "Vides_Simultanes": fmt_kpi(((sm == '') & (sv == '')).sum(), len(sm))
        }

    cols_orphan = [c for c in [col_kpep, col_mail_ciam, col_val_coord] if c]
    if cols_orphan:
        mask_vide = pd.Series([True]*len(df_ns), index=df_ns.index)
        for c in cols_orphan: mask_vide &= (df_ns[c] == '')
        ct_met["Profils_Sans_Donnees_Identifiantes"] = fmt_kpi(mask_vide.sum(), vol)

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