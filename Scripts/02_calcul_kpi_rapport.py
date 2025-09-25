#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import sys
import unicodedata
import difflib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# === KPI Filter: Type_assuré ∈ {"ASSPRI","MPRETR","MPVRET"} (no CONJOI exclusion) ===
_VALID_TYPES = ("ASSPRI","MPRETR","MPVRET")
_INVISIBLE_CHARS = [
    '\ufeff','\u200b','\u200c','\u200d','\u2060','\ufffe','\xa0','\u00a0',
    '\u2000','\u2001','\u2002','\u2003','\u2004','\u2005','\u2006','\u2007',
    '\u2008','\u2009','\u200a','\u202f','\u205f','\u3000','ï»¿','\ufeff'
]
def _strip_invisible_kpi(s: str) -> str:
    if s is None:
        return ""
    out = str(s)
    for ch in _INVISIBLE_CHARS:
        out = out.replace(ch, " ")
    out = re.sub(r"\s+", " ", out).strip()
    return out

def _norm_kpi(s: str) -> str:
    s = _strip_invisible_kpi(str(s))
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"\s+", "", s, flags=re.U).upper()

def apply_kpi_filter(df, *, status_col: str | None):
    """Conserve uniquement les lignes dont Type_assuré est dans {_VALID_TYPES}.
    Ne JAMAIS exclure explicitement "CONJOI".
    - status_col: nom de la colonne Type_assuré (ex: "typeassure").
    """
    import pandas as _pd
    if not isinstance(df, _pd.DataFrame):
        raise TypeError("apply_kpi_filter: df doit être un DataFrame pandas")
    if not status_col or status_col not in df.columns:
        return df
    status_norm = df[status_col].fillna("").map(_norm_kpi)
    mask = status_norm.isin(_VALID_TYPES)
    return df.loc[mask].copy()
# === End KPI Filter block ===

import unicodedata
import re as _re_utils
import csv  


def _normalize_colname(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.strip().lower()
    name = name.replace(' ', '_').replace('-', '_').replace('__', '_')
    name = _re_utils.sub(r'[^a-z0-9_]', '', name)
    return name

def _best_sep(sample: str) -> str:
    candidates = [',',';','	','|']
    lines = [ln for ln in sample.splitlines() if ln.strip()][:20]
    if not lines:
        return ','
    scores = {}
    for sep in candidates:
        counts = [ln.count(sep) for ln in lines]
        if max(counts) == 0:
            scores[sep] = 0
        else:
            import statistics
            scores[sep] = (sum(counts)/len(counts)) - (statistics.pvariance(counts) if len(counts)>1 else 0)
    return max(scores, key=scores.get)

def read_csv_smart(path, purpose=None):
    """Robust CSV reader avec détection sep/encoding + normalisation texte (accents)."""
    import pandas as pd
    if not path:
        return pd.DataFrame(), {'status':'missing'}
    try:
        raw = open(path, 'rb').read(65536)
    except Exception:
        raw = b''
    encodings = ['utf-8-sig','utf-8','cp1252','latin-1']
    sample_text = ''
    for enc in encodings:
        try:
            sample_text = raw.decode(enc, errors='strict')
            encoding = enc
            break
        except Exception:
            continue
    else:
        encoding = 'latin-1'
        try:
            sample_text = raw.decode(encoding, errors='ignore')
        except Exception:
            sample_text = ''
    sep = _best_sep(sample_text) if sample_text else ','
    try:
        df = pd.read_csv(path, sep=sep, encoding=encoding, engine='python',
                         dtype=str, on_bad_lines='skip')
    except Exception:
        df = None
        for sep2 in [';','\t','|',',']:
            try:
                df = pd.read_csv(path, sep=sep2, encoding=encoding, engine='python',
                                 dtype=str, on_bad_lines='skip')
                sep = sep2
                break
            except Exception:
                df = None
        if df is None:
            return pd.DataFrame(), {'status':'error', 'error':'failed_to_parse'}
    # normalise les noms de colonnes et les cellules texte (accents préservés)
    df.columns = [_normalize_colname(c) for c in df.columns]
    for c in df.columns:
        try:
            df[c] = df[c].astype(str).str.strip()
        except Exception:
            pass
    df = normalize_text_cols(df)
    meta = {'status':'ok','sep':sep,'encoding':encoding,'rows':len(df),'cols':len(df.columns),'purpose':purpose or ''}
    print(f"[INFO] Chargé {purpose or 'CSV'}: sep='{sep}', enc='{encoding}', shape={df.shape}")
    return df, meta


def postprocess_iehe(df):
    import pandas as pd
    if df is None or df.empty:
        return df
    # Candidate names for refperboccn
    cands = ['refperboccn','refperbocc_n','refperbocn','refperbocc','refperboccid','idrpp']
    target = None
    for c in cands:
        if c in df.columns:
            target = c; break
    if target is None:
        for c in df.columns:
            if 'refperboc' in c:
                target = c; break
    if target is None and 'numpersonne' in df.columns:
        target = 'numpersonne'
    if target:
        if target != 'refperboccn':
            df = df.rename(columns={target:'refperboccn'})
        df['refperboccn'] = df['refperboccn'].astype(str).str.strip()
        df = df[df['refperboccn']!='']
        df = df.drop_duplicates(subset=['refperboccn'])
    else:
        print('[WARN] IEHE: colonne refperboccn introuvable après normalisation.')
    return df

def postprocess_ciam_email(df):
    """Ensure CIAM_EMAIL has a usable 'mail' column and clean it."""
    import pandas as pd
    if df is None or df.empty:
        return df
    # candidate columns for email
    mail_candidates = ['mail','email','adresse_email','courriel','mailciam']
    mail_col = None
    for c in mail_candidates:
        if c in df.columns:
            mail_col = c; break
    if mail_col is None:
        # try to find one containing 'mail' or 'email'
        for c in df.columns:
            if 'mail' in c or 'email' in c:
                mail_col = c; break
    if mail_col is None:
        print('[WARN] CIAM_EMAIL: colonne mail introuvable après normalisation.')
        return df
    if mail_col != 'mail':
        df = df.rename(columns={mail_col:'mail'})
    # clean/normalize
    df['mail'] = df['mail'].astype(str).str.strip().str.lower()
    df = df[df['mail']!='']
    df = df.drop_duplicates(subset=['mail'])
    return df

def postprocess_ciam_kpep(df):
    """Ensure CIAM_KPEP has 'idkpep' and ideally 'mail' columns, with cleaning and dedup."""
    import pandas as pd
    if df is None or df.empty:
        return df
    id_candidates = ['idkpep','kpep','kp_id','idkpepci','idkpep_id']
    id_col = None
    for c in id_candidates:
        if c in df.columns:
            id_col = c; break
    if id_col is None:
        for c in df.columns:
            if 'idkpep' in c or c.endswith('kpep') or c.startswith('kpep'):
                id_col = c; break
    if id_col is None:
        print('[WARN] CIAM_KPEP: colonne idkpep introuvable après normalisation.')
        mail_candidates = ['mail','email']
        for c in mail_candidates:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip().str.lower()
        return df
    if id_col != 'idkpep':
        df = df.rename(columns={id_col:'idkpep'})
    df['idkpep'] = df['idkpep'].astype(str).str.strip()
    mail_candidates = ['mail','email','mailciam']
    mail_col = None
    for c in mail_candidates:
        if c in df.columns:
            mail_col = c; break
    if mail_col and mail_col != 'mail':
        df = df.rename(columns={mail_col:'mail'})
    if 'mail' in df.columns:
        df['mail'] = df['mail'].astype(str).str.strip().str.lower()
    df = df[df['idkpep']!='']
    df = df.drop_duplicates(subset=['idkpep'])
    return df
    # idkpep candidates
    id_candidates = ['idkpep','kpep','kp_id','idkpepci','idkpep_id']
    id_col = None
    for c in id_candidates:
        if c in df.columns:
            id_col = c; break
    if id_col is None:
        # heuristic
        for c in df.columns:
            if 'idkpep' in c or (c.endswith('kpep') or c.startswith('kpep')):
                id_col = c; break
    if id_col is None:
        print('[WARN] CIAM_KPEP: colonne idkpep introuvable après normalisation.')
        # still try to handle mail cleanup
        mail_candidates = ['mail','email']
        for c in mail_candidates:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip().str.lower()
        return df
    if id_col != 'idkpep':
        df = df.rename(columns={id_col:'idkpep'})
    # clean idkpep
    df['idkpep'] = df['idkpep'].astype(str).str.strip()
    # mail candidates
    mail_candidates = ['mail','email','mailciam']
    mail_col = None
    for c in mail_candidates:
        if c in df.columns:
            mail_col = c; break
    if mail_col and mail_col != 'mail':
        df = df.rename(columns={mail_col:'mail'})
    if 'mail' in df.columns:
        df['mail'] = df['mail'].astype(str).str.strip().str.lower()
    # drop empties and dups
    df = df[df['idkpep']!='']
    # Prefer dedup on idkpep; if duplicates remain with different mail, keep first
    df = df.drop_duplicates(subset=['idkpep'])
    return df

    cands = ['refperboccn','refperbocc_n','refperbocn','refperbocc','refperboccid','idrpp']
    target = None
    for c in cands:
        if c in df.columns:
            target = c; break
    if target is None:
        for c in df.columns:
            if 'refperboc' in c:
                target = c; break
    if target is None and 'numpersonne' in df.columns:
        target = 'numpersonne'
    if target:
        if target != 'refperboccn':
            df = df.rename(columns={target:'refperboccn'})
        df['refperboccn'] = df['refperboccn'].astype(str).str.strip()
        df = df[df['refperboccn']!='']
        df = df.drop_duplicates(subset=['refperboccn'])
    else:
        print('[WARN] IEHE: colonne refperboccn introuvable après normalisation.')
    return df


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dateutil import parser as dtparse

try:
    from jinja2 import Template
except Exception:
    Template = None


# ================================ Logging =====================================

def info(msg: str) -> None:
    print(f"[INFO] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def err(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


# ============================== Fichiers/CSV ==================================

CANDIDATE_SEPARATORS = [",", ";", "\t", "|", ":"]

def sniff_separator(sample_path: Path, default: str = ",") -> str:
    try:
        with open(sample_path, "r", encoding="utf-8", errors="ignore") as f:
            head = "".join([next(f) for _ in range(10)])
    except Exception:
        try:
            with open(sample_path, "r", encoding="cp1252", errors="ignore") as f:
                head = "".join([next(f) for _ in range(10)])
        except Exception:
            return default
    scores = {sep: head.count(sep) for sep in CANDIDATE_SEPARATORS}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else default

def read_csv_safe(path: Optional[Path]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    detected_sep = sniff_separator(path)
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1", "iso-8859-1"):
        try:
            df = pd.read_csv(path, sep=detected_sep, encoding=enc, dtype=str, on_bad_lines="skip")
            if df.shape[1] > 1:
                info(f"Lecture réussie: {path.name} sep='{detected_sep}', encoding={enc}, shape={df.shape}")
                df.columns = [_strip_invisible_enhanced(str(c)) for c in df.columns]
                return normalize_columns(df)
        except Exception:
            continue
    try:
        df = pd.read_csv(path, sep=r"[;,\t|:]", engine="python", encoding="utf-8", dtype=str, on_bad_lines="skip")
        info(f"Fallback regex: {path.name} shape={df.shape}")
        df.columns = [_strip_invisible_enhanced(str(c)) for c in df.columns]
        return normalize_columns(df)
    except Exception:
        return pd.DataFrame()

def write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Écrit toujours en UTF-8-SIG (BOM) pour une ouverture Excel FR sans corruption d'accents.
    Séparateur ; pour compatibilité locale.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = normalize_text_cols(df.copy())
    try:
        df.to_csv(
            path,
            index=False,
            sep=";",
            encoding="utf-8-sig",
            quoting=csv.QUOTE_MINIMAL
        )
    except Exception:
        # dernier filet de sécurité (au cas où le BOM poserait problème dans un outillage tiers)
        df.to_csv(
            path,
            index=False,
            sep=";",
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL
        )



# ============================== Normalisation =================================

def _strip_invisible_enhanced(s: str) -> str:
    if s is None:
        return ""
    invisible_chars = [
        '\ufeff', '\u200b', '\u200c', '\u200d', '\u2060', '\ufffe',
        '\xa0', '\u00a0', '\u2000', '\u2001', '\u2002', '\u2003', '\u2004',
        '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200a',
        '\u202f', '\u205f', '\u3000', 'ï»¿', '​', ' ', ' ', '‎', '‏',
    ]
    result = str(s)
    for ch in invisible_chars:
        result = result.replace(ch, " ")
    result = re.sub(r"\s+", " ", result)
    return result.strip()

def _normalize_token(s: str) -> str:
    s = _strip_invisible_enhanced(str(s)).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s or "col"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols: Dict[str, str] = {}
    for c in df.columns:
        norm = _normalize_token(c)
        if norm in new_cols.values():
            suffix = 2
            while f"{norm}_{suffix}" in new_cols.values():
                suffix += 1
            norm = f"{norm}_{suffix}"
        new_cols[c] = norm
    return df.rename(columns=new_cols)

def _normalize_text_cell(x):
    """Nettoie/normalise une cellule texte pour préserver les accents et retirer les invisibles."""
    if not isinstance(x, str):
        return x
    s = x.replace("\r\n", "\n").replace("\r", "\n")
    s = _strip_invisible_enhanced(s)              # retire BOM, NO-BREAK SPACE, etc.
    s = unicodedata.normalize("NFC", s)           # normalise accents (préserve é, è, ç…)
    return s

def normalize_text_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Applique la normalisation sur toutes les colonnes texte (object)."""
    if df is None or df.empty:
        return df
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        try:
            df[c] = df[c].map(_normalize_text_cell)
        except Exception:
            # en cas de colonne mixte/erreur isolée, on passe
            pass
    return df



# =============================== Découverte ===================================

def _extract_dt_from_name(p: Path) -> datetime:
    # Extract all 8-digit tokens and try YYYYMMDD then DDMMYYYY
    tokens = re.findall(r"(\d{8})", p.name)
    for t in tokens:
        try:
            y, mth, d = int(t[0:4]), int(t[4:6]), int(t[6:8])
            if 1900 <= y <= 2100 and 1 <= mth <= 12 and 1 <= d <= 31:
                return datetime(y, mth, d)
        except Exception:
            pass
        try:
            d, mth, y = int(t[0:2]), int(t[2:4]), int(t[4:8])
            if 1900 <= y <= 2100 and 1 <= mth <= 12 and 1 <= d <= 31:
                return datetime(y, mth, d)
        except Exception:
            pass
    return datetime.fromtimestamp(p.stat().st_mtime)

def read_first_existing_csv(data_dir: Path, patterns: List[str]) -> Optional[Path]:
    matches: List[Path] = []
    for pat in patterns:
        matches.extend(list(data_dir.glob(pat)))
    if not matches:
        return None
    return max(matches, key=_extract_dt_from_name)


# ============================== Détection colonnes ============================

CANDS_EXTENDED = {
    "email": ["email","mail","mailciam","emailciam","e_mail","courriel","adresseemail","adresse_mail","emailns","mailns","mailctc","emailemailotherchannel","emailotherchannel","otherchannel","valeurcoordonneeidkpepmailciam","adrmail","emailemailotherchannel"],
    "id": ["numpersonne","id","identifiant","personid","idkpep","idrealmid","idclient","clientid","idrealmididkpep","kp_id","numctindiv","idrpprefperb"],
    "kpep": ["idkpep","kp_id","idrealmididkpep","kpep","idkp","kp","idkpepref","idrpprefperb"],
    "id_contrat": ["numctindiv","idcontrat","contratid","numcontratindividuel","idct","numct","ctid"],
    "code_soc": ["codesocappart","codesociete","codesoc","societe","code_soc_appart","socappart","occnsoc","boccnsoc"],
    "date_adh": ["dateadhesion","adhesiondate","dateadherent","dateadh","date_adh","datedadhesion"],
    "date_eff": ["dateeffetadhesion","dateeffet","effetdate","date_effet","dateeff","effet","dateevt"],
    "date_update": ["dateupdate","updatedat","maj","datedernieremaj","dateimport","updateat","dateevt","heureevt"],
    "date_rad": ["dateradassure","dateradiation","raddate","datefinadhesion","datefin","resiliationdate","date_resiliation"],
    "birthdate": ["datenaissance","birthdate","datedenaissance","birthday","datenaiss","dateofbirth"],
    "has_tp": ["hascartetp","cartetp","tpok","tpvalide","tp","cartethirdparty","thirdparty"],
    "nom": ["nom","lastname","last_name","nomlongprenom","nomlong","lastnamemiddlename"],
    "prenom": ["prenom","firstname","first_name","prenomtypeassure","firstnamemiddlename"],
    "status": ["typeassure","typeassure2","statut","status","etat","actifradie","isactive","type","origin"],
}

def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    if df.empty:
        return {k: None for k in CANDS_EXTENDED}
    cols = list(df.columns)
    out: Dict[str, Optional[str]] = {}
    for key, candidates in CANDS_EXTENDED.items():
        cand_norms = [_normalize_token(c) for c in candidates]
        found = None
        for col in cols:
            if col in cand_norms: found = col; break
        if not found:
            for col in cols:
                for cand in cand_norms:
                    if cand in col or col in cand: found = col; break
                if found: break
        if not found:
            best_col, best_score = None, 0.0
            for col in cols:
                for cand in cand_norms:
                    score = difflib.SequenceMatcher(None, col, cand).ratio()
                    if score > best_score:
                        best_score, best_col = score, col
            if best_score >= 0.75: found = best_col
        if not found and key == "email":
            for col in cols:
                if "mail" in col: found = col; break
        if not found and key == "id":
            for col in cols:
                if col.endswith("id") or "num" in col: found = col; break
        out[key] = found
    return out

def all_email_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if re.search(r"(email|e_mail|mail)", c, flags=re.I)]


# ============================= Dates & métriques ==============================

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-']+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
PHONE_LIKE = re.compile(r"^\s*(?:\+|00)?\s*[\d\.\-\s]{8,}\s*$")  # ≥ ~9 digits, no '@'

def preferred_ns_mail_ciam(df: pd.DataFrame) -> Optional[str]:
    """Retourne la colonne correspondant à mail_ciam si trouvée.
    Fallback : la première colonne détectée comme email."""
    cols = list(df.columns)
    patterns = [r"^mailciam$", r"^emailciam$", r"mail.?ciam", r"email.?ciam"]
    for pat in patterns:
        for c in cols:
            if re.search(pat, c, flags=re.I):
                return c
    ec = all_email_columns(df)
    return ec[0] if ec else None

def all_phone_columns(df: pd.DataFrame) -> List[str]:
    """Détecte les colonnes correspondant aux numéros de téléphone."""
    pats = [r"tel", r"phone", r"telephone", r"portable", r"gsm", r"mobile"]
    out = []
    for c in df.columns:
        if any(re.search(p, c, flags=re.I) for p in pats):
            out.append(c)
    return out



def parse_date_safe(s: object) -> Optional[pd.Timestamp]:
    if pd.isna(s):
        return None
    if isinstance(s, (pd.Timestamp, datetime)):
        return pd.Timestamp(s)
    txt = _strip_invisible_enhanced(str(s)).strip()
    if not txt: return None
    if re.match(r"^\d{4}-\d{2}-\d{2}", txt):
        dt = pd.to_datetime(txt, format="%Y-%m-%d", errors="coerce")
        if pd.notna(dt): return dt
    try:
        return pd.to_datetime(txt, errors="coerce", dayfirst=True)
    except Exception:
        pass
    try:
        return pd.to_datetime(dtparse.parse(txt, dayfirst=True))
    except Exception:
        return None

def days_between(a: Optional[pd.Timestamp], b: Optional[pd.Timestamp]) -> Optional[int]:
    if a is None or b is None: return None
    try:
        return int((b.normalize() - a.normalize()).days)
    except Exception:
        return None

def pct(n: float, d: float) -> float:
    return round((n / d * 100.0), 2) if d else 0.0

def as_pct_str(val: Optional[float]) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)): return "—"
    s = f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", "")
    return s + " %"

def fmt_delta_stats(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce"); s = s[~np.isnan(s)]
    if s.empty: return "—"
    try:
        mn, avg, med, mx = int(np.nanmin(s)), float(np.nanmean(s)), float(np.nanmedian(s)), int(np.nanmax(s))
        return f"{mn} / {avg:.1f} / {med:.1f} / {mx} j"
    except Exception:
        return "—"


# =============================== Badges =======================================

@dataclass
class ThresholdBand:
    green: float
    warn: float
    red: float
    mode: str = "high_is_good"  # or "low_is_good"

    def badge(self, value_pct: float) -> Tuple[str, str]:
        v = value_pct if value_pct is not None else 0.0
        if self.mode == "high_is_good":
            if v >= self.green: return ("ok", "OK")
            if v < self.red:    return ("bad", "À corriger")
            return ("warn", "À surveiller")
        else:
            if v <= self.green: return ("ok", "OK")
            if v > self.red:    return ("bad", "À corriger")
            return ("warn", "À surveiller")


# ============================ Graphiques base64 ===============================

def fig_to_base64() -> str:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=144, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ============================== Template HTML =================================

HTML_TEMPLATE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8"/>
<title>{{ title }}</title>
<style>
:root{--bg:#0f172a;--text:#0b1324;--muted:#64748b;--card:#f8fafc;--line:#e2e8f0;--ok:#10b981;--warn:#f59e0b;--bad:#ef4444;--link:#0ea5e9;}
*{box-sizing:border-box} html,body{margin:0;padding:0}
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif;color:var(--text);background:#fff}
header{background:var(--bg);color:#fff;padding:20px 28px;position:sticky;top:0;z-index:20;box-shadow:0 1px 0 #0002}
h1{margin:0;font-size:22px}
.container{padding:22px 28px}
a{color:var(--link);text-decoration:none} a:hover{text-decoration:underline}
.small{color:var(--muted);font-size:12px}
.nav{display:flex;flex-wrap:wrap;gap:10px;margin-top:8px}
.nav a{color:#c7e0ff;padding:6px 10px;border-radius:8px}
.nav a.active{background:#1f2937;color:#fff}
.grid{display:grid;gap:16px}
.cards{grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.card h3{margin:0 0 8px;font-size:13px;color:#475569;font-weight:600}
.kpi{display:flex;align-items:baseline;gap:8px}
.kpi .val{font-size:24px;font-weight:800;color:#0f172a}
.badge{display:inline-block;padding:2px 8px;border-radius:9999px;font-size:12px;font-weight:600}
.badge.ok{background:var(--ok);color:#fff}
.badge.warn{background:var(--warn);color:#111}
.badge.bad{background:var(--bad);color:#fff}
.section h2{font-size:18px;margin:26px 0 10px}
.table{border-collapse:collapse;width:100%;font-size:13px}
.table th,.table td{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:top}
.table thead th{background:#f1f5f9;position:sticky;top:64px;z-index:1}
.table tbody tr:nth-child(odd){background:#fcfdff}
th.sortable{cursor:pointer;user-select:none}
th.sortable .arrow{opacity:.35;margin-left:6px}
th.sortable[data-asc="true"] .arrow::after{content:"▲";}
th.sortable[data-asc="false"] .arrow::after{content:"▼";}
.imgbox{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:20px}
.figure{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fff}
.figure img{width:100%;height:auto;display:block}
.figure .cap{padding:8px 10px;font-size:13px;color:var(--muted);border-top:1px solid var(--line)}
footer{padding:22px 28px;color:var(--muted)}
@media print{header,.nav{position:static;box-shadow:none} .table thead th{top:auto} a[href^="#"]::after{content:" (" attr(href) ")";color:#999}}
</style>
<script>
function sortTable(th){
  const table = th.closest('table'), tbody = table.querySelector('tbody');
  const idx = Array.from(th.parentNode.children).indexOf(th);
  const asc = th.dataset.asc !== 'true';
  const rows = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((a,b)=>{
    const A=(a.children[idx].innerText||'').trim();
    const B=(b.children[idx].innerText||'').trim();
    const nA=Number(A.replace(/\s/g,'').replace(',','.'));
    const nB=Number(B.replace(/\s/g,'').replace(',','.'));
    const isNum=!Number.isNaN(nA)&&!Number.isNaN(nB);
    return asc ? (isNum ? nA-nB : A.localeCompare(B)) : (isNum ? nB-nA : B.localeCompare(A));
  });
  rows.forEach(r=>tbody.appendChild(r));
  th.dataset.asc = asc;
  document.querySelectorAll('th.sortable').forEach(x=>{ if(x!==th) x.removeAttribute('data-asc'); });
}
</script>
</head>
<body>
<header>
  <h1>{{ title }}</h1>
  <div class="nav small">
    <a href="#kpi">KPI</a>
    <a href="#graphs">Graphiques</a>
    <a href="#emails">E-mails</a>
    <a href="#xci">NS↔CIAM / IEHE</a>
    <a href="#doublons">Doublons, risques et emails manquants</a>
    <a href="#tops">Tops</a>
    <a href="#exports">Exports</a>
  </div>
</header>

<div class="container">

<!-- ========================= Résumé exécutif ========================= -->
<div id="kpi" class="grid cards">

  {% if kpi_file and kpi_file.volumetry and kpi_file.volumetry.KPI1_vol_contrat %}
  <div class="card auto-nbsp">
    <h3>Contrats (KPI)</h3>
    <div class="kpi"><div class="val">{{ kpi_file.volumetry.KPI1_vol_contrat }}</div></div>
  </div>
  {% endif %}

  {% if kpi_file and kpi_file.resiliations and (kpi_file.resiliations.total is not none) %}
  <div class="card auto-nbsp">
    <h3>Résiliations totales (KPI)</h3>
    <div class="kpi"><div class="val">{{ kpi_file.resiliations.total }}</div></div>
  </div>
  {% endif %}

  {% if kpi_file and kpi_file.anomalies %}
  <div class="card auto-nbsp">
    <h3>Total anomalies (KPI)</h3>
    <div class="kpi"><div class="val">{{ kpi_file.anomalies.total }}</div></div>
    <div class="small">
      {% for t in kpi_file.anomalies.ranking %}
        {{ t.type }}: <b>{{ t.count }}</b> ({{ t.pct if t.pct is not none else '—' }}%){{ ", " if not loop.last else "" }}
      {% endfor %}
    </div>
  </div>
  {% endif %}

  <div class="card auto-nbsp"><h3>Assurés</h3>
    <div class="kpi"><div class="val">{{ kpi.assures_n }}</div></div>
  </div>

  <div class="card auto-nbsp"><h3>Assurés%</h3>
    <div class="kpi"><div class="val">{{ kpi.assures_pct }}</div><span class="small">&nbsp;({{ kpi.assures_n }})</span></div>
  </div>

  <div class="card auto-nbsp"><h3>Souscriptions du jour</h3>
    <div class="kpi"><div class="val">{{ kpi.subs_jour_n }}</div><span class="small">&nbsp;({{ kpi.subs_jour_pct }})</span></div>
    <div class="small">Types: ASSPRI, MPRETR, MPVRET</div>
  </div>

  <div class="card auto-nbsp"><h3>Comptes CIAM créés</h3>
    <div class="kpi"><div class="val">{{ kpi.ciam_crees_n }}</div><span class="small">&nbsp;({{ kpi.ciam_crees_pct }})</span></div>
    <div class="small">Règles: NS.valeur_coordonnee ∈ CIAM_EMAIL.mail OU NS.idkpep ∈ CIAM_KPEP.idkpep OU NS.valeur_coordonnee = CIAM_KPEP.mail</div>
  </div>

  <div class="card auto-nbsp"><h3>Personnes IEHE créées</h3>
    <div class="kpi"><div class="val">{{ kpi.iehe_creees_n }}</div><span class="small">&nbsp;({{ kpi.iehe_creees_pct }})</span></div>
    <div class="small">Règle: NS.num_personne ∈ IEHE.refperboccn</div>
  </div>


  <div class="card auto-nbsp"><h3>Radiés</h3>
    <div class="kpi"><div class="val">{{ kpi.radies_pct }}</div><span class="small">&nbsp;({{ kpi.radies_n }})</span></div>
  </div>

  <div class="card auto-nbsp"><h3>E-mails présents</h3>
  <div class="card auto-nbsp"><h3>E-mails identiques (NS.mail = NS.mail_ciam)</h3>
    <div class="kpi"><div class="val">{{ kpi.emails_identiques_pct }}</div>
      <span class="small">&nbsp;({{ kpi.emails_identiques_n }} / {{ kpi.emails_identiques_total }})</span>
    </div>
  </div>
    <div class="kpi"><div class="val">{{ kpi.emails_present }} <span class="small">({{ kpi.emails_present_pct }})</span></div></div>
  </div>

  <div class="card auto-nbsp"><h3>Valides (regex)</h3>
    <div class="kpi"><div class="val">{{ kpi.emails_valid_pct }}</div>
      <span class="badge {{ kpi.emails_valid_badge_cls }}">{{ kpi.emails_valid_badge }}</span>
    </div>
    <div class="small">{{ kpi.emails_valid }}/{{ kpi.emails_non_empty }} cellules e-mail valides</div>
  </div>

  <div class="card auto-nbsp"><h3>Mismatch NS↔CIAM</h3>
    <div class="kpi"><div class="val">{{ kpi.ciam_email_presence_pct }}</div>
      <span class="badge {{ kpi.ciam_email_presence_badge_cls }}">{{ kpi.ciam_email_presence_badge }}</span>
    </div>
    <div class="small">{{ kpi.ciam_email_presence_n }}/{{ kpi.ciam_email_total }} e-mails NS retrouvés dans CM</div>
  </div>

  <div class="card auto-nbsp"><h3>Mismatch NS↔IEHE</h3>
    <div class="kpi"><div class="val">{{ kpi.iehe_presence_pct }}</div>
      <span class="badge {{ kpi.iehe_presence_badge_cls }}">{{ kpi.iehe_presence_badge }}</span>
    </div>
    <div class="small">{{ kpi.iehe_presence_n }}/{{ kpi.iehe_total }} idrpp NS retrouvés dans IEHE</div>
  </div>

  <div class="card auto-nbsp"><h3>TP conformes (0/{{ kpi.tp_window }}j)</h3>
    <div class="kpi"><div class="val">{{ kpi.tp_ok_pct }}</div>
      <span class="badge {{ kpi.tp_ok_badge_cls }}">{{ kpi.tp_ok_badge }}</span>
    </div>
    <div class="small">{{ kpi.tp_ok }}/{{ kpi.tp_expected }} attendues</div>
  </div>

  <div class="card auto-nbsp"><h3>TP future (≥{{ kpi.tp_window_plus1 }}j)</h3>
    <div class="kpi"><div class="val">{{ kpi.tp_future_pct }}</div></div>
    <div class="small">{{ kpi.tp_future }}/{{ kpi.tp_expected }}</div>
  </div>

  <div class="card auto-nbsp"><h3>TP effet &lt; adhésion</h3>
    <div class="kpi"><div class="val">{{ kpi.tp_negative_pct }}</div></div>
    <div class="small">{{ kpi.tp_negative }}/{{ kpi.tp_expected }}</div>
  </div>

  <div class="card auto-nbsp"><h3>∆ effet (min / moyenne / médiane / max)</h3>
    <div class="kpi"><div class="val">{{ kpi.delta_effet }}</div></div>
  </div>

  <div class="card auto-nbsp"><h3>Doublons</h3>
    <div class="kpi"><div class="val">{{ kpi.dup_pct }}</div>
      <span class="badge {{ kpi.dup_badge_cls }}">{{ kpi.dup_badge }}</span>
    </div>
    <div class="small">{{ kpi.dup_n }} enregistrements en doublon</div>
  </div>
</div>


<!-- ====================== KPI supplémentaires ====================== -->
<div id="kpi-additional" class="section">
  <h2>KPI supplémentaires</h2>
  <div class="grid cards">
    <div class="card auto-nbsp"><h3>Kpi 1 - Nbr de souscriptions du jour</h3>
      <div class="kpi"><div class="val">{{ kpi.subs_jour_n }}</div></div>
    </div>
    <div class="card auto-nbsp"><h3>Kpi 2 - Nombre de comptes CIAM créés</h3>
      <div class="kpi"><div class="val">{{ kpi.ciam_crees_n }}</div><span class="small">&nbsp;({{ kpi.ciam_crees_pct }})</span></div>
    </div>
    <div class="card auto-nbsp"><h3>Kpi 3 - Nombre de personnes créées dans IEHE</h3>
      <div class="kpi"><div class="val">{{ kpi.iehe_creees_n }}</div><span class="small">&nbsp;({{ kpi.iehe_creees_pct }})</span></div>
    </div>
    <div class="card auto-nbsp"><h3>Kpi 4 - Taux de circulation de la donnée</h3>
      <div class="kpi"><div class="val">{{ kpi.kpi4_taux_circulation_donnee_pct }}</div></div>
      <div class="small">min(Kpi2, Kpi3)</div>
    </div>
    <div class="card auto-nbsp"><h3>Kpi 5 - Cartes TP éligibles</h3>
      <div class="kpi"><div class="val">{{ kpi.tp_ok }}</div></div>
      <div class="small">delta ≤ {{ kpi.tp_window }} jours</div>
    </div>
    <div class="card auto-nbsp"><h3>Kpi 6 - Taux Cartes TP présentes en GED</h3>
      <div class="kpi"><div class="val">{{ kpi.kpi6_taux_cartes_tp_presentes_ged_pct }}</div></div>
    </div>
    <div class="card auto-nbsp"><h3>Kpi 7 - Indice de qualité des adresses Emails CIAM</h3>
      <div class="kpi"><div class="val">{{ kpi.emails_identiques_pct }}</div></div>
      <div class="small">{{ kpi.emails_identiques_n }}/{{ kpi.emails_identiques_total }} identiques</div>
    </div>
    <div class="card auto-nbsp"><h3>Kpi 8 - Nombre d'adresse Email en double</h3>
      <div class="kpi"><div class="val">{{ kpi.kpi8_nombre_adresses_email_en_double }}</div></div>
    </div>
    <div class="card auto-nbsp"><h3>Kpi 9 - Indice de compte CIAM créé sans Email</h3>
      <div class="kpi"><div class="val">{{ kpi.kpi9_indice_compte_ciam_sans_email_pct }}</div></div>
      <div class="small">{{ kpi.kpi9_sans_email_n }} comptes sans e-mail</div>
    </div>
    <div class="card auto-nbsp"><h3>Kpi 10 - Souscriptions en double sur la journée</h3>
      <div class="kpi"><div class="val">{{ kpi.kpi10_souscriptions_en_double_du_jour_n }}</div></div>
    </div>
  </div>
</div>
<!-- ============================ Graphiques ============================ -->
<div id="graphs" class="section">
  <h2>Graphiques</h2>
  <div class="imgbox">
    {% if graphs.age %}
    <figure class="figure"><img alt="Distribution des âges" src="data:image/png;base64,{{ graphs.age }}"/>
      <figcaption class="cap">Répartition des âges.</figcaption></figure>
    {% endif %}
    {% if graphs.delta_effet %}
    <figure class="figure"><img alt="Distribution des délais d'effet" src="data:image/png;base64,{{ graphs.delta_effet }}"/>
      <figcaption class="cap">Répartition des délais d'effet (jours).</figcaption></figure>
    {% endif %}

    {# === KPI (depuis le fichier KPI) : inséré juste après "∆ d'effet" === #}
    {% if kpi_file and kpi_file.graphs %}
      {% if kpi_file.graphs.anomalies_pie %}
      <figure class="figure">
        <img alt="Déphasages (camembert)" src="data:image/png;base64,{{ kpi_file.graphs.anomalies_pie }}"/>
        <figcaption class="cap">
          Qualité des données (déphasages).
          {% if kpi_file.graphs_meta and kpi_file.graphs_meta.anomalies %}
            <div>Répartition :
              {% for r in kpi_file.graphs_meta.anomalies %}
                {{ r.label }}: <b>{{ r.count }}</b> ({{ r.pct }}%){{ ", " if not loop.last else "" }}
              {% endfor %}
            </div>
          {% endif %}
        </figcaption>
      </figure>
      {% endif %}

      {% if kpi_file.graphs.volumetry_pie %}
      <figure class="figure">
        <img alt="Volumétrie contrats (camembert)" src="data:image/png;base64,{{ kpi_file.graphs.volumetry_pie }}"/>
        <figcaption class="cap">
          Volumétrie contrats.
          {% if kpi_file.graphs_meta and kpi_file.graphs_meta.volumetry %}
            <div>Détails :
              {% for r in kpi_file.graphs_meta.volumetry %}
                {{ r.label }}: <b>{{ r.count }}</b>{{ ", " if not loop.last else "" }}
              {% endfor %}
            </div>
          {% endif %}
        </figcaption>
      </figure>
      {% endif %}

      {% if kpi_file.graphs.motif_bar %}
      <figure class="figure">
        <img alt="Répartition par motif (barres)" src="data:image/png;base64,{{ kpi_file.graphs.motif_bar }}"/>
        <figcaption class="cap">
          Répartition par motif (TOP).
          {% if kpi_file.graphs_meta and kpi_file.graphs_meta.motif %}
            <div>
              {% for r in kpi_file.graphs_meta.motif %}
                {{ r.label }}: <b>{{ r.count }}</b>{{ ", " if not loop.last else "" }}
              {% endfor %}
            </div>
          {% endif %}
        </figcaption>
      </figure>
      {% endif %}

      {% if kpi_file.graphs.codesoc_bar %}
      <figure class="figure">
        <img alt="Répartition par code_soc_appart (barres)" src="data:image/png;base64,{{ kpi_file.graphs.codesoc_bar }}"/>
        <figcaption class="cap">
          Répartition par code_soc_appart (010/044/052).
          {% if kpi_file.graphs_meta and kpi_file.graphs_meta.codesoc %}
            <div>
              {% for r in kpi_file.graphs_meta.codesoc %}
                {{ r.label }}: <b>{{ r.count }}</b>{{ ", " if not loop.last else "" }}
              {% endfor %}
            </div>
          {% endif %}
        </figcaption>
      </figure>
      {% endif %}
    {% endif %}
  </div>
</div>

<!-- ========================= E-mails invalides / jetables =================== -->
<div id="emails" class="section">
  <h2>E-mails invalides / jetables</h2>
  <table class="table">
    <thead><tr>
      <th class="sortable" onclick="sortTable(this)">Nom <span class="arrow"></span></th>
      <th class="sortable" onclick="sortTable(this)">Prénom <span class="arrow"></span></th>
      <th class="sortable" onclick="sortTable(this)">Naissance <span class="arrow"></span></th>
      <th class="sortable" onclick="sortTable(this)">KPEP <span class="arrow"></span></th>
      <th class="sortable" onclick="sortTable(this)">ID contrat <span class="arrow"></span></th>
      <th class="sortable" onclick="sortTable(this)">ID personne <span class="arrow"></span></th>
      <th class="sortable" onclick="sortTable(this)">E-mail <span class="arrow"></span></th>
      <th class="sortable" onclick="sortTable(this)">Raison anomalie <span class="arrow"></span></th>
    </tr></thead>
    <tbody>
      {% for row in tables.invalid_emails %}
      <tr>
        <td>{{ row.nom }}</td>
        <td>{{ row.prenom }}</td>
        <td>{{ row.birthdate }}</td>
        <td>{{ row.kpep }}</td>
        <td>{{ row.id_contrat }}</td>
        <td>{{ row.id_personne }}</td>
        <td>{{ row.email }}</td>
        <td>{{ row.reason }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>

<!-- ========================= NS ↔ CIAM / IEHE ======================== -->
<div id="xci" class="section">
  <h2>NS ↔ CIAM / IEHE (mismatch)</h2>

  <h3>Mismatch CIAM (détails e-mail)</h3>
  <table class="table">
    <thead><tr>
      <th class="sortable" onclick="sortTable(this)">ID <span class="arrow"></span></th>
      <th class="sortable" onclick="sortTable(this)">E-mail NS <span class="arrow"></span></th>
      <th class="sortable" onclick="sortTable(this)">E-mail CIAM <span class="arrow"></span></th>
    </tr></thead>
    <tbody>
      {% for row in tables.mismatch %}
      <tr><td>{{ row.id }}</td><td>{{ row.ns }}</td><td>{{ row.ciam }}</td></tr>
      {% endfor %}
    </tbody>
  </table>

  {% if tables.mismatch_iehe %}
  <h3 style="margin-top:16px">Mismatch IEHE (détails e-mail)</h3>
  <table class="table">
    <thead><tr>
      <th class="sortable" onclick="sortTable(this)">ID <span class="arrow"></span></th>
      <th class="sortable" onclick="sortTable(this)">E-mail NS <span class="arrow"></span></th>
      <th class="sortable" onclick="sortTable(this)">E-mail IEHE <span class="arrow"></span></th>
    </tr></thead>
    <tbody>
      {% for row in tables.mismatch_iehe %}
      <tr><td>{{ row.id }}</td><td>{{ row.ns }}</td><td>{{ row.iehe }}</td></tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}
</div>

<!-- ==================== Doublons / risques / manquants ================= -->
<div id="doublons" class="section">
  <h2>Doublons, risques et emails manquants</h2>
  <table class="table">
    <thead><tr>
      <th class="sortable" onclick="sortTable(this)">Clé <span class="arrow"></span></th>
      <th class="sortable" onclick="sortTable(this)">Email <span class="arrow"></span></th>
      <th class="sortable" onclick="sortTable(this)">Occurrences <span class="arrow"></span></th>
      <th class="sortable" onclick="sortTable(this)">IDs <span class="arrow"></span></th>
    </tr></thead>
    <tbody>
      {% for row in tables.dups %}
      <tr><td>{{ row.key }}</td><td>{{ row.email }}</td><td>{{ row.count }}</td><td>{{ row.ids }}</td></tr>
      {% endfor %}
    </tbody>
  </table>
</div>

<!-- ============================== Tops ================================ -->
<div id="tops" class="section">
  <h2>Tops</h2>
  <div class="grid cards">
    <div class="card auto-nbsp">
      <h3>Top domaines d'e-mail</h3>
      <div class="small">{% for dom in tables.top_domains %}<div>{{ dom.domain }} – <b>{{ dom.count }}</b></div>{% endfor %}</div>
    </div>
    <div class="card auto-nbsp">
      <h3>Top doublons e-mail</h3>
      <div class="small">{% for dup in tables.top_dup_emails %}<div>{{ dup.email }} – <b>{{ dup.count }}</b></div>{% endfor %}</div>
    </div>
  </div>
</div>

<!-- ============================ Exports =============================== -->
<div id="exports" class="section">
  <h2>Exports</h2>
  <ul>
    {% for exp in exports %}
    <li><a href="{{ exp.href }}">{{ exp.label }}</a> <span class="small">({{ exp.rows }} lignes)</span></li>
    {% endfor %}
  </ul>
</div>

</div>

<footer class="small">Généré le {{ now }}</footer>
</body>
</html>"""


# ======================= Parsing KPI.csv (robuste) ============================

def parse_kpi_file_text(text: str) -> Dict:
    """
    Parse un KPI.csv libre (mélange texte/valeurs + table code_soc/motif/count).
    Renvoie un dict prêt à être injecté dans le template : kpi_file = {
      volumetry: {...}, anomalies: {total, ranking[...]}, resiliations: {total,...},
      graphs: {anomalies_pie, volumetry_pie, motif_bar, codesoc_bar},
      graphs_meta: {anomalies[], volumetry[], motif[], codesoc[]}
    }
    """
    out = {"volumetry": {}, "anomalies": {}, "resiliations": {}, "graphs": {}, "graphs_meta": {}}

    # ---- Anomalies (texte libre) ----
    def get_num(keywords):
        for kw in keywords:
            m = re.search(rf"{kw}[^0-9]*:\s*([0-9]+)", text, flags=re.I)
            if m: return int(m.group(1))
        return None

    counts = {
        "mail": get_num(["déphasages mails","dephasages mails","mails"]),
        "telephone": get_num(["déphasage téléphone","dephasage telephone","téléphone","telephone"]),
        "naissance": get_num(["naissance"]),
        "deces": get_num(["décès","deces"]),
        "adresse": get_num(["adresses","adresse"]),
    }
    counts = {k: (v or 0) for k, v in counts.items()}
    total_anom = sum(counts.values())
    out["anomalies"] = {
        "total": total_anom,
        "ranking": sorted(
            [{"type": k, "count": v, "pct": round(100*v/total_anom, 2) if total_anom else None} for k, v in counts.items()],
            key=lambda x: x["count"], reverse=True
        )
    }

    # ---- Volumétrie (paires clé/valeur sur 2 lignes) ----
    for k in ["KPI1_vol_contrat", "KPI1_vol_contrat2", "KPI1_vol_contrat5", "KPI1_vol_contrat12"]:
        m = re.search(rf"^{k}\s*[\r\n]+([0-9]+)", text, flags=re.M)
        if m:
            out["volumetry"][k] = int(m.group(1))

    # ---- Résiliations (table code_soc_appart,motif_resiliation,count2) ----
    needle = "code_soc_appart,motif_resiliation,count2"
    pos = text.find(needle)
    res_df = None
    if pos != -1:
        rows = []
        for line in text[pos:].splitlines()[1:]:
            s = line.strip()
            if not s: break
            if s.count(",") != 2: break  # sort des autres blocs
            a, b, c = s.split(",")
            if not c.isdigit(): break
            rows.append([a, b, int(c)])
        if rows:
            res_df = pd.DataFrame(rows, columns=["code_soc_appart", "motif", "count"])
            out["resiliations"]["total"] = int(res_df["count"].sum())
            out["resiliations"]["by_motif"] = res_df.groupby("motif", as_index=False)["count"].sum().sort_values("count", ascending=False)
            out["resiliations"]["by_code"]  = res_df.groupby("code_soc_appart", as_index=False)["count"].sum().sort_values("count", ascending=False)

    # ---- Graphiques KPI (base64) ----
    # Anomalies pie
    if total_anom > 0:
        labels = ["mail", "téléphone", "adresse", "naissance", "décès"]
        vals = [counts.get("mail", 0), counts.get("telephone", 0), counts.get("adresse", 0),
                counts.get("naissance", 0), counts.get("deces", 0)]
        plt.figure()
        plt.pie(vals, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title("Qualité des données – déphasages")
        out["graphs"]["anomalies_pie"] = fig_to_base64()
        out["graphs_meta"]["anomalies"] = [{"label": lab, "count": int(v), "pct": round(100*v/total_anom, 1)} for lab, v in zip(labels, vals)]

    # Volumétrie pie
    vols = [("≤2j", "KPI1_vol_contrat2"), ("≤5j", "KPI1_vol_contrat5"), ("≤12j", "KPI1_vol_contrat12")]
    vvals = [out["volumetry"].get(k, 0) for _, k in vols]
    if any(vvals):
        plt.figure()
        plt.pie(vvals, labels=[a for a, _ in vols], autopct='%1.1f%%', startangle=90)
        plt.title("Volumétrie contrats")
        out["graphs"]["volumetry_pie"] = fig_to_base64()
        out["graphs_meta"]["volumetry"] = [{"label": a, "count": int(out["volumetry"].get(k, 0))} for a, k in vols]

    # Barres motif + codesoc
    if res_df is not None and not res_df.empty:
        top_m = res_df.groupby("motif", as_index=False)["count"].sum().sort_values("count", ascending=False).head(12)
        plt.figure()
        plt.bar(top_m["motif"], top_m["count"])
        plt.title("Résiliations — répartition par motif")
        plt.xticks(rotation=45, ha='right')
        out["graphs"]["motif_bar"] = fig_to_base64()
        out["graphs_meta"]["motif"] = [{"label": r["motif"], "count": int(r["count"])} for _, r in top_m.iterrows()]

        filt = res_df[res_df["code_soc_appart"].isin(["010", "044", "052"])].groupby("code_soc_appart", as_index=False)["count"].sum()
        if not filt.empty:
            plt.figure()
            plt.bar(filt["code_soc_appart"], filt["count"])
            plt.title("Résiliations — par code_soc_appart (010/044/052)")
            out["graphs"]["codesoc_bar"] = fig_to_base64()
            out["graphs_meta"]["codesoc"] = [{"label": r["code_soc_appart"], "count": int(r["count"])} for _, r in filt.iterrows()]

    return out


# ============================ Calcul KPI & tables =============================

@dataclass
class ArgsNS:
    data_dir: Path
    out: Path
    title: str
    tp_window_days: int
    exclude_soc: List[str]
    warn_threshold_email: float
    red_threshold_email: float
    warn_threshold_tp: float
    red_threshold_tp: float
    warn_threshold_dup: float
    green_threshold_dup: float

def try_parse_dates(df: pd.DataFrame, cols: List[Optional[str]]) -> None:
    for c in cols:
        if c and c in df.columns:
            df[c] = df[c].map(parse_date_safe)

def compute_payload(NS: pd.DataFrame,
                    CM: pd.DataFrame,
                    IEHE: pd.DataFrame,
                    args: ArgsNS,
                    col_ns: Dict[str, Optional[str]],
                    col_ci: Dict[str, Optional[str]],
                    col_iehe: Dict[str, Optional[str]]) -> Tuple[Dict, Dict[str, pd.DataFrame]]:
    anomalies: Dict[str, pd.DataFrame] = {}

    total = len(NS)

    # Radiés
    radies_n = 0
    date_rad = col_ns.get("date_rad")
    if date_rad and date_rad in NS.columns:
        rad_col = NS[date_rad]
        if not np.issubdtype(rad_col.dtype, np.datetime64):
            import pandas as pd
            rad_col = rad_col.map(parse_date_safe)
        radies_n = int(rad_col.notna().sum())
    radies_pct_val = pct(radies_n, total)

    # Assurés (porteurs)
    assures_n = conjoints_n = 0
    status_col = col_ns.get("status")
    if status_col and status_col in NS.columns:
        status_norm = NS[status_col].fillna("").astype(str).map(_strip_invisible_enhanced).str.replace(r"\s+","", regex=True).str.upper()
        conjoin_mask = status_norm.str.startswith("CONJOI")
        assur_mask = status_norm.str.startswith("ASSPRI")

        s = NS[status_col].fillna("").astype(str).map(_strip_invisible_enhanced).str.replace(r"\s+", "", regex=True).str.upper()
        assures_n = int(s.str.startswith("ASSPRI").sum())
        conjoints_n = int(s.str.startswith("CONJOI").sum())
    assures_pct_val   = pct(assures_n, total)
    conjoints_pct_val = pct(conjoints_n, total)

    # =================== Scan multi-colonnes e-mail (NEW_S / CK) ===================
    email_cols = all_email_columns(NS)
    mail_ciam_col = preferred_ns_mail_ciam(NS)

    nom_col = col_ns.get("nom") or ""
    prenom_col = col_ns.get("prenom") or ""
    birth_col = col_ns.get("birthdate") or ""
    id_personne_col = col_ns.get("id") or ""
    kpep_col = col_ns.get("kpep") or ""
    id_contrat_col = col_ns.get("id_contrat") or ""

    def get_cell(row, colname):
        try:
            return row[colname] if colname in row and pd.notna(row[colname]) else ""
        except Exception:
            return ""

    disposable_domains = {
        "yopmail.com","yopmail.fr","yopmail.net","mailinator.com","mailinator.net",
        "guerrillamail.com","10minutemail.com","tempmail.com","trashmail.com",
        "maildrop.cc","moakt.com","sharklasers.com","getnada.com","linshiyouxiang.net",
        "dispostable.com","mailnesia.com","mintemail.com","tempsky.com"
    }

    invalid_rows: List[Dict] = []
    non_empty_cells = 0
    valid_cells = 0

    if email_cols:
        cols_needed = email_cols + [c for c in [nom_col, prenom_col, birth_col, id_personne_col, kpep_col, id_contrat_col] if c]
        for _, row in NS[cols_needed].fillna("").iterrows():
            for c in email_cols:
                raw = str(row[c]).strip()
                if raw == "" or raw.lower() in {"nan","none","null"}:
                    continue
                non_empty_cells += 1
                v = _strip_invisible_enhanced(raw)
                v_lower = v.lower()

                reasons = []
                if "@" not in v_lower and PHONE_LIKE.match(v_lower):
                    reasons.append("Numéro de téléphone détecté")
                if "@" in v_lower and not EMAIL_REGEX.match(v):
                    reasons.append("Format e-mail invalide")
                if "@" in v_lower:
                    dom = v_lower.split("@")[-1]
                    if dom in disposable_domains:
                        reasons.append("Domaine jetable")

                if not reasons:
                    valid_cells += 1
                else:
                    invalid_rows.append({
                        "nom": get_cell(row, nom_col),
                        "prenom": get_cell(row, prenom_col),
                        "birthdate": get_cell(row, birth_col),
                        "kpep": get_cell(row, kpep_col),
                        "id_contrat": get_cell(row, id_contrat_col),
                        "id_personne": get_cell(row, id_personne_col),
                        "email": v,
                        "reason": ", ".join(reasons)
                    })

    emails_present = non_empty_cells
    emails_valid = valid_cells
    emails_present_pct = pct(emails_present, total if total else emails_present)
    emails_valid_pct = pct(emails_valid, emails_present if emails_present else 0)

    anomalies["invalid_emails"] = pd.DataFrame(invalid_rows)

    
    # ====== Couverture e-mail NS -> CIAM (NS.mail_ciam vs CM.email, fallback via KPEP). Inclut CONJOI. ======
    ciam_email_presence_n = 0
    ciam_email_total = 0
    if mail_ciam_col and not CM.empty and "email" in col_ci and col_ci["email"] in CM.columns:
        ns_series = NS[[mail_ciam_col] + ([col_ns.get("kpep")] if col_ns.get("kpep") in NS.columns else [])].copy()
        ns_series[mail_ciam_col] = ns_series[mail_ciam_col].fillna("").astype(str).map(_strip_invisible_enhanced).str.strip().str.lower()
        ns_series['__kpep'] = (NS[col_ns["kpep"]] if col_ns.get("kpep") in NS.columns else "").fillna("").astype(str).map(_strip_invisible_enhanced).str.strip()
        has_mail = ns_series[mail_ciam_col] != ""
        ciam_email_total = int(has_mail.sum())  # denom: NS with mail_ciam present

        cm_emails = CM[col_ci["email"]].fillna("").astype(str).map(_strip_invisible_enhanced).str.strip().str.lower().drop_duplicates()
        cm_set = set(cm_emails.tolist())
        cm_kpep = (CM[col_ci["kpep"]].fillna("").astype(str).map(_strip_invisible_enhanced).str.strip() if (col_ci.get("kpep") in CM.columns) else pd.Series([], dtype=str))
        cm_kpep_set = set(cm_kpep.tolist())

        def present(row):
            m = row[mail_ciam_col]
            if m and m in cm_set:
                return True
            kp = row["__kpep"]
            return bool(kp and kp in cm_kpep_set)

        ciam_email_presence_n = int(ns_series.apply(present, axis=1).sum())
    ciam_email_presence_pct_val = pct(ciam_email_presence_n, ciam_email_total)
    
    
    # ====== Couverture idrpp NS -> IEHE (NS.num_personne vs IEHE.refperboccn). Fallback via mail ou téléphone. Inclut CONJOI. ======
    iehe_presence_n = 0
    iehe_total = int(len(NS))
    if col_ns.get("id") and col_iehe.get("id"):
        base_df = NS.copy()

        ns_ids = base_df[col_ns["id"]].fillna("").astype(str).map(_strip_invisible_enhanced).str.strip() if col_ns["id"] in base_df.columns else pd.Series([], dtype=str)
        iehe_ids = IEHE[col_iehe["id"]].fillna("").astype(str).map(_strip_invisible_enhanced).str.strip().drop_duplicates() if col_iehe["id"] in IEHE.columns else pd.Series([], dtype=str)
        iehe_id_set = set(iehe_ids.tolist())

        ns_mail = base_df[mail_ciam_col].fillna("").astype(str).map(_strip_invisible_enhanced).str.strip().str.lower() if mail_ciam_col else pd.Series([], dtype=str)
        ns_phone_cols = all_phone_columns(base_df)
        ns_phone = base_df[ns_phone_cols[0]].fillna("").astype(str).map(_strip_invisible_enhanced).str.strip() if ns_phone_cols else pd.Series([], dtype=str)

        iehe_mail_col = col_iehe.get("email")
        iehe_mail = IEHE[iehe_mail_col].fillna("").astype(str).map(_strip_invisible_enhanced).str.strip().str.lower() if (iehe_mail_col in IEHE.columns) else pd.Series([], dtype=str)
        iehe_phone_cols = all_phone_columns(IEHE)
        iehe_phone = IEHE[iehe_phone_cols[0]].fillna("").astype(str).map(_strip_invisible_enhanced).str.strip() if iehe_phone_cols else pd.Series([], dtype=str)

        iehe_total = int(len(base_df))

        iehe_mail_set = set(iehe_mail.tolist())
        iehe_phone_set = set(iehe_phone.tolist())

        match_series = ns_ids.map(lambda x: bool(x and x in iehe_id_set))
        if len(ns_mail) == len(match_series):
            match_series = match_series | ns_mail.map(lambda x: bool(x and x in iehe_mail_set))
        if len(ns_phone) == len(match_series):
            match_series = match_series | ns_phone.map(lambda x: bool(x and x in iehe_phone_set))

        iehe_presence_n = int(match_series.sum())
    iehe_presence_pct_val = pct(iehe_presence_n, iehe_total)
    
    # Mismatch tables (facultatif)
    mismatch_rows: List[Dict] = []
    if not NS.empty and not CM.empty and email_cols and col_ns.get("id") and col_ci.get("email"):
        key = col_ns["id"]
        ns_key = NS[key].fillna("").astype(str).map(_strip_invisible_enhanced).str.strip() if key in NS.columns else pd.Series([], dtype=str)
        ci_email_series = CM[col_ci["email"]].fillna("").astype(str).map(_strip_invisible_enhanced)
        ci_key = (CM[col_ci.get("id")] if col_ci.get("id") in CM.columns else ci_email_series).fillna("").astype(str).map(_strip_invisible_enhanced).str.strip()
        ci_email_map = dict(zip(ci_key, ci_email_series))
        ns_email_pref = NS[email_cols[0]].fillna("").astype(str).map(_strip_invisible_enhanced)  # affichage
        for i, k in ns_key.items():
            if not k: continue
            ci_email = ci_email_map.get(k, "")
            if ci_email and ci_email.strip().lower() != ns_email_pref.iloc[i].strip().lower():
                mismatch_rows.append({"id": k, "ns": ns_email_pref.iloc[i], "ciam": ci_email})
    anomalies["mismatch"] = pd.DataFrame(mismatch_rows)

    mismatch_iehe_rows: List[Dict] = []
    if not NS.empty and not IEHE.empty and email_cols and col_ns.get("id") and col_iehe.get("email"):
        key = col_ns["id"]
        ns_key = NS[key].fillna("").astype(str).map(_strip_invisible_enhanced).str.strip()
        ie_email_series = IEHE[col_iehe["email"]].fillna("").astype(str).map(_strip_invisible_enhanced)
        ie_key = (IEHE[col_iehe.get("id")] if col_iehe.get("id") in IEHE.columns else ie_email_series).fillna("").astype(str).map(_strip_invisible_enhanced).str.strip()
        ie_email_map = dict(zip(ie_key, ie_email_series))
        ns_email_pref = NS[email_cols[0]].fillna("").astype(str).map(_strip_invisible_enhanced)
        for i, k in ns_key.items():
            if not k: continue
            ie_email = ie_email_map.get(k, "")
            if ie_email and ie_email.strip().lower() != ns_email_pref.iloc[i].strip().lower():
                mismatch_iehe_rows.append({"id": k, "ns": ns_email_pref.iloc[i], "iehe": ie_email})
    anomalies["mismatch_iehe"] = pd.DataFrame(mismatch_iehe_rows)

    # (TP pour cartes uniquement)
    date_adh = col_ns.get("date_adh"); date_eff = col_ns.get("date_eff"); code_soc = col_ns.get("code_soc")
    try_parse_dates(NS, [date_adh, date_eff, col_ns.get("date_update"), col_ns.get("birthdate"), col_ns.get("date_rad")])
    tp_expected = tp_ok = 0
    df_tp = NS.copy()
    if 'conjoin_mask' in locals():
        df_tp = df_tp.loc[~conjoin_mask].copy()
        # Garder uniquement les assurés éligibles : ASSPRI, MPRETR, MPVRET
    status_col = col_ns.get("status")
    if status_col and status_col in df_tp.columns:
        df_tp = df_tp[
            df_tp[status_col].fillna("").astype(str).str.upper().isin(["ASSPRI", "MPRETR", "MPVRET"])
        ].copy()

    if code_soc and code_soc in df_tp.columns and args.exclude_soc:
        excl = df_tp[code_soc].astype(str).map(_strip_invisible_enhanced).isin(set(args.exclude_soc))
        df_tp = df_tp.loc[~excl].copy()
    if date_adh in df_tp.columns and date_eff in df_tp.columns:
        tp_expected = len(df_tp)
        delta = df_tp.apply(lambda r: days_between(r.get(date_adh), r.get(date_eff)), axis=1)
        df_tp["delta"] = delta
        ok_mask = df_tp["delta"].map(lambda d: (pd.notna(d) and (d >= 0) and (d <= args.tp_window_days)))
    tp_ok = int(ok_mask.sum())
    tp_pct = pct(tp_ok, tp_expected if tp_expected else 0)
    tp_future = int(df_tp["delta"].map(lambda d: (pd.notna(d) and d >= args.tp_window_days + 1)).sum()) if "delta" in df_tp else 0
    tp_future_pct = pct(tp_future, tp_expected if tp_expected else 0)
    tp_negative = int(df_tp["delta"].map(lambda d: (pd.notna(d) and d < 0)).sum()) if "delta" in df_tp else 0
    
    # TP conformes = toutes les cartes avec delta <= fenêtre (compte déjà les négatifs)
    tp_conformes = int(min(tp_ok + tp_negative, tp_expected))
    tp_conformes_pct_val = pct(tp_conformes, tp_expected if tp_expected else 0)
    tp_negative_pct = pct(tp_negative, tp_expected if tp_expected else 0)
    delta_vals = df_tp["delta"].astype(float) if "delta" in df_tp.columns else pd.Series(dtype=float)

    
    # Doublons (actifs): id_personne or (nom, prenom, date_naissance), with date_radiation null; hors CONJOI
    dups_rows = []
    dup_pct = 0.0
    dup_n = 0
    working = NS.copy()
    if 'conjoin_mask' in locals():
        working = working.loc[~conjoin_mask].copy()

    dr_col = col_ns.get("date_rad")
    if dr_col and dr_col in working.columns:
        dr = working[dr_col].map(parse_date_safe)
        working = working[dr.isna()].copy()

    if col_ns.get("id") and col_ns["id"] in working.columns:
        key_series = working[col_ns["id"]].fillna("").astype(str).map(_strip_invisible_enhanced).str.strip()
        counts = key_series.value_counts()
        dup_keys = counts[counts > 1]
        dup_n = int((key_series.isin(dup_keys.index) & (key_series != "")).sum())
        dup_pct = pct(dup_n, len(working) if len(working) else 0)
        for key, cnt in dup_keys.items():
            ids = working[key_series == key][col_ns["id"]].astype(str).tolist()
            dups_rows.append({"key": key, "email": "", "count": int(cnt), "ids": ", ".join(ids)})
    else:
        nom_col = col_ns.get("nom") or ""
        prenom_col = col_ns.get("prenom") or ""
        birth_col = col_ns.get("birthdate") or ""
        if nom_col and prenom_col and birth_col and all(c in working.columns for c in [nom_col, prenom_col, birth_col]):
            grp = working.groupby([nom_col, prenom_col, birth_col]).size().reset_index(name="count")
            dup_rows = grp[grp["count"] > 1]
            dup_n = int(dup_rows["count"].sum())
            dup_pct = pct(dup_n, len(working) if len(working) else 0)
            for _, r in dup_rows.iterrows():
                key = f"{r[nom_col]}|{r[prenom_col]}|{r[birth_col]}"
                dups_rows.append({"key": key, "email": "", "count": int(r['count']), "ids": ""})
    anomalies["dups"] = pd.DataFrame(dups_rows)
    
    # Tops
    top_domains: List[Dict] = []
    if email_cols:
        dom = NS[email_cols[0]].fillna("").astype(str).str.strip().str.lower().map(lambda x: x.split("@")[-1] if "@" in x else "")
        vc = dom[dom != ""].value_counts().head(10)
        top_domains = [{"domain": d, "count": int(c)} for d, c in vc.items()]
    top_dup_emails: List[Dict] = []
    if "dups" in anomalies and not anomalies["dups"].empty:
        top_dup_emails = [{"email": r["email"], "count": int(r["count"])} for _, r in anomalies["dups"].nlargest(10, "count").iterrows()]

    # Badges
    email_band    = ThresholdBand(green=93.0, warn=85.0, red=0, mode="high_is_good")
    tp_band       = ThresholdBand(green=90.0, warn=80.0, red=0, mode="high_is_good")
    presence_band = ThresholdBand(green=99.0, warn=80.0, red=0, mode="high_is_good")
    dup_band      = ThresholdBand(green=1.0, warn=3.0, red=1000, mode="low_is_good")

    emails_valid_badge_cls, emails_valid_badge = email_band.badge(emails_valid_pct)
    tp_badge_cls, tp_badge                     = tp_band.badge(tp_pct)
    ciam_email_presence_badge_cls, ciam_email_presence_badge = presence_band.badge(ciam_email_presence_pct_val)
    iehe_presence_badge_cls, iehe_presence_badge = presence_band.badge(iehe_presence_pct_val)
    dup_badge_cls, dup_badge                   = dup_band.badge(dup_pct)

    # Graphs (âge + ∆ effet)
    graphs = {"age": None, "delta_effet": None}
    if col_ns.get("birthdate") and col_ns["birthdate"] in NS.columns:
        b = NS[col_ns["birthdate"]].map(parse_date_safe)
        age_vals = b.map(lambda d: (datetime.now().date().year - d.year) if pd.notna(d) else np.nan)
        if not age_vals.dropna().empty:
            plt.figure(); plt.hist(age_vals.dropna().astype(float).values, bins=20)
            plt.title("Distribution des âges"); plt.xlabel("Âge (années)"); plt.ylabel("Effectifs")
            graphs["age"] = fig_to_base64()
    if not delta_vals.dropna().empty:
        plt.figure(); plt.hist(delta_vals.dropna().astype(float).values, bins=30)
        plt.title("Distribution des délais d'effet"); plt.xlabel("∆ jours"); plt.ylabel("Effectifs")
        graphs["delta_effet"] = fig_to_base64()


    # E-mails identiques (NS.mail vs NS.mail_ciam), normalisés; hors CONJOI
    mail_plain_col = None
    for pat in [r"^mail$", r"^email$", r"^emailns$", r"mail.?ns", r"email.?ns"]:
        for c in NS.columns:
            if re.search(pat, c, flags=re.I):
                mail_plain_col = c; break
        if mail_plain_col: break
    emails_identiques_n = emails_identiques_total = 0
    if mail_ciam_col and mail_plain_col:
        subset = NS[[mail_ciam_col, mail_plain_col]].copy()
        if 'conjoin_mask' in locals():
            subset = subset.loc[~conjoin_mask].copy()
        subset["m1"] = subset[mail_plain_col].fillna("").astype(str).map(_strip_invisible_enhanced).str.strip().str.lower()
        subset["m2"] = subset[mail_ciam_col].fillna("").astype(str).map(_strip_invisible_enhanced).str.strip().str.lower()
        valid_rows = (subset["m1"] != "") | (subset["m2"] != "")
        subset = subset.loc[valid_rows]
        emails_identiques_total = int(len(subset))
        emails_identiques_n = int((subset["m1"] == subset["m2"]).sum())
    emails_identiques_pct_val = pct(emails_identiques_n, emails_identiques_total if emails_identiques_total else 0)
    
    # KPI dict
    kpi = {
        "assures_n": f"{assures_n}",
        "assures_pct": as_pct_str(assures_pct_val),
        "conjoints_pct": as_pct_str(conjoints_pct_val), "conjoints_n": f"{conjoints_n}",
        "radies_pct": as_pct_str(radies_pct_val), "radies_n": f"{radies_n}",

        "emails_present": f"{emails_present}",
        "emails_present_pct": as_pct_str(emails_present_pct),
        "emails_valid_pct": as_pct_str(emails_valid_pct),
        "emails_valid": f"{emails_valid}",
        "emails_non_empty": f"{emails_present}",
        "emails_valid_badge_cls": emails_valid_badge_cls, "emails_valid_badge": emails_valid_badge,

        "emails_identiques_pct": as_pct_str(emails_identiques_pct_val),
        "emails_identiques_n": f"{emails_identiques_n}", "emails_identiques_total": f"{emails_identiques_total}",

        "ciam_email_presence_pct": as_pct_str(ciam_email_presence_pct_val),
        "ciam_email_presence_n": f"{ciam_email_presence_n}",
        "ciam_email_total": f"{ciam_email_total}",
        "ciam_email_presence_badge_cls": ciam_email_presence_badge_cls,
        "ciam_email_presence_badge": ciam_email_presence_badge,

        "iehe_presence_pct": as_pct_str(iehe_presence_pct_val),
        "iehe_presence_n": f"{iehe_presence_n}",
        "iehe_total": f"{iehe_total}",
        "iehe_presence_badge_cls": iehe_presence_badge_cls,
        "iehe_presence_badge": iehe_presence_badge,

        "tp_ok_pct": as_pct_str(tp_pct), "tp_ok_badge_cls": tp_badge_cls, "tp_ok_badge": tp_badge, "tp_ok": f"{tp_ok}", "tp_expected": f"{tp_expected}", "tp_conformes": f"{tp_conformes}", "tp_conformes_pct": as_pct_str(tp_conformes_pct_val), "tp_conformes": f"{tp_conformes}", "tp_conformes_pct": as_pct_str(tp_pct),
        "tp_future_pct": as_pct_str(tp_future_pct), "tp_future": f"{tp_future}",
        "tp_negative_pct": as_pct_str(tp_negative_pct), "tp_negative": f"{tp_negative}",
        "delta_effet": fmt_delta_stats(delta_vals),

        "dup_pct": as_pct_str(dup_pct), "dup_badge_cls": dup_badge_cls, "dup_badge": dup_badge, "dup_n": f"{dup_n}",

        "tp_window": f"{21}", "tp_window_plus1": f"{22}",
    }

    def to_list_of_dicts(df: pd.DataFrame) -> List[Dict]:
        if df is None or df.empty: return []
        return [{k: ("" if pd.isna(v) else v) for k, v in row.items()} for row in df.to_dict(orient="records")]

    tables = {
        "invalid_emails": to_list_of_dicts(anomalies["invalid_emails"]),
        "mismatch": to_list_of_dicts(anomalies["mismatch"]),
        "mismatch_iehe": to_list_of_dicts(anomalies["mismatch_iehe"]),
        "dups": to_list_of_dicts(anomalies["dups"]),
        "top_domains": [{"domain": d, "count": int(c)} for d, c in (NS[email_cols[0]].fillna("").astype(str).str.lower().map(lambda x: x.split("@")[-1] if "@" in x else "").value_counts().head(10).items())] if email_cols else [],
        "top_dup_emails": [{"email": r["email"], "count": int(r["count"])} for _, r in anomalies["dups"].nlargest(10, "count").iterrows()] if not anomalies["dups"].empty else [],
    }

    # === [PATCH] Exports de validation : UNIQUEMENT 5 CSV demandés ================
    import pandas as pd

    def _first_col(df, names):
        for n in names:
            if n in df.columns: return n
        lower = {c.lower().replace("_",""): c for c in df.columns}
        for n in names:
            key = n.lower().replace("_","")
            if key in lower: return lower[key]
        return None

    def _safe_series(df, col, default=""):
        return df[col] if (col and col in df.columns) else pd.Series([default]*len(df))

    def _subset_and_rename(df, mapping):
        out = {}
        for out_col, aliases in mapping.items():
            src = _first_col(df, aliases + [out_col])
            out[out_col] = _safe_series(df, src, "")
        return pd.DataFrame(out)

    # ------------------ 1) Souscriptions du jour ----------------------------------
    ns_map = {
        "code_soc_appart": ["code_soc_appart","codesocappr","socappr","code_soc"],
        "num_ctr_indiv":   ["num_ctr_indiv","numcontratindiv","num_ctr","numcontrat"],
        "dateradassure":   ["dateradassure","date_rad_assure","date_rad","daterad"],
        "num_personne":    ["num_personne","numpersonne","idrpp","id_rpp"],
        "nom_long":        ["nom_long","nom","lastname","last_name"],
        "prenom":          ["prenom","firstname","first_name"],
        "type_assure":     ["type_assure","typeassure","type"],
        "date_naissance":  ["date_naissance","datenaissance","birthdate","dob","nais"],
        "valeur_coordonnee":["valeur_coordonnee","valeurcoordonnee","email","mail"],
        "idkpep":          ["idkpep","kpep","kp_id","idrealmididkpep"],
        "mailciam":        ["mailciam","mail_ciam","email_ciam"]
    }
    _ns_src = NS.copy()
    type_col = _first_col(_ns_src, ["type_assure","typeassure","type"])
    if type_col:
        mask_types = _ns_src[type_col].fillna("").astype(str).str.upper().str.replace(r"\s+","", regex=True).isin({"ASSPRI","MPRETR","MPVRET"})
        ns_jour = _ns_src.loc[mask_types].copy()
    else:
        ns_jour = _ns_src.copy()
    csv_souscriptions_du_jour = _subset_and_rename(ns_jour, ns_map)

    # ------------------ 2) Personnes IEHE créées ----------------------------------
    iehe_map = {
        "idrpp":      ["idrpp","id_rpp","numpersonne","num_personne"],
        "refperboccn":["refperboccn","ref_per_boccn","refper","refpersonne"],
        "siboccn":    ["siboccn","si_boccn","siege_boccn"],
        "socappr":    ["socappr","code_soc_appart","codesocappr","code_soc"],
        "telmbictc":  ["telmbictc","tel_mbi_ctc","telephone","tel"],
        "adrmailctc": ["adrmailctc","adr_mail_ctc","email","mail"]
    }
    csv_iehe = _subset_and_rename(IEHE, iehe_map)

    # ------------------ 3) Radiés -------------------------------------------------
    rad_col = _first_col(NS, ["dateradassure","date_rad_assure","date_rad","daterad"])
    if rad_col:
        rad_mask = NS[rad_col].notna() & (NS[rad_col].astype(str).str.strip() != "")
        csv_rades = _subset_and_rename(NS.loc[rad_mask], ns_map)
    else:
        csv_rades = _subset_and_rename(NS.iloc[0:0], ns_map)

    # ------------------ 4) Nombre d'adresse Email en double -----------------------
    mail_cols = [
        _first_col(NS, ["valeur_coordonnee","valeurcoordonnee","email","mail"]),
        _first_col(NS, ["mailciam","mail_ciam","email_ciam"]),
    ]
    emails = []
    for c in mail_cols:
        if c and c in NS.columns:
            emails.extend(NS[c].fillna("").astype(str).str.strip().str.lower().tolist())
    s = pd.Series([e for e in emails if e])
    dups = s.groupby(s).size()
    dups = dups[dups > 1].index.tolist()
    csv_mails_dupliques = pd.DataFrame({"mail": dups})

    # ------------------ 5) TP -----------------------------------------------------
    date_eff_col = _first_col(NS, ["date_effet_adhesion","dateeffetadhesion","date_effet","dateeffet"])
    date_adh_col = _first_col(NS, ["date_adhesion","dateadhesion","date_adh"])
    num_pers_col = _first_col(NS, ["num_personne","numpersonne","idrpp","id_rpp"])
    num_ctr_col  = _first_col(NS, ["num_ctr_indiv","numcontratindiv","num_ctr","numcontrat"])
    type_ass_col = _first_col(NS, ["type_assure","typeassure","type"])

    csv_tp = pd.DataFrame({
        "num_personne":        _safe_series(NS, num_pers_col, ""),
        "num_ctr_indiv":       _safe_series(NS, num_ctr_col, ""),
        "type_assure":         _safe_series(NS, type_ass_col, ""),
        "date_effet_adhesion": _safe_series(NS, date_eff_col, ""),
        "date_adhesion":       _safe_series(NS, date_adh_col, ""),
    }).copy()

    def _parse_dt(x):
        try:
            return pd.to_datetime(x, errors="coerce", dayfirst=True, utc=False)
        except Exception:
            return pd.NaT

    def _days_between(eff, adh):
        if pd.isna(eff) or pd.isna(adh): return None
        try:
            return int((eff - adh).days)
        except Exception:
            return None

    csv_tp["_eff"] = csv_tp["date_effet_adhesion"].map(_parse_dt)
    csv_tp["_adh"] = csv_tp["date_adhesion"].map(_parse_dt)
    csv_tp["delta"] = csv_tp.apply(lambda r: _days_between(r["_eff"], r["_adh"]), axis=1)
    csv_tp = csv_tp.drop(columns=["_eff","_adh"])

    # ------------------ Publication : NE GARDER QUE CES 5 CSV ---------------------
    exports_validation = {
        "kpi_souscriptions_du_jour": csv_souscriptions_du_jour,
        "kpi_iehe_creees":           csv_iehe,
        "kpi_radies":                csv_rades,
        "kpi_mails_dupliques":       csv_mails_dupliques,
        "kpi_tp":                    csv_tp,
    }
    anomalies.clear()
    anomalies.update(exports_validation)
    # === [/PATCH] =================================================================


    return {"kpi": kpi, "tables": tables, "graphs": graphs}, anomalies


# ================================ Sorties =====================================

def export_anomalies(anomalies: Dict[str, pd.DataFrame], out_dir: Path) -> List[Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    exports = []
    for key, df in anomalies.items():
        p = out_dir / f"export_{key}.csv"
        write_csv(df, p)
        exports.append({"href": p.name, "label": f"{p.name}", "rows": len(df)})
    return exports

def render_html(payload: Dict, title: str) -> str:
    ctx = {
        "title": title,
        "kpi": payload["kpi"],
        "tables": payload["tables"],
        "graphs": payload["graphs"],
        "exports": payload.get("exports", []),
        "kpi_file": payload.get("kpi_file", {}),
        "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    if Template is None:
        html = HTML_TEMPLATE
        html = html.replace("{{ title }}", ctx["title"]).replace("{{ now }}", ctx["now"]).replace("{% endif %}", "").replace("{% if tables.invalid_emails %}", "")
        import re as _re
        def repl_kpi(m):
            key = m.group(1)
            return str(ctx.get("kpi", {}).get(key, ""))
        html = _re.sub(r"\{\{\s*kpi\.([a-zA-Z0-9_]+)\s*\}\}", repl_kpi, html)
        return html
    return Template(HTML_TEMPLATE).render(**ctx)


# ================================== Main =====================================

def main():
    ap = argparse.ArgumentParser(
        prog="nsac_quality_pipeline_v4",
        description="NS/CIAM Quality Report (KPI + e-mails multi-colonnes + présence CIAM/IEHE).",
    )
    ap.add_argument("--data-dir", required=True, help="Dossier CSV (CK, CM, IEHE, NEW_S, KPI).")
    ap.add_argument("--out", required=True, help="Dossier de sortie.")
    ap.add_argument("--title", required=True, help="Titre du rapport HTML.")
    ap.add_argument("--tp-window-days", type=int, default=21, help="Fenêtre 0..W pour TP conforme (pour cartes).")
    ap.add_argument("--exclude-soc", type=str, default="", help='Codes société à exclure (CSV), ex: "73,74"')
    # Seuils KPI (valeurs par défaut cohérentes)
    ap.add_argument("--warn-threshold-email", type=float, default=93.0)
    ap.add_argument("--red-threshold-email", type=float, default=85.0)
    ap.add_argument("--warn-threshold-tp", type=float, default=90.0)
    ap.add_argument("--red-threshold-tp", type=float, default=80.0)
    ap.add_argument("--warn-threshold-dup", type=float, default=3.0)
    ap.add_argument("--green-threshold-dup", type=float, default=1.0)
    args_ns = ap.parse_args()

    data_dir = Path(args_ns.data_dir)
    out_dir  = Path(args_ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = ArgsNS(
        data_dir=data_dir,
        out=out_dir,
        title=args_ns.title,
        tp_window_days=args_ns.tp_window_days,
        exclude_soc=[x.strip() for x in args_ns.exclude_soc.split(",") if x.strip()],
        warn_threshold_email=args_ns.warn_threshold_email,
        red_threshold_email=args_ns.red_threshold_email,
        warn_threshold_tp=args_ns.warn_threshold_tp,
        red_threshold_tp=args_ns.red_threshold_tp,
        warn_threshold_dup=args_ns.warn_threshold_dup,
        green_threshold_dup=args_ns.green_threshold_dup,
    )

    print("[INFO] ╔════════════════════════════════════════════════╗")
    print("[INFO] ║            Contrôle Qualité Données            ║")
    print("[INFO] ║  Supervision KPI, Nouvelles Souscriptions      ║")
    print("[INFO] ╚════════════════════════════════════════════════╝")
    info(f"Dossier données : {data_dir}")

    # Découverte fichiers
    ck_path   = read_first_existing_csv(data_dir, ["*_CK.csv", "*CK*.csv"])
    cm_path   = read_first_existing_csv(data_dir, ["*_CM.csv", "*CM*.csv"])
    iehe_path = read_first_existing_csv(data_dir, ["*_IEHE.csv", "*IEHE*.csv"])
    ns_path   = read_first_existing_csv(data_dir, ["*_NEW_S.csv", "*_New_S.csv", "*NEW_S*.csv", "*New_S*.csv"])
    kpi_path  = read_first_existing_csv(data_dir, ["*_KPI.csv", "*KPI*.csv"])

    print("[INFO] Fichiers détectés :")
    ciam_email_path = read_first_existing_csv(data_dir, ["*CIAM_EMAIL*.csv", "*_CIAM_EMAIL.csv"])
    ciam_kpep_path  = read_first_existing_csv(data_dir, ["*CIAM_KPEP*.csv", "*_CIAM_KPEP.csv"])
    for tag, p in [("CK", ck_path), ("CM", cm_path), ("IEHE", iehe_path), ("New_S", ns_path), ("CIAM_EMAIL", ciam_email_path), ("CIAM_KPEP", ciam_kpep_path), ("KPI", kpi_path)]:
        if p and p.exists():
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    nrows = sum(1 for _ in f) - 1
                print(f"       ✓ {tag:<8}: {p.name} (~{max(0,nrows)} lignes)")
            except Exception:
                print(f"       ✓ {tag:<8}: {p.name}")
        else:
            print(f"       • {tag:<8}: (absent)")

    info("Chargement des données…")
    CK    = read_csv_safe(ck_path)   if ck_path else pd.DataFrame()
    CM    = read_csv_safe(cm_path)   if cm_path else pd.DataFrame()
    IEHE, _meta_iehe   = read_csv_smart(iehe_path, purpose="IEHE") if iehe_path else (pd.DataFrame(), {"status":"missing"})
    IEHE = postprocess_iehe(IEHE)
    NEW_S, _meta_new_s = read_csv_smart(ns_path, purpose="NEW_S")   if ns_path else (pd.DataFrame(), {"status":"missing"})
    CIAM_EMAIL, _meta_cemail = read_csv_smart(ciam_email_path, purpose="CIAM_EMAIL") if "ciam_email_path" in locals() and ciam_email_path else (pd.DataFrame(), {"status":"missing"})
    CIAM_EMAIL = postprocess_ciam_email(CIAM_EMAIL)
    CIAM_KPEP, _meta_ckpep   = read_csv_smart(ciam_kpep_path, purpose="CIAM_KPEP")   if "ciam_kpep_path"  in locals() and ciam_kpep_path  else (pd.DataFrame(), {"status":"missing"})
    CIAM_KPEP = postprocess_ciam_kpep(CIAM_KPEP)

    # Safety guards: ensure DataFrames, not None
    if IEHE is None:
        IEHE = pd.DataFrame()
    if NEW_S is None:
        NEW_S = pd.DataFrame()
    if CIAM_EMAIL is None:
        CIAM_EMAIL = pd.DataFrame()
    if CIAM_KPEP is None:
        CIAM_KPEP = pd.DataFrame()
    # === Aliasing (user clarified): use CM as CIAM_EMAIL and CK as CIAM_KPEP if CIAM files absent ===
    try:
        if (CIAM_EMAIL is None or CIAM_EMAIL.empty) and (not CM.empty):
            CIAM_EMAIL = postprocess_ciam_email(CM.copy())
            info("Alias activé : CIAM_EMAIL ← CM")
        if (CIAM_KPEP is None or CIAM_KPEP.empty) and (not CK.empty):
            CIAM_KPEP = postprocess_ciam_kpep(CK.copy())
            info("Alias activé : CIAM_KPEP ← CK")
    except Exception as _e_alias:
        warn(f"Aliasing CIAM échoué: {_e_alias}")



    # Choix NS = NEW_S si présent sinon CK
    NS = NEW_S.copy() if not NEW_S.empty else CK.copy()

    # Analyse colonnes
    info("Analyse des colonnes…")
    col_ns   = detect_columns(NS)
    col_ci   = detect_columns(CM)
    col_iehe = detect_columns(IEHE)
    info("Colonnes détectées :")
    for k, v in col_ns.items():
        if v: print(f"       • {k:<12}→ {v}")

    # KPIs & anomalies (NS/CM/IEHE)
    info("Calcul des KPI (base NS/CM/IEHE)…")
    payload, anomalies = compute_payload(NS, CM, IEHE, args, col_ns, col_ci, col_iehe)

    # === KPIs demandés ===
    # 1) Souscriptions du jour (NEW_S.type_assure in {ASSPRI, MPRETR, MPVRET})
    try:
        ns_df = NEW_S.copy() if not NEW_S.empty else NS.copy()
        total_ns_rows = int(len(ns_df))
        if total_ns_rows == 0:
            subs_jour_n = 0
        else:
            # Columns are normalized already, so 'type_assure' likely 'typeassure'
            type_col = None
            for cand in ["typeassure", "type_assure", "status", "type"]:
                if cand in ns_df.columns:
                    type_col = cand
                    break
            if type_col:
                s = ns_df[type_col].fillna("").astype(str).str.upper().str.replace(r"\s+", "", regex=True)
                subs_jour_n = int(s.isin({"ASSPRI", "MPRETR", "MPVRET"}).sum())
            else:
                subs_jour_n = 0
        subs_jour_pct_val = round((subs_jour_n / total_ns_rows * 100.0), 2) if total_ns_rows else 0.0
        payload["kpi"]["subs_jour_n"] = f"{subs_jour_n}"
        payload["kpi"]["subs_jour_pct"] = f"{subs_jour_pct_val:.2f} %"
    except Exception as e:
        payload["kpi"]["subs_jour_n"] = "0"
        payload["kpi"]["subs_jour_pct"] = "0.00 %"

    # 2) Comptes CIAM créés
    # Règles : NEW_S.valeur_coordonnee ∈ CIAM_EMAIL.mail
    #       OU NEW_S.idkpep ∈ CIAM_KPEP.idkpep
    #       OU NEW_S.valeur_coordonnee = CIAM_KPEP.mail
    try:
        # --- [NOUVEAU] Filtre porteurs pour CE KPI seulement ---
        ns_ciam = ns_df.copy()
        types_valides = {"ASSPRI", "MPRETR", "MPVRET"}
        type_col = next((c for c in ["typeassure", "type_assure", "status", "type"] if c in ns_ciam.columns), None)
        if type_col:
            s = ns_ciam[type_col].fillna("").astype(str).str.upper().str.replace(r"\s+", "", regex=True)
            ns_ciam = ns_ciam[s.isin(types_valides)].copy()
        # ce dénominateur est désormais le nombre de porteurs éligibles
        total_ns_rows_ciam = int(len(ns_ciam))
        ciam_crees_pct_val = round((ciam_crees_n / total_ns_rows_ciam * 100.0), 2) if total_ns_rows_ciam else 0.0


        # Prepare NS columns (sur le DataFrame FILTRÉ)
        vcol = None  # valeur_coordonnee
        for cand in ["valeurcoordonnee", "valeur_coordonnee", "mail", "email", "mailciam"]:
            if cand in ns_ciam.columns:
                vcol = cand
                break
        kpep_col = None
        for cand in ["idkpep", "kpep", "kp_id", "idrealmididkpep"]:
            if cand in ns_ciam.columns:
                kpep_col = cand
                break

        ns_mail_series = ns_ciam[vcol].fillna("").astype(str).str.strip().str.lower() if vcol else ns_ciam.apply(lambda r: "", axis=1)
        ns_kpep_series = ns_ciam[kpep_col].fillna("").astype(str).str.strip() if kpep_col else ns_ciam.apply(lambda r: "", axis=1)

        # Prepare CIAM sets (inchangé)
        def col_in(df, choices):
            for c in choices:
                if c in df.columns:
                    return c
            return None

        ce_mail_col = col_in(CIAM_EMAIL, ["mail", "email"])
        ciam_email_set = set(CIAM_EMAIL[ce_mail_col].fillna("").astype(str).str.strip().str.lower().tolist()) if (not CIAM_EMAIL.empty and ce_mail_col) else set()

        ck_kpep_col = col_in(CIAM_KPEP, ["idkpep", "kpep"])
        ck_mail_col = col_in(CIAM_KPEP, ["mail", "email"])
        ciam_kpep_set = set(CIAM_KPEP[ck_kpep_col].fillna("").astype(str).str.strip().tolist()) if (not CIAM_KPEP.empty and ck_kpep_col) else set()
        ciam_kpep_mail_set = set(CIAM_KPEP[ck_mail_col].fillna("").astype(str).str.strip().str.lower().tolist()) if (not CIAM_KPEP.empty and ck_mail_col) else set()

        def is_ciam_created(mail_val, kpep_val):
            m = (mail_val in ciam_email_set) if mail_val else False
            k = (kpep_val in ciam_kpep_set) if kpep_val else False
            m2 = (mail_val in ciam_kpep_mail_set) if mail_val else False
            return m or k or m2

        ciam_crees_n = int(sum(is_ciam_created(m, k) for m, k in zip(ns_mail_series, ns_kpep_series)))
        ciam_crees_pct_val = round((ciam_crees_n / total_ns_rows * 100.0), 2) if total_ns_rows else 0.0
        payload["kpi"]["ciam_crees_n"] = f"{ciam_crees_n}"
        payload["kpi"]["ciam_crees_pct"] = f"{ciam_crees_pct_val:.2f} %"
    except Exception as e:
        payload["kpi"]["ciam_crees_n"] = "0"
        payload["kpi"]["ciam_crees_pct"] = "0.00 %"


    
    # 3) Personnes IEHE créées : NEW_S.numpersonne/idrpp ∈ IEHE.refperboccn/idrpp
    # Recharge IEHE brut pour éviter un IEHE vidé par compute_payload
    IEHE_KPI = IEHE.copy()
    try:
        id_ns_col = None
        for cand in ["numpersonne", "num_personne", "idrpp"]:
            if cand in ns_df.columns:
                id_ns_col = cand; break
        id_iehe_col = None
        for cand in ["refperboccn", "idrpp", "numpersonne", "num_personne"]:
            if cand in IEHE_KPI.columns:
                id_iehe_col = cand; break

        ns_ids = ns_df[id_ns_col].fillna("").astype(str).str.strip() if id_ns_col else []
        iehe_ids = set(IEHE_KPI[id_iehe_col].fillna("").astype(str).str.strip().tolist()) if id_iehe_col else set()
        payload["kpi"]["_dbg_iehe_ns_col"] = id_ns_col if id_ns_col else "(none)"
        payload["kpi"]["_dbg_iehe_iehe_col"] = id_iehe_col if id_iehe_col else "(none)"
        payload["kpi"]["_dbg_iehe_ns_len"] = str(len(ns_df))
        payload["kpi"]["_dbg_iehe_iehe_len"] = str(len(IEHE_KPI))
        payload["kpi"]["_dbg_iehe_ids_len"] = str(len(iehe_ids))
        iehe_creees_n = int(sum((x in iehe_ids) for x in ns_ids))
        iehe_creees_pct_val = round((iehe_creees_n / total_ns_rows * 100.0), 2) if total_ns_rows else 0.0
        payload["kpi"]["iehe_creees_n"] = f"{iehe_creees_n}"
        payload["kpi"]["iehe_creees_pct"] = f"{iehe_creees_pct_val:.2f} %"
    except Exception:
        pass
    # ---- KPI 4: Taux de circulation de la donnée
    try:
        def _p2f(s):
            try:
                s = str(s).replace("%","").replace(",",".").strip()
                return float(s)
            except Exception:
                return 0.0
        p2 = _p2f(payload["kpi"].get("ciam_crees_pct", ""))
        p3 = _p2f(payload["kpi"].get("iehe_creees_pct", ""))
        payload["kpi"]["kpi4_taux_circulation_donnee_pct"] = f"{min(p2,p3):.2f} %"
    except Exception:
        payload["kpi"]["kpi4_taux_circulation_donnee_pct"] = ""

    # ---- KPI 8: Nombre d'adresse Email en double (même mail, identité différente)
    try:
        types_valides = {"ASSPRI","MPRETR","MPVRET"}
        type_col = next((c for c in ["typeassure","type_assure","status","type"] if c in ns_df.columns), None)
        mail_col = next((c for c in ["valeurcoordonnee","valeur_coordonnee","mail","email","mailciam"] if c in ns_df.columns), None)
        nom_col = next((c for c in ["nom","lastname","last_name","nomtypeassure","nom_assure"] if c in ns_df.columns), None)
        prenom_col = next((c for c in ["prenom","firstname","first_name","prenomtypeassure"] if c in ns_df.columns), None)
        birth_col = next((c for c in ["birthdate","datenaissance","date_naissance","nais","dob"] if c in ns_df.columns), None)
        dfp = ns_df.copy()
        if type_col:
            s = dfp[type_col].fillna("").astype(str).str.upper().str.replace(r"\s+","", regex=True)
            dfp = dfp[s.isin(types_valides)].copy()
        if mail_col:
            dfp["__mail"] = dfp[mail_col].fillna("").astype(str).str.strip().str.lower()
            dfp = dfp[dfp["__mail"]!=""].copy()
            dup_emails = 0
            for mail, g in dfp.groupby("__mail"):
                if len(g) < 2: continue
                names = set((str(g[nom_col].iloc[i]) if nom_col else "", str(g[prenom_col].iloc[i]) if prenom_col else "", str(g[birth_col].iloc[i]) if birth_col else "") for i in range(len(g)))
                if len(names) > 1: dup_emails += 1
            payload["kpi"]["kpi8_nombre_adresses_email_en_double"] = f"{dup_emails}"
        else:
            payload["kpi"]["kpi8_nombre_adresses_email_en_double"] = ""
    except Exception:
        payload["kpi"]["kpi8_nombre_adresses_email_en_double"] = ""

    # ---- KPI 9: Indice de compte CIAM créé sans Email
    try:
        def col_in(df, choices):
            for c in choices:
                if c in df.columns: return c
            return None
        ce_mail_col = col_in(CIAM_EMAIL, ["mail","email"])
        ck_kpep_col = col_in(CIAM_KPEP, ["idkpep","kpep"])
        ck_mail_col = col_in(CIAM_KPEP, ["mail","email"])
        ciam_email_set = set(CIAM_EMAIL[ce_mail_col].fillna("").astype(str).str.strip().str.lower().tolist()) if (not CIAM_EMAIL.empty and ce_mail_col) else set()
        ciam_kpep_set = set(CIAM_KPEP[ck_kpep_col].fillna("").astype(str).str.strip().tolist()) if (not CIAM_KPEP.empty and ck_kpep_col) else set()
        ciam_kpep_mail_set = set(CIAM_KPEP[ck_mail_col].fillna("").astype(str).str.strip().str.lower().tolist()) if (not CIAM_KPEP.empty and ck_mail_col) else set()
        vcol = next((c for c in ["valeurcoordonnee","valeur_coordonnee","mail","email","mailciam"] if c in ns_df.columns), None)
        kpep_col = next((c for c in ["idkpep","kpep","kp_id","idrealmididkpep"] if c in ns_df.columns), None)
        ns_mail_series = ns_df[vcol].fillna("").astype(str).str.strip().str.lower() if vcol else pd.Series([], dtype=str)
        ns_kpep_series = ns_df[kpep_col].fillna("").astype(str).str.strip() if kpep_col else pd.Series([], dtype=str)
        def is_ciam_created(mail_val, kpep_val):
            m = (mail_val in ciam_email_set) if mail_val else False
            k = (kpep_val in ciam_kpep_set) if kpep_val else False
            m2 = (mail_val in ciam_kpep_mail_set) if mail_val else False
            return m or k or m2
        created_flags = [is_ciam_created(m,k) for m,k in zip(ns_mail_series, ns_kpep_series)]
        no_email_count = 0
        for created, m in zip(created_flags, ns_mail_series):
            if not created: continue
            if (not m) or ((m not in ciam_email_set) and (m not in ciam_kpep_mail_set)):
                no_email_count += 1
        try:
            ciam_crees_n_val = int(str(payload["kpi"].get("ciam_crees_n","0")).strip())
        except Exception:
            ciam_crees_n_val = 0
        pct_no = (no_email_count / ciam_crees_n_val * 100.0) if ciam_crees_n_val else 0.0
        payload["kpi"]["kpi9_sans_email_n"] = f"{no_email_count}"
        payload["kpi"]["kpi9_indice_compte_ciam_sans_email_pct"] = f"{pct_no:.2f} %"
    except Exception:
        payload["kpi"]["kpi9_sans_email_n"] = ""
        payload["kpi"]["kpi9_indice_compte_ciam_sans_email_pct"] = ""

    # ---- KPI 10: Souscriptions en double sur la journée
    try:
        nom_col = next((c for c in ["nom","lastname","last_name","nomtypeassure","nom_assure"] if c in ns_df.columns), None)
        prenom_col = next((c for c in ["prenom","firstname","first_name","prenomtypeassure"] if c in ns_df.columns), None)
        birth_col = next((c for c in ["birthdate","datenaissance","date_naissance","nais","dob"] if c in ns_df.columns), None)
        dr_col = next((c for c in ["dateradiation","date_radiation","date_rad","radiationdate"] if c in ns_df.columns), None)
        dfp = ns_df.copy()
        if dr_col:
            dr = dfp[dr_col].map(parse_date_safe)
            dfp = dfp[dr.isna()].copy()
        for c in [nom_col, prenom_col, birth_col]:
            if c and c in dfp.columns:
                dfp[c] = dfp[c].fillna("")
        grp_cols = [c for c in [nom_col, prenom_col, birth_col] if c]
        dup_n = 0
        if grp_cols:
            grp = dfp.groupby(grp_cols).size().reset_index(name="_cnt")
            dup_n = int((grp["_cnt"]-1).clip(lower=0).sum())
        payload["kpi"]["kpi10_souscriptions_en_double_du_jour_n"] = f"{max(dup_n,0)}"
    except Exception:
        payload["kpi"]["kpi10_souscriptions_en_double_du_jour_n"] = ""

    except Exception as e:
        payload["kpi"]["iehe_creees_n"] = "0"
        payload["kpi"]["iehe_creees_pct"] = "0.00 %"
    except Exception:
        pass
    
    # ---- KPI 11: Nombre de personnes par type_assure ----
    try:
        type_col = next((c for c in ["typeassure","type_assure","status","type"] if c in NS.columns), None)
        if type_col:
            s = NS[type_col].fillna("").astype(str).str.upper().str.replace(r"\s+","", regex=True)
            counts = s.value_counts().to_dict()
            total_types = int(s.count())
            payload["kpi"]["kpi11_personnes_par_type"] = json.dumps(counts, ensure_ascii=False)
            payload["kpi"]["kpi11_personnes_total"] = total_types
        else:
            payload["kpi"]["kpi11_personnes_par_type"] = "{}"
            payload["kpi"]["kpi11_personnes_total"] = 0
    except Exception:
        payload["kpi"]["kpi11_personnes_par_type"] = "{}"
        payload["kpi"]["kpi11_personnes_total"] = 0



    # KPI File (KPI.csv)
    kpi_file = {}
    if kpi_path and kpi_path.exists():
        info(f"Lecture KPI (texte): {kpi_path.name}")
        try:
            with open(kpi_path, "r", encoding="utf-8", errors="ignore") as f:
                ktext = f.read()
            kpi_file = parse_kpi_file_text(ktext)
        except Exception as e:
            warn(f"Echec parsing KPI.csv: {e}")

    # Exports CSV (anomalies)
    info("Export des anomalies…")
    exports = export_anomalies(anomalies, out_dir)

    payload["exports"] = exports
    payload["kpi_file"] = kpi_file

    # JSON KPI (pipeline NS uniquement)
    json_path = out_dir / "kpi_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload["kpi"], f, ensure_ascii=False, indent=2)
    info(f"✅ Export JSON : {json_path}")

    # HTML
    info("Génération du rapport HTML…")
    html = render_html(payload, args.title)
    html_path = out_dir / "rapport_qualite.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    info(f"✅ Rapport généré : {html_path}")
    info("Pipeline terminé avec succès !")

    # Console: résumé
    k = payload["kpi"]

    def _to_int(x):
        try:
            return int(str(x).split()[0].replace("\u202f","").replace(" ", ""))
        except Exception:
            return 0

    # Reconstituer TP conformes pour l'affichage à partir du dict k
    tp0_21 = _to_int(k.get("tp_ok", 0))
    tp_neg = _to_int(k.get("tp_negative", 0))
    tp_exp = _to_int(k.get("tp_expected", 0))
    tp_conf = tp0_21 + tp_neg
    tp_conf_pct = k.get("tp_conformes_pct") or as_pct_str(pct(tp_conf, tp_exp))


    # === Affichage résumé dans le terminal ===
    print("[INFO] === Résumé des KPI (pipeline) ===")
    print(f"       Souscriptions du jour : {k.get('subs_jour_n','')} ({k.get('subs_jour_pct','')})")
    print(f"       Personnes par type_assure : {k.get('kpi11_personnes_par_type','')}")
    print(f"       Total personnes (tous types): {k.get('kpi11_personnes_total','0')}")
    print(f"       Personnes IEHE créées : {k.get('iehe_creees_n','')} ({k.get('iehe_creees_pct','')})")
    print(f"       Nombre de comptes CIAM créés : {k.get('ciam_crees_n','')} ({k.get('ciam_crees_pct','')})")
    print(f"       Taux de circulation de la donnée : {k.get('kpi4_taux_circulation_donnee_pct','')}")
    print(f"       Radiés : {k.get('radies_n','')} ({k.get('radies_pct','')})")

    print("       --- KPI Cartes TP (uniquement ASSPRI / MPRETR / MPVRET) ---")
    print(f"       TP attendues : {k.get('tp_expected','')}")
    print(f"       TP 0/{k.get('tp_window','21')}j : {k.get('tp_ok','')}/{k.get('tp_expected','')} ({k.get('tp_ok_pct','')})")
    print(f"       TP delta négatif : {k.get('tp_negative','')}/{k.get('tp_expected','')} ({k.get('tp_negative_pct','')})")
    print(f"       TP conformes : {tp_conf}/{k.get('tp_expected','')} ({tp_conf_pct})")
    print(f"       TP future (≥{k.get('tp_window_plus1','22')}j) : {k.get('tp_future','')}/{k.get('tp_expected','')} ({k.get('tp_future_pct','')})")
    print(f"       ∆ effet (min / moyenne / médiane / max) : {k.get('delta_effet','')}")

if __name__ == "__main__":
    main()