#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script : 01_export_referentiels.py
---------------------------------
Ce script sert à extraire des données depuis la base IEHE à partir des identifiants trouvés
dans le fichier New_S.csv (fichier de nouvelles souscriptions).

Fonctionnalités :
- Lecture du fichier New_S.csv (colonnes clients).
- Extraction de la colonne num_personne (ou autre colonne choisie).
- Connexion automatique à la base PostgreSQL IEHE (multi-host, multi-port, multi-bdd).
- Récupération des lignes correspondantes dans la table IEHE.refkpep.
- Génération de 3 fichiers de sortie :
    1. {STAMP}_IEHE.csv              → déposé dans Input_Data
    2. {STAMP}_IEHE_NOT_FOUND.csv    → déposé dans Output_Data/{STAMP}
    3. {STAMP}_SUMMARY.json          → déposé dans Output_Data/{STAMP}

Notation :
- {STAMP} = date JJMMYYYY extraite du nom du fichier New_S.csv (ex: 19092025_New_S.csv → 19092025).
"""

import os, sys, csv, argparse, re, json, time, traceback
from pathlib import Path
from typing import List, Sequence, Tuple, Optional
from datetime import datetime

# Import du fuseau horaire (utile pour logguer à l’heure locale Europe/Paris)
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

# Librairies externes utilisées
import pandas as pd   # pour lire facilement le CSV New_S
import psycopg        # pour se connecter à PostgreSQL


# ========= IDENTIFIANTS DE CONNEXION =========
# ⚠️ Ici, user/password sont codés en dur pour la démo (à sécuriser en prod : .env ou vault)
PG_USER = "u_lpillon"
PG_PASSWORD = "T_Run_Asc_2025#"


# ========= PARAMÈTRES DE LA BASE IEHE =========
# Le script va tester toutes les combinaisons host/port/db pour trouver un serveur accessible.
IEHE_HOSTS  = ["bdd-X0ED0550.alias", "100.54.41.6"]
IEHE_PORTS  = [5559, 5432]
IEHE_DBS    = ["choregie_db", "postgres"]
IEHE_SCHEMA = "iehe"        # schéma de la table
IEHE_TABLE  = "refkpep"     # table cible
IEHE_COL_ID = "refperboccn" # colonne utilisée pour matcher les identifiants


# ========= LOGGING =========
class RunLog:
    """
    Petit utilitaire de logging :
    - Ajoute un timestamp devant chaque message
    - Conserve l’historique des logs et erreurs dans des listes (utile pour résumé JSON)
    """
    def __init__(self):
        self.lines: List[str] = []   # tous les messages loggés
        self.errors: List[str] = []  # uniquement les erreurs

    def _ts(self) -> str:
        """Retourne l’heure actuelle (format YYYY-MM-DD HH:MM:SS)"""
        dt = datetime.now(ZoneInfo("Europe/Paris")) if ZoneInfo else datetime.now()
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def log(self, msg: str):
        """Message normal (stdout)"""
        line = f"[{self._ts()}] {msg}"
        print(line)
        self.lines.append(line)

    def err(self, msg: str, exc: Optional[BaseException] = None):
        """Message d’erreur (stderr) avec stacktrace optionnelle"""
        base = f"[{self._ts()}] [ERROR] {msg}"
        if exc:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            full = f"{base}\n{tb}"
        else:
            full = base
        print(full, file=sys.stderr)
        self.lines.append(full)
        self.errors.append(full)

# Instance globale de logger
RUNLOG = RunLog()
log = RUNLOG.log
log_err = RUNLOG.err


# ========= FONCTIONS UTILES =========
def norm_trim_unique(values: Sequence[str], lower: bool=False) -> List[str]:
    """
    Nettoie une liste de chaînes :
    - Supprime espaces avant/après
    - Supprime doublons
    - Option : convertit en minuscule
    """
    out, seen = [], set()
    for v in values:
        v = (v or "").strip()
        if not v:
            continue
        if lower:
            v = v.lower()
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def write_csv(path: Path, header: List[str], rows: List[Tuple]):
    """Écrit un fichier CSV avec entête (header) + lignes (rows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(rows)

def write_list_csv(path: Path, header_name: str, values: List[str]):
    """Écrit un CSV à une seule colonne (ex: liste des identifiants non trouvés)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([header_name])
        for v in values:
            w.writerow([v])

def today_stamp_paris() -> str:
    """Retourne la date du jour en format JJMMYYYY (fuseau Paris)."""
    dt = datetime.now(ZoneInfo("Europe/Paris")) if ZoneInfo else datetime.now()
    return dt.strftime("%d%m%Y")

def extract_stamp_from_name(p: Path) -> Optional[str]:
    """
    Extrait la date (STAMP) d’un fichier New_S.
    Exemple : 19092025_New_S.csv → "19092025"
    """
    m = re.match(r"^(\d{8})[_-]?new_s\.csv$", p.name, flags=re.IGNORECASE)
    return m.group(1) if m else None


# ========= LECTURE New_S.csv =========
def read_new_s(csv_path: Path, iehe_col: str, sep: str = ",", encoding: Optional[str] = None) -> List[str]:
    """
    Lit le fichier New_S.csv et retourne une liste unique d’identifiants IEHE.
    - csv_path : chemin du New_S
    - iehe_col : nom de la colonne qui contient les identifiants (ex: num_personne)
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, sep=sep, encoding=encoding)
    refs = df[iehe_col].astype(str).tolist() if iehe_col in df.columns else []
    return norm_trim_unique(refs, lower=False)


# ========= CONNEXION IEHE =========
def connect_pg(host: str, port: int, db: str) -> psycopg.Connection:
    """Tentative de connexion PostgreSQL sur un host/port/db donné."""
    return psycopg.connect(
        host=host, port=port, dbname=db,
        user=PG_USER, password=PG_PASSWORD,
        connect_timeout=5
    )

def connect_iehe_auto() -> Optional[psycopg.Connection]:
    """
    Essaie toutes les combinaisons (host, port, db).
    Renvoie la 1ère connexion valide ou None si toutes échouent.
    """
    for host in IEHE_HOSTS:
        for port in IEHE_PORTS:
            for db in IEHE_DBS:
                try:
                    log(f"[IEHE] tentative: host={host} port={port} db={db}")
                    conn = connect_pg(host, port, db)
                    log(f"[IEHE] connecté: {host}:{port}/{db}")
                    return conn
                except Exception as e:
                    log(f"[IEHE] échec: {e}")
    return None


# ========= SQL =========
SQL_IEHE = f"""
WITH ids AS (SELECT unnest(%(vals)s::text[]) AS v)
SELECT r.*
FROM {IEHE_SCHEMA}.{IEHE_TABLE} r
JOIN ids ON ids.v = r.{IEHE_COL_ID}
"""


# ========= MAIN =========
def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description="Export IEHE depuis New_S.csv")
    ap.add_argument("--new-s", default=".", help="Fichier New_S.csv ou dossier contenant DDMMYYYY_New_S.csv")
    ap.add_argument("--iehe-col", default="num_personne")
    ap.add_argument("--batch", type=int, default=20000)  # (pas utilisé ici, prévu pour limiter batch SQL)
    ap.add_argument("--sep", default=",")
    ap.add_argument("--encoding", default=None)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    # 1) Localisation du fichier New_S.csv
    new_s_arg = Path(args.new_s)
    if new_s_arg.is_file():
        csv_path = new_s_arg
    else:
        csv_path = None
        for p in Path(args.new_s).iterdir():
            if re.match(r"^\d{8}_New_S\.csv$", p.name, re.IGNORECASE):
                csv_path = p
                break
    if not csv_path or not csv_path.exists():
        log_err(f"Fichier New_S introuvable dans {args.new_s}")
        sys.exit(1)

    log(f"[info] Fichier New_S sélectionné: {csv_path}")
    stamp = extract_stamp_from_name(csv_path) or today_stamp_paris()

    # 2) Définir les chemins de sortie
    # - IEHE.csv dans Input_Data
    # - NOT_FOUND + SUMMARY dans Output_Data/{STAMP}
    input_data_dir = csv_path.parent
    iehe_path = input_data_dir / f"{stamp}_IEHE.csv"

    base_out = Path(args.out_dir)
    def _normname(p: Path) -> str:
        return p.name.lower().replace("-", "_")

    # Si l’utilisateur a donné Input_Data comme --out-dir, on dérive vers Output_Data
    if base_out.resolve() == input_data_dir.resolve() or _normname(base_out) in {"input_data", "inputdata"}:
        other_out_base = input_data_dir.parent / "Output_Data"
    else:
        other_out_base = base_out

    other_out_dir = other_out_base / stamp
    other_out_dir.mkdir(parents=True, exist_ok=True)

    # chemins des autres fichiers
    iehe_nf_path = other_out_dir / f"{stamp}_IEHE_NOT_FOUND.csv"
    summary_json = other_out_dir / f"{stamp}_SUMMARY.json"

    # 3) Lecture des identifiants depuis New_S
    refs_in = read_new_s(csv_path, args.iehe_col, sep=args.sep, encoding=args.encoding)
    log(f"[info] iehe_ids_in={len(refs_in)}")

    header_iehe, rows_iehe = [], []
    iehe_status = "SKIPPED"

    # 4) Connexion à IEHE et exécution SQL
    if refs_in:
        conn_iehe = connect_iehe_auto()
        if conn_iehe:
            try:
                with conn_iehe.cursor() as cur:
                    cur.execute(SQL_IEHE, {"vals": refs_in})
                    rows_iehe = cur.fetchall()
                    header_iehe = [d.name for d in cur.description]
                iehe_status = "CONNECTED"
                log(f"[IEHE] lecture OK ({len(rows_iehe)} lignes)")
            except Exception as e:
                log_err("[IEHE] erreur lecture", e)
                iehe_status = f"ERROR: {e}"
            finally:
                conn_iehe.close()

    # 5) Écriture des fichiers de sortie
    write_csv(iehe_path, header_iehe, rows_iehe)  # résultats trouvés
    found_refs = {(row[0] or "").strip() for row in rows_iehe} if rows_iehe else set()
    refs_nf = [r for r in refs_in if r not in found_refs]
    write_list_csv(iehe_nf_path, "num_personne_not_found", refs_nf)  # identifiants non trouvés

    # 6) Génération du résumé JSON (métadonnées de l’exécution)
    summary_obj = {
        "run": {"stamp_prefix": stamp},
        "inputs": {"new_s_path": str(csv_path), "iehe_col": args.iehe_col, "count_refs": len(refs_in)},
        "status": {"iehe": iehe_status},
        "outputs": {
            "iehe_csv": str(iehe_path),
            "iehe_not_found_csv": str(iehe_nf_path),
            "summary_json": str(summary_json),
        },
        "counts": {"iehe_rows": len(rows_iehe), "iehe_not_found": len(refs_nf)},
        "errors": RUNLOG.errors,
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)

    log(f"[done] Résumés écrits: {summary_json}")

if __name__ == "__main__":
    main()
