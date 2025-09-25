#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kpi_type_assure_filter.py
--------------------------------------------------------------------
Règle unique et officielle pour filtrer les lignes utilisées
dans le calcul des KPI :

  Type_assuré ∈ {"ASSPRI", "MPRETR", "MPVRET"}

⚠️ Important :
  - AUCUN autre filtre sur le statut ne doit s'appliquer ici
    (ex.: ne PAS exclure explicitement "CONJOI" ; cette règle
    d'exclusion est SUPPRIMÉE).
  - Le nom de colonne (status/type assuré) peut varier ; on
    le passe en paramètre.
  - La comparaison est insensible à la casse et tolère les
    espaces/invisibles.

Intégration (exemple) :
--------------------------------------------------------------------
    from kpi_type_assure_filter import apply_kpi_filter

    df_kpi = apply_kpi_filter(NS, status_col=col_ns.get("status"))
    # ... ensuite, utilisez df_kpi pour tous les KPI (TP, CIAM, IEHE, etc.)
--------------------------------------------------------------------
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

import pandas as pd


VALID_TYPES: tuple[str, ...] = ("ASSPRI", "MPRETR", "MPVRET")


_INVISIBLE = [
    '\ufeff','\u200b','\u200c','\u200d','\u2060','\ufffe',
    '\xa0','\u00a0','\u2000','\u2001','\u2002','\u2003','\u2004',
    '\u2005','\u2006','\u2007','\u2008','\u2009','\u200a',
    '\u202f','\u205f','\u3000','ï»¿','\ufeff'
]


def _strip_invisible(s: str) -> str:
    if s is None:
        return ""
    out = str(s)
    for ch in _INVISIBLE:
        out = out.replace(ch, " ")
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _norm(s: str) -> str:
    """Normalise pour comparer proprement les statuts."""
    s = _strip_invisible(str(s))
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"\s+", "", s, flags=re.U).upper()


def apply_kpi_filter(df: pd.DataFrame, *, status_col: Optional[str]) -> pd.DataFrame:
    """Retourne une vue filtrée du DataFrame pour le calcul des KPI.

    Paramètres
    ----------
    df : pd.DataFrame
        Données de souscription (NS).
    status_col : Optional[str]
        Nom de la colonne représentant le *Type_assuré*.
        Si None ou absente, retourne df inchangé (aucun filtrage
        ne peut être appliqué de façon fiable).

    Règle
    -----
    Conserver UNIQUEMENT les lignes dont Type_assuré est dans
    {ASSPRI, MPRETR, MPVRET}. Aucune exclusion explicite de CONJOI.

    Returns
    -------
    pd.DataFrame
        DataFrame filtré.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un DataFrame pandas")

    if not status_col or status_col not in df.columns:
        # Impossible de filtrer proprement : on renvoie tel quel.
        return df

    # Normalise la colonne et applique l'inclusion stricte
    status_norm = df[status_col].fillna("").map(_norm)
    mask = status_norm.isin(VALID_TYPES)
    return df.loc[mask].copy()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Applique le filtre KPI sur un CSV NS et exporte le CSV filtré."
    )
    p.add_argument("csv_in", help="Chemin du CSV source (NS)")
    p.add_argument("--status-col", default="typeassure",
                   help="Nom de la colonne Type_assuré (défaut: typeassure)")
    p.add_argument("--out", help="Chemin du CSV filtré à écrire (défaut: <csv_in>.filtered.csv)")
    args = p.parse_args()

    df = pd.read_csv(args.csv_in, dtype=str)
    out_df = apply_kpi_filter(df, status_col=args.status_col)
    out = args.out or (args.csv_in + ".filtered.csv")
    out_df.to_csv(out, index=False, encoding="utf-8-sig", sep=";")
    print(f"[OK] Filtré: {len(out_df)}/{len(df)} lignes conservées → {out}")
