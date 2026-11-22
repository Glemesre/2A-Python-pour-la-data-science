import json
from urllib.parse import quote
import pandas as pd

BASE_URL = "https://minio.lab.sspcloud.fr/k1000ensae/diffusion/projet-2a/data"
FICHIERS = ['decp-2024-01.parquet',
            'decp-2024-02.parquet',
            'decp-2024-03.parquet',
            'decp-2024-04.parquet',
            'decp-2024-05.parquet',
            'decp-2024-06.parquet',
            'decp-2024-07.parquet',
            'decp-2024-08.parquet',
            'decp-2024-09.parquet',
            'decp-2024-10.parquet',
            'decp-2024-11.parquet',
            'decp-2024-12.parquet',
            'decp-2024.parquet',
            'decp-2025-01.parquet',
            'decp-2025-02.parquet',
            'decp-2025-03.parquet',
            'decp-2025-04.parquet',
            'decp-2025-05.parquet',
            'decp-2025-06.parquet',
            'decp-2025-07.parquet',
            'decp-2025-08.parquet',
            'decp-2025-09.parquet',
            'decp-2025-10.parquet',
            'decp-2025-11.parquet',
            'decp-2025-12.parquet',
            'decp-2025.parquet',
            'decp-2026-01.parquet',
            'decp-2026-02.parquet',
            'decp-2026-03.parquet',
            'decp-2026-04.parquet',
            'decp-2026-05.parquet',
            'decp-2026-06.parquet',
            'decp-2026-07.parquet',
            'decp-2026-08.parquet',
            'decp-2026-09.parquet',
            'decp-2026-10.parquet',
            'decp-2026-11.parquet',
            'decp-2026.parquet',
            'decp-global.parquet'
            ]


def _mettre_a_plat(df):
    """Transforme la colonne unique 'marches' en vraies colonnes tabulaires."""
    if list(df.columns) == ["marches"]:
        serie = df["marches"].dropna().apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        if serie.apply(lambda x: isinstance(x, list)).any():
            serie = serie.explode().dropna()
        df_propre = pd.json_normalize(serie.tolist())

        # Gère le cas où la structure JSON était {"marches": {"marche": [...]}}
        if list(df_propre.columns) == ["marche"]:
            serie_m = df_propre["marche"].dropna().explode().dropna()
            return pd.json_normalize(serie_m.tolist())
        return df_propre
    return df


def charger_donnees():
    """Charge chaque fichier séparément dans un dictionnaire {nom: DataFrame}."""
    donnees = {}
    for nom in FICHIERS:
        df_brut = pd.read_parquet(f"{BASE_URL}/{quote(nom)}")
        donnees[nom] = _mettre_a_plat(df_brut)
    return donnees

