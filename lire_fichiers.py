import json
import pandas as pd
import s3fs


def charger_donnees():
    # Accès public anonyme : fonctionne sur n'importe quel ordinateur sans clés
    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"},
    )

    fichiers_s3 = sorted(fs.glob("k1000ensae/diffusion/projet-2a/data/*.parquet"))

    dfs = []
    for chemin in fichiers_s3:
        with fs.open(chemin, "rb") as f:
            dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)

    if "marches" in df.columns:
        df["marches"] = df["marches"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
    return df
