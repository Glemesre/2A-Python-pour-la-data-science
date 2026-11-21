import json
import os
import pandas as pd
import s3fs

fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://" + os.environ["AWS_S3_ENDPOINT"]},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    token=os.environ["AWS_SESSION_TOKEN"],
)

fichiers_s3 = fs.glob("k1000ensae/diffusion/projet-2a/data/*.parquet")

dfs = []
for chemin in fichiers_s3:
    with fs.open(chemin, "rb") as f:
        dfs.append(pd.read_parquet(f))

df = pd.concat(dfs, ignore_index=True)

# Si tu as besoin de reconvertir la colonne "marches" en listes/dictionnaires Python :
if "marches" in df.columns:
    df["marches"] = df["marches"].apply(
        lambda x: json.loads(x) if pd.notnull(x) else None
    )
