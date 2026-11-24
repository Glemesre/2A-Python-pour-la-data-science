"""Fonctions de nettoyage, jointure, visualisation et analyse financière (DECP x Sirene x INPI)."""

import gc
import fsspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plotly.express as px
import pyarrow.parquet as pq
import requests
import seaborn as sns


# ==============================================================================
# 1. DIAGNOSTIC, NETTOYAGE DES DECP ET PRÉPARATION DE SIRENE
# ==============================================================================

def analyser_valeurs_manquantes_decp(donnees: dict):
    """Calcule le bilan des valeurs manquantes sur l'ensemble des millésimes DECP."""
    total_lignes = sum(len(table) for table in donnees.values())

    valeurs_valides = pd.concat(
        [table.notna().sum() for table in donnees.values()], axis=1
    ).sum(axis=1)

    bilan_nan = pd.DataFrame({
        "nb_nan": (total_lignes - valeurs_valides).astype(int),
        "pourcentage_nan": ((total_lignes - valeurs_valides) / total_lignes * 100).round(2),
    }).sort_values(by="nb_nan", ascending=False)

    colonnes_vides = list(bilan_nan[bilan_nan["pourcentage_nan"] == 100].index)
    print(f"Colonnes complètement vides ({len(colonnes_vides)}) :", colonnes_vides)
    return bilan_nan, colonnes_vides


def inspecter_parquet(url: str, nom: str = "", batch_size: int = 5) -> pd.DataFrame:
    """Inspecte légèrement un fichier Parquet distant (schéma + aperçu) sans tout charger."""
    with fsspec.open(url, "rb") as f:
        pf = pq.ParquetFile(f)
        prefix = f"[{nom}] " if nom else ""
        print(f"{prefix}{pf.metadata.num_rows:,} lignes | {len(pf.schema_arrow.names)} colonnes")
        print(f"{prefix}Colonnes exactes :", pf.schema_arrow.names)
        apercu = next(pf.iter_batches(batch_size=batch_size)).to_pandas()
    return apercu


def clean_decp(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et aplatit la table consolidée des DECP (1 ligne par marché x titulaire)."""
    df = df.copy()

    id_col = "uid" if "uid" in df.columns else "id"
    cols_keep = [
        c for c in [
            id_col, "id", "nature", "procedure", "codeCPV",
            "montant", "dureeMois", "dateNotification",
            "acheteur_id", "acheteur_nom",
            "lieuExecution_code", "lieuExecution_typeCode",
            "titulaires",
        ] if c in df.columns
    ]
    df = df[list(dict.fromkeys(cols_keep))]

    df["montant"] = pd.to_numeric(df["montant"], errors="coerce")
    df["dureeMois"] = pd.to_numeric(df["dureeMois"], errors="coerce")
    df["dateNotification"] = pd.to_datetime(df["dateNotification"], errors="coerce")

    has_tit = df["titulaires"].apply(
        lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0
    )
    df = df[has_tit].copy()

    df["nb_titulaires"] = df["titulaires"].apply(len)
    df["montant_part"] = df["montant"] / df["nb_titulaires"]

    df = df.explode("titulaires").reset_index(drop=True)

    tit_records = [
        t["titulaire"] if isinstance(t, dict) and "titulaire" in t
        else (t if isinstance(t, dict) else {})
        for t in df["titulaires"]
    ]
    df_tit = pd.json_normalize(tit_records).rename(columns={
        "id": "titulaire_id",
        "typeIdentifiant": "titulaire_type_id",
        "denominationSociale": "titulaire_nom",
    })

    cols_tit = [c for c in ["titulaire_id", "titulaire_type_id", "titulaire_nom"] if c in df_tit.columns]
    df = pd.concat([df.drop(columns=["titulaires"]), df_tit[cols_tit]], axis=1)

    df["titulaire_id"] = df["titulaire_id"].astype(str).str.strip()
    is_siret = (df["titulaire_type_id"] == "SIRET") & (df["titulaire_id"].str.match(r"^\d{14}$"))
    print(f"Part de lignes avec SIRET valide (14 chiffres) : {is_siret.mean():.1%}")

    df = df[is_siret].copy()
    df["siret"] = df["titulaire_id"]
    df["siren"] = df["siret"].str[:9]

    avant_dedup = len(df)
    df = df.drop_duplicates(subset=[id_col, "siret"]).copy()
    print(f"Doublons (marché, siret) supprimés : {avant_dedup - len(df):,}")

    avant_filt = len(df)
    df = df[(df["montant_part"] > 100) & (df["montant_part"] < 5e8)].copy()
    print(f"Lignes écartées pour montant aberrant (<= 100 € ou >= 500 M€) : {avant_filt - len(df):,}")

    duree_valide = df["dureeMois"].where(df["dureeMois"] > 0, np.nan)
    df["montant_annualise"] = df["montant_part"] / duree_valide * 12

    if "codeCPV" in df.columns:
        df["cpv_division"] = df["codeCPV"].astype(str).str.slice(0, 2)

    return df


def nettoyer_decp(donnees, colonnes_vides=None) -> pd.DataFrame:
    """Consolide un dictionnaire de DataFrames DECP (ou un DataFrame) puis applique clean_decp."""
    if isinstance(donnees, dict):
        df_raw = pd.concat(donnees.values(), ignore_index=True)
    else:
        df_raw = donnees.copy()

    if colonnes_vides:
        df_raw = df_raw.drop(columns=colonnes_vides, errors="ignore")
    df_raw.columns = df_raw.columns.str.replace(".", "_", regex=False)

    df_decp = clean_decp(df_raw)
    del df_raw
    gc.collect()
    print(f"DECP nettoyées prêtes pour la jointure : {len(df_decp):,} lignes")
    return df_decp


def map_naf_section(code_naf: pd.Series) -> pd.Series:
    """Regroupe les codes NAF Rev. 2 (ex: '41.20A') en 21 sections officielles (A à U)."""
    div = pd.to_numeric(code_naf.astype(str).str.slice(0, 2), errors="coerce")

    conditions = [
        div.between(1, 3),
        div.between(5, 9),
        div.between(10, 33),
        div == 35,
        div.between(36, 39),
        div.between(41, 43),
        div.between(45, 47),
        div.between(49, 53),
        div.between(55, 56),
        div.between(58, 63),
        div.between(64, 66),
        div == 68,
        div.between(69, 75),
        div.between(77, 82),
        div == 84,
        div == 85,
        div.between(86, 88),
        div.between(90, 93),
        div.between(94, 96),
        div.between(97, 98),
        div == 99,
    ]
    labels = [
        "A - Agriculture, sylviculture et pêche",
        "B - Industries extractives",
        "C - Industrie manufacturière",
        "D - Électricité, gaz, vapeur et air conditionné",
        "E - Eau, assainissement et déchets",
        "F - Construction (BTP)",
        "G - Commerce et réparation auto",
        "H - Transports et entreposage",
        "I - Hébergement et restauration",
        "J - Information et communication",
        "K - Activités financières et d'assurance",
        "L - Activités immobilières",
        "M - Activités spécialisées, scientifiques et techniques",
        "N - Services administratifs et de soutien",
        "O - Administration publique",
        "P - Enseignement",
        "Q - Santé humaine et action sociale",
        "R - Arts, spectacles et loisirs",
        "S - Autres activités de services",
        "T - Activités des ménages employeurs",
        "U - Organisations extraterritoriales",
    ]
    return pd.Series(np.select(conditions, labels, default="Inconnu"), index=code_naf.index)


def charger_et_preparer_sirene(url_siret: str):
    """Charge sélectivement StockEtablissement (Sirene) et construit le référentiel des sections NAF actives."""
    cols_siret_cibles = [
        "siret",
        "siren",
        "etablissementSiege",
        "activitePrincipaleEtablissement",
        "nomenclatureActivitePrincipaleEtablissement",
        "trancheEffectifsEtablissement",
        "codeCommuneEtablissement",
        "etatAdministratifEtablissement",
    ]

    with fsspec.open(url_siret, "rb") as f:
        pf = pq.ParquetFile(f)
        cols_dispo = [c for c in cols_siret_cibles if c in pf.schema_arrow.names]
        df_sirene = pf.read(columns=cols_dispo).to_pandas()

    df_sirene["siret"] = df_sirene["siret"].astype(str).str.strip().str.zfill(14)
    df_sirene = df_sirene.drop_duplicates(subset=["siret"]).copy()

    df_sirene["naf_code"] = df_sirene["activitePrincipaleEtablissement"].astype(str).str.strip()
    df_sirene["naf_division"] = df_sirene["naf_code"].str.slice(0, 2)
    df_sirene["naf_section"] = map_naf_section(df_sirene["naf_code"])

    if "codeCommuneEtablissement" in df_sirene.columns:
        commune = df_sirene["codeCommuneEtablissement"].astype(str).str.strip()
        df_sirene["dep_titulaire"] = np.where(
            commune.str.startswith(("97", "98")),
            commune.str.slice(0, 3),
            commune.str.slice(0, 2),
        )

    ref_secteurs_sirene = (
        df_sirene[df_sirene["etatAdministratifEtablissement"] == "A"]
        .groupby("naf_section", as_index=False)
        .agg(
            nb_etablissements_actifs=("siret", "count"),
            nb_entreprises_actives=("siren", "nunique"),
        )
    )
    ref_secteurs_sirene["part_etablissements_sirene"] = (
        ref_secteurs_sirene["nb_etablissements_actifs"]
        / ref_secteurs_sirene["nb_etablissements_actifs"].sum()
    )

    print(f"Sirene nettoyé : {len(df_sirene):,} établissements uniques")
    return df_sirene, ref_secteurs_sirene


# ==============================================================================
# 2. JOINTURE DECP x SIRENE ET SYNTHÈSE SECTORIELLE
# ==============================================================================

def joindre_decp_sirene(
    df_decp: pd.DataFrame,
    df_sirene: pd.DataFrame,
    ref_secteurs_sirene: pd.DataFrame = None,
    export_path: str = "decp_sirene_clean.parquet",
    export_ref_path: str = "ref_secteurs_sirene.parquet",
) -> pd.DataFrame:
    """Joint les DECP nettoyées à Sirene (many-to-one), affiche le contrôle qualité et exporte en Parquet."""
    df_final = pd.merge(
        df_decp,
        df_sirene,
        on="siret",
        how="left",
        suffixes=("", "_sirene"),
        validate="many_to_one",
    )

    gc.collect()

    taux_lignes = df_final["naf_code"].notna().mean()
    montant_tot = df_final["montant_part"].sum()
    montant_ok = df_final.loc[df_final["naf_code"].notna(), "montant_part"].sum()

    print(f"Nombre final de lignes (marché x titulaire) : {len(df_final):,}")
    print(f"Taux d'appariement Sirene (en lignes)  : {taux_lignes:.1%}")
    print(f"Taux d'appariement Sirene (en montant) : {montant_ok / montant_tot:.1%}")

    if export_path:
        df_final.to_parquet(export_path, index=False)
    if ref_secteurs_sirene is not None and export_ref_path:
        ref_secteurs_sirene.to_parquet(export_ref_path, index=False)

    return df_final


def construire_synthese_secteurs(
    df_final: pd.DataFrame, ref_secteurs_sirene: pd.DataFrame
) -> pd.DataFrame:
    """Construit le tableau de synthèse sectorielle avec l'indice de surreprésentation."""
    synthese = (
        df_final.groupby("naf_section", as_index=False)
        .agg(
            nb_contrats=("siret", "count"),
            nb_entreprises=("siren", "nunique"),
            montant_total_M=("montant_part", lambda s: s.sum() / 1e6),
            montant_median=("montant_part", "median"),
        )
        .merge(ref_secteurs_sirene, on="naf_section", how="left")
    )
    synthese["part_montant_decp"] = (
        synthese["montant_total_M"] / synthese["montant_total_M"].sum()
    )
    synthese["indice_surrepresentation"] = (
        synthese["part_montant_decp"] / synthese["part_etablissements_sirene"]
    )
    return synthese


# ==============================================================================
# 3. VISUALISATIONS EXPLORATOIRES (DECP x SIRENE)
# ==============================================================================

SIZE_MAPPING = {
    "NN": "Non renseigné / 0 salarié",
    "00": "Micro (0-9 salariés)",
    "01": "Micro (0-9 salariés)",
    "02": "Micro (0-9 salariés)",
    "03": "Micro (0-9 salariés)",
    "11": "Petite (10-49 salariés)",
    "12": "Petite (10-49 salariés)",
    "21": "Moyenne (50-249 salariés)",
    "22": "Moyenne (50-249 salariés)",
    "31": "Moyenne (50-249 salariés)",
    "32": "ETI (250-4 999 salariés)",
    "41": "ETI (250-4 999 salariés)",
    "42": "ETI (250-4 999 salariés)",
    "51": "ETI (250-4 999 salariés)",
    "52": "Grande Entreprise (5 000+ salariés)",
    "53": "Grande Entreprise (5 000+ salariés)",
}

CATEGORY_ORDER = [
    "Micro (0-9 salariés)",
    "Petite (10-49 salariés)",
    "Moyenne (50-249 salariés)",
    "ETI (250-4 999 salariés)",
    "Grande Entreprise (5 000+ salariés)",
    "Non renseigné / 0 salarié",
]


def _filtrer_apparies(df_final: pd.DataFrame) -> pd.DataFrame:
    """Conserve les lignes valides appariées dans Sirene avec un montant positif."""
    col_match = "siren_sirene" if "siren_sirene" in df_final.columns else "naf_code"
    return df_final[df_final[col_match].notna() & (df_final["montant_part"] > 0)].copy()


def plot_montants_par_secteur(df_final: pd.DataFrame):
    """Graphique 1 : Volume financier total attribué par section NAF (tous secteurs)."""
    df_valid = _filtrer_apparies(df_final)
    df_valid = df_valid[df_valid["naf_section"].notna()].copy()

    sector_totals = (
        df_valid.groupby("naf_section", observed=True, as_index=False)
        .agg(
            montant_total=("montant_part", "sum"),
            nb_marches=("id", "nunique"),
        )
        .sort_values("montant_total", ascending=False)
    )
    sector_totals["montant_millions"] = sector_totals["montant_total"] / 1e6

    sns.set_theme(style="whitegrid")
    fig_height = max(7, len(sector_totals) * 0.45)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    sns.barplot(
        data=sector_totals,
        x="montant_millions",
        y="naf_section",
        palette="viridis",
        hue="naf_section",
        legend=False,
        ax=ax,
    )

    max_val = sector_totals["montant_millions"].max()
    for container in ax.containers:
        ax.bar_label(container, fmt=" %.1f M€", padding=4, fontsize=9, color="#222222")

    ax.set_xlim(0, max_val * 1.15)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f} M€"))
    ax.set_xlabel("Montant total attribué (en millions d'euros)", fontsize=11, labelpad=10)
    ax.set_ylabel("")
    ax.set_title(
        "Montant total des marchés publics attribués par section NAF de l'établissement",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_distribution_montants_annualises(df_final: pd.DataFrame):
    """Graphique 2 : Distribution des montants annualisés par grande section NAF (Top 10)."""
    top_sections = df_final["naf_section"].value_counts().nlargest(10).index
    df_plot = df_final[
        df_final["naf_section"].isin(top_sections) & (df_final["montant_annualise"] > 0)
    ]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 6))
    sns.boxplot(
        data=df_plot,
        x="montant_annualise",
        y="naf_section",
        hue="naf_section",
        order=top_sections,
        palette="crest",
        legend=False,
        showfliers=False,
    )
    plt.xscale("log")
    plt.xlabel("Montant annualisé (€, échelle logarithmique)")
    plt.ylabel("")
    plt.title("Distribution des montants annualisés par grande section NAF (Top 10)")
    plt.tight_layout()
    plt.show()


def plot_repartition_taille_etablissement(df_final: pd.DataFrame):
    """Graphique 3 : Volume total capté et distribution unitaire par taille d'établissement."""
    df_size = _filtrer_apparies(df_final)
    df_size["categorie_taille"] = (
        df_size["trancheEffectifsEtablissement"]
        .fillna("NN")
        .map(SIZE_MAPPING)
        .fillna("Non renseigné / 0 salarié")
    )

    agg_size = (
        df_size.groupby("categorie_taille", observed=False)
        .agg(
            montant_total=("montant_part", "sum"),
            nb_marches=("id", "nunique"),
            montant_median=("montant_part", "median"),
        )
        .reindex(CATEGORY_ORDER, fill_value=0)
        .reset_index()
    )
    agg_size["montant_millions"] = agg_size["montant_total"] / 1e6
    agg_size["part_volume_pct"] = (
        agg_size["montant_total"] / agg_size["montant_total"].sum() * 100
    )

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.1, 1]})

    sns.barplot(
        data=agg_size,
        x="montant_millions",
        y="categorie_taille",
        palette="Blues_d",
        hue="categorie_taille",
        legend=False,
        ax=axes[0],
    )
    max_m = agg_size["montant_millions"].max()
    for idx, row in agg_size.iterrows():
        axes[0].text(
            row["montant_millions"] + max_m * 0.015,
            idx,
            f"{row['montant_millions']:,.1f} M€ ({row['part_volume_pct']:.1f}%)",
            va="center",
            fontsize=9.5,
            color="#222222",
        )
    axes[0].set_xlim(0, max_m * 1.32)
    axes[0].xaxis.set_major_formatter("{x:,.0f} M€")
    axes[0].set_xlabel("Montant total cumulé (M€)", fontsize=11)
    axes[0].set_ylabel("")
    axes[0].set_title(
        "1. Volume total capté par taille d'établissement", fontsize=12, fontweight="bold"
    )

    sns.boxplot(
        data=df_size,
        x="montant_part",
        y="categorie_taille",
        order=CATEGORY_ORDER,
        palette="Blues_d",
        hue="categorie_taille",
        legend=False,
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Montant par marché (€, échelle log)", fontsize=11)
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])
    axes[1].set_title(
        "2. Distribution des montants unitaires (médiane & quartiles)",
        fontsize=12,
        fontweight="bold",
    )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_montants_par_procedure(df_final: pd.DataFrame):
    """Graphique 4 : Montant total attribué par type de procédure de passation."""
    df_proc = _filtrer_apparies(df_final)
    df_proc = df_proc[df_proc["procedure"].notna()].copy()

    proc_agg = (
        df_proc.groupby("procedure", as_index=False)
        .agg(
            montant_total=("montant_part", "sum"),
            nb_marches=("id", "nunique"),
            montant_median=("montant_part", "median"),
        )
        .sort_values("montant_total", ascending=False)
        .reset_index(drop=True)
    )
    proc_agg["montant_millions"] = proc_agg["montant_total"] / 1e6
    proc_agg["part_volume_pct"] = (
        proc_agg["montant_total"] / proc_agg["montant_total"].sum() * 100
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, max(5, len(proc_agg) * 0.55)))

    sns.barplot(
        data=proc_agg,
        x="montant_millions",
        y="procedure",
        palette="mako",
        hue="procedure",
        legend=False,
        ax=ax,
    )

    max_val = proc_agg["montant_millions"].max()
    for idx, row in proc_agg.iterrows():
        ax.text(
            row["montant_millions"] + max_val * 0.015,
            idx,
            f"{row['montant_millions']:,.1f} M€ ({row['part_volume_pct']:.1f}% | n={row['nb_marches']:,})",
            va="center",
            fontsize=9.5,
            color="#222222",
        )

    ax.set_xlim(0, max_val * 1.38)
    ax.xaxis.set_major_formatter("{x:,.0f} M€")
    ax.set_xlabel("Montant total attribué (en millions d'euros)", fontsize=11, labelpad=10)
    ax.set_ylabel("")
    ax.set_title(
        "Montant total de la commande publique par type de procédure de passation",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_procedure_par_taille(df_final: pd.DataFrame):
    """Graphique 5 : Croisement Procédure x Taille (Nombre de marchés vs Volume financier)."""
    df_proc = _filtrer_apparies(df_final)
    df_proc = df_proc[df_proc["procedure"].notna()].copy()

    df_proc["categorie_taille"] = (
        df_proc["trancheEffectifsEtablissement"]
        .fillna("NN")
        .map(SIZE_MAPPING)
        .fillna("Non renseigné / 0 salarié")
    )

    proc_order = (
        df_proc.groupby("procedure")["montant_part"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    ct_count = (
        pd.crosstab(df_proc["procedure"], df_proc["categorie_taille"], normalize="index")
        .reindex(index=proc_order, columns=CATEGORY_ORDER, fill_value=0)
        * 100
    )
    ct_volume = (
        pd.crosstab(
            df_proc["procedure"],
            df_proc["categorie_taille"],
            values=df_proc["montant_part"],
            aggfunc="sum",
            normalize="index",
        )
        .reindex(index=proc_order, columns=CATEGORY_ORDER, fill_value=0)
        * 100
    )

    sns.set_theme(style="whitegrid")
    colors = ["#2b5c8f", "#4c8cb5", "#7fb3d5", "#f39c12", "#d35400", "#bdc3c7"]

    fig, axes = plt.subplots(
        1, 2, figsize=(16, max(5.5, len(proc_order) * 0.6)), sharey=True
    )

    ct_count.plot(kind="barh", stacked=True, color=colors, width=0.72, ax=axes[0], legend=False)
    axes[0].set_title("A. Répartition en NOMBRE de marchés (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Part des attributions (%)")
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, 100)

    ct_volume.plot(kind="barh", stacked=True, color=colors, width=0.72, ax=axes[1], legend=False)
    axes[1].set_title("B. Répartition en VOLUME financier (% des €)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Part des montants attribués (%)")
    axes[1].set_xlim(0, 100)

    for ax in axes:
        ax.invert_yaxis()
        for container in ax.containers:
            labels = [f"{v:.0f}%" if v >= 8 else "" for v in container.datavalues]
            ax.bar_label(
                container,
                labels=labels,
                label_type="center",
                color="white",
                fontsize=8.5,
                fontweight="bold",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=6,
        frameon=False,
        fontsize=9.5,
    )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_carte_departements(df_final: pd.DataFrame):
    """Graphique 6 : Carte choroplèthe interactive des montants par département du titulaire."""
    geojson_url = (
        "https://raw.githubusercontent.com/gregoiredavid/france-geojson/"
        "master/departements-version-simplifiee.geojson"
    )
    departements_geo = requests.get(geojson_url, timeout=30).json()

    df_matched = _filtrer_apparies(df_final)
    dep_stats = (
        df_matched.groupby("dep_titulaire", as_index=False)
        .agg(
            nb_marches=("id", "nunique"),
            montant_total=("montant_part", "sum"),
            montant_median=("montant_part", "median"),
        )
    )
    dep_stats["montant_millions"] = dep_stats["montant_total"] / 1e6

    fig = px.choropleth(
        dep_stats,
        geojson=departements_geo,
        locations="dep_titulaire",
        featureidkey="properties.code",
        color="montant_millions",
        hover_name="dep_titulaire",
        hover_data={"nb_marches": True, "montant_millions": ":.1f"},
        color_continuous_scale="Viridis",
        labels={"montant_millions": "Montant attribué (M€)", "nb_marches": "Nombre de marchés"},
        title="Montant des marchés publics par département de l'établissement titulaire",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.show()


# ==============================================================================
# 4. APPROFONDISSEMENT FINANCIER (AGRÉGATION SIREN x INPI / BCE)
# ==============================================================================

def first_mode(series: pd.Series):
    """Retourne la modalité la plus fréquente (mode) d'une série pandas."""
    m = series.dropna().mode()
    return m.iloc[0] if not m.empty else None


def agreger_et_joindre_inpi(df_final: pd.DataFrame, url_inpi: str) -> pd.DataFrame:
    """Agrège df_final au niveau SIREN (9 chiffres) et joint le dernier bilan INPI/BCE (1-pour-1)."""
    df_work = df_final[df_final["siret"].notna()].copy()
    df_work["siren"] = df_work["siret"].astype(str).str.strip().str.zfill(14).str[:9]

    agg_dict = {
        "nb_marches": ("id", "nunique"),
        "nb_etablissements_gagnants": ("siret", "nunique"),
        "montant_total": ("montant_part", "sum"),
        "montant_moyen": ("montant_part", "mean"),
        "montant_median": ("montant_part", "median"),
        "montant_annualise_total": ("montant_annualise", "sum"),
        "naf_section_principale": ("naf_section", first_mode),
        "naf_division_principale": ("naf_division", first_mode),
        "tranche_effectifs_max": ("trancheEffectifsEtablissement", "max"),
        "dep_principal": ("dep_titulaire", first_mode),
        "a_gagne_via_siege": ("etablissementSiege", "any"),
    }

    for col_nom in ["titulaire_nom", "denominationUniteLegale", "nom"]:
        if col_nom in df_work.columns:
            agg_dict["entreprise_nom"] = (col_nom, "first")
            break

    df_siren_decp = df_work.groupby("siren", as_index=False).agg(**agg_dict)

    sirens_decp = df_siren_decp["siren"].dropna().unique().tolist()
    df_inpi = pd.read_parquet(url_inpi, filters=[("siren", "in", sirens_decp)])

    df_inpi["siren"] = df_inpi["siren"].astype(str).str.strip().str.zfill(9)
    df_inpi["date_cloture_exercice"] = pd.to_datetime(
        df_inpi["date_cloture_exercice"], errors="coerce"
    )

    df_inpi_latest = (
        df_inpi.sort_values(
            by=["siren", "date_cloture_exercice", "type_bilan"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["siren"], keep="first")
    )

    df_entreprise = pd.merge(
        df_siren_decp,
        df_inpi_latest,
        on="siren",
        how="left",
        validate="one_to_one",
    )

    n_total = len(df_entreprise)
    n_inpi_present = df_entreprise["date_cloture_exercice"].notna().sum()
    n_ca_disponible = df_entreprise["chiffre_d_affaires"].notna().sum()

    print(f"Entreprises uniques (SIREN) dans DECP : {n_total:,}")
    print(f"Entreprises présentes dans la base INPI : {n_inpi_present:,} ({n_inpi_present / n_total:.1%})")
    print(f"Entreprises avec chiffre d'affaires non confidentiel : {n_ca_disponible:,} ({n_ca_disponible / n_total:.1%})")

    return df_entreprise


def diagnostiquer_base_entreprises(df_entreprise: pd.DataFrame, url_inpi: str = None):
    """Affiche le diagnostic des valeurs manquantes et la répartition de confidentialité."""
    nan_summary = pd.DataFrame({
        "nb_valeurs_presentes": df_entreprise.notna().sum(),
        "nb_NaN": df_entreprise.isna().sum(),
        "taux_NaN_pct": (df_entreprise.isna().mean() * 100).round(1),
        "type_colonne": df_entreprise.dtypes,
    })
    print("=== DIAGNOSTIC DES VALEURS MANQUANTES (df_entreprise) ===")
    print(nan_summary.to_string())

    if url_inpi:
        sample_inpi = pd.read_parquet(url_inpi, columns=["siren", "confidentiality"]).head(10)
        print("\n=== APERÇU DU FORMAT BRUT DANS URL_INPI ===")
        print("Type de la colonne siren :", sample_inpi["siren"].dtype)
        print(sample_inpi)

    if "confidentiality" in df_entreprise.columns:
        print("\n=== STATUT DE CONFIDENTIALITÉ DES COMPTES (ENTREPRISES APPARIÉES) ===")
        print(df_entreprise["confidentiality"].value_counts(dropna=False))


def plot_dependance_loglog(df_entreprise: pd.DataFrame):
    """Graphique 7 : Nuage de points log-log (Montant annualisé des marchés vs Chiffre d'affaires)."""
    df_plot = df_entreprise[
        (df_entreprise["chiffre_d_affaires"] > 0)
        & (df_entreprise["montant_annualise_total"] > 0)
    ].copy()

    df_plot["taux_dependance_pct"] = (
        df_plot["montant_annualise_total"] / df_plot["chiffre_d_affaires"] * 100
    )

    top_naf = df_plot["naf_section_principale"].value_counts().nlargest(6).index
    df_plot["secteur_affiche"] = np.where(
        df_plot["naf_section_principale"].isin(top_naf),
        df_plot["naf_section_principale"],
        "Autres sections NAF",
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))

    sns.scatterplot(
        data=df_plot,
        x="chiffre_d_affaires",
        y="montant_annualise_total",
        hue="secteur_affiche",
        alpha=0.55,
        s=35,
        edgecolor=None,
        ax=ax,
    )

    x_min, x_max = df_plot["chiffre_d_affaires"].min(), df_plot["chiffre_d_affaires"].max()
    x_vals = np.logspace(np.log10(max(x_min, 1e3)), np.log10(x_max), 200)

    for ratio, label, style in [
        (1.0, "100 % du CA (dépendance totale)", "--"),
        (0.1, "10 % du CA", ":"),
        (0.01, "1 % du CA", "-."),
    ]:
        ax.plot(x_vals, x_vals * ratio, linestyle=style, color="#444444", linewidth=1.1, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Chiffre d'affaires annuel de l'entreprise (€, échelle log)", fontsize=11, labelpad=10)
    ax.set_ylabel("Montant annualisé des marchés gagnés (€, échelle log)", fontsize=11, labelpad=10)
    ax.set_title(
        f"Montant des marchés publics vs. Chiffre d'affaires (n = {len(df_plot):,} entreprises)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=9)

    sns.despine()
    plt.tight_layout()
    plt.show()

    print(
        "Médiane du ratio (Montant annualisé / CA) :",
        f"{df_plot['taux_dependance_pct'].median():.2f} %",
    )
    print(
        "Part des entreprises réalisant > 10 % de leur CA via la commande publique :",
        f"{(df_plot['taux_dependance_pct'] > 10).mean():.1%}",
    )


def plot_tranches_dependance_par_secteur(df_entreprise: pd.DataFrame):
    """Graphique 8 : Répartition par tranche de dépendance (< 1 %, 1-100 %, > 100 % du CA) par section NAF."""
    df_dep = df_entreprise[
        (df_entreprise["chiffre_d_affaires"] > 0)
        & (df_entreprise["montant_annualise_total"] > 0)
        & df_entreprise["naf_section_principale"].notna()
    ].copy()

    df_dep["taux_dependance_pct"] = (
        df_dep["montant_annualise_total"] / df_dep["chiffre_d_affaires"] * 100
    )

    bins = [0, 1, 100, np.inf]
    labels_tranches = [
        "< 1 % du CA (sous la ligne 1 %)",
        "1 % à 100 % du CA (entre les deux)",
        "> 100 % du CA (au-dessus de 100 %)",
    ]
    df_dep["tranche_dependance"] = pd.cut(
        df_dep["taux_dependance_pct"],
        bins=bins,
        labels=labels_tranches,
        right=True,
    )

    ct_counts = pd.crosstab(df_dep["naf_section_principale"], df_dep["tranche_dependance"])
    ct_counts = ct_counts.loc[ct_counts.sum(axis=1).sort_values(ascending=False).index]
    ct_pct = ct_counts.div(ct_counts.sum(axis=1), axis=0) * 100

    sns.set_theme(style="whitegrid")
    colors = ["#4c8cb5", "#f39c12", "#c0392b"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, max(7, len(ct_counts) * 0.45)),
        sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1]},
    )

    ct_counts.plot(kind="barh", color=colors, width=0.78, ax=axes[0], legend=False)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Nombre d'entreprises (SIREN)", fontsize=11)
    axes[0].set_ylabel("")
    axes[0].set_title("A. Nombre d'entreprises par tranche et secteur", fontsize=12, fontweight="bold")

    for container in axes[0].containers:
        labels = [f" {int(v):,}" if v > 0 else "" for v in container.datavalues]
        axes[0].bar_label(container, labels=labels, fontsize=8, color="#222222")
    axes[0].set_xlim(0, ct_counts.values.max() * 1.15)

    ct_pct.plot(kind="barh", stacked=True, color=colors, width=0.72, ax=axes[1], legend=False)
    axes[1].set_xlabel("Part des entreprises du secteur (%)", fontsize=11)
    axes[1].set_xlim(0, 100)
    axes[1].set_title("B. Part relative au sein de chaque secteur (%)", fontsize=12, fontweight="bold")

    for container in axes[1].containers:
        labels = [f"{v:.0f}%" if v >= 7 else "" for v in container.datavalues]
        axes[1].bar_label(
            container,
            labels=labels,
            label_type="center",
            color="white",
            fontsize=8.5,
            fontweight="bold",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()

    print("=== RÉPARTITION GLOBALE PAR TRANCHE DE DÉPENDANCE ===")
    print(
        ct_counts.sum()
        .to_frame("nb_entreprises")
        .assign(part_pct=lambda d: (d["nb_entreprises"] / d["nb_entreprises"].sum() * 100).round(1))
    )
