import fsspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plotly.express as px
import pyarrow.parquet as pq
import requests
import seaborn as sns


def plot_montant_par_naf(df_final: pd.DataFrame) -> None:
    """
    Affiche un graphique en barres horizontales du montant total attribué (en M€)
    par section NAF de l'établissement titulaire.

    Paramètres
    ----------
    df_final : pd.DataFrame
        Table fusionnée (marchés DECP x Sirene) contenant les colonnes
        'naf_code', 'naf_section', 'montant_part' et 'id'.
    """
    # Sous-ensemble des lignes valides appariées dans Sirene avec montant positif
    df_apparies = df_final[
        df_final["naf_code"].notna()
        & (df_final["naf_section"] != "Inconnu")
        & (df_final["montant_part"] > 0)
    ].copy()

    # Agrégation des montants (en M€) et du nombre de marchés distincts par section NAF
    totaux_secteurs = (
        df_apparies.groupby("naf_section", observed=True, as_index=False)
        .agg(
            montant_total=("montant_part", "sum"),
            nb_marches=("id", "nunique"),
        )
        .sort_values("montant_total", ascending=False)
    )
    totaux_secteurs["montant_millions"] = totaux_secteurs["montant_total"] / 1e6

    # Graphique en barres horizontales avec hauteur adaptée au nombre de sections
    sns.set_theme(style="whitegrid")
    fig_height = max(7, len(totaux_secteurs) * 0.45)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    sns.barplot(
        data=totaux_secteurs,
        x="montant_millions",
        y="naf_section",
        hue="naf_section",
        palette="viridis",
        legend=False,
        ax=ax,
    )

    # Ajout des valeurs directement au bout de chaque barre pour lire les petits secteurs
    max_val = totaux_secteurs["montant_millions"].max()
    for container in ax.containers:
        ax.bar_label(container, fmt=" %.1f M€", padding=4, fontsize=9, color="#222222")

    ax.set_xlim(0, max_val * 1.18)
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


def plot_boxplot_montant_an(df_final: pd.DataFrame) -> None:
    """
    Affiche la distribution des montants annualisés (en échelle logarithmique)
    sous forme de boîtes à moustaches (boxplots) pour les 10 sections NAF
    les plus fréquentes en nombre de contrats.

    Paramètres
    ----------
    df_final : pd.DataFrame
        Table fusionnée (marchés DECP x Sirene) contenant les colonnes
        'naf_code', 'naf_section', 'montant_part' et 'montant_annualise'.
    """
    # Sous-ensemble des lignes valides appariées dans Sirene avec montant positif
    df_apparies = df_final[
        df_final["naf_code"].notna()
        & (df_final["naf_section"] != "Inconnu")
        & (df_final["montant_part"] > 0)
    ].copy()

    # Sélection des 10 sections les plus fréquentes en nombre de contrats
    top_10_sections = df_apparies["naf_section"].value_counts().nlargest(10).index
    df_boxplot = df_apparies[
        df_apparies["naf_section"].isin(top_10_sections)
        & (df_apparies["montant_annualise"] > 0)
    ]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 6))
    sns.boxplot(
        data=df_boxplot,
        x="montant_annualise",
        y="naf_section",
        hue="naf_section",
        order=top_10_sections,
        palette="crest",
        legend=False,
        showfliers=False,  # Masque les valeurs extrêmes pour garder les boîtes interquartiles lisibles
    )

    plt.xscale("log")
    plt.xlabel("Montant annualisé (€, échelle logarithmique)")
    plt.ylabel("")
    plt.title("Distribution des montants annualisés par grande section NAF (Top 10)")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_siege_procedure(df_final: pd.DataFrame) -> None:
    """
    Affiche la répartition en volume financier (%) entre agences secondaires (False)
    et sièges sociaux (True) selon le type de procédure de marché public.

    Paramètres
    ----------
    df_final : pd.DataFrame
        Table fusionnée (marchés DECP x Sirene) contenant les colonnes
        'naf_code', 'naf_section', 'montant_part', 'procedure',
        'etablissementSiege' et 'trancheEffectifsEtablissement'.
    """
    # Sous-ensemble des lignes valides appariées dans Sirene avec montant positif
    df_apparies = df_final[
        df_final["naf_code"].notna()
        & (df_final["naf_section"] != "Inconnu")
        & (df_final["montant_part"] > 0)
    ].copy()

    # Table de correspondance des codes Insee de tranches d'effectifs salariés
    libelles_effectifs = {
        "NN": "0 salarié / Non renseigné",
        "00": "0 salarié",
        "01": "1-2 sal.",
        "02": "3-5 sal.",
        "03": "6-9 sal.",
        "11": "10-19 sal.",
        "12": "20-49 sal.",
        "21": "50-99 sal.",
        "22": "100-199 sal.",
        "31": "200-249 sal.",
        "32": "250-499 sal.",
        "41": "500-999 sal.",
        "42": "1000-1999 sal.",
        "51": "2000-4999 sal.",
        "52": "5000+ sal.",
        "53": "5000+ sal.",
    }
    df_apparies["taille_etab"] = df_apparies["trancheEffectifsEtablissement"].map(libelles_effectifs)

    # Répartition en volume financier (%) entre agences locales (False) et sièges sociaux (True)
    ct_volume = (
        pd.crosstab(
            df_apparies["procedure"],
            df_apparies["etablissementSiege"],
            values=df_apparies["montant_part"],
            aggfunc="sum",
            normalize="index",
        )
        * 100
    )

    sns.set_theme(style="whitegrid")
    ax = ct_volume.plot(
        kind="barh",
        stacked=True,
        figsize=(10, 5),
        color=["#4c72b0", "#dd8452"],
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Part du montant total attribué (%)")
    ax.set_ylabel("")
    ax.legend(
        ["Agence secondaire (False)", "Siège social (True)"],
        title="Type d'établissement",
        loc="lower right",
    )
    ax.set_title("Part des montants captés par les sièges sociaux selon la procédure")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_carte_montants(
    df_final: pd.DataFrame,
    geojson_url: str = (
        "https://raw.githubusercontent.com/gregoiredavid/france-geojson/"
        "master/departements-version-simplifiee.geojson"
    ),
) -> None:
    """
    Affiche une carte choroplèthe interactive des montants attribués (en M€)
    et du nombre de marchés par département d'implantation de l'établissement titulaire.

    Paramètres
    ----------
    df_final : pd.DataFrame
        Table fusionnée (marchés DECP x Sirene) contenant les colonnes
        'naf_code', 'naf_section', 'montant_part', 'dep_titulaire' et 'id'.
    geojson_url : str, optionnel
        URL vers le contour GeoJSON des départements français.
    """
    # Sous-ensemble des lignes valides appariées dans Sirene avec montant positif
    df_apparies = df_final[
        df_final["naf_code"].notna()
        & (df_final["naf_section"] != "Inconnu")
        & (df_final["montant_part"] > 0)
    ].copy()

    # Chargement du contour simplifié des départements français (GeoJSON)
    departements_geo = requests.get(geojson_url, timeout=30).json()

    # Agrégation par département d'implantation de l'établissement titulaire
    stats_dep = (
        df_apparies.groupby("dep_titulaire", as_index=False)
        .agg(
            nb_marches=("id", "nunique"),
            montant_total=("montant_part", "sum"),
            montant_median=("montant_part", "median"),
        )
    )
    stats_dep["montant_millions"] = stats_dep["montant_total"] / 1e6

    # Carte choroplèthe interactive
    fig = px.choropleth(
        stats_dep,
        geojson=departements_geo,
        locations="dep_titulaire",
        featureidkey="properties.code",
        color="montant_millions",
        hover_name="dep_titulaire",
        hover_data={"nb_marches": True, "montant_millions": ":.1f"},
        color_continuous_scale="Viridis",
        labels={
            "montant_millions": "Montant attribué (M€)",
            "nb_marches": "Nombre de marchés",
        },
        title="Montant des marchés publics par département de l'établissement titulaire",
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.show()


def plot_montants_taille(df_final: pd.DataFrame) -> None:
    """
    Affiche deux graphiques côte à côte selon la taille de l'établissement (tranches d'effectifs Insee) :
    1. Le volume total cumulé capté (en M€ et en % du total).
    2. La distribution des montants unitaires par marché (échelle logarithmique, hors valeurs extrêmes).

    Paramètres
    ----------
    df_final : pd.DataFrame
        Table fusionnée (marchés DECP x Sirene) contenant les colonnes
        'trancheEffectifsEtablissement', 'montant_part', 'id' ainsi qu'une
        colonne témoin de l'appariement Sirene ('naf_code' ou 'siren_sirene').
    """
    # 1. Table de correspondance des codes Insee de tranches d'effectifs salariés
    # Note : trancheEffectifsEtablissement mesure l'effectif de l'établissement (proxy de la taille)
    size_mapping = {
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

    category_order = [
        "Micro (0-9 salariés)",
        "Petite (10-49 salariés)",
        "Moyenne (50-249 salariés)",
        "ETI (250-4 999 salariés)",
        "Grande Entreprise (5 000+ salariés)",
        "Non renseigné / 0 salarié",
    ]

    # 2. Filtrer les lignes valides appariées dans Sirene avec un montant positif
    col_temoin = "siren_sirene" if "siren_sirene" in df_final.columns else "naf_code"
    df_size = df_final[
        df_final[col_temoin].notna() & (df_final["montant_part"] > 0)
    ].copy()

    df_size["categorie_taille"] = (
        df_size["trancheEffectifsEtablissement"]
        .fillna("NN")
        .map(size_mapping)
        .fillna("Non renseigné / 0 salarié")
    )

    # 3. Agréger le montant total, le nombre de marchés et la médiane par catégorie de taille
    agg_size = (
        df_size.groupby("categorie_taille", observed=False)
        .agg(
            montant_total=("montant_part", "sum"),
            nb_marches=("id", "nunique"),
            montant_median=("montant_part", "median"),
        )
        .reindex(category_order, fill_value=0)
        .reset_index()
    )

    agg_size["montant_millions"] = agg_size["montant_total"] / 1e6
    agg_size["part_volume_pct"] = (
        agg_size["montant_total"] / agg_size["montant_total"].sum() * 100
    )

    # 4. Graphique : Gauche = Volume cumulé (barres) | Droite = Distribution unitaire (boxplots)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.1, 1]})

    # --- Panneau 1 : Volume total capté ---
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
    axes[0].set_title("1. Volume total capté par taille d'établissement", fontsize=12, fontweight="bold")

    # --- Panneau 2 : Distribution des montants individuels (échelle logarithmique) ---
    sns.boxplot(
        data=df_size,
        x="montant_part",
        y="categorie_taille",
        order=category_order,
        palette="Blues_d",
        hue="categorie_taille",
        legend=False,
        showfliers=False,
        ax=axes[1],
    )

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Montant par marché (€, échelle log)", fontsize=11)
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])  # Partage proprement les étiquettes de l'axe Y avec le panneau gauche
    axes[1].set_title("2. Distribution des montants unitaires (médiane & quartiles)", fontsize=12, fontweight="bold")

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_montant_procedure(df_final: pd.DataFrame) -> None:
    """
    Affiche un graphique en barres horizontales du montant total attribué (en M€),
    de la part en volume (%) et du nombre de marchés par type de procédure de passation.

    Paramètres
    ----------
    df_final : pd.DataFrame
        Table fusionnée (marchés DECP x Sirene) contenant les colonnes
        'procedure', 'montant_part', 'id' ainsi qu'une colonne témoin
        de l'appariement Sirene ('siren_sirene' ou 'naf_code').
    """
    # 1. Préparer le sous-ensemble valide apparié dans Sirene avec procédure renseignée
    col_temoin = "siren_sirene" if "siren_sirene" in df_final.columns else "naf_code"
    df_proc = df_final[
        df_final[col_temoin].notna()
        & df_final["procedure"].notna()
        & (df_final["montant_part"] > 0)
    ].copy()

    # 2. Agréger le montant total, le nombre de marchés distincts et la médiane par procédure
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

    # 3. Tracer le graphique en barres horizontales
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

    # Annoter chaque barre avec les M€, la part en % de la dépense et le nombre de contrats
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


def plot_procedure_taille_etablissement(df_final: pd.DataFrame) -> None:
    """
    Affiche côte à côte deux graphiques en barres horizontales empilées (100 %)
    croisant le type de procédure de passation et la taille de l'établissement :
    - Panneau A : Répartition en nombre de marchés (%).
    - Panneau B : Répartition en volume financier (% des euros attribués).

    Paramètres
    ----------
    df_final : pd.DataFrame
        Table fusionnée (marchés DECP x Sirene) contenant les colonnes
        'procedure', 'trancheEffectifsEtablissement', 'montant_part' ainsi qu'une
        colonne témoin de l'appariement Sirene ('siren_sirene' ou 'naf_code').
    """
    size_mapping = {
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

    category_order = [
        "Micro (0-9 salariés)",
        "Petite (10-49 salariés)",
        "Moyenne (50-249 salariés)",
        "ETI (250-4 999 salariés)",
        "Grande Entreprise (5 000+ salariés)",
        "Non renseigné / 0 salarié",
    ]

    # 1. Préparer le sous-ensemble valide et catégoriser la taille des établissements
    col_temoin = "siren_sirene" if "siren_sirene" in df_final.columns else "naf_code"
    df_proc = df_final[
        df_final[col_temoin].notna()
        & df_final["procedure"].notna()
        & (df_final["montant_part"] > 0)
    ].copy()

    df_proc["categorie_taille"] = (
        df_proc["trancheEffectifsEtablissement"]
        .fillna("NN")
        .map(size_mapping)
        .fillna("Non renseigné / 0 salarié")
    )

    # Ordonner les procédures du plus grand volume financier au plus petit
    proc_order = (
        df_proc.groupby("procedure")["montant_part"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # 2. Calculer les tableaux croisés normalisés à 100 % : (A) en nombre, (B) en volume
    ct_count = (
        pd.crosstab(df_proc["procedure"], df_proc["categorie_taille"], normalize="index")
        .reindex(index=proc_order, columns=category_order, fill_value=0)
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
        .reindex(index=proc_order, columns=category_order, fill_value=0)
        * 100
    )

    # 3. Tracer les barres horizontales empilées à 100 % côte à côte
    sns.set_theme(style="whitegrid")
    colors = ["#2b5c8f", "#4c8cb5", "#7fb3d5", "#f39c12", "#d35400", "#bdc3c7"]

    fig, axes = plt.subplots(
        1, 2, figsize=(16, max(5.5, len(proc_order) * 0.6)), sharey=True
    )

    # Panneau de gauche : Part du nombre de contrats
    ct_count.plot(kind="barh", stacked=True, color=colors, width=0.72, ax=axes[0], legend=False)
    axes[0].set_title("A. Répartition en NOMBRE de marchés (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Part des attributions (%)")
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, 100)

    # Panneau de droite : Part du volume financier total
    ct_volume.plot(kind="barh", stacked=True, color=colors, width=0.72, ax=axes[1], legend=False)
    axes[1].set_title("B. Répartition en VOLUME financier (% des €)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Part des montants attribués (%)")
    axes[1].set_xlim(0, 100)

    # Ajouter les étiquettes de pourcentage à l'intérieur des segments suffisamment larges (>= 8 %)
    for ax in axes:
        ax.invert_yaxis()  # Conserver la procédure au plus fort volume en haut
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

    # Légende commune unique en haut de la figure
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