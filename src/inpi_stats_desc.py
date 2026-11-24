import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.express as px

def plot_dependance_ca(df_entreprise: pd.DataFrame) -> None:
    """
    Affiche un nuage de points en échelle log-log croisant le chiffre d'affaires annuel
    des entreprises et le montant annualisé de leurs marchés publics gagnés, avec les
    droites repères de dépendance (1 %, 10 % et 100 % du CA) et le résumé statistique.

    Paramètres
    ----------
    df_entreprise : pd.DataFrame
        Table consolidée au niveau SIREN contenant les colonnes
        'chiffre_d_affaires', 'montant_annualise_total' et 'naf_section_principale'.
    """
    # 1. Filtrer les entreprises ayant un CA positif et un montant de marchés positif
    df_plot = df_entreprise[
        (df_entreprise["chiffre_d_affaires"] > 0)
        & (df_entreprise["montant_annualise_total"] > 0)
    ].copy()

    # Calculer le taux de dépendance à la commande publique (en %)
    df_plot["taux_dependance_pct"] = (
        df_plot["montant_annualise_total"] / df_plot["chiffre_d_affaires"] * 100
    )

    # Regrouper les sections NAF les plus fréquentes pour garder une légende lisible (Top 6 + Autres)
    top_naf = df_plot["naf_section_principale"].value_counts().nlargest(6).index
    df_plot["secteur_affiche"] = np.where(
        df_plot["naf_section_principale"].isin(top_naf),
        df_plot["naf_section_principale"],
        "Autres sections NAF",
    )

    # 2. Tracer le nuage de points en échelle log-log
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

    # 3. Ajouter les droites repères de ratio (Marchés annualisés / Chiffre d'affaires)
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

    # Placer la légende à l'extérieur à droite pour ne pas masquer les points
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=9)

    sns.despine()
    plt.tight_layout()
    plt.show()

    # Résumé statistique du taux de dépendance
    print("Médiane du ratio (Montant annualisé / CA) :", f"{df_plot['taux_dependance_pct'].median():.2f} %")
    print(
        "Part des entreprises réalisant > 10 % de leur CA via la commande publique :",
        f"{(df_plot['taux_dependance_pct'] > 10).mean():.1%}",
    )


def plot_dependance_secteur(df_entreprise: pd.DataFrame) -> pd.DataFrame:
    """
    Affiche deux graphiques en barres côte à côte répartissant les entreprises selon leur
    taux de dépendance à la commande publique (< 1 %, 1 % à 100 %, > 100 % du CA) et leur
    section NAF principale :
    - Panneau A : Effectifs bruts d'entreprises (SIREN) par tranche et secteur.
    - Panneau B : Répartition normalisée à 100 % au sein de chaque secteur.

    Paramètres
    ----------
    df_entreprise : pd.DataFrame
        Table consolidée au niveau SIREN contenant les colonnes
        'chiffre_d_affaires', 'montant_annualise_total' et 'naf_section_principale'.

    Retourne
    --------
    pd.DataFrame
        Tableau récapitulatif global du nombre et de la part (%) d'entreprises par tranche.
    """
    # 1. Filtrer les entreprises avec un CA et un montant de marchés strictement positifs
    df_dep = df_entreprise[
        (df_entreprise["chiffre_d_affaires"] > 0)
        & (df_entreprise["montant_annualise_total"] > 0)
        & df_entreprise["naf_section_principale"].notna()
    ].copy()

    # 2. Calculer le ratio (Montant annualisé des marchés / Chiffre d'affaires) en %
    df_dep["taux_dependance_pct"] = (
        df_dep["montant_annualise_total"] / df_dep["chiffre_d_affaires"] * 100
    )

    # 3. Découper en 3 catégories autour des seuils de 1 % et 100 %
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

    # 4. Créer les tableaux croisés : effectifs bruts (count) et parts en %
    ct_counts = pd.crosstab(df_dep["naf_section_principale"], df_dep["tranche_dependance"])
    # Trier les secteurs par nombre total d'entreprises décroissant
    ct_counts = ct_counts.loc[ct_counts.sum(axis=1).sort_values(ascending=False).index]
    ct_pct = ct_counts.div(ct_counts.sum(axis=1), axis=0) * 100

    # 5. Tracer les deux graphiques côte à côte
    sns.set_theme(style="whitegrid")
    colors = ["#4c8cb5", "#f39c12", "#c0392b"]  # Bleu (<1%), Orange (1-100%), Rouge (>100%)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, max(7, len(ct_counts) * 0.45)),
        sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1]},
    )

    # --- Panneau A : Nombre brut d'entreprises (barres horizontales groupées) ---
    ct_counts.plot(
        kind="barh",
        color=colors,
        width=0.78,
        ax=axes[0],
        legend=False,
    )
    axes[0].invert_yaxis()  # Secteur avec le plus d'entreprises en haut
    axes[0].set_xlabel("Nombre d'entreprises (SIREN)", fontsize=11)
    axes[0].set_ylabel("")
    axes[0].set_title("A. Nombre d'entreprises par tranche et secteur", fontsize=12, fontweight="bold")

    # Ajouter les effectifs au bout des barres (si > 0)
    for container in axes[0].containers:
        labels = [f" {int(v):,}" if v > 0 else "" for v in container.datavalues]
        axes[0].bar_label(container, labels=labels, fontsize=8, color="#222222")

    axes[0].set_xlim(0, ct_counts.values.max() * 1.15)

    # --- Panneau B : Répartition normalisée à 100 % par secteur ---
    ct_pct.plot(
        kind="barh",
        stacked=True,
        color=colors,
        width=0.72,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_xlabel("Part des entreprises du secteur (%)", fontsize=11)
    axes[1].set_xlim(0, 100)
    axes[1].set_title("B. Part relative au sein de chaque secteur (%)", fontsize=12, fontweight="bold")

    # Afficher les pourcentages à l'intérieur des segments suffisamment larges (>= 7 %)
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

    # Légende commune en haut de la figure
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

    # Tableau récapitulatif global
    resume_tranches = (
        ct_counts.sum()
        .to_frame("nb_entreprises")
        .assign(part_pct=lambda d: (d["nb_entreprises"] / d["nb_entreprises"].sum() * 100).round(1))
    )
    print("=== RÉPARTITION GLOBALE PAR TRANCHE DE DÉPENDANCE ===")
    print(resume_tranches)

    return resume_tranches