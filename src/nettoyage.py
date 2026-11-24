import gc

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Paramètres méthodologiques de filtrage des DECP
ANNEE_MIN = 2024          # Fenêtre d'étude retenue (écarte les anciens contrats mis à jour tardivement)
MONTANT_MIN_EUROS = 100   # Écarte les accords-cadres saisis à 1 € symbolique
MONTANT_MAX_EUROS = 5e8   # Écarte les erreurs manifestes de saisie (> 500 M€)


def analyser_valeurs_manquantes_decp(
    donnees_decp: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Calcule le bilan global des valeurs manquantes (NaN) sur l'ensemble
    des tables contenues dans le dictionnaire donnees_decp.
    Renvoie:
    - bilan_nan : DataFrame trié par nombre décroissant de valeurs manquantes.
    - colonnes_vides : Liste des colonnes entièrement vides (100 % de NaN).
    """
    total_lignes = sum(len(table) for table in donnees_decp.values())
    valeurs_valides = pd.concat(
        [table.notna().sum() for table in donnees_decp.values()],
        axis=1,
    ).sum(axis=1)

    bilan_nan = pd.DataFrame({
        "nb_nan": (total_lignes - valeurs_valides).astype(int),
        "pourcentage_nan": ((total_lignes - valeurs_valides) / total_lignes * 100).round(2),
    }).sort_values(by="nb_nan", ascending=False)

    colonnes_vides = bilan_nan.index[bilan_nan["pourcentage_nan"] == 100].tolist()
    print(f"Colonnes 100 % vides dans les DECP ({len(colonnes_vides)}) : {colonnes_vides}")

    return bilan_nan, colonnes_vides


def inspecter_parquet(url: str, nom_source: str, n_lignes: int = 5) -> pd.DataFrame:
    """Lit uniquement les métadonnées et un premier lot de lignes d'un fichier Parquet
    distant sans charger l'intégralité de la table en mémoire vive (RAM)."""
    with fsspec.open(url, "rb") as f:
        pf = pq.ParquetFile(f)
        print(f"\n[{nom_source}] {pf.metadata.num_rows:,} lignes | {len(pf.schema_arrow.names)} colonnes")
        print("Colonnes :", pf.schema_arrow.names)
        return next(pf.iter_batches(batch_size=n_lignes)).to_pandas()


def nettoyer_decp(
    df: pd.DataFrame,
    annee_min: int | None = ANNEE_MIN,
    montant_min: float = MONTANT_MIN_EUROS,
    montant_max: float = MONTANT_MAX_EUROS,
) -> pd.DataFrame:
    """Nettoie et normalise la table des marchés publics (DECP) :
    1. Sélectionne les variables utiles et convertit les types numériques/dates.
    2. Filtre sur la période d'étude (écarte les anciens contrats republiés tardivement).
    3. Éclate les groupements d'entreprises (1 ligne = 1 couple marché x titulaire)
       et divise le montant par le nombre de co-titulaires pour éviter le double comptage.
    4. Normalise les identifiants SIRET (14 chiffres) et extrait le SIREN (9 chiffres).
    5. Dédoublonne sur le couple (identifiant marché, SIRET).
    6. Écarte les montants aberrants et calcule un montant annualisé.
    """
    df = df.copy()

    # 1. Sélection des variables d'intérêt
    id_col = "uid" if "uid" in df.columns else "id"
    cols_cibles = [
        id_col, "id", "nature", "procedure", "codeCPV",
        "montant", "dureeMois", "dateNotification",
        "acheteur_id", "acheteur_nom",
        "lieuExecution_code", "lieuExecution_typeCode",
        "titulaires",
    ]
    cols_keep = list(dict.fromkeys([c for c in cols_cibles if c in df.columns]))
    df = df[cols_keep]

    # 2. Conversion des types et filtrage temporel
    df["montant"] = pd.to_numeric(df["montant"], errors="coerce")
    df["dureeMois"] = pd.to_numeric(df["dureeMois"], errors="coerce")
    df["dateNotification"] = pd.to_datetime(df["dateNotification"], errors="coerce")

    if annee_min is not None:
        avant_date = len(df)
        df = df[df["dateNotification"].dt.year >= annee_min].copy()
        print(f"Contrats antérieurs à {annee_min} écartés : {avant_date - len(df):,}")

    # 3. Gestion des titulaires et des groupements d'entreprises
    has_tit = df["titulaires"].apply(
        lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0
    )
    df = df[has_tit].copy()

    # Répartition équitable du montant entre les co-titulaires d'un groupement
    df["nb_titulaires"] = df["titulaires"].apply(len)
    df["montant_part"] = df["montant"] / df["nb_titulaires"]

    # Passage de la maille 'contrat' à la maille 'contrat x titulaire'
    df = df.explode("titulaires").reset_index(drop=True)

    # Aplatissement des dictionnaires titulaires (gère l'encapsulation {'titulaire': {...}})
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
    cols_tit = [
        c for c in ["titulaire_id", "titulaire_type_id", "titulaire_nom"]
        if c in df_tit.columns
    ]
    df = pd.concat([df.drop(columns=["titulaires"]), df_tit[cols_tit]], axis=1)

    # 4. Filtrage sur les SIRET français valides (14 chiffres) et extraction du SIREN (9 chiffres)
    df["titulaire_id"] = df["titulaire_id"].astype(str).str.strip()
    is_siret = (
        (df["titulaire_type_id"] == "SIRET")
        & df["titulaire_id"].str.match(r"^\d{14}$", na=False)
    )
    print(f"Part de lignes avec SIRET valide (14 chiffres) : {is_siret.mean():.1%}")

    df = df[is_siret].copy()
    df["siret"] = df["titulaire_id"]
    df["siren"] = df["siret"].str[:9]

    # 5. Suppression des doublons de publication pour un même couple (marché, SIRET)
    avant_dedup = len(df)
    df = df.drop_duplicates(subset=[id_col, "siret"]).copy()
    print(f"Doublons (marché, SIRET) supprimés : {avant_dedup - len(df):,}")

    # 6. Filtrage des montants extrêmes et calcul du montant annualisé
    avant_filt = len(df)
    df = df[
        (df["montant_part"] > montant_min) & (df["montant_part"] < montant_max)
    ].copy()
    print(
        f"Lignes écartées pour montant aberrant "
        f"(<= {montant_min} € ou >= {montant_max:,.0f} €) : {avant_filt - len(df):,}"
    )

    duree_valide = df["dureeMois"].where(df["dureeMois"] > 0, np.nan)
    df["montant_annualise"] = df["montant_part"] / duree_valide * 12

    # Extraction de la division CPV (2 premiers chiffres = grande famille d'achat)
    if "codeCPV" in df.columns:
        df["cpv_division"] = df["codeCPV"].astype(str).str.slice(0, 2)

    return df


def associer_section_naf(code_naf: pd.Series) -> pd.Series:
    """Regroupe les 732 sous-classes NAF Rev. 2 (ex: '41.20A') en 21 sections
    officielles (A à U) à partir des deux premiers chiffres (division NAF)."""
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


def charger_et_preparer_sirene(
    url_sirene_etab: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les colonnes utiles de la base Sirene Établissements depuis un fichier Parquet,
    normalise les clés SIRET, crée les variables sectorielles (NAF) et géographiques,
    calcule le référentiel économique national des établissements actifs et libère la RAM.
    Renvoie:
        - df_sirene : Table Sirene nettoyée et dédoublonnée au niveau SIRET (14 chiffres).
        - ref_secteurs_sirene : Référentiel national des établissements/entreprises actifs par section NAF.
    """
    # Lecture uniquement des 7 colonnes utiles pour limiter l'empreinte mémoire
    cols_siret_cibles = [
        "siret",
        "etablissementSiege",
        "activitePrincipaleEtablissement",
        "nomenclatureActivitePrincipaleEtablissement",
        "trancheEffectifsEtablissement",
        "codeCommuneEtablissement",
        "etatAdministratifEtablissement",
    ]

    with fsspec.open(url_sirene_etab, "rb") as f:
        pf = pq.ParquetFile(f)
        cols_dispo = [c for c in cols_siret_cibles if c in pf.schema_arrow.names]
        df_sirene = pf.read(columns=cols_dispo).to_pandas()

    # Normalisation de la clé SIRET (14 chiffres) et garantie d'unicité avant jointure
    df_sirene["siret"] = df_sirene["siret"].astype(str).str.strip().str.zfill(14)
    df_sirene = df_sirene.drop_duplicates(subset=["siret"]).copy()

    # Création des variables sectorielles (code APE, division à 2 chiffres, section A-U)
    df_sirene["naf_code"] = df_sirene["activitePrincipaleEtablissement"].astype(str).str.strip()
    df_sirene["naf_division"] = df_sirene["naf_code"].str.slice(0, 2)
    df_sirene["naf_section"] = associer_section_naf(df_sirene["naf_code"])

    # Extraction du département de l'établissement (3 chiffres pour DROM 97x/98x, 2 en métropole)
    if "codeCommuneEtablissement" in df_sirene.columns:
        commune = df_sirene["codeCommuneEtablissement"].astype(str).str.strip()
        df_sirene["dep_titulaire"] = np.where(
            commune.str.startswith(("97", "98")),
            commune.str.slice(0, 3),
            commune.str.slice(0, 2),
        )

    # Référentiel économique national : poids démographique de chaque secteur actif en France
    sirene_actifs = df_sirene[df_sirene["etatAdministratifEtablissement"] == "A"].copy()
    sirene_actifs["siren_tmp"] = sirene_actifs["siret"].str[:9]

    ref_secteurs_sirene = (
        sirene_actifs.groupby("naf_section", as_index=False)
        .agg(
            nb_etablissements_actifs=("siret", "count"),
            nb_entreprises_actives=("siren_tmp", "nunique"),
        )
    )
    ref_secteurs_sirene["part_etablissements_sirene"] = (
        ref_secteurs_sirene["nb_etablissements_actifs"]
        / ref_secteurs_sirene["nb_etablissements_actifs"].sum()
    )

    del sirene_actifs
    gc.collect()
    print(f"Sirene nettoyé : {len(df_sirene):,} établissements uniques prêts pour la jointure")

    return df_sirene, ref_secteurs_sirene


def joindre_decp_et_sirene(
    df_decp: pd.DataFrame,
    df_sirene: pd.DataFrame,
    ref_secteurs_sirene: pd.DataFrame,
    chemin_export_final: str | None = "decp_sirene_clean.parquet",
    chemin_export_ref: str | None = "ref_secteurs_sirene.parquet",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Réalise la jointure à gauche (many-to-one) entre les marchés publics DECP et
    la base Sirene Établissements, affiche le diagnostic de couverture, sauvegarde
    les tables nettoyées au format Parquet et calcule le tableau de synthèse sectorielle.
    Renvoie:
    - df_final : Table fusionnée (marché x titulaire enrichi Sirene).
    - synthese : Tableau comparant le poids financier DECP et le poids économique Sirene par section NAF.
    """
    # `how="left"` conserve tous les marchés publics ;
    # `validate="many_to_one"` garantit mathématiquement l'absence de duplication de montants.
    df_final = pd.merge(
        df_decp,
        df_sirene,
        on="siret",
        how="left",
        validate="many_to_one",
    )

    # Libération immédiate des références locales avant le passage du ramasse-miettes
    del df_decp, df_sirene
    gc.collect()

    # Diagnostic de couverture de la jointure
    taux_lignes = df_final["naf_code"].notna().mean()
    montant_tot = df_final["montant_part"].sum()
    montant_ok = df_final.loc[df_final["naf_code"].notna(), "montant_part"].sum()

    print(f"Table finale (marché x titulaire)      : {len(df_final):,} lignes, {df_final.shape[1]} colonnes")
    print(f"Taux d'appariement Sirene (en lignes)  : {taux_lignes:.1%}")
    print(f"Taux d'appariement Sirene (en montant) : {montant_ok / montant_tot:.1%}")

    # Sauvegarde au format Parquet pour accélérer les sessions suivantes sans recharger Sirene
    if chemin_export_final:
        df_final.to_parquet(chemin_export_final, index=False)
    if chemin_export_ref:
        ref_secteurs_sirene.to_parquet(chemin_export_ref, index=False)

    # Tableau de synthèse : poids financier dans les DECP vs poids économique dans Sirene
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
    # Indice > 1 : le secteur capte davantage de dépense publique que son poids économique
    synthese["indice_surrepresentation"] = (
        synthese["part_montant_decp"] / synthese["part_etablissements_sirene"]
    )

    return df_final, synthese


def construire_base_entreprises_inpi(
    df_final: pd.DataFrame,
    url_inpi: str = "https://minio.lab.sspcloud.fr/glemesre/diffusion/ratios_inpi_bce(1).parquet",
) -> pd.DataFrame:
    """
    Agrège la table des marchés publics au niveau entreprise (SIREN à 9 chiffres),
    charge les ratios financiers INPI/BCE filtrés sur ces entreprises, conserve
    le bilan le plus récent (en priorisant les bilans complets/consolidés) et
    réalise une jointure 1-pour-1.
    """
    # ==============================================================================
    # ÉTAPE 1 : Agréger df_final à 1 ligne par SIREN (niveau entreprise à 9 chiffres)
    # ==============================================================================
    df_work = df_final[df_final["siret"].notna()].copy()

    # Extraire le SIREN à 9 chiffres en tronquant le numéro SIRET
    df_work["siren"] = df_work["siret"].astype(str).str.strip().str.zfill(14).str[:9]

    # Fonction auxiliaire pour récupérer la modalité la plus fréquente (mode) par SIREN
    def first_mode(series: pd.Series):
        m = series.dropna().mode()
        return m.iloc[0] if not m.empty else None

    # Construction dynamique du dictionnaire d'agrégation
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

    # Ajoute le nom de l'entreprise uniquement si une colonne de nom existe dans df_final
    for col_nom in ["titulaire_nom", "denominationUniteLegale", "nom"]:
        if col_nom in df_work.columns:
            agg_dict["entreprise_nom"] = (col_nom, "first")
            break

    df_siren_decp = df_work.groupby("siren", as_index=False).agg(**agg_dict)

    # ==============================================================================
    # ÉTAPE 2 : Charger (avec filtre) et dédoublonner les ratios INPI/BCE par SIREN
    # ==============================================================================
    # Récupérer la liste des SIREN uniques présents dans la base DECP
    sirens_decp = df_siren_decp["siren"].dropna().unique().tolist()

    # Charger uniquement les lignes correspondant à vos entreprises DECP
    df_inpi = pd.read_parquet(
        url_inpi,
        filters=[("siren", "in", sirens_decp)],
    )

    # Normaliser le format de la colonne 'siren' sur 9 caractères
    df_inpi["siren"] = df_inpi["siren"].astype(str).str.strip().str.zfill(9)

    # Convertir 'date_cloture_exercice' au format datetime pour trier chronologiquement
    df_inpi["date_cloture_exercice"] = pd.to_datetime(
        df_inpi["date_cloture_exercice"], errors="coerce"
    )

    # Trier pour garder par SIREN :
    # 1. L'exercice le plus récent ('date_cloture_exercice' décroissant)
    # 2. En cas de doublon sur la même date, privilégier les bilans complets ('C') ou consolidés ('K') avant simplifiés ('S')
    df_inpi_latest = (
        df_inpi.sort_values(
            by=["siren", "date_cloture_exercice", "type_bilan"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["siren"], keep="first")
    )

    # ==============================================================================
    # ÉTAPE 3 : Jointure à gauche (1-pour-1) et diagnostic du taux d'appariement
    # ==============================================================================
    df_entreprise = pd.merge(
        df_siren_decp,
        df_inpi_latest,
        on="siren",
        how="left",
        validate="one_to_one",  # Garantit l'absence de multiplication accidentelle des lignes
    )

    # Affichage de contrôle
    n_total = len(df_entreprise)
    n_inpi_present = df_entreprise["date_cloture_exercice"].notna().sum()
    n_ca_disponible = df_entreprise["chiffre_d_affaires"].notna().sum()

    print(f"Entreprises uniques (SIREN) dans DECP : {n_total:,}")
    print(f"Entreprises présentes dans la base INPI : {n_inpi_present:,} ({n_inpi_present / n_total:.1%})")
    print(f"Entreprises avec chiffre d'affaires non confidentiel : {n_ca_disponible:,} ({n_ca_disponible / n_total:.1%})")

    return df_entreprise


def diagnostiquer_appariement_inpi(
    df_entreprise: pd.DataFrame,
    url_inpi: str = "https://minio.lab.sspcloud.fr/glemesre/diffusion/ratios_inpi_bce(1).parquet",
) -> pd.DataFrame:
    """
    Affiche un diagnostic complet de la qualité de l'appariement entre df_entreprise
    et la base INPI/BCE :
    1. Taux de valeurs manquantes (NaN) et types par colonne dans df_entreprise.
    2. Vérification du format brut de la colonne 'siren' dans le fichier Parquet INPI.
    3. Répartition des statuts de confidentialité des comptes parmi les entreprises appariées.
    """
    # 1. Tableau récapitulatif des valeurs manquantes (NaN) par colonne dans df_entreprise
    nan_summary = pd.DataFrame({
        "nb_valeurs_presentes": df_entreprise.notna().sum(),
        "nb_NaN": df_entreprise.isna().sum(),
        "taux_NaN_pct": (df_entreprise.isna().mean() * 100).round(1),
        "type_colonne": df_entreprise.dtypes,
    })
    print("=== DIAGNOSTIC DES VALEURS MANQUANTES (df_entreprise) ===")
    print(nan_summary.to_string())

    # 2. Vérifier le type et le format brut de 'siren' dans le fichier Parquet INPI
    sample_inpi = pd.read_parquet(url_inpi, columns=["siren", "confidentiality"]).head(10)
    print("\n=== APERÇU DU FORMAT BRUT DANS URL_INPI ===")
    print("Type de la colonne siren :", sample_inpi["siren"].dtype)
    print(sample_inpi)

    # 3. Répartition de la confidentialité parmi les entreprises appariées
    if "confidentiality" in df_entreprise.columns:
        print("\n=== STATUT DE CONFIDENTIALITÉ DES COMPTES (ENTREPRISES APPARIÉES) ===")
        print(df_entreprise["confidentiality"].value_counts(dropna=False))

    return nan_summary


