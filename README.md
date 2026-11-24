# 2A-Python-pour-la-data-science
# Analyse de la dépense publique par secteur d'activité et dépendance financière des entreprises (2024–2026)

> **Projet Python pour la Data Science — ENSAE Paris (2026)**  
> **Auteurs :** BENNANI Kamil, LEMESRE Grégoire

---

## 1. Problématique et objectifs

La commande publique représente un levier économique majeur en France, au cœur des débats sur l'efficacité de la dépense publique, l'accès des PME aux appels d'offres et le poids de l'argent public dans le modèle économique des entreprises privées. 

Ce projet répond à la double problématique suivante :
> *La commande publique française bénéficie-t-elle de manière disproportionnée à certains secteurs économiques et tailles d'entreprises, et quel degré de dépendance financière crée-t-elle pour les sociétés titulaires ?*

Pour y répondre, nous construisons un pipeline de données reproductible croisant **trois référentiels publics massifs** :
1. Les **Données Essentielles de la Commande Publique (DECP)** sur les millésimes 2024 à 2026.
2. Le répertoire **Sirene (`StockEtablissement`, Insee)** pour caractériser le secteur (`NAF Rev. 2`), la taille salariale et la localisation des titulaires, ainsi que la structure démographique du tissu productif français.
3. Les **ratios financiers et comptes annuels (INPI / BCE)** afin de rapporter les montants publics captés au chiffre d'affaires réel des entreprises.

---

## 2. Architecture du dépôt

Le rapport final du projet est disponible dans le notebook **`rapport_depense_publique.ipynb`**.

Dans une logique de modularité et de séparation entre le code d'ingénierie, la narration scientifique et les données volumineuses, le projet est structuré de la manière suivante :

```text
├── rapport_depense_publique.ipynb  # Rapport final : analyse narrative, tableaux synthétiques et visualisations
├── lire_fichiers.py                # Module de collecte et consolidation des fichiers DECP (2024–2026)
├── src/                            # Package métier : nettoyage, appariements, agrégations et graphiques
│   ├── __init__.py                 # Initialisation du package src
│   ├── nettoyage.py                # Diagnostic qualité, nettoyage et jointures DECP × Sirene × INPI
│   ├── sirene_stats_desc.py        # Analyses sectorielles, géographiques et structurelles (DECP × Sirene)
│   └── inpi_stats_desc.py          # Calculs de dépendance financière et analyses des bilans (INPI / BCE)
├── requirements.txt                # Dépendances Python nécessaires à l'exécution
└── README.md                       # Documentation du projet
```

*(Note : Les fichiers volumineux au format Parquet ne sont pas versionnés sur GitHub ; ils sont lus à la volée depuis le stockage objet MinIO du SSP Cloud via `fsspec` et `pyarrow`.)*