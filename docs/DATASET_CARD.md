# Fiche de données — Flickr8k

Flickr8k contient environ 8 000 photographies, chacune accompagnée de cinq légendes humaines en anglais. Le projet ne réentraîne pas les VLMs sur ces images : il utilise un sous-ensemble d’évaluation de 500 à 1 000 images.

## Provenance

Le script télécharge la copie Kaggle `dibyansudiptiman/flickr-8k`. Il enregistre l’URL, la date, les nombres détectés, la seed et le SHA-256 du manifeste dans `data/processed/dataset_metadata.json`.

## Schéma normalisé

| Champ | Type | Description |
|---|---|---|
| `image_id` | texte | nom de fichier stable |
| `image_path` | texte | chemin local absolu |
| `references` | liste JSON | une à cinq légendes humaines |

## Découpage

Les noms sont triés, mélangés avec `random.Random(42)`, puis les 1 000 premiers forment le manifest de test. Le benchmark complet en lit les 500 premiers. Cette procédure est déterministe pour une même version des fichiers.

## Risques

Le contenu peut inclure des personnes et refléter les biais culturels des contributeurs. Les légendes peuvent contenir des hypothèses sur les personnes. Ne pas utiliser pour identifier des individus ni pour déduire des caractéristiques sensibles. Ne pas redistribuer les images sans vérifier leurs conditions d’utilisation.

