# Données

Le dépôt ne versionne ni les images Flickr8k, ni les poids des modèles.

## Préparation

```bash
pip install -e ".[data]"
python data/download_flickr8k.py
```

Le script utilise `kagglehub`, télécharge `adityajn105/flickr8k`, détecte les images et la table de légendes, puis produit :

- `data/processed/flickr8k_all.csv` ;
- `data/processed/flickr8k_test.csv` (1 000 images, seed 42) ;
- `data/processed/dataset_metadata.json` avec provenance et empreinte du manifeste.

Chaque ligne contient `image_id`, `image_path` et `references`, encodé comme une liste JSON. La licence et les conditions d’utilisation restent celles de Flickr8k et de la copie Kaggle ; elles doivent être vérifiées avant redistribution.

