# Prism — benchmark de VLMs légers

Prism est une plateforme reproductible qui compare deux modèles vision-langage de légendage d’images — **BLIP Base** et **GIT Base** — en pleine précision, 8 bits et 4 bits. Le projet mesure simultanément la qualité des légendes, la latence, la RAM et la VRAM, puis conserve chaque expérience dans SQLite pour la rendre vérifiable et exportable.

> Projet 7 · IA générative visuelle · Option A : produire des légendes exploitables pour documenter ou préparer un jeu LoRA.

## Ce qui en fait un projet d’ingénierie

- backend FastAPI versionné avec validation des entrées et erreurs normalisées ;
- moteur d’inférence à chargement paresseux et cache borné pour ne pas saturer le GPU ;
- jobs de benchmark asynchrones, persistants, annulables et récupérés proprement après redémarrage ;
- matrice contrôlée `2 modèles × 3 précisions × N images` ;
- métriques BLEU-4, ROUGE-L, latence moyenne/p95, RAM et VRAM de pic ;
- frontend responsive same-origin, export CSV et inspection des prédictions ;
- tests unitaires, tests de contrat API, CI, Docker et configuration Colab/Kaggle ;
- documentation données, modèle, éthique et rapport académique.

## Architecture

```text
Image / manifest
      │
      ▼
FastAPI ── validation ──► ExperimentRunner ──► CaptionEngine
   │                            │                    │
   │                            ▼                    ├─ BLIP Base
   ├─ dashboard             SQLite WAL              └─ GIT Base
   ├─ OpenAPI                   │                         │
   └─ export CSV ◄──────────────┴──── métriques ◄─────────┘
```

Le worker est volontairement limité à un thread par défaut : plusieurs chargements concurrents de modèles quantifiés rendraient les mesures VRAM non comparables et risqueraient un OOM sur un GPU 12 Go.

## Installation locale

Prérequis : Python 3.10–3.13. Pour les modes 8/4 bits, utiliser de préférence Python 3.11 sous Linux avec un GPU NVIDIA CUDA (Colab ou Kaggle recommandé).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[ml,dev,data]"
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/macOS
python data/download_flickr8k.py
vlm-lab --reload
```

Ouvrir <http://127.0.0.1:8000>. La documentation OpenAPI est disponible sur <http://127.0.0.1:8000/docs>.

Sur CPU, l’inférence `full` fonctionne en float32. Les modes `int8` et `int4` retournent volontairement une erreur explicite : ils nécessitent CUDA et bitsandbytes.

## Exécuter le protocole

1. Préparer Flickr8k avec `python data/download_flickr8k.py`.
2. Lancer le serveur puis ouvrir la section **Construire la matrice**.
3. Commencer par 10 images et pleine précision (`configs/benchmark_quick.json`).
4. Sur un GPU 12 Go, lancer 500 images avec les six combinaisons (`configs/benchmark_full.json`).
5. Garder constants le GPU, le manifest, `num_beams=3` et `max_new_tokens=40`.
6. Exporter le CSV depuis le détail de l’expérience et produire les figures du rapport.

Exemple API :

```bash
curl -X POST http://127.0.0.1:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  --data @configs/benchmark_quick.json
```

## Endpoints

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/api/v1/health` | état du service |
| `GET` | `/api/v1/models` | registre des modèles et précisions |
| `POST` | `/api/v1/caption` | légende une image validée |
| `POST` | `/api/v1/experiments` | crée un benchmark asynchrone |
| `GET` | `/api/v1/experiments` | historique persistant |
| `GET` | `/api/v1/experiments/{id}/metrics` | agrégats qualité/coût |
| `GET` | `/api/v1/experiments/{id}/predictions` | sorties auditables |
| `POST` | `/api/v1/experiments/{id}/cancel` | annulation coopérative |
| `GET` | `/api/v1/experiments/{id}/export.csv` | export brut |

Les erreurs ont toujours la forme `{"error": "…", "code": "…", "details": null}`. Les traces serveur et réponses des fournisseurs ne sont jamais exposées au client.

## Mesures et interprétation

- **BLEU-4** mesure le recouvrement de n-grammes avec les cinq références ; il pénalise fortement les paraphrases valides.
- **ROUGE-L** mesure la plus longue sous-séquence commune ; il est un peu plus souple sur la formulation.
- **Latence p95** représente un cas lent réaliste, après chargement du modèle.
- **VRAM de pic** est mesurée par PyTorch pendant la génération.
- **RAM processus** est relevée après chaque inférence.

Le chargement initial n’est pas inclus dans la latence de génération, mais la mémoire du modèle l’est. Comparer les configurations sur le même matériel ; ne pas comparer directement un run CPU et un run GPU.

## Qualité, sécurité et limites

- uploads limités à 10 Mo et 25 Mpx, décodés et vérifiés par Pillow ;
- seuls JPEG, PNG et WebP sont acceptés ;
- les manifests doivent rester sous `data/processed` ;
- aucun secret ni poids n’est commité ;
- cache limité à un modèle pour respecter le budget VRAM ;
- SQLite convient à une instance de laboratoire, pas à plusieurs serveurs distribués ;
- les légendes peuvent contenir des erreurs, biais ou attributs sensibles inventés : elles ne doivent pas servir de vérité terrain sans revue humaine.

Voir [docs/ETHICS.md](docs/ETHICS.md), [docs/MODEL_CARD.md](docs/MODEL_CARD.md) et [docs/DATASET_CARD.md](docs/DATASET_CARD.md).

## Tests et livraison

```bash
ruff check .
pytest --cov=vlm_lab --cov-report=term-missing
python -m compileall src
docker build -t prism-vlm-lab .
```

Les tests n’installent ni ne téléchargent les poids : le moteur d’inférence est remplacé par un faux déterministe au niveau du contrat.

## Structure du dépôt

```text
configs/     paramètres versionnés
data/        téléchargement et fiche des données
demo/        guide et script de démonstration
docs/        cartes, éthique et décisions techniques
notebooks/   parcours Colab/Kaggle numéroté
report/      rapport final en Markdown
results/     figures et exports légers
src/         API, worker, inférence, métriques et frontend
tests/       tests unitaires et de contrat
```

## Références

- [BLIP Base — carte officielle Hugging Face](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [GIT Base — carte officielle Hugging Face](https://huggingface.co/microsoft/git-base)
- [Flickr8k — copie Kaggle utilisée par le script](https://www.kaggle.com/datasets/dibyansudiptiman/flickr-8k)

## Licence

Le code est sous licence MIT. Les modèles et données conservent leurs propres licences ; consultez les cartes correspondantes avant toute redistribution.

