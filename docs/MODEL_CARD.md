# Carte du système modèle

## Usage prévu

Comparer des VLMs légers pour produire des légendes descriptives en anglais et préparer un corpus qui sera ensuite vérifié par un humain. Ce système est un outil d’étude, pas un service de décision.

## Modèles

| ID interne | Modèle | Taille approximative | Licence |
|---|---|---:|---|
| `blip-base` | `Salesforce/blip-image-captioning-base` | 247 M | BSD-3-Clause |
| `git-base` | `microsoft/git-base` | 177 M | MIT |

Le nombre de paramètres est indicatif. Les poids sont chargés depuis Hugging Face au premier usage et ne sont pas redistribués par ce dépôt.

## Modes

- `full` : float16 sur GPU, float32 sur CPU ;
- `int8` : quantification bitsandbytes 8 bits, CUDA requis ;
- `int4` : NF4, calcul float16, CUDA requis.

## Évaluation

Le protocole principal utilise 500 images Flickr8k figées par seed, cinq références par image, beam search 3 et 40 nouveaux tokens maximum. La qualité est évaluée avec BLEU-4 et ROUGE-L ; le coût avec latence, RAM et VRAM.

## Limites

Les métriques lexicales ne reconnaissent pas toutes les paraphrases valides. Flickr8k est ancien, centré sur des scènes photographiques et des légendes anglaises. Les modèles peuvent halluciner des objets, compter incorrectement ou attribuer un genre, un âge ou une activité sans preuve suffisante.

## Usages exclus

Accessibilité sans contrôle qualité, surveillance, diagnostic médical, inférence d’attributs sensibles, notation de personnes ou production automatique de vérité terrain.

