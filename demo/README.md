# Démonstration (10 minutes)

## Préparation

Lancer le serveur sur un GPU, précharger BLIP avec une image et créer un quick run terminé. Garder une image avec personnes et une image avec plusieurs objets pour montrer les limites.

## Scénario

1. **0:00–1:00 — problème.** Produire des légendes coûte cher ; la quantification économise de la mémoire mais peut dégrader le sens.
2. **1:00–2:00 — architecture.** Montrer le schéma README, le worker mono-GPU et SQLite.
3. **2:00–4:00 — image unique.** Charger une image, comparer BLIP/GIT et lire latence/VRAM.
4. **4:00–6:00 — expérience.** Créer une matrice 2 × 3 sur 10 images ; expliquer les paramètres figés.
5. **6:00–8:00 — résultats.** Ouvrir un run de 500 images, comparer BLEU/ROUGE/latence/VRAM et montrer deux erreurs qualitatives.
6. **8:00–9:00 — robustesse.** Montrer l’API, l’export CSV et les tests.
7. **9:00–10:00 — limites.** Biais, hallucinations, revue humaine et extension CLIPScore/human evaluation.

Enregistrer la courte vidéo avec ce même scénario et placer uniquement son lien dans `demo/video.md` ; ne pas commiter un fichier vidéo lourd.

