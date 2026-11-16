# Prism : impact de la quantification sur deux VLMs légers de légendage

**Projet 7 — IA générative visuelle**  
Équipe : _à compléter_ · Année universitaire : 2026–2027

> Ce document est la source du rapport final de 6 à 8 pages. Les emplacements `[RÉSULTAT]` doivent être remplacés après l’expérience complète ; aucun chiffre n’est prérempli sans mesure.

## Résumé

Ce projet étudie le compromis entre qualité linguistique et coût d’inférence de petits modèles vision-langage. Nous comparons BLIP Base et GIT Base sur un sous-ensemble déterministe de 500 images Flickr8k, en pleine précision, 8 bits et 4 bits. La plateforme Prism exécute les six configurations selon un protocole identique, conserve chaque prédiction et mesure BLEU-4, ROUGE-L, latence et mémoire. L’objectif n’est pas de désigner un modèle universellement meilleur, mais d’identifier la configuration la plus efficace sous une contrainte de 12 Go de VRAM. `[RÉSULTAT PRINCIPAL EN UNE PHRASE]`.

## 1. Problématique

Les modèles vision-langage associent un encodeur visuel et un décodeur textuel afin de produire une description d’image. Ils peuvent automatiser l’annotation d’un corpus destiné à l’entraînement d’une LoRA texte-image, mais leur déploiement reste contraint par la mémoire GPU. La quantification réduit la précision numérique des poids et rend les modèles plus accessibles ; elle peut aussi modifier la sortie discrète du décodeur, parfois à partir de faibles variations de logits.

Notre question de recherche est : **dans quelle mesure le passage de la pleine précision à 8 puis 4 bits réduit-il le coût de BLIP Base et GIT Base sans dégrader de manière importante leurs légendes sur Flickr8k ?**

Hypothèses : (H1) 8 bits réduit nettement la VRAM avec une perte faible de BLEU/ROUGE ; (H2) 4 bits maximise le gain mémoire mais produit davantage d’erreurs qualitatives ; (H3) le modèle dominant en qualité n’est pas nécessairement celui qui offre le meilleur compromis qualité-latence.

## 2. État de l’art et modèles

BLIP combine un encodeur visuel et un décodeur de texte préentraînés avec des objectifs d’alignement image-texte. La variante Base utilisée compte environ 247 millions de paramètres. GIT formule plusieurs tâches vision-langage comme une génération autorégressive conditionnée par des tokens visuels ; la variante Base compte environ 177 millions de paramètres.

Nous utilisons les poids publics `Salesforce/blip-image-captioning-base` et `microsoft/git-base` sans fine-tuning. Ce choix isole l’impact de la quantification et respecte la consigne de ne pas affiner complètement un VLM. La quantification bitsandbytes est appliquée au chargement : INT8 pour le mode 8 bits, NF4 avec calcul float16 pour le mode 4 bits. Le mode « pleine précision » utilise float16 sur GPU.

## 3. Données et protocole

Flickr8k contient environ 8 000 images avec cinq légendes humaines en anglais par image. Le script de préparation regroupe ces cinq références, trie les identifiants, applique une permutation déterministe avec la seed 42, puis écrit un manifest de test dont l’empreinte SHA-256 est conservée. Nous sélectionnons les 500 premières images du manifest pour le benchmark final.

Les six configurations partagent exactement les mêmes images et paramètres : beam search de taille 3, maximum 40 nouveaux tokens, température 1, aucune adaptation. L’exécution a lieu sur `[GPU]`, avec `[VRAM]` Go, versions `[CUDA / PyTorch / Transformers]`. Un seul worker GPU est utilisé. Le modèle peut rester en cache entre deux images ; la latence rapportée exclut donc le téléchargement et le chargement initial, mais la mémoire comprend les poids chargés.

### 3.1 Métriques

BLEU-4 compare les n-grammes de la légende aux cinq références ; ROUGE-L mesure leur plus longue sous-séquence commune. Ces métriques favorisent le recouvrement lexical et sous-estiment certaines paraphrases correctes. Nous ajoutons la latence moyenne et p95, la VRAM de pic et la longueur moyenne des sorties. L’analyse qualitative classe au moins 30 erreurs : objet, action, comptage, relation spatiale, attribut de personne et texte visible.

### 3.2 Reproductibilité

Les configurations JSON, le hash du manifest, chaque prédiction et la version de l’API sont conservés. Les gros fichiers et poids restent hors Git. Le backend valide les images, rend les erreurs structurées et persiste les runs dans SQLite. Les tests de contrat remplacent les poids par un moteur déterministe : la CI reste rapide et ne dépend pas de Hugging Face.

## 4. Résultats

| Modèle | Précision | BLEU-4 ↑ | ROUGE-L ↑ | Latence p95 (ms) ↓ | VRAM pic (Mo) ↓ |
|---|---|---:|---:|---:|---:|
| BLIP Base | FP16 | [RÉSULTAT] | [RÉSULTAT] | [RÉSULTAT] | [RÉSULTAT] |
| BLIP Base | INT8 | [RÉSULTAT] | [RÉSULTAT] | [RÉSULTAT] | [RÉSULTAT] |
| BLIP Base | INT4 | [RÉSULTAT] | [RÉSULTAT] | [RÉSULTAT] | [RÉSULTAT] |
| GIT Base | FP16 | [RÉSULTAT] | [RÉSULTAT] | [RÉSULTAT] | [RÉSULTAT] |
| GIT Base | INT8 | [RÉSULTAT] | [RÉSULTAT] | [RÉSULTAT] | [RÉSULTAT] |
| GIT Base | INT4 | [RÉSULTAT] | [RÉSULTAT] | [RÉSULTAT] | [RÉSULTAT] |

**Figure 1.** Qualité en fonction de la VRAM — insérer `results/quality_vs_vram.png`.  
**Figure 2.** Distribution des latences — insérer `results/latency_boxplot.png`.  
**Figure 3.** Six exemples comparés — insérer `results/qualitative_grid.png`.

Rapporter les variations relatives par rapport à FP16, ainsi que les intervalles bootstrap si le temps le permet. Ne pas conclure uniquement à partir de différences inférieures à la variabilité observée.

## 5. Discussion

`[INTERPRÉTER H1–H3 À PARTIR DES RÉSULTATS]`. Une faible variation moyenne peut masquer des changements importants sur le comptage ou les personnes. Pour chaque configuration, discuter un succès représentatif et un échec. Vérifier si les gains mémoire se traduisent réellement par une meilleure déployabilité sous 12 Go, et si la quantification modifie la longueur des sorties.

La comparaison a quatre limites. Premièrement, les métriques lexicales ne mesurent pas directement la fidélité visuelle. Deuxièmement, Flickr8k est petit, ancien et anglophone. Troisièmement, un seul GPU ne permet pas de généraliser les latences à tout matériel. Enfin, les modèles ont des tailles et préentraînements différents ; l’étude compare des systèmes réels, pas uniquement deux architectures contrôlées.

## 6. Éthique et usages

Une légende automatique peut halluciner un objet ou attribuer à tort un âge, un genre, une origine ou une intention. Pour l’annotation d’un corpus LoRA, Prism doit servir de pré-annotation : un humain valide ou corrige chaque phrase. Les sorties concernant des personnes sont examinées en priorité. Les images et poids gardent leurs licences d’origine et ne sont pas redistribués dans le dépôt.

## 7. Conclusion

Prism transforme une comparaison de modèles en expérience reproductible : données figées, matrice de précision, mesures qualité/coût, prédictions auditables et interface de démonstration. `[CONFIGURATION RECOMMANDÉE]` offre le meilleur compromis observé sous la contrainte de 12 Go parce que `[JUSTIFICATION CHIFFRÉE]`. Une extension naturelle consisterait à ajouter CLIPScore, une évaluation humaine en aveugle et un troisième VLM récent, tout en conservant le même contrat expérimental.

## Références

1. Li et al., *BLIP: Bootstrapping Language-Image Pre-training*, ICML 2022.
2. Wang et al., *GIT: A Generative Image-to-text Transformer for Vision and Language*, 2022.
3. Hodosh et al., *Framing Image Description as a Ranking Task*, JAIR 2013.
4. Papineni et al., *BLEU: a Method for Automatic Evaluation of Machine Translation*, ACL 2002.
5. Lin, *ROUGE: A Package for Automatic Evaluation of Summaries*, 2004.

