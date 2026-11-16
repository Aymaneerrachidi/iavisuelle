# Réflexion éthique

Le système réduit le coût de production de légendes, mais une phrase fluide n’est pas nécessairement vraie. Le risque principal est l’automatisation de l’hallucination : objets inexistants, nombres faux et attributs personnels supposés.

Mesures intégrées : conservation de la sortie brute et de sa configuration, comparaison à plusieurs références, inspection visuelle dans le dashboard, absence de score de « vérité », et documentation des usages exclus. Pour créer un dataset LoRA, chaque légende doit passer par une revue humaine. Les exemples comportant des personnes doivent être vérifiés en priorité et les attributs sensibles supprimés lorsqu’ils ne sont pas explicitement nécessaires.

La quantification peut dégrader différemment certains concepts. Le rapport doit donc inclure une analyse qualitative par catégories (personnes, animaux, nombre d’objets, actions, texte dans l’image) et pas seulement une moyenne globale.

