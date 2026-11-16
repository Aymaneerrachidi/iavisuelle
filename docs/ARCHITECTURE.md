# Architecture et décisions

## Frontières

Le navigateur ne charge jamais de poids et ne reçoit aucun chemin arbitraire. Il parle uniquement à `/api/v1`. L’API valide les fichiers et configurations, puis délègue le calcul lourd au `ExperimentRunner`. Le runner est la seule couche qui orchestre les modèles et écrit les prédictions.

## Choix structurants

### Deux adaptateurs, une sortie commune

BLIP et GIT utilisent des classes Transformers différentes, mais `CaptionEngine.caption()` retourne toujours `CaptionResult`. Le reste du système ignore ainsi l’architecture précise.

### Quantification au chargement

`BitsAndBytesConfig` charge les poids en 8 ou 4 bits. Le mode 4 bits utilise NF4 et un calcul float16. Aucune conversion permanente des poids n’est écrite sur disque.

### Cache borné

Un `OrderedDict` conserve au plus un modèle/configuration par défaut. À l’éviction, les références sont libérées, le garbage collector est exécuté, puis le cache CUDA est vidé. C’est plus lent lors d’un changement de configuration, mais compatible avec 12 Go de VRAM.

### Persistance

SQLite en mode WAL stocke expériences et prédictions. Une contrainte unique rend l’écriture idempotente pour `(expérience, image, modèle, précision)`. Après un arrêt brutal, les jobs restés `queued/running` passent à `failed` au redémarrage plutôt que d’afficher un faux état actif.

### Exécution

Le worker en mémoire est adapté à un déploiement mono-instance de laboratoire. Pour une production multi-utilisateur, remplacer `ThreadPoolExecutor` par Celery/RQ et SQLite par PostgreSQL sans changer le contrat HTTP.

## États

```text
queued ──► running ──► completed
   │           │
   └───────────┴────► cancelled
               └────► failed
```

L’annulation est coopérative entre deux images. Une génération déjà lancée n’est pas interrompue au milieu d’un kernel CUDA.

## Menaces traitées

- bombe de décompression et taille excessive ;
- fichier non image ou format non autorisé ;
- manifest hors du répertoire de données ;
- fuite de stack trace ;
- OOM favorisé par plusieurs workers GPU ;
- doublons après réessai ;
- dérive de protocole grâce aux configs et empreintes du manifest.

