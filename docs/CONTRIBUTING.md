# Contribution

Créer une issue par expérience ou fonctionnalité. Les branches suivent `feat/...`, `fix/...` ou `exp/...`. Un changement est accepté quand `ruff check .`, `pytest` et `python -m compileall src` passent, que les paramètres d’expérience sont versionnés et qu’aucune donnée lourde n’est ajoutée.

Messages recommandés : `feat(api): add cancellable benchmark jobs`, `exp(blip): run int4 on 500 Flickr8k images`, `docs(report): analyze quantization trade-off`.

