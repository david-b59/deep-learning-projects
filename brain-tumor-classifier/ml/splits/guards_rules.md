# Règles des garde-fous (étape 11c)

Écrites AVANT toute mesure (étape 11c-0). Modèle porteur : A1 (`models/a1/best.pt`,
`configs/train_a1.yaml`, prétraitement `head_crop`). Les garde-fous servent à refuser ou signaler
les images pour lesquelles le modèle n'est pas fait (IRM T1 avec contraste du cerveau).

## Ordre du verdict

1. `color` : image en couleur -> **refusée**.
2. `t2_like` : l'image ressemble à une IRM T2 -> **signalée**, classe masquée, bouton
   « Voir quand même la réponse du modèle (non fiable) ».
3. `out_of_distribution` ou `low_confidence` -> **« incertain »**, aucune classe affichée.
4. Sinon -> **accepté** (classe et confiance affichées).

## Définition de chaque garde-fou

- **Couleur** : calculée sur les pixels natifs, avant recadrage. On ne compte que les pixels dont
  le canal max est > 16. Statistique = part de ces pixels avec (max - min entre canaux) > 24.
  Seuil = 2 x le maximum observé sur Kaggle train + val, plancher 1 %.
- **Température** : un scalaire, ajusté par LBFGS sur la NLL de la validation A1.
- **Mahalanobis** : ACP à 256 composantes sur les caractéristiques 2048 du train A1 ; covariance
  commune estimée par LedoitWolf ; centres de classe calculés sur le train ; score = distance
  minimale aux 4 centres ; tau_D = 97e percentile de la validation.
- **Confiance** : tau_P = la plus grande valeur telle que la part de la validation refusée par
  distance OU confiance soit <= 5 %.
- **Détecteur T2** : régression logistique L2 sur les caractéristiques A1 ; C choisi par
  validation croisée groupée par patient (5 plis) sur Pandit développement (exactitude
  équilibrée) ; tau_T2 = plus petit seuil donnant <= 2 % de fausses alertes sur les images figshare
  de la validation Kaggle (T1c certain) ET <= 5 % sur les coupes T1c de Pandit développement
  (prédictions hors pli).

## Critères (fixés avant mesure)

- **G0** : <= 5 % de refus (distance ou confiance) sur la validation.
- **G1** : 0 image Kaggle (train + val + test, 7 176) refusée pour la couleur ET >= 99 % des
  photos et fonds d'oeil refusés.
- **G2** : >= 90 % du jeu hors sujet refusé (toutes raisons) et aucune catégorie < 75 %.
- **G3** : >= 80 % des 116 coupes T2 du test Pandit refusées ou signalées ET <= 10 % des coupes T1c
  du test Pandit signalées T2.
- **G4** : test Kaggle groupé : couverture >= 90 % et exactitude sur les acceptées >= 90,3 %,
  <= 2 % des images figshare du test signalées T2.
- **G5** : ECE du test après température <= avant (les deux sont publiés).
- **G6** : parité : ONNX vs PyTorch <= 1e-4 (probabilités, `t2_prob` ; écart relatif sur la
  distance) ; navigateur vs Python : même boîte de recadrage à +-1 px sur >= 99 % des images,
  même verdict sur >= 99 %, écart d'exactitude <= 0,5 pt ; ONNX <= 100 Mo.

## Conséquences

- G1, G2 ou G3 ratés : on garde le garde-fou et on publie le chiffre, sans réglage sur le test.
- G0 ou G4 ratés : arrêt, on demande à David.

## Données interdites à l'ajustement

Une assertion dans le futur `fit_guards.py` vérifie qu'aucune de ces données n'est utilisée pour
ajuster un seuil, un modèle ou un paramètre :

- la partie `test` de `ml/splits/kaggle_grouped.csv` ;
- les 30 patients Pandit tirés à l'étape 11b-7 (dont le patient 16, écarté faute de grille
  commune) ; liste : `ml/splits/external_pandit.csv` + règle de tirage de `external_rules.md`
  (graine 42, parmi ceux qui ont T1, T2 et masque) ;
- le jeu hors sujet (`ml/splits/offtopic.csv`).

## Publiés dans tous les cas

- couverture et exactitude sur les images acceptées pour Pandit T1c ;
- chiffres par séquence ;
- FLAIR : « non mesuré, aucune donnée libre ».
