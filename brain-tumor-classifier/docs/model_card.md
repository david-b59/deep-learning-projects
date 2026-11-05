---
language: fr
license: mit
library_name: pytorch
pipeline_tag: image-classification
datasets:
  - masoudnickparvar/brain-tumor-mri-dataset
metrics:
  - accuracy
tags:
  - medical
  - mri
  - brain-tumor
  - grad-cam
  - transfer-learning
---

# Brain Tumor Classifier : classification d'IRM cérébrales (ResNet-50)

> ⚠️ **Démonstration technique, pas un dispositif médical.** Ce modèle ne doit jamais servir à
> un diagnostic ni à une décision de soin. Il n'a fait l'objet d'aucune validation clinique.

## En bref
Ce modèle regarde une IRM du cerveau (une coupe) et indique laquelle de ces **4 catégories** elle
ressemble le plus : `glioma`, `meningioma`, `pituitary` (tumeur de l'hypophyse) ou `notumor`
(pas de tumeur). Il fournit aussi une carte **Grad-CAM** qui montre les zones de l'image ayant
influencé sa réponse.

Projet réalisé dans le cadre de mon Master Machine Learning Engineering, puis prolongé à titre
personnel. Code complet, tests et application web : dépôt GitHub `david-b59/deep-learning-projects`
(dossier `brain-tumor-classifier`).

## Utilisation

```python
from PIL import Image
from brain_mri.predict import Predictor

predictor = Predictor.from_hub("david-b59/brain-tumor-classifier")
resultat = predictor.predict(Image.open("irm.jpg"))
print(resultat.predicted_class, resultat.probabilities)
```

Le dossier contient `model.safetensors` (poids), `model.onnx` (le même modèle au format ONNX, pour
le navigateur), `config.json` (noms des classes, prétraitement, scores) et cette fiche. Entrée :
image redimensionnée en 224 x 224, convertie en RGB, normalisée avec la moyenne et l'écart-type
d'ImageNet.

### Format ONNX (navigateur)
`model.onnx` (94 Mo, décimaux 32 bits) prend l'image prétraitée `[1, 3, 224, 224]` et renvoie deux
sorties : `probs` `[1, 4]` (probabilités) et `cams` `[1, 4, 7, 7]` (cartes d'activation de chaque
classe, à normaliser puis agrandir). Comme la dernière couche est linéaire après un pooling moyen,
la carte Grad-CAM se calcule sans gradient. Vérifié contre PyTorch : écart maximal de probabilité
de l'ordre de 1e-6. Une version quantifiée (int8, 24 Mo) a été testée puis écartée : elle perdait
1,07 point d'accuracy sur le jeu de test (94,38 % → 93,31 %).

## Données
- **Source** : *Brain Tumor MRI Dataset* de Masoud Nickparvar (Kaggle), licence **CC BY 4.0**.
- 7 200 images, 4 classes, dossiers `Training` et `Testing`.
- **187 copies identiques** ont été retirées (détection par empreinte SHA-256) pour éviter qu'une
  même image se retrouve à la fois en entraînement et en validation.
- Jeux utilisés : entraînement **4 614**, validation **815** (découpage stratifié de `Training`),
  test **1 584** (dossier `Testing`, utilisé **une seule fois**, à la fin).

## Entraînement
- ResNet-50 pré-entraîné sur ImageNet (`IMAGENET1K_V1`), nouvelle dernière couche à 4 sorties.
- **Phase 1** : seule la nouvelle couche est entraînée (3 epochs, lr 1e-3).
- **Phase 2** : dégel du dernier bloc `layer4` (5 epochs, lr 1e-4). Les couches de normalisation
  gelées restent en mode évaluation.
- Augmentation légère (retournement, rotation ±10°, luminosité et contraste).
- Optimiseur Adam, réduction du lr sur plateau, arrêt anticipé possible. Meilleur modèle choisi
  sur la **loss de validation** (epoch 7 sur 8).
- Entraîné sur CPU (environ 2 h 44).

## Résultats
| Jeu | Accuracy |
|---|---|
| Validation (sert à choisir le modèle) | 97,7 % |
| **Test** (1 584 images, jamais vues) | **94,4 %** |

Par classe, sur le test :

| Classe | Précision | Rappel | F1 | Images |
|---|---|---|---|---|
| glioma | 0,985 | **0,837** | 0,905 | 386 |
| meningioma | 0,917 | 0,942 | 0,929 | 398 |
| notumor | 0,901 | 1,000 | 0,948 | 400 |
| pituitary | 0,985 | 0,993 | 0,989 | 400 |

## Limites (à lire)
- **Les gliomes sont les moins bien reconnus** : 16 % ne sont pas retrouvés. 33 sont pris pour des
  méningiomes et **29 pour « pas de tumeur »**. Au total, 44 tumeurs (gliomes et méningiomes) sont
  classées « notumor » : c'est l'erreur la plus grave dans un contexte réel.
- **Le modèle ne regarde pas toujours la tumeur.** Les cartes Grad-CAM montrent que, dans plusieurs
  erreurs, la zone « chaude » est ailleurs (crâne, orbite), et qu'une bonne réponse peut s'appuyer
  sur le fond de l'image. La carte est grossière (7 x 7 agrandie) : c'est une aide de lecture, pas
  une preuve.
- **Confiance trompeuse** : certaines erreurs ont une probabilité de 100 %.
- **Écart validation / test** (97,7 % contre 94,4 %) : la validation a servi à choisir le modèle,
  elle est donc un peu optimiste. Le chiffre à retenir est celui du test.
- **Doublons** : seules les copies *identiques* ont été détectées. Des quasi-doublons (retournés,
  recadrés) peuvent subsister entre entraînement et test et rendre le score un peu plus optimiste.
- **Une seule source de données**, des images 2D déjà extraites, sans informations sur les
  patients, les machines ni les protocoles. Le modèle n'a pas été testé sur d'autres hôpitaux.
- Aucune validation clinique, aucun avis médical.

## Citation et licences
- Données : Masoud Nickparvar, *Brain Tumor MRI Dataset*, Kaggle, CC BY 4.0.
- Poids de départ : ResNet-50 de torchvision, pré-entraîné sur ImageNet.
- Code et poids de ce projet : licence MIT.
