# Règles d'extraction du test externe « intra-source » (Pandit 2025)

Écrites AVANT toute mesure des modèles (étape 11b-7). Elles sont reprises dans le docstring de
`ml/src/brain_mri/external_slices.py`. Ce jeu sert uniquement à TESTER, jamais à entraîner.

## Jeu

Pandit AS et coll., « Mapping Pituitary Neuroendocrine Tumors: An Annotated MRI Dataset
Profiling Tumor and Carotid Characteristics », Scientific Data, 2025,
doi 10.1038/s41597-024-04218-8. Données : figshare, doi 10.6084/m9.figshare.27894084.v1,
licence CC BY 4.0. 136 patients, IRM 3D T1 avec contraste (0,5 mm isotrope), T2 recalé sur la T1
(100 patients), masque du complexe tumeur-glande hypophysaire. Pas de FLAIR.

## Choix des patients

30 patients tirés au sort (graine 42) parmi ceux qui ont T1, T2 ET masque (99 candidats).

**Exclusion** : un patient est écarté si son masque n'est pas sur la même grille (dimensions,
affine) que sa T1, car les indices de coupe ne correspondraient plus. Observé le 2026-11-11, avant
toute mesure de modèle, pour le patient 16 de ce tirage. **29 patients retenus sur 30 tirés.**

## Coupes

1. **Plan** : coupes axiales (axe z après réorientation en RAS).
2. **Coupe tumeur** : la coupe axiale où le masque a la plus grande surface (en cas d'égalité,
   la plus basse). Étiquette `tumor`, distance 0 mm.
3. **Coupes saines** : coupes axiales du MÊME patient situées **au-dessus** de la tumeur et dont la
   distance à la coupe masquée la plus proche est d'au moins **20 mm** (distance = écart d'indices
   x espacement réel des voxels selon z, lu dans l'en-tête). Étiquette `healthy`.
   *Modification du 2026-11-11, avant toute mesure de modèle* : la première version autorisait
   aussi les coupes en dessous ; la planche de contrôle (patient 25) a montré qu'elles tombent sur
   le cou et la base du crâne, pas sur du cerveau. Seules les coupes sus-tentorielles restent.
4. **Assez de cerveau** : une coupe est « riche » si sa surface de tête (voxels de la T1 au-dessus
   de 15 % du percentile 99,5 du volume) vaut au moins **60 %** de la surface maximale sur toutes
   les coupes. Cela écarte le cou et le sommet du crâne.
5. **Nombre** : 3 coupes saines par patient, aux quantiles 1/6, 3/6 et 5/6 de la liste des coupes
   admissibles (ordre croissant des indices) ; 2 s'il n'y en a que 2 ; patient écarté s'il y en a
   moins de 2.
6. **Séquences** : mêmes indices pour T1 et T2, seulement si la grille (dimensions, affine) du T2
   est identique à celle de la T1 (vérifié par patient). Sinon : T1 seule pour ce patient.
7. **Orientation** : radiologique, comme les images Kaggle : face vers le haut, côté droit du
   patient à gauche de l'image.
8. **Intensité** : fenêtre entre les percentiles 0,5 et 99,5 des voxels non nuls du volume
   (une fenêtre par volume et par séquence, commune à toutes ses coupes), valeurs écrasées dans
   cet intervalle puis converties en 8 bits. L'image est complétée de noir en carré (pas
   d'étirement).
9. **Sortie** : PNG 8 bits en niveaux de gris dans `data/external/pandit/slices/` (ignoré par git) ;
   manifeste versionné `ml/splits/external_pandit.csv` (fichier, patient, séquence, étiquette,
   indice de coupe, distance à la tumeur en mm, sha256). Aucune image dans git.

## Critères de la mesure (fixés avant)

- **E1** : part des coupes tumeur prédites « notumor » <= 10 %.
- **E2** : spécificité sur les coupes saines (prédit « notumor ») >= 80 %.
- Intervalles de confiance à 95 % : bootstrap (graine 42).

## Limites connues

Le masque couvre le complexe tumeur-glande, pas la tumeur seule. Une coupe « saine » à 20 mm peut
quand même contenir de l'hypophyse ou une tumeur étendue verticalement : le seuil de 20 mm limite
ce risque sans l'annuler. Une seule pathologie (adénome hypophysaire) : on ne peut pas juger la
classification en 4 classes.

**Niveau anatomique confondu avec l'étiquette** : les coupes tumeur sont toutes à la selle turcique
(base du crâne, orbites visibles) et les coupes saines toutes plus haut (24,5 mm et plus au-dessus).
E2 mesure donc en partie « coupe haute du cerveau → notumor », pas seulement « pas de tumeur ».
Un modèle peut réussir E2 sans savoir reconnaître une tumeur ; E1 et E2 doivent se lire ensemble.

**Portée** : un seul hôpital, 29 patients (intervalles de confiance larges) ; coupes axiales à la
selle, rares dans les images Kaggle d'hypophyse (surtout sagittales et coronales). Le résultat dit
« pas fiable hors du jeu d'origine », il ne mesure pas une exactitude générale sur d'autres hôpitaux.
