"""Audit du biais de source : les métadonnées des images trahissent-elles la classe ?

Dans le jeu Kaggle, les images d'une même classe viennent de sources différentes, donc
de tailles et de modes (RGB ou niveaux de gris) différents. Un modèle peut alors
« tricher » en reconnaissant la taille plutôt que la tumeur. Ce module contient la
logique testable de l'audit :

    - lecture de la taille et du mode de chaque image (sans décoder les pixels) ;
    - tableau par classe (part de 512x512, nombre de tailles distinctes, part RGB) ;
    - exactitude d'un modèle par classe et par groupe (512x512 ou non) ;
    - « classifieur idiot » qui ne voit que largeur, hauteur et mode.

Le script ``scripts/audit_source_bias.py`` assemble ces briques et écrit le rapport.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.tree import DecisionTreeClassifier

from brain_mri.duplicates import IMAGE_SUFFIXES

REFERENCE_SIZE = (512, 512)  # taille majoritaire du jeu, utilisée comme repère


@dataclass(frozen=True)
class ImageInfo:
    """Métadonnées d'une image : où elle est, sa classe, sa taille et son mode."""

    path: Path  # chemin du fichier
    label: str  # nom de la classe (nom du sous-dossier)
    width: int  # largeur en pixels
    height: int  # hauteur en pixels
    mode: str  # mode Pillow : "RGB", "L"...

    @property
    def is_reference_size(self) -> bool:
        """Vrai si l'image fait exactement 512x512."""
        return (self.width, self.height) == REFERENCE_SIZE


def read_metadata(root: Path) -> list[ImageInfo]:
    """Lit taille et mode de toutes les images d'un dossier ``root/<classe>/<image>``.

    Pillow n'ouvre que l'en-tête du fichier : c'est rapide et les pixels ne sont pas lus.

    Args:
        root: Dossier ``Training`` ou ``Testing``.

    Returns:
        Une :class:`ImageInfo` par image, triée par chemin.
    """
    infos = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        with Image.open(path) as image:
            width, height = image.size  # Pillow donne (largeur, hauteur)
            infos.append(ImageInfo(path, path.parent.name, width, height, image.mode))
    return infos


def size_table(infos: list[ImageInfo]) -> dict[str, dict]:
    """Résume, par classe, la répartition des tailles et des modes.

    Args:
        infos: Métadonnées des images.

    Returns:
        ``{classe: {n, share_512, n_distinct_sizes, share_rgb}}`` (parts entre 0 et 1).
    """
    groups: dict[str, list[ImageInfo]] = defaultdict(list)
    for info in infos:
        groups[info.label].append(info)
    table = {}
    for label in sorted(groups):
        items = groups[label]
        n = len(items)
        table[label] = {
            "n": n,
            "share_512": sum(i.is_reference_size for i in items) / n,
            "n_distinct_sizes": len({(i.width, i.height) for i in items}),
            "share_rgb": sum(i.mode == "RGB" for i in items) / n,
        }
    return table


def accuracy_by_group(
    infos: list[ImageInfo], y_true: list[str], y_pred: list[str]
) -> dict[str, dict]:
    """Calcule l'exactitude par classe, séparément pour les images 512x512 et les autres.

    Args:
        infos: Métadonnées des images, dans le même ordre que les prédictions.
        y_true: Vraie classe de chaque image.
        y_pred: Classe prédite de chaque image.

    Returns:
        ``{classe: {"512x512": {n, correct, accuracy}, "autre": {...}}}`` ; l'exactitude
        vaut ``None`` si le groupe est vide.
    """
    counts: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"512x512": [0, 0], "autre": [0, 0]}
    )
    for info, truth, pred in zip(infos, y_true, y_pred, strict=True):
        group = "512x512" if info.is_reference_size else "autre"
        counts[truth][group][0] += 1  # nombre d'images
        counts[truth][group][1] += int(truth == pred)  # bonnes réponses
    return {
        label: {
            group: {"n": n, "correct": ok, "accuracy": ok / n if n else None}
            for group, (n, ok) in groups.items()
        }
        for label, groups in sorted(counts.items())
    }


def metadata_features(infos: list[ImageInfo]) -> np.ndarray:
    """Transforme les métadonnées en tableau numérique : largeur, hauteur, mode RGB (0/1).

    Args:
        infos: Métadonnées des images.

    Returns:
        Tableau de forme ``(n_images, 3)``.
    """
    return np.array([[i.width, i.height, int(i.mode == "RGB")] for i in infos], dtype=float)


def metadata_only_classifier(train: list[ImageInfo], test: list[ImageInfo]) -> dict:
    """Entraîne un arbre de décision qui ne voit QUE largeur, hauteur et mode.

    Si cet « idiot » devine la classe bien mieux que le hasard, c'est que les
    métadonnées contiennent un raccourci que le vrai modèle peut exploiter.

    Args:
        train: Images d'entraînement.
        test: Images de test.

    Returns:
        ``{accuracy, per_class, chance_level}`` : exactitude globale, par classe, et
        exactitude obtenue en prédisant toujours la classe majoritaire du test.
    """
    tree = DecisionTreeClassifier(max_depth=5, random_state=0)  # petit arbre : pas de par cœur
    tree.fit(metadata_features(train), [i.label for i in train])
    y_true = [i.label for i in test]
    y_pred = list(tree.predict(metadata_features(test)))

    per_class = {}
    for label in sorted(set(y_true)):
        pairs = [(t, p) for t, p in zip(y_true, y_pred, strict=True) if t == label]
        per_class[label] = {
            "n": len(pairs),
            "accuracy": sum(t == p for t, p in pairs) / len(pairs),
        }
    majority = max(y_true.count(label) for label in set(y_true))
    return {
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true),
        "per_class": per_class,
        "chance_level": majority / len(y_true),
    }
