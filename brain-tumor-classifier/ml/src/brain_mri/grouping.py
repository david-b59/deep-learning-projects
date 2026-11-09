"""Groupes « même patient » et découpage honnête train / val / test (étape 11b-1).

Problème : le jeu Kaggle ne donne aucun identifiant de patient, et un même patient apparaît
sur plusieurs coupes voisines. Si ces coupes se retrouvent des deux côtés du découpage, le
test mesure de la mémorisation, pas de la généralisation (77,6 % des images de test ont une
jumelle à l'entraînement).

Méthode, en quatre temps :
    1. :func:`match_patient_ids` donne un vrai identifiant de patient aux images Kaggle qui
       sont des copies d'images figshare (jeu de Cheng, qui fournit les identifiants) ;
    2. :func:`label_conflicts` repère les images quasi identiques mais de classes différentes
       (étiquette douteuse) ;
    3. :func:`build_groups` regroupe (union-find) les images liées par un cosinus DINOv2 élevé
       de même classe, ou par un même identifiant de patient ;
    4. :func:`group_split` découpe en 7 plis en gardant chaque groupe d'un seul côté ;
       :func:`residual_leak` mesure ce qui fuit malgré tout.

Le module ne contient que de la logique pure (tableaux numpy) : les scripts fournissent les
fichiers et les embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from brain_mri.leakage import BIN_LABELS, assign_bins
from brain_mri.near_duplicates import CHUNK, similar_pairs

LINK_COSINE = 0.98  # cosinus à partir duquel deux images sont « la même image »
N_SPLITS = 7  # plis : 5 pour l'entraînement, 1 pour la validation, 1 pour le test
SEED = 42  # graine du découpage
PARTS = ("train", "val", "test")  # noms des trois parties


def match_patient_ids(
    kaggle_emb: np.ndarray,
    kaggle_labels: np.ndarray,
    figshare_emb: np.ndarray,
    figshare_labels: np.ndarray,
    figshare_patients: Sequence[str],
    threshold: float = LINK_COSINE,
) -> tuple[list[str], np.ndarray]:
    """Donne à chaque image Kaggle l'identifiant de patient de sa copie figshare, si elle existe.

    Une image Kaggle est rattachée à l'image figshare la plus proche (cosinus) de MÊME classe
    si ce cosinus atteint ``threshold``.

    Args:
        kaggle_emb: Embeddings Kaggle ``(N, dim)``, de norme 1.
        kaggle_labels: Classe (indice) de chaque image Kaggle.
        figshare_emb: Embeddings figshare ``(M, dim)``, de norme 1.
        figshare_labels: Classe (indice) de chaque image figshare (même codage que Kaggle).
        figshare_patients: Identifiant de patient de chaque image figshare.
        threshold: Cosinus minimal du rattachement.

    Returns:
        ``(patients, cosinus)`` : l'identifiant de patient de chaque image Kaggle (chaîne vide
        si aucune copie figshare) et le meilleur cosinus de même classe (``-inf`` si la classe
        n'existe pas dans figshare).
    """
    patients = [""] * len(kaggle_emb)
    best = np.full(len(kaggle_emb), -np.inf, dtype=np.float32)
    for start in range(0, len(kaggle_emb), CHUNK):
        block = kaggle_emb[start : start + CHUNK]
        sims = block @ figshare_emb.T  # (bloc, M)
        other = kaggle_labels[start : start + len(block), None] != figshare_labels[None, :]
        sims[other] = -np.inf  # seules les images de la même classe comptent
        nearest = sims.argmax(axis=1)
        top = sims.max(axis=1)
        best[start : start + len(block)] = top
        for offset, (index, value) in enumerate(zip(nearest.tolist(), top.tolist(), strict=True)):
            if value >= threshold:
                patients[start + offset] = str(figshare_patients[index])
    return patients, best


def label_conflicts(
    embeddings: np.ndarray, labels: np.ndarray, threshold: float = LINK_COSINE
) -> list[tuple[int, int, float]]:
    """Liste les paires quasi identiques (cosinus >= ``threshold``) de classes différentes.

    Args:
        embeddings: Embeddings de norme 1, ``(N, dim)``.
        labels: Classe (indice) de chaque image.
        threshold: Cosinus minimal de la paire.

    Returns:
        Liste de ``(i, j, cosinus)`` avec ``i < j`` et ``labels[i] != labels[j]``.
    """
    return [(i, j, c) for i, j, c in similar_pairs(embeddings, threshold) if labels[i] != labels[j]]


class UnionFind:
    """Structure « union-find » : regroupe des éléments liés deux à deux."""

    def __init__(self, size: int) -> None:
        """Crée ``size`` éléments, chacun dans son propre groupe."""
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        """Renvoie le représentant du groupe de ``item`` (avec compression de chemin)."""
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # on raccourcit le chemin pour les appels suivants
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: int, b: int) -> None:
        """Fusionne les groupes de ``a`` et de ``b``."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)  # le plus petit indice gagne


def build_groups(
    embeddings: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    patient_ids: Sequence[str] | None = None,
    sha256: Sequence[str] | None = None,
) -> np.ndarray:
    """Regroupe les images supposées du même patient (union-find).

    Deux images sont dans le même groupe si, de proche en proche : elles ont un cosinus
    ``>= threshold`` ET la même classe, OU le même identifiant de patient (non vide), OU le
    même SHA-256 (copie exacte).

    Args:
        embeddings: Embeddings de norme 1, ``(N, dim)``.
        labels: Classe (indice) de chaque image.
        threshold: Cosinus minimal d'un lien « même classe ».
        patient_ids: Identifiant de patient de chaque image (chaîne vide si inconnu).
        sha256: Empreinte SHA-256 de chaque image (optionnelle).

    Returns:
        Un entier de groupe par image, numérotés de 0 à ``k - 1`` dans l'ordre de première
        apparition.
    """
    n = len(embeddings)
    union = UnionFind(n)
    for i, j, _ in similar_pairs(embeddings, threshold):
        if labels[i] == labels[j]:
            union.union(i, j)
    for keys in (patient_ids, sha256):
        if keys is None:
            continue
        first: dict[str, int] = {}
        for index, key in enumerate(keys):
            if not key:
                continue  # identifiant inconnu : aucun lien
            if key in first:
                union.union(first[key], index)
            else:
                first[key] = index
    numbering: dict[int, int] = {}
    return np.array([numbering.setdefault(union.find(i), len(numbering)) for i in range(n)])


def group_split(
    labels: np.ndarray,
    groups: np.ndarray,
    extra_strata: np.ndarray | None = None,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> np.ndarray:
    """Découpe en train / val / test sans jamais séparer un groupe.

    ``StratifiedGroupKFold`` fabrique ``n_splits`` plis de groupes entiers, avec des strates
    (classe, puis éventuellement une variable en plus, ex. « image 512 x 512 ou non »)
    réparties au mieux. Le pli 0 sert de test, le pli 1 de validation, les autres
    (5 sur 7) d'entraînement.

    Args:
        labels: Classe (indice) de chaque image.
        groups: Groupe de chaque image (voir :func:`build_groups`).
        extra_strata: Variable entière de stratification en plus de la classe (optionnelle).
        n_splits: Nombre de plis.
        seed: Graine du mélange.

    Returns:
        Tableau de chaînes ``"train"`` / ``"val"`` / ``"test"``, une par image.
    """
    strata = labels.astype(np.int64)
    if extra_strata is not None:
        strata = strata * (int(extra_strata.max()) + 1) + extra_strata.astype(np.int64)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    parts = np.full(len(labels), "train", dtype=object)
    for fold, (_, held_out) in enumerate(splitter.split(np.zeros(len(labels)), strata, groups)):
        if fold == 0:
            parts[held_out] = "test"
        elif fold == 1:
            parts[held_out] = "val"
    return parts.astype(str)


def max_cosine_to(queries: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Cosinus maximal de chaque requête avec l'ensemble de référence (toutes classes).

    Args:
        queries: Embeddings ``(Q, dim)``, de norme 1.
        references: Embeddings ``(R, dim)``, de norme 1.

    Returns:
        Tableau ``(Q,)`` ; ``-inf`` partout si ``references`` est vide.
    """
    result = np.full(len(queries), -np.inf, dtype=np.float32)
    if len(references) == 0:
        return result
    for start in range(0, len(queries), CHUNK):
        result[start : start + CHUNK] = (queries[start : start + CHUNK] @ references.T).max(axis=1)
    return result


def residual_leak(
    test_emb: np.ndarray, train_emb: np.ndarray, threshold: float = LINK_COSINE
) -> dict:
    """Mesure la fuite restante : images de test ayant un voisin proche dans le train.

    Args:
        test_emb: Embeddings des images de test ``(T, dim)``.
        train_emb: Embeddings des images d'entraînement ``(R, dim)``.
        threshold: Cosinus au-dessus duquel un voisin compte comme « jumeau ».

    Returns:
        ``{"n_test", "n_above_threshold", "rate_above_threshold", "by_cosine_bin"}`` ;
        ``by_cosine_bin`` compte les images de test par tranche de leur cosinus maximal avec
        le train (tranches de :data:`brain_mri.leakage.BIN_LABELS`).
    """
    best = max_cosine_to(test_emb, train_emb)
    bins = assign_bins(best)
    above = int((best >= threshold).sum())
    return {
        "n_test": len(test_emb),
        "threshold": threshold,
        "n_above_threshold": above,
        "rate_above_threshold": above / len(test_emb) if len(test_emb) else None,
        "by_cosine_bin": {
            label: int((bins == index).sum()) for index, label in enumerate(BIN_LABELS)
        },
    }
