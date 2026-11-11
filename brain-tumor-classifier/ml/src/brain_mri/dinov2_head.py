"""Modèle B : DINOv2-small gelé + une couche linéaire (étape 11b-4).

Le corps du modèle est DINOv2-small (Hugging Face, révision épinglée), entièrement gelé : on
ne l'entraîne jamais. Seule une couche ``nn.Linear(768, n_classes)`` apprend. Pour une image :

    1. DINOv2 produit une suite de jetons ``(1 + 256, 384)`` : le jeton CLS (résumé global) puis
       un jeton par patch de 14 x 14 pixels (image 224 x 224 = 16 x 16 patchs) ;
    2. la caractéristique de l'image est la concaténation du jeton CLS (384 valeurs) et de la
       moyenne des jetons de patch (384 valeurs) : 768 valeurs ;
    3. la couche linéaire transforme ces 768 valeurs en un score par classe.

L'entrée du modèle est le même tenseur ``(B, 3, 224, 224)`` normalisé ImageNet que celui du
ResNet : ``predict``, ``evaluate`` et les mesures d'invariance l'utilisent donc tel quel.

Comme le corps est gelé, ses caractéristiques ne changent jamais pour une même image : on les
calcule une seule fois (:func:`get_features`) et on les range dans un cache sur disque, puis on
entraîne la couche linéaire sur ces caractéristiques en mémoire (:func:`train_head`), en
quelques secondes. Le cache est indexé par une clé qui contient la révision de DINOv2, le
prétraitement, le nombre de vues, la graine et la liste des images : on ne recalcule jamais ce
qui existe déjà, et on ne relit jamais un cache qui ne correspond pas à la demande.

Format du checkpoint : le ``state_dict`` ne contient QUE la tête (clés ``head.weight`` et
``head.bias``, 3 076 valeurs). Le corps est rechargé depuis Hugging Face à la révision épinglée
par :class:`Dinov2Classifier` : le fichier reste minuscule et le corps ne peut pas dériver.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from brain_mri.config import DataConfig, Dinov2HeadConfig
from brain_mri.data import ManifestDataset
from brain_mri.embeddings import MODEL_REVISION, load_model
from brain_mri.head_crop import CROP_VERSION
from brain_mri.transforms import build_transforms

LOGGER = logging.getLogger(__name__)

ARCHITECTURE_NAME = "dinov2_s"  # valeur de ``architecture`` dans la configuration et le checkpoint


def pool_tokens(hidden: torch.Tensor) -> torch.Tensor:
    """Résume la suite de jetons DINOv2 en un vecteur par image : CLS + moyenne des patchs.

    Args:
        hidden: Jetons ``(B, 1 + n_patchs, d)`` ; le jeton 0 est le jeton CLS.

    Returns:
        Tenseur ``(B, 2 * d)`` : jeton CLS puis moyenne des jetons de patch.
    """
    cls_token = hidden[:, 0]  # (B, d) : résumé global de l'image
    patch_mean = hidden[:, 1:].mean(dim=1)  # (B, d) : moyenne des jetons de patch
    return torch.cat([cls_token, patch_mean], dim=1)


@torch.no_grad()
def backbone_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Passe un lot d'images dans le corps gelé et renvoie les caractéristiques poolées.

    Args:
        backbone: Corps DINOv2 (``Dinov2Model`` ou équivalent), en mode évaluation.
        images: Lot ``(B, 3, 224, 224)`` normalisé ImageNet.

    Returns:
        Caractéristiques ``(B, 2 * hidden_size)``, sans gradient.
    """
    hidden = backbone(pixel_values=images).last_hidden_state
    return pool_tokens(hidden)


def head_state_dict(head: nn.Linear) -> dict[str, torch.Tensor]:
    """État de la tête avec le préfixe ``head.`` (format du checkpoint du modèle B).

    Args:
        head: La couche linéaire.

    Returns:
        Dictionnaire ``{"head.weight": ..., "head.bias": ...}`` (copies détachées).
    """
    return {f"head.{key}": value.detach().clone() for key, value in head.state_dict().items()}


class Dinov2Classifier(nn.Module):
    """DINOv2-small gelé suivi d'une couche linéaire (modèle B).

    Attributes:
        backbone: Le corps DINOv2, gelé et toujours en mode évaluation.
        head: ``nn.Linear(2 * hidden_size, n_classes)`` : la seule partie entraînable.
    """

    def __init__(self, n_classes: int, backbone: nn.Module | None = None) -> None:
        """Construit le modèle.

        Args:
            n_classes: Nombre de classes en sortie.
            backbone: Corps à utiliser. Par défaut, DINOv2-small est chargé depuis Hugging Face
                à la révision épinglée ; on ne passe un autre corps que dans les tests.
        """
        super().__init__()
        self.backbone = backbone if backbone is not None else load_model()
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False  # corps gelé : jamais mis à jour
        feature_dim = 2 * self.backbone.config.hidden_size  # CLS + moyenne des patchs
        self.head = nn.Linear(feature_dim, n_classes)

    def train(self, mode: bool = True) -> Dinov2Classifier:
        """Change le mode du modèle, mais garde le corps en ``eval()`` (gelé, sans dropout).

        Args:
            mode: True pour l'entraînement, False pour l'évaluation.

        Returns:
            Le modèle lui-même.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Calcule les caractéristiques d'un lot d'images (corps gelé, sans gradient).

        Args:
            images: Lot ``(B, 3, 224, 224)`` normalisé ImageNet.

        Returns:
            Caractéristiques ``(B, 768)`` : CLS (384) puis moyenne des patchs (384).
        """
        return backbone_features(self.backbone, images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Calcule un score par classe pour chaque image.

        Args:
            images: Lot ``(B, 3, 224, 224)`` normalisé ImageNet.

        Returns:
            Scores ``(B, n_classes)``.
        """
        return self.head(self.features(images))

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Ne renvoie que la tête : le corps est rechargé depuis la révision épinglée."""
        return head_state_dict(self.head)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True, **kwargs):
        """Recharge la tête depuis un checkpoint (clés ``head.*``).

        Args:
            state_dict: Dictionnaire produit par :meth:`state_dict`.
            strict: Si True, refuse les clés manquantes ou inattendues.
            **kwargs: Ignorés (compatibilité avec ``nn.Module``).

        Returns:
            Le résultat de ``load_state_dict`` de la couche linéaire.

        Raises:
            RuntimeError: Si les clés ne correspondent pas (mode strict).
        """
        unexpected = [k for k in state_dict if not k.startswith("head.")]
        if strict and unexpected:
            raise RuntimeError(f"Clés inattendues dans le checkpoint DINOv2 : {unexpected}")
        head_only = {
            k.removeprefix("head."): v for k, v in state_dict.items() if k not in unexpected
        }
        return self.head.load_state_dict(head_only, strict=strict)


# ---------------------------------------------------------------------------------------------
# Extraction des caractéristiques avec cache disque
# ---------------------------------------------------------------------------------------------


def feature_cache_stem(
    part: str,
    paths: Sequence[Path],
    data_config: DataConfig,
    n_views: int,
    seed: int,
    augmented: bool,
    revision: str = MODEL_REVISION,
) -> str:
    """Construit le début du nom des fichiers de cache (la « clé » des caractéristiques).

    La clé change dès que l'un de ces éléments change : révision de DINOv2, mode de prétraitement,
    version du recadrage, mode (augmenté ou non), nombre de vues, graine, taille et statistiques
    de normalisation, paramètres d'augmentation (si augmenté) et liste des images.
    Limite : la clé repose sur les CHEMINS des images, pas sur leur contenu ; si une image est
    remplacée sous le même nom, il faut vider le cache à la main (cas absent ici : le jeu
    Kaggle est figé et son manifeste contient le SHA-256 de chaque image).

    Args:
        part: Nom de la partie (``train``, ``val`` ou ``test``).
        paths: Chemins des images, dans l'ordre.
        data_config: Configuration des données (prétraitement, augmentation, normalisation).
        n_views: Nombre de vues par image.
        seed: Graine de base des augmentations.
        augmented: True si les vues sont augmentées, False pour la vue d'évaluation.
        revision: Révision de DINOv2.

    Returns:
        Un texte du type ``train_head_crop_aug_8v_s42_1a2b3c4d5e6f``.
    """
    mode = "aug" if augmented else "eval"
    ingredients = {
        "revision": revision,
        "preprocessing": data_config.preprocessing,
        "crop_version": CROP_VERSION,
        "mode": mode,
        "n_views": n_views,
        "seed": seed,
        "image_size": data_config.image_size,
        "mean": list(data_config.mean),
        "std": list(data_config.std),
        "aug": json.loads(data_config.head_crop_aug.model_dump_json()) if augmented else None,
        "legacy_aug": [data_config.flip_prob, data_config.rotation_degrees, data_config.jitter]
        if augmented and data_config.preprocessing == "legacy"
        else None,
        "paths": [p.as_posix() for p in paths],
    }
    digest = hashlib.sha256(json.dumps(ingredients, sort_keys=True).encode()).hexdigest()[:12]
    return f"{part}_{data_config.preprocessing}_{mode}_{n_views}v_s{seed}_{digest}"


def extract_view(
    backbone: nn.Module,
    dataset: ManifestDataset,
    seed: int,
    batch_size: int,
    workers: int,
    desc: str = "",
) -> torch.Tensor:
    """Calcule les caractéristiques d'une vue de chaque image du jeu de données.

    Args:
        backbone: Corps DINOv2 en mode évaluation (sur CPU ou GPU).
        dataset: Jeu de données (la transformation du jeu fixe la vue : évaluation ou augmentée).
        seed: Graine de cette vue : même graine, mêmes augmentations.
        batch_size: Images par lot.
        workers: Processus de chargement des images.
        desc: Texte de la barre de progression.

    Returns:
        Caractéristiques ``(N, 768)`` sur CPU, dans l'ordre du jeu de données.
    """
    device = next(backbone.parameters()).device
    torch.manual_seed(seed)  # augmentations du processus principal (workers = 0)
    generator = torch.Generator().manual_seed(seed)  # graine de base des processus de chargement
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers, generator=generator
    )
    outputs: list[torch.Tensor] = []
    for images, _ in tqdm(loader, desc=desc, leave=False, mininterval=30):
        outputs.append(backbone_features(backbone, images.to(device)).cpu())
    return torch.cat(outputs)


class CacheMissingError(RuntimeError):
    """Une vue manque dans le cache et aucun corps DINOv2 n'a été fourni pour la calculer."""


@dataclass
class ExtractionResult:
    """Caractéristiques d'une partie du jeu de données et statistiques de calcul."""

    features: torch.Tensor  # (N, n_views, 768)
    labels: torch.Tensor  # (N,) indices de classe
    computed_views: int  # nombre de vues calculées (les autres venaient du cache)
    compute_seconds: float  # temps passé dans DINOv2 (hors cache)
    computed_images: int  # nombre d'images passées dans DINOv2 (vues calculées x N)


def get_features(
    backbone: nn.Module | None,
    part: str,
    samples: Sequence[tuple[Path, int]],
    classes: list[str],
    data_config: DataConfig,
    head_config: Dinov2HeadConfig,
    n_views: int,
    augmented: bool,
    seed: int,
    revision: str = MODEL_REVISION,
    progress: Callable[[str], None] | None = None,
) -> ExtractionResult:
    """Calcule (ou relit dans le cache) les caractéristiques de toutes les vues d'une partie.

    Chaque vue a son fichier de cache ``<clé>_view<k>.pt`` : une extraction interrompue reprend
    à la première vue manquante, et rien n'est recalculé si le cache existe. Le corps n'est
    chargé (``backbone``) que si au moins une vue manque : on peut passer ``None`` si tout est
    en cache, mais une vue manquante lève alors une erreur.

    Args:
        backbone: Corps DINOv2, ou None si tout est déjà en cache.
        part: ``train``, ``val`` ou ``test``.
        samples: Couples ``(chemin absolu, indice de classe)``.
        classes: Noms des classes triés.
        data_config: Configuration des données (le prétraitement fixe les transformations).
        head_config: Configuration de la tête (taille des lots, processus, dossier du cache).
        n_views: Nombre de vues par image.
        augmented: True : transformation d'entraînement (aléatoire, une graine par vue) ;
            False : transformation d'évaluation (la même vue, donc ``n_views`` doit valoir 1).
        seed: Graine de base ; la vue ``k`` utilise ``seed + k``.
        revision: Révision de DINOv2 (entre dans la clé du cache).
        progress: Fonction optionnelle appelée avec un message à chaque vue traitée.

    Returns:
        Un :class:`ExtractionResult` : caractéristiques ``(N, n_views, 768)`` et étiquettes.

    Raises:
        ValueError: Si ``augmented`` est False avec plus d'une vue (les vues seraient identiques).
        CacheMissingError: Si une vue manque dans le cache et qu'aucun corps n'est fourni.
    """
    if not augmented and n_views != 1:
        raise ValueError(
            "Sans augmentation, une seule vue a un sens (les vues seraient identiques)"
        )
    paths = [path for path, _ in samples]
    stem = feature_cache_stem(part, paths, data_config, n_views, seed, augmented, revision)
    cache_dir = head_config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = ManifestDataset(
        list(samples), build_transforms(data_config, train=augmented), classes
    )

    views: list[torch.Tensor] = []
    computed, seconds = 0, 0.0
    for view in range(n_views):
        cache_file = cache_dir / f"{stem}_view{view}.pt"
        if cache_file.is_file():
            views.append(torch.load(cache_file, weights_only=True))  # jamais recalculé
            continue
        if backbone is None:
            raise CacheMissingError(
                f"Cache absent ({cache_file.name}) et aucun corps DINOv2 fourni"
            )
        start = time.time()
        features = extract_view(
            backbone,
            dataset,
            seed + view,
            head_config.extract_batch_size,
            head_config.extract_workers,
            desc=f"{part} vue {view + 1}/{n_views}",
        )
        seconds += time.time() - start
        computed += 1
        tmp_file = cache_file.with_suffix(".tmp")
        torch.save(features, tmp_file)  # écriture puis renommage : pas de cache à moitié écrit
        tmp_file.replace(cache_file)
        views.append(features)
        if progress:
            progress(f"{part} : vue {view + 1}/{n_views} calculée ({len(dataset)} images)")
    labels = torch.tensor([target for _, target in samples], dtype=torch.long)
    return ExtractionResult(
        features=torch.stack(views, dim=1),
        labels=labels,
        computed_views=computed,
        compute_seconds=seconds,
        computed_images=computed * len(dataset),
    )


# ---------------------------------------------------------------------------------------------
# Entraînement de la tête sur les caractéristiques en mémoire
# ---------------------------------------------------------------------------------------------


@dataclass
class HeadTrainingResult:
    """Résultat de :func:`train_head`."""

    head: nn.Linear  # la tête, avec les poids du meilleur epoch (validation)
    history: list[dict]  # une entrée par epoch
    best_epoch: int  # numéro (à partir de 1) du meilleur epoch
    best_val_loss: float  # loss de validation de ce meilleur epoch
    best_val_acc: float  # accuracy de validation de ce meilleur epoch


@torch.no_grad()
def _evaluate_head(
    head: nn.Linear, features: torch.Tensor, labels: torch.Tensor, criterion: nn.Module
) -> tuple[float, float]:
    """Loss et accuracy de la tête sur un jeu de caractéristiques (sans gradient)."""
    head.eval()
    logits = head(features)
    return criterion(logits, labels).item(), (logits.argmax(dim=1) == labels).float().mean().item()


def train_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    n_classes: int,
    config: Dinov2HeadConfig,
    seed: int = 42,
    on_epoch: Callable[[dict], None] | None = None,
    sample_weights: torch.Tensor | None = None,
) -> HeadTrainingResult:
    """Entraîne la couche linéaire sur des caractéristiques déjà calculées.

    AdamW, entropie croisée, mini-lots mélangés (graine fixée), arrêt précoce sur la loss de
    validation (``config.patience``). On garde les poids du meilleur epoch de validation.

    Args:
        train_x: Caractéristiques d'entraînement ``(M, d)`` (toutes les vues mises bout à bout).
        train_y: Étiquettes d'entraînement ``(M,)``.
        val_x: Caractéristiques de validation ``(V, d)``.
        val_y: Étiquettes de validation ``(V,)``.
        n_classes: Nombre de classes.
        config: Réglages (lr, weight_decay, epochs, patience, min_delta, batch_size).
        seed: Graine (initialisation de la tête et mélange des lots).
        on_epoch: Fonction optionnelle appelée avec la ligne d'historique de chaque epoch.
        sample_weights: Poids par exemple d'entraînement ``(M,)`` (repondération des groupes,
            étape 11b-6b). La perte d'un lot est alors la moyenne pondérée des entropies croisées
            (somme des ``poids x perte`` divisée par la somme des poids du lot : l'échelle des
            poids n'a pas d'importance). ``None`` (défaut) : comportement strictement inchangé.
            La validation et l'arrêt précoce restent non pondérés.

    Returns:
        Un :class:`HeadTrainingResult`.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)  # mélange des mini-lots reproductible
    head = nn.Linear(train_x.shape[1], n_classes)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    per_sample_loss = nn.CrossEntropyLoss(reduction="none")  # utilisée seulement avec des poids

    history: list[dict] = []
    best = {"loss": float("inf"), "acc": 0.0, "epoch": 0, "state": head_state_dict(head)}
    waited = 0  # epochs consécutives sans progrès
    for epoch in range(1, config.epochs + 1):
        start = time.time()
        head.train()
        order = torch.randperm(len(train_x), generator=generator)
        total_loss, correct, total_weight = 0.0, 0, 0.0
        for first in range(0, len(order), config.batch_size):
            batch = order[first : first + config.batch_size]
            optimizer.zero_grad()
            logits = head(train_x[batch])
            if sample_weights is None:
                loss = criterion(logits, train_y[batch])
                batch_weight = float(len(batch))  # pondéré par la taille du lot
                total_loss += loss.item() * batch_weight
            else:
                weights = sample_weights[batch]
                loss = (per_sample_loss(logits, train_y[batch]) * weights).sum() / weights.sum()
                batch_weight = float(weights.sum())  # la loss d'epoch est une moyenne pondérée
                total_loss += loss.item() * batch_weight
            total_weight += batch_weight
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(dim=1) == train_y[batch]).sum().item()
        val_loss, val_acc = _evaluate_head(head, val_x, val_y, criterion)
        row = {
            "phase": "head",
            "epoch": epoch,
            "train_loss": total_loss / total_weight,
            "train_acc": correct / len(train_x),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": config.lr,
            "seconds": time.time() - start,
        }
        history.append(row)
        if on_epoch:
            on_epoch(row)
        if val_loss < best["loss"] - config.min_delta:
            waited = 0
        else:
            waited += 1
        if val_loss < best["loss"]:  # meilleur modèle de validation
            best = {
                "loss": val_loss,
                "acc": val_acc,
                "epoch": epoch,
                "state": head_state_dict(head),
            }
        if waited >= config.patience:
            LOGGER.info("Arrêt précoce après %d epoch(s)", epoch)
            break
    head.load_state_dict({k.removeprefix("head."): v for k, v in best["state"].items()})
    head.eval()
    return HeadTrainingResult(head, history, best["epoch"], best["loss"], best["acc"])
