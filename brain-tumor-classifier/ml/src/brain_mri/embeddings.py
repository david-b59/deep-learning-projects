"""Embeddings d'images IRM avec DINOv2 (sert à repérer les quasi-doublons).

Un « embedding » est un vecteur de nombres qui résume le contenu d'une image : deux
images très ressemblantes ont des embeddings proches (similarité cosinus proche de 1).
On utilise ``facebook/dinov2-small`` (384 dimensions), un modèle généraliste, sans
aucun entraînement supplémentaire.

Chaîne de traitement d'une image :
    1. :func:`brain_mri.head_crop.crop_head` : tête recadrée en carré, intensité normalisée ;
    2. redimensionnement 224 x 224 puis recopie du canal gris sur 3 canaux ;
    3. normalisation ImageNet (moyenne et écart-type du processeur DINOv2) ;
    4. DINOv2 : on garde le vecteur du jeton CLS (384 valeurs), normalisé en norme L2.

Les embeddings sont mis en cache (un petit fichier ``.npz`` par image, nommé d'après le
SHA-256 du fichier, la révision du modèle et la version du recadrage) : on ne recalcule
jamais deux fois la même image, et on ne relit jamais d'anciens embeddings.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Dinov2Model

from brain_mri.duplicates import sha256_file
from brain_mri.head_crop import CROP_VERSION, crop_head

MODEL_ID = "facebook/dinov2-small"
# Révision (commit) épinglée sur Hugging Face : garantit les mêmes poids à chaque exécution
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
IMAGE_SIZE = 224  # côté de l'image donnée au modèle (16 x 16 patchs de 14 pixels)
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # valeurs du processeur DINOv2 (preprocessor_config.json)
IMAGENET_STD = (0.229, 0.224, 0.225)
EMBEDDING_DIM = 384  # taille de l'embedding de dinov2-small


def load_model(
    model_id: str = MODEL_ID, revision: str = MODEL_REVISION, device: str = "cpu"
) -> Dinov2Model:
    """Charge DINOv2 en mode évaluation (téléchargé une seule fois puis mis en cache local).

    Args:
        model_id: Identifiant du modèle sur Hugging Face.
        revision: Hash du commit à charger (révision épinglée).
        device: ``"cpu"`` ou ``"cuda"``.

    Returns:
        Le modèle, en mode évaluation, sur ``device``.
    """
    model = Dinov2Model.from_pretrained(model_id, revision=revision)
    return model.to(device).eval()


def preprocess(image: Image.Image) -> torch.Tensor:
    """Transforme une image PIL en tenseur ``(3, 224, 224)`` prêt pour DINOv2.

    Args:
        image: Image PIL de n'importe quelle taille ou mode.

    Returns:
        Tenseur float32 : tête recadrée, 224 x 224, 3 canaux identiques, normalisé ImageNet.
    """
    head = crop_head(image).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    array = np.asarray(head, dtype=np.float32) / 255.0  # (224, 224) entre 0 et 1
    tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)  # 3 canaux identiques
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


@torch.inference_mode()
def embed_batch(model: Dinov2Model, batch: torch.Tensor) -> np.ndarray:
    """Calcule les embeddings CLS normalisés (norme L2) d'un lot d'images prétraitées.

    Args:
        model: Modèle DINOv2 en mode évaluation.
        batch: Tenseur ``(N, 3, 224, 224)``.

    Returns:
        Tableau float32 ``(N, dim)`` dont chaque ligne a une norme égale à 1.
    """
    device = next(model.parameters()).device
    hidden = model(pixel_values=batch.to(device)).last_hidden_state  # (N, jetons, dim)
    cls = hidden[:, 0]  # le jeton CLS résume l'image entière
    cls = torch.nn.functional.normalize(cls, p=2, dim=1)
    return cls.cpu().numpy().astype(np.float32)


def embed_images(
    model: Dinov2Model,
    images: Iterable[Image.Image],
    batch_size: int = 32,
) -> np.ndarray:
    """Calcule les embeddings d'une liste d'images PIL, par lots.

    Args:
        model: Modèle DINOv2 en mode évaluation.
        images: Images PIL.
        batch_size: Nombre d'images par lot.

    Returns:
        Tableau float32 ``(N, dim)`` (tableau vide ``(0, 0)`` si aucune image).
    """
    outputs: list[np.ndarray] = []
    batch: list[torch.Tensor] = []
    for image in images:
        batch.append(preprocess(image))
        if len(batch) == batch_size:
            outputs.append(embed_batch(model, torch.stack(batch)))
            batch = []
    if batch:
        outputs.append(embed_batch(model, torch.stack(batch)))
    return np.concatenate(outputs) if outputs else np.zeros((0, 0), dtype=np.float32)


def _load_preprocessed(path: Path) -> torch.Tensor:
    """Ouvre un fichier image et le prétraite (fonction exécutée dans un thread)."""
    with Image.open(path) as image:
        return preprocess(image)


def embed_files(
    model: Dinov2Model,
    paths: Sequence[Path],
    cache_dir: Path,
    revision: str = MODEL_REVISION,
    batch_size: int = 32,
    workers: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Calcule (ou relit dans le cache) les embeddings d'une liste de fichiers.

    Le cache contient un fichier ``<sha256>_<révision>_<version du recadrage>.npz`` par
    image : seules les images absentes du cache passent par le modèle.

    Args:
        model: Modèle DINOv2 en mode évaluation.
        paths: Chemins des images.
        cache_dir: Dossier du cache (créé si besoin), par ex. ``data/cache/embeddings``.
        revision: Révision du modèle, incluse dans le nom des fichiers du cache.
        batch_size: Nombre d'images par lot.
        workers: Nombre de threads pour ouvrir et prétraiter les images.
        progress: Fonction optionnelle appelée avec ``(faites, total)`` après chaque lot.

    Returns:
        Tableau float32 ``(len(paths), dim)``, dans l'ordre de ``paths``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_files = [cache_dir / f"{sha256_file(p)}_{revision[:8]}_{CROP_VERSION}.npz" for p in paths]
    result: list[np.ndarray | None] = [None] * len(paths)
    missing = []
    for index, cache_file in enumerate(cache_files):
        if cache_file.is_file():
            with np.load(cache_file) as stored:
                result[index] = stored["embedding"]
        else:
            missing.append(index)

    done = len(paths) - len(missing)
    if progress:
        progress(done, len(paths))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            tensors = list(pool.map(_load_preprocessed, [paths[i] for i in chunk]))
            vectors = embed_batch(model, torch.stack(tensors))
            for index, vector in zip(chunk, vectors, strict=True):
                np.savez_compressed(cache_files[index], embedding=vector)
                result[index] = vector
            done += len(chunk)
            if progress:
                progress(done, len(paths))
    if not paths:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([v for v in result if v is not None])
