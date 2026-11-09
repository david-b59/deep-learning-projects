"""Tests de brain_mri.config et brain_mri.data."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from pydantic import ValidationError
from torch.utils.data import RandomSampler, SequentialSampler

from brain_mri.config import Config, DataConfig, load_config
from brain_mri.data import build_dataloaders, build_transforms, make_datasets, stratified_split

# ----- configuration ------------------------------------------------------


def test_default_config_file_is_valid() -> None:
    """Le fichier configs/train.yaml du dépôt est valide et ses chemins sont absolus."""
    config = load_config()
    assert config.data.image_size == 224
    assert config.data.raw_dir.is_absolute()


def test_config_rejects_unknown_key() -> None:
    """Une clé inconnue (faute de frappe) est refusée."""
    with pytest.raises(ValidationError):
        Config.model_validate({"data": {"image_sise": 224}})


def test_config_rejects_invalid_value() -> None:
    """Une valeur hors limites (val_fraction > 1) est refusée."""
    with pytest.raises(ValidationError):
        DataConfig(val_fraction=1.5)


# ----- transformations ----------------------------------------------------


def test_transforms_output_shape_and_dtype(config: DataConfig, raw_dir: Path) -> None:
    """Avec ou sans augmentation, la sortie est un tenseur (3, taille, taille) en float32."""
    image = Image.open(raw_dir / "Training/glioma/img_0.jpg").convert("RGB")
    for train in (True, False):
        tensor = build_transforms(config, train=train)(image)
        assert tensor.shape == (3, 32, 32)
        assert tensor.dtype == torch.float32


def test_eval_transform_is_deterministic(config: DataConfig, raw_dir: Path) -> None:
    """Sans augmentation, la même image donne toujours exactement le même tenseur."""
    image = Image.open(raw_dir / "Training/glioma/img_0.jpg").convert("RGB")
    transform = build_transforms(config, train=False)
    assert torch.equal(transform(image), transform(image))


# ----- découpage ----------------------------------------------------------


def test_split_is_disjoint_complete_and_stratified() -> None:
    """Le découpage est disjoint, complet, et garde la proportion de la petite classe."""
    targets = [0] * 40 + [1] * 40 + [2] * 20
    train_idx, val_idx = stratified_split(targets, val_fraction=0.25, seed=1)
    assert set(train_idx).isdisjoint(val_idx)
    assert sorted(train_idx + val_idx) == list(range(100))
    assert [targets[i] for i in val_idx].count(2) == 5  # 25 % de la petite classe (20)


def test_split_is_reproducible() -> None:
    """La même graine donne exactement le même découpage."""
    targets = [0, 1] * 30
    assert stratified_split(targets, 0.2, seed=3) == stratified_split(targets, 0.2, seed=3)


# ----- DataLoaders --------------------------------------------------------


def test_dataloaders_shapes_classes_and_exclusions(config: DataConfig) -> None:
    """Classes triées, doublons écartés, et un lot a les bonnes formes et types."""
    bundle = build_dataloaders(config)
    assert bundle.class_names == ["glioma", "meningioma", "notumor", "pituitary"]
    # 81 fichiers d'entraînement - 1 copie = 80 (puis 25 % pour la validation) ;
    # test : 25 fichiers - 1 fuite = 24.
    assert bundle.sizes["train"] + bundle.sizes["val"] == 80
    assert bundle.sizes["test"] == 24
    images, labels = next(iter(bundle.train_loader))
    assert images.shape == (8, 3, 32, 32)
    assert labels.shape == (8,) and labels.dtype == torch.int64


def test_only_train_loader_is_shuffled(config: DataConfig) -> None:
    """Seul le DataLoader d'entraînement est mélangé (validation et test en ordre fixe)."""
    bundle = build_dataloaders(config)
    assert isinstance(bundle.train_loader.sampler, RandomSampler)
    assert isinstance(bundle.val_loader.sampler, SequentialSampler)
    assert isinstance(bundle.test_loader.sampler, SequentialSampler)


def test_shuffle_order_is_reproducible(config: DataConfig) -> None:
    """Avec la même graine, l'ordre du mélange est identique d'une construction à l'autre."""
    first = [labels.tolist() for _, labels in build_dataloaders(config).train_loader]
    second = [labels.tolist() for _, labels in build_dataloaders(config).train_loader]
    assert first == second


def test_validation_images_are_not_augmented(config: DataConfig) -> None:
    """La validation n'est jamais augmentée, même avec une augmentation très forte."""
    # Si l'augmentation fuitait dans la validation, deux passes donneraient des images différentes.
    strong = config.model_copy(update={"rotation_degrees": 45.0, "jitter": 0.5})
    first = torch.cat([x for x, _ in build_dataloaders(strong).val_loader])
    second = torch.cat([x for x, _ in build_dataloaders(strong).val_loader])
    assert torch.equal(first, second)


def test_train_and_val_do_not_share_files(config: DataConfig) -> None:
    """Aucun fichier n'est à la fois dans l'entraînement et dans la validation."""
    train, val, _, _ = make_datasets(config)
    train_files = {train.dataset.samples[i][0] for i in train.indices}
    val_files = {val.dataset.samples[i][0] for i in val.indices}
    assert train_files.isdisjoint(val_files)


# ----- manifeste de découpage ---------------------------------------------


def _write_manifest(path: Path, header: str = "file,sha256,class,part") -> None:
    """Écrit un CSV de découpage synthétique : 4 train, 1 val, 1 test par classe."""
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    lines = [header]
    for name in classes:
        for i in range(6):
            part = "train" if i < 4 else ("val" if i == 4 else "test")
            lines.append(f"Training/{name}/img_{i}.jpg,abc,{name},{part}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_manifest_builds_the_three_sets(config: DataConfig, tmp_path: Path) -> None:
    """Avec split_manifest, les trois jeux viennent du CSV (pas du découpage automatique)."""
    manifest = tmp_path / "split.csv"
    _write_manifest(manifest)
    cfg = config.model_copy(update={"split_manifest": manifest})
    train, val, test, classes = make_datasets(cfg)
    assert (len(train), len(val), len(test)) == (16, 4, 4)
    assert classes == ["glioma", "meningioma", "notumor", "pituitary"]
    image, target = train[0]
    assert image.shape == (3, 32, 32) and target == 0
    # Les chemins de test restent lisibles comme avec ImageFolder (.samples)
    assert all(Path(p).is_file() for p, _ in test.samples)


def test_manifest_custom_columns(config: DataConfig, tmp_path: Path) -> None:
    """Les noms de colonnes sont configurables (ex. partie / classe en français)."""
    manifest = tmp_path / "split.csv"
    _write_manifest(manifest, header="chemin,sha256,classe,partie")
    cfg = config.model_copy(
        update={
            "split_manifest": manifest,
            "manifest_path_column": "chemin",
            "manifest_label_column": "classe",
            "manifest_split_column": "partie",
        }
    )
    assert len(make_datasets(cfg)[2]) == 4


def test_manifest_errors(config: DataConfig, tmp_path: Path) -> None:
    """Colonne absente, partie inconnue et image introuvable sont refusées clairement."""
    manifest = tmp_path / "split.csv"
    _write_manifest(manifest)
    wrong_col = config.model_copy(update={"split_manifest": manifest, "manifest_split_column": "x"})
    with pytest.raises(ValueError, match="Colonnes absentes"):
        make_datasets(wrong_col)
    manifest.write_text(manifest.read_text().replace(",test", ",val"), encoding="utf-8")
    with pytest.raises(ValueError, match="Parties vides"):  # plus aucune ligne « test »
        make_datasets(config.model_copy(update={"split_manifest": manifest}))
    manifest.write_text("file,class,part\nNope/x.jpg,glioma,train\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        make_datasets(config.model_copy(update={"split_manifest": manifest}))


def test_manifest_dataloaders_and_train_only_augmentation(
    config: DataConfig, tmp_path: Path
) -> None:
    """Les DataLoaders fonctionnent avec un manifeste ; val et test ne sont pas augmentés."""
    manifest = tmp_path / "split.csv"
    _write_manifest(manifest)
    cfg = config.model_copy(update={"split_manifest": manifest})
    bundle = build_dataloaders(cfg)
    assert bundle.sizes == {"train": 16, "val": 4, "test": 4}
    x, y = next(iter(bundle.train_loader))
    assert x.shape[1:] == (3, 32, 32) and len(x) == len(y) == 8
    train, val, _, _ = make_datasets(cfg)
    assert len(train.transform.transforms) > len(val.transform.transforms)


def test_manifest_real_format_ignores_conflict_and_purge(
    config: DataConfig, tmp_path: Path
) -> None:
    """Format réel du CSV (colonnes file, class, part) : conflit et purge sont ignorés."""
    manifest = tmp_path / "kaggle_grouped.csv"
    lines = ["file,sha256,class,is_512,group,part,max_cos_to_train"]
    for i, part in enumerate(["train", "train", "val", "test", "conflit", "purge"]):
        lines.append(f"Training/glioma/img_{i}.jpg,abc,glioma,1,{i},{part},0.5")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg = config.model_copy(update={"split_manifest": manifest})
    assert cfg.manifest_path_column == "file" and cfg.manifest_split_column == "part"
    train, val, test, classes = make_datasets(cfg)
    assert (len(train), len(val), len(test)) == (2, 1, 1)
    assert classes == ["glioma"]
