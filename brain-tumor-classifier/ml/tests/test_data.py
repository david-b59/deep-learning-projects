"""Tests for brain_mri.config and brain_mri.data."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from pydantic import ValidationError
from torch.utils.data import RandomSampler, SequentialSampler

from brain_mri.config import Config, DataConfig, load_config
from brain_mri.data import build_dataloaders, build_transforms, make_datasets, stratified_split

# ----- config -------------------------------------------------------------


def test_default_config_file_is_valid() -> None:
    config = load_config()
    assert config.data.image_size == 224
    assert config.data.raw_dir.is_absolute()


def test_config_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"data": {"image_sise": 224}})


def test_config_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError):
        DataConfig(val_fraction=1.5)


# ----- transforms ---------------------------------------------------------


def test_transforms_output_shape_and_dtype(config: DataConfig, raw_dir: Path) -> None:
    image = Image.open(raw_dir / "Training/glioma/img_0.jpg").convert("RGB")
    for train in (True, False):
        tensor = build_transforms(config, train=train)(image)
        assert tensor.shape == (3, 32, 32)
        assert tensor.dtype == torch.float32


def test_eval_transform_is_deterministic(config: DataConfig, raw_dir: Path) -> None:
    image = Image.open(raw_dir / "Training/glioma/img_0.jpg").convert("RGB")
    transform = build_transforms(config, train=False)
    assert torch.equal(transform(image), transform(image))


# ----- split --------------------------------------------------------------


def test_split_is_disjoint_complete_and_stratified() -> None:
    targets = [0] * 40 + [1] * 40 + [2] * 20
    train_idx, val_idx = stratified_split(targets, val_fraction=0.25, seed=1)
    assert set(train_idx).isdisjoint(val_idx)
    assert sorted(train_idx + val_idx) == list(range(100))
    assert [targets[i] for i in val_idx].count(2) == 5  # 25 % of the small class


def test_split_is_reproducible() -> None:
    targets = [0, 1] * 30
    assert stratified_split(targets, 0.2, seed=3) == stratified_split(targets, 0.2, seed=3)


# ----- dataloaders --------------------------------------------------------


def test_dataloaders_shapes_classes_and_exclusions(config: DataConfig) -> None:
    bundle = build_dataloaders(config)
    assert bundle.class_names == ["glioma", "meningioma", "notumor", "pituitary"]
    # 81 train files - 1 duplicate copy = 80 (then 25 % validation); test: 25 files - 1 leak = 24
    assert bundle.sizes["train"] + bundle.sizes["val"] == 80
    assert bundle.sizes["test"] == 24
    images, labels = next(iter(bundle.train_loader))
    assert images.shape == (8, 3, 32, 32)
    assert labels.shape == (8,) and labels.dtype == torch.int64


def test_only_train_loader_is_shuffled(config: DataConfig) -> None:
    bundle = build_dataloaders(config)
    assert isinstance(bundle.train_loader.sampler, RandomSampler)
    assert isinstance(bundle.val_loader.sampler, SequentialSampler)
    assert isinstance(bundle.test_loader.sampler, SequentialSampler)


def test_shuffle_order_is_reproducible(config: DataConfig) -> None:
    first = [labels.tolist() for _, labels in build_dataloaders(config).train_loader]
    second = [labels.tolist() for _, labels in build_dataloaders(config).train_loader]
    assert first == second  # same seed, same order


def test_validation_images_are_not_augmented(config: DataConfig) -> None:
    # Strong augmentation settings: if they leaked into validation, two passes would differ.
    strong = config.model_copy(update={"rotation_degrees": 45.0, "jitter": 0.5})
    first = torch.cat([x for x, _ in build_dataloaders(strong).val_loader])
    second = torch.cat([x for x, _ in build_dataloaders(strong).val_loader])
    assert torch.equal(first, second)


def test_train_and_val_do_not_share_files(config: DataConfig) -> None:
    train, val, _, _ = make_datasets(config)
    train_files = {train.dataset.samples[i][0] for i in train.indices}
    val_files = {val.dataset.samples[i][0] for i in val.indices}
    assert train_files.isdisjoint(val_files)
