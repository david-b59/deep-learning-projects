"""Tests de brain_mri.train (faux jeu de données, poids aléatoires, quelques lots)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from brain_mri.config import Config, DataConfig, ModelConfig, PhaseConfig, TrainConfig
from brain_mri.data import build_dataloaders
from brain_mri.model import build_model, unfreeze_last_block
from brain_mri.train import (
    EarlyStopping,
    build_optimizer,
    get_device,
    keep_awake,
    load_checkpoint,
    plot_curves,
    run_epoch,
    save_checkpoint,
    set_frozen_batchnorm_eval,
    train,
)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


@pytest.fixture
def small_model() -> nn.Module:
    """Fournit un ResNet-50 à poids aléatoires, corps gelé, tête à 4 classes."""
    return build_model(4, ModelConfig(pretrained=False, freeze_backbone=True))


@pytest.fixture
def full_config(config: DataConfig, tmp_path: Path) -> Config:
    """Configuration complète pour le faux jeu : 1 epoch par phase, sorties dans tmp_path."""
    training = TrainConfig(
        phase1=PhaseConfig(epochs=1, lr=1e-3),
        phase2=PhaseConfig(epochs=1, lr=1e-4),
        checkpoint_path=tmp_path / "models" / "best.pt",
        figures_dir=tmp_path / "figures",
        mlflow_dir=tmp_path / "mlruns",
    )
    return Config(data=config, model=ModelConfig(pretrained=False), training=training)


# ----- briques de base ----------------------------------------------------


def test_early_stopping_triggers_after_patience() -> None:
    """L'arrêt se déclenche après ``patience`` epochs sans amélioration, pas avant."""
    stopper = EarlyStopping(patience=2, min_delta=0.0)
    assert stopper.step(1.0) is False  # premier progrès
    assert stopper.step(1.1) is False  # 1 epoch sans progrès
    assert stopper.step(1.2) is True  # 2 epochs sans progrès : on arrête


def test_early_stopping_resets_on_improvement() -> None:
    """Une amélioration remet le compteur à zéro."""
    stopper = EarlyStopping(patience=2)
    stopper.step(1.0)
    stopper.step(1.1)
    stopper.step(0.5)  # amélioration
    assert stopper.counter == 0
    assert stopper.step(0.6) is False


def test_optimizer_receives_only_trainable_parameters(small_model: nn.Module) -> None:
    """L'optimiseur ne contient que la tête quand le corps est gelé."""
    optimizer = build_optimizer(small_model, lr=1e-3)
    n_optimized = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
    assert n_optimized == 2048 * 4 + 4


def test_frozen_batchnorm_stay_in_eval_mode(small_model: nn.Module) -> None:
    """Après ``train()``, les BatchNorm gelés repassent en eval, sauf ceux de layer4 dégelé."""
    small_model.train()
    assert small_model.bn1.training is True  # train() a tout réactivé
    set_frozen_batchnorm_eval(small_model)
    assert small_model.bn1.training is False
    assert small_model.layer4[0].bn1.training is False

    unfreeze_last_block(small_model)  # layer4 devient entraînable : ses BatchNorm doivent apprendre
    small_model.train()
    set_frozen_batchnorm_eval(small_model)
    assert small_model.layer4[0].bn1.training is True
    assert small_model.layer1[0].bn1.training is False  # le reste reste figé


def test_run_epoch_eval_does_not_change_weights(config: DataConfig, small_model: nn.Module) -> None:
    """L'évaluation renvoie loss et accuracy valides sans modifier les poids."""
    bundle = build_dataloaders(config)
    before = small_model.fc.weight.detach().clone()
    loss, acc = run_epoch(
        small_model, bundle.val_loader, nn.CrossEntropyLoss(), torch.device("cpu"), max_batches=1
    )
    assert loss > 0 and 0.0 <= acc <= 1.0
    assert torch.equal(before, small_model.fc.weight)


def test_run_epoch_train_updates_only_the_head(config: DataConfig, small_model: nn.Module) -> None:
    """L'entraînement modifie la tête et laisse le corps gelé intact."""
    bundle = build_dataloaders(config)
    head_before = small_model.fc.weight.detach().clone()
    body_before = small_model.conv1.weight.detach().clone()
    run_epoch(
        small_model,
        bundle.train_loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        optimizer=build_optimizer(small_model, 1e-2),
        max_batches=1,
    )
    assert not torch.equal(head_before, small_model.fc.weight)
    assert torch.equal(body_before, small_model.conv1.weight)


def test_checkpoint_keeps_class_names(small_model: nn.Module, tmp_path: Path) -> None:
    """Le fichier sauvegardé conserve les noms de classes et les poids."""
    path = tmp_path / "ckpt" / "best.pt"
    save_checkpoint(path, small_model, CLASS_NAMES, {"val_loss": 0.5})
    loaded = load_checkpoint(path)
    assert loaded["class_names"] == CLASS_NAMES
    assert loaded["val_loss"] == 0.5
    assert torch.equal(loaded["state_dict"]["fc.weight"], small_model.fc.weight)


def test_plot_curves_writes_an_image(tmp_path: Path) -> None:
    """Les courbes sont enregistrées dans un fichier PNG non vide."""
    row = {"train_loss": 1.0, "val_loss": 1.1, "train_acc": 0.5, "val_acc": 0.4}
    history = [{"phase": "phase1", **row}, {"phase": "phase2", **row}]
    path = tmp_path / "figs" / "curves.png"
    plot_curves(history, path)
    assert path.is_file() and path.stat().st_size > 0


def test_keep_awake_runs_the_block_and_can_be_disabled() -> None:
    """Le bloc s'exécute normalement, activé ou non, et l'exception éventuelle est propagée."""
    ran = []
    with keep_awake(True):
        ran.append("actif")
    with keep_awake(False):
        ran.append("inactif")
    assert ran == ["actif", "inactif"]
    with pytest.raises(ValueError), keep_awake(True):
        raise ValueError("erreur dans le bloc")


def test_get_device_returns_a_device() -> None:
    """L'appareil choisi est cpu ou cuda."""
    assert get_device().type in {"cpu", "cuda"}


# ----- entraînement complet -----------------------------------------------


def test_full_training_writes_checkpoint_curves_and_history(full_config: Config) -> None:
    """Un entraînement minimal produit historique, meilleur modèle, courbes et suivi MLflow."""
    trainer = train(full_config, max_batches=1, use_mlflow=True)
    assert [row["phase"] for row in trainer.history] == ["phase1", "phase2"]
    assert trainer.best_val_loss < float("inf")
    checkpoint = load_checkpoint(full_config.training.checkpoint_path)
    assert checkpoint["class_names"] == CLASS_NAMES
    assert (full_config.training.figures_dir / "training_curves.png").is_file()
    assert (full_config.data.processed_dir / "training_history.json").is_file()
    assert (full_config.training.mlflow_dir / "mlflow.db").is_file()
