"""Entraînement en deux phases (phase C du pipeline du cours).

Phase 1 : seule la nouvelle tête ``fc`` est entraînée (``lr`` 1e-3), le corps du
ResNet-50 reste gelé. Phase 2 : on dégèle le dernier bloc ``layer4`` (fine-tuning) avec
un ``lr`` dix fois plus petit (1e-4).

À chaque epoch : entraînement sur ``train``, mesure sur ``val``, mise à jour du
scheduler ``ReduceLROnPlateau`` et de l'early stopping sur la loss de validation, et
sauvegarde du meilleur modèle (poids + noms des classes). Le jeu de test n'est jamais
utilisé ici : il sert une seule fois, à l'étape 4.

Écarts avec le cours (corrections déjà identifiées) :
    - les couches BatchNorm gelées restent en ``eval()`` (voir
      :func:`set_frozen_batchnorm_eval`) ;
    - l'optimiseur ne reçoit que les paramètres entraînables ;
    - le scheduler est créé sur l'optimiseur réellement utilisé ;
    - la sélection du modèle se fait sur la validation, pas sur le test.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python -m brain_mri.train
    uv run python -m brain_mri.train --max-batches 2   # essai rapide
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import logging
import math
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : les figures sont écrites dans des fichiers
import matplotlib.pyplot as plt
import mlflow
import torch
import truststore
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from brain_mri.config import Config, PhaseConfig, load_config
from brain_mri.data import DataBundle, build_dataloaders
from brain_mri.model import build_model, count_parameters, unfreeze_last_block

LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def keep_awake(enabled: bool = True) -> Iterator[None]:
    """Empêche Windows de se mettre en veille pendant un long entraînement.

    Demande à Windows de rester éveillé tant que le bloc ``with`` s'exécute, puis
    relâche la demande à la sortie : aucun réglage du système n'est modifié. Sans effet
    sur les autres systèmes ou si ``enabled`` vaut False.

    Args:
        enabled: Si False, ne fait rien.
    """
    active = enabled and sys.platform == "win32"
    if active:
        # ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        yield
    finally:
        if active:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS seul


def get_device() -> torch.device:
    """Choisit le GPU s'il est disponible, sinon le CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Remet en mode ``eval()`` les BatchNorm dont tous les paramètres sont gelés.

    ``model.train()`` réactive toutes les couches BatchNorm, même gelées : leurs
    statistiques (moyenne et variance) continueraient alors à dériver vers nos IRM et
    abîmeraient ce que le modèle a appris sur ImageNet. On appelle cette fonction
    après ``model.train()`` à chaque epoch.

    Args:
        model: Le modèle en cours d'entraînement.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
            p.requires_grad for p in module.parameters()
        ):
            module.eval()


def build_optimizer(model: nn.Module, lr: float) -> Adam:
    """Crée un optimiseur Adam sur les seuls paramètres entraînables.

    Args:
        model: Le modèle (certains paramètres sont gelés).
        lr: Taux d'apprentissage.

    Returns:
        L'optimiseur.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    return Adam(trainable, lr=lr)


class EarlyStopping:
    """Arrête une phase quand la loss de validation ne s'améliore plus.

    Args:
        patience: Nombre d'epochs consécutives sans progrès avant d'arrêter.
        min_delta: Amélioration minimale pour compter comme un progrès.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        """Initialise le compteur à zéro, sans meilleure valeur connue."""
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf  # meilleure loss vue dans cette phase
        self.counter = 0  # epochs consécutives sans progrès

    def step(self, val_loss: float) -> bool:
        """Enregistre la loss d'une epoch.

        Args:
            val_loss: Loss de validation de l'epoch.

        Returns:
            True s'il faut arrêter la phase.
        """
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
    max_batches: int | None = None,
    desc: str = "",
) -> tuple[float, float]:
    """Parcourt un DataLoader une fois, en entraînement ou en évaluation.

    Si ``optimizer`` est fourni, le modèle apprend (``train()``, gradients, mise à
    jour). Sinon, il est évalué (``eval()`` + ``no_grad``), comme dans le cours.

    Args:
        model: Le modèle.
        loader: Les lots à parcourir.
        criterion: La fonction de perte.
        device: CPU ou GPU.
        optimizer: Optimiseur, ou None pour une simple évaluation.
        max_batches: Si fourni, s'arrête après ce nombre de lots (essais rapides).
        desc: Texte de la barre de progression.

    Returns:
        ``(loss moyenne, accuracy)`` sur les images parcourues.
    """
    training = optimizer is not None
    if training:
        model.train()
        set_frozen_batchnorm_eval(model)  # écart avec le cours : voir le docstring du module
    else:
        model.eval()

    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):  # pas de gradients en évaluation
        for i, (images, labels) in enumerate(tqdm(loader, desc=desc, leave=False, mininterval=30)):
            if max_batches is not None and i >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()  # calcule les gradients
                optimizer.step()  # met à jour les poids entraînables
            total_loss += loss.item() * labels.size(0)  # pondéré par la taille du lot
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
    return total_loss / seen, correct / seen


def save_checkpoint(path: Path, model: nn.Module, class_names: list[str], info: dict) -> None:
    """Enregistre le modèle avec ses noms de classes (sinon l'ordre serait perdu).

    Args:
        path: Fichier de destination (``.pt``, hors git).
        model: Le modèle à sauvegarder.
        class_names: Noms des classes, dans l'ordre des étiquettes.
        info: Informations utiles (epoch, phase, métriques, prétraitement).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "class_names": class_names, **info}, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> dict:
    """Relit un fichier produit par :func:`save_checkpoint`.

    Args:
        path: Fichier à lire.
        device: Appareil sur lequel charger les tenseurs.

    Returns:
        Le dictionnaire enregistré (``state_dict``, ``class_names``, etc.).
    """
    return torch.load(path, map_location=device, weights_only=True)


def plot_curves(history: list[dict], path: Path) -> None:
    """Trace les courbes de loss et d'accuracy (entraînement et validation).

    Args:
        history: Une entrée par epoch, avec ``train_loss``, ``val_loss``, ``train_acc``,
            ``val_acc`` et ``phase``.
        path: Fichier image de destination.
    """
    epochs = list(range(1, len(history) + 1))
    # Position du passage de la phase 1 à la phase 2 (trait vertical sur les courbes).
    phase1_epochs = sum(1 for row in history if row["phase"] == "phase1")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title, ylabel in (
        (ax_loss, "loss", "Loss", "loss"),
        (ax_acc, "acc", "Accuracy", "accuracy"),
    ):
        ax.plot(epochs, [r[f"train_{key}"] for r in history], marker="o", label="entraînement")
        ax.plot(epochs, [r[f"val_{key}"] for r in history], marker="o", label="validation")
        if 0 < phase1_epochs < len(history):
            ax.axvline(phase1_epochs + 0.5, color="gray", linestyle="--", label="dégel de layer4")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


@dataclass
class Trainer:
    """Entraîne le modèle phase par phase et garde le meilleur modèle de validation.

    Attributs d'entrée : ``model``, ``bundle`` (DataLoaders), ``config``, ``device``.
    Les autres attributs sont l'état de l'entraînement.
    """

    model: nn.Module
    bundle: DataBundle
    config: Config
    device: torch.device
    max_batches: int | None = None  # essais rapides : limite le nombre de lots par epoch
    use_mlflow: bool = False  # envoie les métriques à MLflow
    criterion: nn.Module = field(default_factory=nn.CrossEntropyLoss)
    history: list[dict] = field(default_factory=list)  # une entrée par epoch
    best_val_loss: float = math.inf  # meilleure loss de validation, toutes phases confondues

    def run_phase(self, name: str, phase: PhaseConfig) -> None:
        """Entraîne le modèle pendant une phase (jusqu'à ``phase.epochs`` ou l'early stopping).

        Args:
            name: Nom de la phase (``phase1`` ou ``phase2``), pour les journaux.
            phase: Nombre d'epochs et taux d'apprentissage de la phase.
        """
        cfg = self.config.training
        optimizer = build_optimizer(self.model, phase.lr)
        trainable, total = count_parameters(self.model)
        LOGGER.info(
            "%s : %s paramètres entraînables sur %s, lr=%g, %d epoch(s) max",
            name,
            f"{trainable:,}".replace(",", " "),
            f"{total:,}".replace(",", " "),
            phase.lr,
            phase.epochs,
        )
        # Le scheduler est créé sur l'optimiseur de CETTE phase (correction du cours).
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
        )
        stopper = EarlyStopping(cfg.early_stopping_patience, cfg.early_stopping_min_delta)

        for epoch in range(1, phase.epochs + 1):
            start = time.time()
            train_loss, train_acc = self._run(self.bundle.train_loader, optimizer, f"{name} train")
            val_loss, val_acc = self._run(self.bundle.val_loader, None, f"{name} val")
            scheduler.step(val_loss)  # baisse le lr si la validation stagne
            self._record(name, epoch, train_loss, train_acc, val_loss, val_acc, optimizer, start)
            if stopper.step(val_loss):
                LOGGER.info("%s : early stopping après %d epoch(s)", name, epoch)
                break

    def _run(self, loader: DataLoader, optimizer: Adam | None, desc: str) -> tuple[float, float]:
        """Lance :func:`run_epoch` avec les réglages du Trainer."""
        return run_epoch(
            self.model, loader, self.criterion, self.device, optimizer, self.max_batches, desc
        )

    def _record(
        self,
        phase: str,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        optimizer: Adam,
        start: float,
    ) -> None:
        """Enregistre les métriques d'une epoch et sauvegarde le modèle s'il est le meilleur."""
        row = {
            "phase": phase,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - start,
        }
        self.history.append(row)
        LOGGER.info(
            "%s epoch %d | train loss %.4f acc %.3f | val loss %.4f acc %.3f | lr %g | %.0f s",
            phase,
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            row["lr"],
            row["seconds"],
        )
        if self.use_mlflow:
            step = len(self.history)  # numéro d'epoch global, phases confondues
            mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, float)}, step=step)

        if val_loss < self.best_val_loss:  # nouveau meilleur modèle (sur la validation)
            self.best_val_loss = val_loss
            data_cfg = self.config.data
            save_checkpoint(
                self.config.training.checkpoint_path,
                self.model,
                self.bundle.class_names,
                {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "phase": phase,
                    "epoch": epoch,
                    "image_size": data_cfg.image_size,
                    "mean": data_cfg.mean,
                    "std": data_cfg.std,
                },
            )
            LOGGER.info("Meilleur modèle enregistré (val loss %.4f)", val_loss)
        # L'historique est réécrit à chaque epoch : rien n'est perdu si l'entraînement s'interrompt.
        history_path = self.config.data.processed_dir / "training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")


def _flatten(prefix: str, value: object) -> dict[str, object]:
    """Aplatit un dictionnaire imbriqué en clés ``a.b.c`` (format des paramètres MLflow)."""
    if isinstance(value, dict):
        return {
            k: v for key, sub in value.items() for k, v in _flatten(f"{prefix}.{key}", sub).items()
        }
    return {prefix: value}


def setup_mlflow(config: Config) -> None:
    """Configure MLflow : base SQLite et artefacts dans ``mlflow_dir`` (hors git)."""
    cfg = config.training
    cfg.mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(cfg.mlflow_dir / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(cfg.mlflow_experiment) is None:
        mlflow.create_experiment(
            cfg.mlflow_experiment, artifact_location=(cfg.mlflow_dir / "artifacts").as_uri()
        )
    mlflow.set_experiment(cfg.mlflow_experiment)


def train(config: Config, max_batches: int | None = None, use_mlflow: bool = True) -> Trainer:
    """Entraîne le modèle en deux phases et enregistre le meilleur modèle et les courbes.

    Args:
        config: Configuration complète.
        max_batches: Si fourni, limite le nombre de lots par epoch (essai rapide).
        use_mlflow: Si True, suit l'expérience avec MLflow.

    Returns:
        Le :class:`Trainer` (historique et meilleure loss de validation).
    """
    torch.manual_seed(config.data.seed)  # tête aléatoire et augmentation reproductibles
    device = get_device()
    LOGGER.info("Appareil : %s", device)
    bundle = build_dataloaders(config.data)
    model = build_model(len(bundle.class_names), config.model).to(device)
    trainer = Trainer(model, bundle, config, device, max_batches, use_mlflow)

    def run_all() -> None:
        """Enchaîne la phase 1 (tête seule) puis la phase 2 (dégel de layer4)."""
        trainer.run_phase("phase1", config.training.phase1)
        unfreeze_last_block(model)  # fine-tuning : layer4 devient entraînable
        trainer.run_phase("phase2", config.training.phase2)

    if use_mlflow:
        setup_mlflow(config)
        with mlflow.start_run():
            mlflow.log_params(_flatten("config", json.loads(config.model_dump_json())))
            run_all()
            curves = config.training.figures_dir / "training_curves.png"
            plot_curves(trainer.history, curves)
            mlflow.log_artifact(str(curves))
            mlflow.log_metric("best_val_loss", trainer.best_val_loss)
    else:
        run_all()
        plot_curves(trainer.history, config.training.figures_dir / "training_curves.png")
    LOGGER.info("Terminé. Meilleure loss de validation : %.4f", trainer.best_val_loss)
    return trainer


def main() -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraîne le classifieur d'IRM cérébrales.")
    parser.add_argument("--config", type=Path, default=None, help="fichier YAML de configuration")
    parser.add_argument(
        "--max-batches", type=int, default=None, help="limite de lots par epoch (essai rapide)"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="désactive le suivi MLflow")
    parser.add_argument(
        "--keep-awake", action="store_true", help="empêche la mise en veille de Windows"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    truststore.inject_into_ssl()  # téléchargement des poids ImageNet derrière un proxy
    config = load_config(args.config) if args.config else load_config()
    with keep_awake(args.keep_awake):
        train(config, max_batches=args.max_batches, use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
