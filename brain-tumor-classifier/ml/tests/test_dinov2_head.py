"""Tests de brain_mri.dinov2_head (faux corps injecté : DINOv2 n'est jamais téléchargé)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from brain_mri.config import DataConfig, Dinov2HeadConfig, ModelConfig
from brain_mri.dinov2_head import (
    Dinov2Classifier,
    feature_cache_stem,
    get_features,
    pool_tokens,
    train_head,
)
from brain_mri.model import build_model, build_model_for_checkpoint
from brain_mri.train_dinov2 import save_head_checkpoint

HIDDEN = 8  # taille des jetons du faux corps (384 pour le vrai DINOv2-small)
N_CLASSES = 4
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


class FakeBackbone(nn.Module):
    """Faux corps DINOv2 : projette chaque patch 14 x 14 en jetons de taille ``HIDDEN``."""

    def __init__(self) -> None:
        """Crée une convolution (un jeton par patch) et un jeton CLS appris."""
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HIDDEN)
        self.patch = nn.Conv2d(3, HIDDEN, kernel_size=14, stride=14)
        self.cls = nn.Parameter(torch.randn(1, 1, HIDDEN))

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        """Renvoie ``last_hidden_state`` de forme ``(B, 1 + n_patchs, HIDDEN)``."""
        tokens = self.patch(pixel_values).flatten(2).transpose(1, 2)  # (B, n_patchs, HIDDEN)
        cls = self.cls.expand(len(pixel_values), -1, -1)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, tokens], dim=1))


@pytest.fixture
def model() -> Dinov2Classifier:
    """Un classifieur avec le faux corps (aucun téléchargement)."""
    torch.manual_seed(0)
    return Dinov2Classifier(N_CLASSES, backbone=FakeBackbone())


def test_pool_tokens_concatenates_cls_and_patch_mean() -> None:
    """La caractéristique est [CLS, moyenne des patchs] : 2 x d valeurs par image."""
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    pooled = pool_tokens(hidden)
    assert pooled.shape == (2, 6)
    assert torch.equal(pooled[:, :3], hidden[:, 0])
    assert torch.allclose(pooled[:, 3:], hidden[:, 1:].mean(dim=1))


def test_output_shapes(model: Dinov2Classifier) -> None:
    """Un lot (B, 3, 224, 224) donne des caractéristiques (B, 2d) puis des scores (B, classes)."""
    images = torch.randn(3, 3, 224, 224)
    assert model.features(images).shape == (3, 2 * HIDDEN)
    assert model(images).shape == (3, N_CLASSES)


def test_backbone_is_frozen_and_stays_in_eval(model: Dinov2Classifier) -> None:
    """Aucun gradient ne va au corps ; seule la tête est entraînable, même après ``train()``."""
    model.train()
    assert not model.backbone.training
    trainable = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"head"}
    model(torch.randn(2, 3, 224, 224)).sum().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.head.weight.grad is not None


def test_state_dict_holds_only_the_head_and_reloads(model: Dinov2Classifier) -> None:
    """Le state_dict ne contient que ``head.*`` et se recharge dans un autre modèle."""
    state = model.state_dict()
    assert set(state) == {"head.weight", "head.bias"}
    other = Dinov2Classifier(N_CLASSES, backbone=FakeBackbone())
    other.backbone.load_state_dict(model.backbone.state_dict())  # même (faux) corps
    other.load_state_dict(state)
    images = torch.randn(2, 3, 224, 224)
    model.eval()
    other.eval()
    assert torch.allclose(model(images), other(images))
    with pytest.raises(RuntimeError):
        other.load_state_dict({**state, "backbone.cls": torch.zeros(1)})


def test_cache_key_depends_on_mode_views_and_preprocessing() -> None:
    """La clé de cache change avec le mode, le nombre de vues, la graine et le prétraitement."""
    paths = [Path("a.jpg"), Path("b.jpg")]
    head_crop = DataConfig(preprocessing="head_crop")
    legacy = DataConfig(preprocessing="legacy")
    base = feature_cache_stem("train", paths, head_crop, 8, 42, True)
    assert base == feature_cache_stem("train", paths, head_crop, 8, 42, True)  # stable
    others = [
        feature_cache_stem("train", paths, head_crop, 8, 42, False),  # mode d'évaluation
        feature_cache_stem("train", paths, head_crop, 4, 42, True),  # autre nombre de vues
        feature_cache_stem("train", paths, head_crop, 8, 43, True),  # autre graine
        feature_cache_stem("train", paths, legacy, 8, 42, True),  # autre prétraitement
        feature_cache_stem("train", paths[:1], head_crop, 8, 42, True),  # autres images
        feature_cache_stem("train", paths, head_crop, 8, 42, True, revision="autre"),
        feature_cache_stem("val", paths, head_crop, 8, 42, True),  # autre partie
    ]
    assert len({base, *others}) == len(others) + 1


def test_get_features_uses_the_disk_cache(tmp_path: Path, raw_dir: Path) -> None:
    """Une 2e extraction relit le cache (même résultat, sans corps) ; B1 fait K vues distinctes."""
    paths = sorted((raw_dir / "Training" / "glioma").glob("img_*.jpg"))[:4]
    samples = [(p, 0) for p in paths]
    data_cfg = DataConfig(raw_dir=raw_dir, preprocessing="head_crop")
    head_cfg = Dinov2HeadConfig(cache_dir=tmp_path / "cache", extract_batch_size=2)
    backbone = FakeBackbone().eval()

    first = get_features(backbone, "train", samples, CLASSES, data_cfg, head_cfg, 3, True, 42)
    assert first.features.shape == (4, 3, 2 * HIDDEN)
    assert first.computed_views == 3
    assert not torch.equal(first.features[:, 0], first.features[:, 1])  # vues augmentées
    again = get_features(None, "train", samples, CLASSES, data_cfg, head_cfg, 3, True, 42)
    assert again.computed_views == 0  # tout vient du cache
    assert torch.equal(first.features, again.features)
    evaluation = get_features(backbone, "val", samples, CLASSES, data_cfg, head_cfg, 1, False, 42)
    assert evaluation.features.shape == (4, 1, 2 * HIDDEN)
    with pytest.raises(ValueError):  # plusieurs vues sans augmentation : inutile
        get_features(backbone, "val", samples, CLASSES, data_cfg, head_cfg, 2, False, 42)


def test_train_head_decreases_the_loss() -> None:
    """Sur de fausses caractéristiques séparables, la perte baisse et l'accuracy monte."""
    torch.manual_seed(0)
    labels = torch.arange(200) % N_CLASSES
    centers = torch.randn(N_CLASSES, 16) * 3
    x = centers[labels] + torch.randn(200, 16)
    cfg = Dinov2HeadConfig(epochs=30, patience=30, batch_size=32, lr=1e-2)
    result = train_head(x[:150], labels[:150], x[150:], labels[150:], N_CLASSES, cfg, seed=42)
    losses = [row["train_loss"] for row in result.history]
    assert losses[-1] < losses[0] * 0.5
    assert result.best_val_acc > 0.9
    # Même graine, même résultat.
    again = train_head(x[:150], labels[:150], x[150:], labels[150:], N_CLASSES, cfg, seed=42)
    assert again.best_val_loss == pytest.approx(result.best_val_loss)


def test_train_head_stops_early() -> None:
    """Avec des étiquettes aléatoires, la validation ne progresse pas : arrêt avant le maximum."""
    torch.manual_seed(1)
    x, y = torch.randn(80, 16), torch.randint(0, N_CLASSES, (80,))
    vx, vy = torch.randn(40, 16), torch.randint(0, N_CLASSES, (40,))
    cfg = Dinov2HeadConfig(epochs=200, patience=3, batch_size=16, lr=1e-2)
    result = train_head(x, y, vx, vy, N_CLASSES, cfg, seed=0)
    assert len(result.history) < 200


def test_checkpoint_roundtrip(tmp_path: Path, model: Dinov2Classifier, monkeypatch) -> None:
    """Le checkpoint B a les clés du ResNet + architecture, et se recharge dans un modèle B."""
    cfg = Dinov2HeadConfig(epochs=3, patience=3)
    x, y = torch.randn(20, 2 * HIDDEN), torch.arange(20) % N_CLASSES
    result = train_head(x, y, x, y, N_CLASSES, cfg)
    path = tmp_path / "b" / "best.pt"
    save_head_checkpoint(path, result, CLASSES, {"image_size": 224})
    checkpoint = torch.load(path, weights_only=True)
    for key in ("state_dict", "class_names", "phase", "epoch", "val_loss", "val_acc"):
        assert key in checkpoint
    assert checkpoint["architecture"] == "dinov2_s"
    assert set(checkpoint["state_dict"]) == {"head.weight", "head.bias"}

    # build_model_for_checkpoint choisit DINOv2 d'après le checkpoint (faux corps injecté).
    monkeypatch.setattr("brain_mri.dinov2_head.load_model", lambda *a, **k: FakeBackbone())
    reloaded = build_model_for_checkpoint(checkpoint)
    assert isinstance(reloaded, Dinov2Classifier)
    reloaded.load_state_dict(checkpoint["state_dict"])
    assert torch.equal(reloaded.head.weight, result.head.weight)


def test_default_architecture_is_still_resnet50() -> None:
    """Sans réglage, ``build_model`` renvoie toujours un ResNet-50 (comportement inchangé)."""
    assert ModelConfig().architecture == "resnet50"
    resnet = build_model(N_CLASSES, ModelConfig(pretrained=False))
    assert hasattr(resnet, "fc")
    checkpoint = {"class_names": CLASSES}  # ancien checkpoint : pas de clé « architecture »
    assert hasattr(build_model_for_checkpoint(checkpoint), "fc")


def test_train_head_sample_weights_none_or_uniform_changes_nothing() -> None:
    """Sans poids, le résultat est strictement celui d'avant ; des poids égaux donnent le même."""
    torch.manual_seed(2)
    labels = torch.arange(120) % N_CLASSES
    x = torch.randn(N_CLASSES, 16)[labels] * 2 + torch.randn(120, 16)
    cfg = Dinov2HeadConfig(epochs=5, patience=5, batch_size=32, lr=1e-2)
    args = (x[:90], labels[:90], x[90:], labels[90:], N_CLASSES, cfg)
    plain = train_head(*args, seed=7)
    explicit_none = train_head(*args, seed=7, sample_weights=None)
    assert plain.history[-1]["train_loss"] == explicit_none.history[-1]["train_loss"]
    assert torch.equal(plain.head.weight, explicit_none.head.weight)
    uniform = train_head(*args, seed=7, sample_weights=torch.full((90,), 0.01))
    assert torch.allclose(plain.head.weight, uniform.head.weight, atol=1e-5)


def test_train_head_sample_weights_shift_the_decision() -> None:
    """Un poids fort sur une classe rare la fait mieux reconnaître qu'avec des poids égaux."""
    torch.manual_seed(3)
    labels = torch.cat([torch.zeros(180, dtype=torch.long), torch.ones(20, dtype=torch.long)])
    x = torch.randn(200, 4) + labels[:, None].float() * 0.8  # classes qui se chevauchent
    cfg = Dinov2HeadConfig(epochs=40, patience=40, batch_size=32, lr=1e-2)
    weights = torch.where(labels == 1, 9.0, 1.0)  # la classe 1 pèse autant que la classe 0
    plain = train_head(x, labels, x, labels, 2, cfg, seed=0)
    weighted = train_head(x, labels, x, labels, 2, cfg, seed=0, sample_weights=weights)
    recall = lambda head: (head(x[labels == 1]).argmax(dim=1) == 1).float().mean()  # noqa: E731
    assert recall(weighted.head) > recall(plain.head)
