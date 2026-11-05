"""Export ONNX du classifieur, pour faire tourner le modèle ET Grad-CAM dans un navigateur.

Le fichier produit (``model.onnx``) contient tout le calcul, **sans gradient** :
    - entrée ``image`` : ``(1, 3, 224, 224)`` en décimaux 32 bits, **déjà normalisée**
      (moyenne et écart-type d'ImageNet : le prétraitement est fait côté navigateur) ;
    - sortie ``probs`` : ``(1, 4)``, les probabilités (softmax) ;
    - sortie ``cams`` : ``(1, 4, 7, 7)``, une carte par classe, ReLU appliquée mais **pas**
      normalisée (le navigateur normalisera, comme :func:`brain_mri.explain._normalize`).

Pourquoi Grad-CAM sans gradient (justification mathématique) : la tête du modèle est une seule
couche linéaire placée après un pooling moyen. Avec ``A`` les activations de ``layer4``
(``K = 2048`` cartes de taille ``H x W``), le score de la classe ``c`` est
``y_c = sum_k w[c,k] * mean_{h,w}(A[k,h,w]) + b_c``. Son gradient par rapport à ``A[k,h,w]``
vaut donc ``w[c,k] / (H*W)`` : il est **constant** (il ne dépend ni de l'image ni de la
position). Le poids de Grad-CAM, la moyenne spatiale du gradient, est alors
``alpha_k = w[c,k] / (H*W)`` et la carte vaut ``ReLU(sum_k alpha_k * A[k])``. C'est
exactement la carte CAM ``ReLU(sum_k w[c,k] * A[k])`` divisée par la constante ``H*W``, qui
disparaît à la normalisation (division par le maximum). Le biais ne joue aucun rôle (sa
dérivée est nulle). On calcule donc la carte avec une simple multiplication par les poids de
``fc``, sans rétropropagation.

Version allégée : :func:`quantize_onnx` produit ``model.int8.onnx`` (quantification statique
int8 des convolutions, calibrée sur des images d'entraînement). La vérification
(:func:`run_verification`) compare les fichiers à PyTorch et décide lequel recommander.

Utilisation (depuis ``brain-tumor-classifier/ml/``, après ``train``) ::

    uv run python -m brain_mri.export_onnx             # écrit model.onnx et model.int8.onnx
    uv run python -m brain_mri.export_onnx --verify    # + vérifications (onnx_check.json)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
import truststore
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
from torch import nn

from brain_mri.config import PROJECT_ROOT, Config, load_config
from brain_mri.data import build_dataloaders
from brain_mri.explain import gradcam_captum
from brain_mri.predict import Predictor

LOGGER = logging.getLogger(__name__)

DEFAULT_EXPORT_DIR = PROJECT_ROOT / "models" / "export"  # à côté de l'export safetensors
FP32_NAME = "model.onnx"  # modèle complet, décimaux 32 bits
INT8_NAME = "model.int8.onnx"  # modèle allégé, quantifié en int8
CHECK_NAME = "onnx_check.json"  # résultat des vérifications (dans le dossier des figures)
INPUT_NAME = "image"
OUTPUT_NAMES = ("probs", "cams")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Seuils de la décision (consigne de l'étape 10a)
MAX_PROB_GAP = 1e-4  # écart max toléré entre ONNX fp32 et PyTorch (probabilités et cartes)
MAX_ACCURACY_DROP = 0.5  # baisse d'accuracy tolérée pour l'int8, en points de pourcentage
MIN_CAM_CORRELATION = 0.95  # corrélation moyenne minimale des cartes int8 / fp32


class OnnxExportModule(nn.Module):
    """Enveloppe du ResNet-50 entraîné : sorties ``probs`` et ``cams``, sans gradient.

    Args:
        model: Le ResNet-50 (torchvision) avec sa tête ``fc`` entraînée.
    """

    def __init__(self, model: nn.Module) -> None:
        """Garde le modèle en mode évaluation (BatchNorm figée) ; rien n'est copié."""
        super().__init__()
        self.model = model.eval()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calcule les probabilités et les cartes de chaque classe.

        Args:
            image: Image normalisée, forme ``(N, 3, H, W)``.

        Returns:
            ``(probs, cams)`` : ``probs`` de forme ``(N, classes)``, ``cams`` de forme
            ``(N, classes, h, w)`` (7 x 7 pour une image de 224), valeurs positives.
        """
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(image))))  # même enchaînement que ResNet.forward
        activations = m.layer4(m.layer3(m.layer2(m.layer1(x))))  # A : (N, 2048, h, w)
        logits = m.fc(activations.mean(dim=(2, 3)))  # pooling moyen puis tête linéaire
        probs = F.softmax(logits, dim=1)
        # CAM = Grad-CAM ici (voir l'en-tête) : somme des cartes pondérée par les poids de fc,
        # SANS biais ; la ReLU garde ce qui pousse vers la classe.
        cams = torch.relu(torch.einsum("ck,nkhw->nchw", m.fc.weight, activations))
        return probs, cams


def build_export_module(model: nn.Module) -> OnnxExportModule:
    """Construit l'enveloppe d'export autour d'un modèle entraîné.

    Args:
        model: Le ResNet-50 avec ses poids (mis en ``eval()`` par l'enveloppe).

    Returns:
        L'enveloppe, prête à être exportée ou appelée directement.
    """
    return OnnxExportModule(model).eval()  # eval aussi pour l'enveloppe : l'exporteur le vérifie


def export_onnx(model: nn.Module, path: Path, image_size: int = 224) -> Path:
    """Exporte le modèle en ONNX (fp32, un seul fichier, lot de taille 1).

    On utilise l'exporteur actuel de PyTorch (``dynamo=True``, valeur par défaut en 2.15) avec
    l'opset recommandé par PyTorch (``opset_version=None``), et ``external_data=False`` pour
    que les poids (environ 94 Mo) restent dans le fichier unique.

    Args:
        model: Le ResNet-50 entraîné.
        path: Fichier ``.onnx`` à écrire (le dossier est créé si besoin).
        image_size: Côté de l'image d'entrée (224 en production, plus petit pour les tests).

    Returns:
        Le chemin du fichier écrit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    module = build_export_module(model)
    example = torch.zeros(1, 3, image_size, image_size)
    torch.onnx.export(
        module,
        (example,),
        path,
        input_names=[INPUT_NAME],
        output_names=list(OUTPUT_NAMES),
        dynamo=True,
        external_data=False,
        verbose=False,  # sans messages à emojis de l'exporteur (plantent sous Windows redirigé)
    )
    LOGGER.info("ONNX fp32 écrit : %s (%.1f Mo)", path, path.stat().st_size / 1e6)
    return path


class ImageCalibrationReader(CalibrationDataReader):
    """Fournit les images de calibration à la quantification, une par une.

    Args:
        arrays: Images prétraitées, chacune de forme ``(1, 3, H, W)`` en float32.
        input_name: Nom de l'entrée du graphe ONNX.
    """

    def __init__(self, arrays: list[np.ndarray], input_name: str = INPUT_NAME) -> None:
        """Prépare l'itérateur sur les images."""
        self._iterator = iter([{input_name: a} for a in arrays])

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Renvoie l'image suivante (``None`` quand il n'y en a plus)."""
        return next(self._iterator, None)


def quantize_onnx(fp32_path: Path, int8_path: Path, calibration: list[np.ndarray]) -> Path:
    """Quantifie statiquement les convolutions du modèle en int8.

    Choix (documentation ONNX Runtime, « Choosing parameters for CPU inference ») : format
    ``QDQ`` (celui que les noyaux CPU et WASM optimisent), activations ``uint8`` et poids
    ``int8`` par canal avec ``reduce_range`` (précaution sur les processeurs sans VNNI, donc
    la plupart des navigateurs). Seules les convolutions sont quantifiées : la tête (``fc``)
    et le calcul des cartes restent en fp32, car ils pèsent très peu et c'est là que la
    précision compte (probabilités et cartes).

    Args:
        fp32_path: Modèle ONNX fp32 d'entrée.
        int8_path: Fichier quantifié à écrire.
        calibration: Images de calibration (liste de tableaux ``(1, 3, H, W)``).

    Returns:
        Le chemin du fichier quantifié.
    """
    with tempfile.TemporaryDirectory() as tmp:
        prepared = Path(tmp) / "prepared.onnx"
        # Pré-traitement recommandé avant toute quantification (inférence de formes, fusions).
        quant_pre_process(str(fp32_path), str(prepared), skip_symbolic_shape=True)
        quantize_static(
            prepared,
            int8_path,
            ImageCalibrationReader(calibration),
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=["Conv"],
            per_channel=True,
            reduce_range=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )
    LOGGER.info("ONNX int8 écrit : %s (%.1f Mo)", int8_path, int8_path.stat().st_size / 1e6)
    return int8_path


def load_predictor(config: Config) -> Predictor:
    """Charge le meilleur modèle (``models/best.pt``) sur CPU.

    Args:
        config: Configuration complète.

    Returns:
        Le Predictor (modèle en ``eval()``, prétraitement, noms des classes).

    Raises:
        FileNotFoundError: Si ``best.pt`` n'existe pas.
    """
    path = config.training.checkpoint_path
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {path} (lance d'abord train)")
    return Predictor.from_checkpoint(path, torch.device("cpu"))


def list_images(folder: Path, class_names: list[str]) -> list[tuple[Path, int]]:
    """Liste les images d'un dossier ``Training`` ou ``Testing`` avec leur classe.

    Args:
        folder: Dossier contenant un sous-dossier par classe.
        class_names: Noms des classes, dans l'ordre des indices du modèle.

    Returns:
        ``(chemin, indice de la classe)``, triés par chemin (ordre reproductible).
    """
    items = []
    for index, name in enumerate(class_names):
        files = sorted(p for p in (folder / name).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        items += [(p, index) for p in files]
    return items


def evaluation_images(config: Config) -> list[tuple[Path, int]]:
    """Donne les images du jeu de test **utilisé par l'évaluation** (doublons écartés).

    C'est le même jeu que celui des scores de ``evaluation.json`` (1 584 images, et non les
    1 600 fichiers du dossier ``Testing``, car 16 doublons de l'entraînement sont retirés).

    Args:
        config: Configuration complète.

    Returns:
        ``(chemin, indice de la classe)`` de chaque image de test.
    """
    dataset = build_dataloaders(config.data).test_loader.dataset
    return [(Path(path), label) for path, label in dataset.samples]


def preprocess(predictor: Predictor, path: Path) -> torch.Tensor:
    """Charge une image et la prétraite comme à l'inférence.

    Args:
        predictor: Le Predictor (fournit la chaîne de prétraitement).
        path: Chemin de l'image.

    Returns:
        Le tenseur ``(1, 3, H, W)``.
    """
    image = Image.open(path).convert("RGB")  # les IRM sont parfois en niveaux de gris
    return predictor.transform(image).unsqueeze(0)


def calibration_arrays(
    predictor: Predictor, training_dir: Path, count: int = 100, seed: int = 42
) -> list[np.ndarray]:
    """Choisit ``count`` images d'entraînement (réparties entre les classes) pour calibrer.

    Args:
        predictor: Le Predictor (prétraitement et classes).
        training_dir: Dossier ``Training``.
        count: Nombre total d'images.
        seed: Graine du tirage (reproductible).

    Returns:
        Les images prétraitées, tableaux float32 ``(1, 3, H, W)``.
    """
    rng = random.Random(seed)
    per_class = count // len(predictor.class_names)
    chosen: list[Path] = []
    for index in range(len(predictor.class_names)):
        files = [
            p for p, label in list_images(training_dir, predictor.class_names) if label == index
        ]
        chosen += rng.sample(files, per_class)
    return [preprocess(predictor, p).numpy() for p in chosen]


def make_session(path: Path) -> ort.InferenceSession:
    """Ouvre un fichier ONNX avec ONNX Runtime (CPU).

    Args:
        path: Fichier ``.onnx``.

    Returns:
        La session d'inférence.
    """
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run_session(session: ort.InferenceSession, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fait tourner une session sur une image.

    Args:
        session: Session ONNX Runtime.
        x: Image prétraitée ``(1, 3, H, W)``.

    Returns:
        ``(probs (classes,), cams (classes, h, w))`` pour cette image.
    """
    probs, cams = session.run(list(OUTPUT_NAMES), {INPUT_NAME: x})
    return probs[0], cams[0]


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """Normalise une carte par son maximum (même règle que ``explain._normalize``).

    Args:
        cam: Carte à valeurs positives.

    Returns:
        La carte divisée par son maximum (inchangée si elle est nulle).
    """
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def compare_with_pytorch(predictor: Predictor, onnx_path: Path, images: list[Path]) -> dict:
    """Compare ONNX fp32 et PyTorch : probabilités et cartes (classe prédite).

    Args:
        predictor: Le Predictor PyTorch.
        onnx_path: Fichier ONNX fp32.
        images: Images de test.

    Returns:
        Un dictionnaire : nombre d'images, écart max des probabilités, écart max des cartes
        normalisées 7 x 7 (ONNX contre ``gradcam_captum``).
    """
    session = make_session(onnx_path)
    prob_gap, cam_gap = 0.0, 0.0
    for path in images:
        x = preprocess(predictor, path)
        with torch.no_grad():
            expected = F.softmax(predictor.model(x), dim=1)[0].numpy()
        probs, cams = run_session(session, x.numpy())
        prob_gap = max(prob_gap, float(np.abs(probs - expected).max()))
        target = int(expected.argmax())  # on explique la classe prédite
        reference = gradcam_captum(predictor.model, x, target).numpy()  # déjà normalisée
        mine = normalize_cam(cams[target])
        cam_gap = max(cam_gap, float(np.abs(mine - reference).max()))
    return {"images": len(images), "max_prob_gap": prob_gap, "max_cam_gap": cam_gap}


def cam_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Corrélation de Pearson entre deux cartes (1 si elles sont toutes deux constantes).

    Args:
        a: Première carte.
        b: Seconde carte.

    Returns:
        La corrélation entre -1 et 1.
    """
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_files(
    predictor: Predictor, fp32_path: Path, int8_path: Path, test_items: list[tuple[Path, int]]
) -> dict:
    """Mesure fp32 et int8 sur tout le jeu de test (accuracy, écart de probabilités, cartes).

    Args:
        predictor: Le Predictor (prétraitement).
        fp32_path: Fichier ONNX fp32.
        int8_path: Fichier ONNX int8.
        test_items: ``(chemin, classe)`` de chaque image de test.

    Returns:
        Un dictionnaire avec les accuracies (en %), l'écart moyen et maximal des probabilités,
        et la corrélation moyenne des cartes de la classe prédite par le fp32.
    """
    fp32, int8 = make_session(fp32_path), make_session(int8_path)
    ok32 = ok8 = 0
    gaps, correlations = [], []
    for i, (path, label) in enumerate(test_items, start=1):
        x = preprocess(predictor, path).numpy()
        p32, c32 = run_session(fp32, x)
        p8, c8 = run_session(int8, x)
        ok32 += int(p32.argmax() == label)
        ok8 += int(p8.argmax() == label)
        gaps.append(float(np.abs(p32 - p8).mean()))
        target = int(p32.argmax())
        correlations.append(cam_correlation(normalize_cam(c32[target]), normalize_cam(c8[target])))
        if i % 200 == 0:
            LOGGER.info("%d / %d images évaluées", i, len(test_items))
    n = len(test_items)
    return {
        "images": n,
        "accuracy_fp32": 100 * ok32 / n,
        "accuracy_int8": 100 * ok8 / n,
        "mean_prob_gap": float(np.mean(gaps)),
        "max_mean_prob_gap_per_image": float(np.max(gaps)),
        "mean_cam_correlation": float(np.mean(correlations)),
    }


def decide(accuracy_drop: float, cam_correlation_mean: float) -> str:
    """Applique la règle de décision : int8 recommandé ou fp32.

    Args:
        accuracy_drop: Baisse d'accuracy de l'int8 en points (fp32 moins int8).
        cam_correlation_mean: Corrélation moyenne des cartes int8 / fp32.

    Returns:
        ``"int8"`` si la baisse est au plus de 0,5 point ET la corrélation supérieure à 0,95,
        sinon ``"fp32"``.
    """
    ok = accuracy_drop <= MAX_ACCURACY_DROP and cam_correlation_mean > MIN_CAM_CORRELATION
    return "int8" if ok else "fp32"


def run_verification(config: Config, out_dir: Path, n_compare: int = 20) -> dict:
    """Lance toutes les vérifications et écrit ``docs/figures/onnx_check.json``.

    Args:
        config: Configuration complète.
        out_dir: Dossier contenant ``model.onnx`` et ``model.int8.onnx``.
        n_compare: Nombre d'images de test comparées à PyTorch (au moins 20).

    Returns:
        Le dictionnaire écrit dans le fichier JSON.
    """
    predictor = load_predictor(config)
    test_items = evaluation_images(config)
    fp32_path, int8_path = out_dir / FP32_NAME, out_dir / INT8_NAME

    rng = random.Random(config.data.seed)
    sample = [p for p, _ in rng.sample(test_items, n_compare)]
    comparison = compare_with_pytorch(predictor, fp32_path, sample)
    LOGGER.info("ONNX fp32 contre PyTorch : %s", comparison)

    measures = evaluate_files(predictor, fp32_path, int8_path, test_items)
    drop = measures["accuracy_fp32"] - measures["accuracy_int8"]
    decision = decide(drop, measures["mean_cam_correlation"])
    result = {
        "opset": max(
            o.version for o in onnx.load(fp32_path, load_external_data=False).opset_import
        ),
        "versions": {"onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "fp32_vs_pytorch": comparison
        | {"prob_gap_ok": comparison["max_prob_gap"] <= MAX_PROB_GAP}
        | {"cam_gap_ok": comparison["max_cam_gap"] <= MAX_PROB_GAP},
        "test_set": measures,
        "accuracy_drop_points": drop,
        "size_mb": {
            "fp32": fp32_path.stat().st_size / 1e6,
            "int8": int8_path.stat().st_size / 1e6,
        },
        "rule": (
            f"int8 si baisse <= {MAX_ACCURACY_DROP} point et corrélation > {MIN_CAM_CORRELATION}"
        ),
        "recommended": decision,
    }
    target = config.training.figures_dir / CHECK_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Résultat écrit : %s (recommandé : %s)", target, decision)
    return result


def main() -> None:
    """Point d'entrée en ligne de commande : exporte, quantifie et (option) vérifie."""
    parser = argparse.ArgumentParser(description="Exporte le modèle en ONNX (fp32 et int8).")
    parser.add_argument("--config", type=Path, default=None, help="Fichier YAML de configuration")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_EXPORT_DIR, help="Dossier de sortie"
    )
    parser.add_argument("--verify", action="store_true", help="Lance les vérifications")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    truststore.inject_into_ssl()
    config = load_config(args.config) if args.config else load_config()

    predictor = load_predictor(config)
    fp32_path = export_onnx(predictor.model, args.out_dir / FP32_NAME, predictor.image_size)
    calibration = calibration_arrays(predictor, config.data.raw_dir / "Training")
    quantize_onnx(fp32_path, args.out_dir / INT8_NAME, calibration)
    if args.verify:
        print(json.dumps(run_verification(config, args.out_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
