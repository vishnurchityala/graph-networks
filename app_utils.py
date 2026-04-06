import contextlib
import io
import os
import warnings
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from modules.models.misogyny_model import (
    MisogynyModel,
    MisogynyModelNoGraph,
    MisogynyModelPCAOnly,
)


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data_csv.csv"
LOGS_DIR = ROOT / "logs"
IMAGES_DIR = ROOT / "images"
SAVED_MODELS_DIR = ROOT / "saved_models"

LABEL_NAMES = {
    0: "Kitchen",
    1: "Shopping",
    2: "Working",
    3: "Leadership",
}

DISPLAY_TO_DATASET_LABEL = {
    "Kitchen": "kitchen",
    "Shopping": "shopping",
    "Working": "working",
    "Leadership": "leadership",
}

MODEL_SPECS = {
    "text_image_graph": {
        "title": "Multi-Graph Context Model",
        "summary": "Full multimodal pipeline with text, image, and class-aware graph reasoning.",
        "checkpoint": SAVED_MODELS_DIR / "text_image_graph_20260314_0012_BEST_ep10_acc0.948_f10.901.pth",
        "log": LOGS_DIR / "logs_text_image_graph.csv",
        "model_class": MisogynyModel,
        "components": ["BERT", "OpenCLIP", "LDA graph", "Text graph", "Image graph"],
    },
    "no_graph": {
        "title": "No-Graph Ablation",
        "summary": "BERT + OpenCLIP + PCA/LDA fusion without graph propagation.",
        "checkpoint": SAVED_MODELS_DIR / "no_graph_20260217_1208_BEST_ep18_acc0.906_f10.814.pth",
        "log": LOGS_DIR / "logs_no_graph.csv",
        "model_class": MisogynyModelNoGraph,
        "components": ["BERT", "OpenCLIP", "PCA", "LDA"],
    },
    "pca_only": {
        "title": "PCA-Only Baseline",
        "summary": "Lean baseline using PCA-compressed text and image features only.",
        "checkpoint": SAVED_MODELS_DIR / "pca_only_20260217_1217_BEST_ep10_acc0.951_f10.910.pth",
        "log": LOGS_DIR / "logs_pca_only.csv",
        "model_class": MisogynyModelPCAOnly,
        "components": ["BERT", "OpenCLIP", "PCA"],
    },
}

ARCHITECTURE_IMAGES = [
    {
        "path": IMAGES_DIR / "MISOGYNY-MODEL.jpeg",
        "caption": "High-level architecture sketch from the project README.",
    },
    {
        "path": IMAGES_DIR / "model_design.png",
        "caption": "Supporting design diagram for the multimodal graph pipeline.",
    },
]

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)


def configure_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    try:
        from transformers import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
    except Exception:
        pass


@contextlib.contextmanager
def suppress_noisy_output():
    buffer = io.StringIO()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            yield


def get_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df = df.copy()
    df["sample_id"] = df.index
    df["label_display"] = df["image_label"].str.title()
    return df


def get_label_distribution(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_dataset()

    distribution = (
        df["label_display"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
    )
    return distribution


def get_samples_for_label(label_display: str, limit: int = 24) -> pd.DataFrame:
    df = load_dataset()
    dataset_label = DISPLAY_TO_DATASET_LABEL.get(label_display, "").lower()
    samples = df[df["image_label"] == dataset_label].head(limit).copy()
    samples["sample_name"] = samples.apply(
        lambda row: f"#{row['sample_id']} | {row['image_caption'][:72]}",
        axis=1,
    )
    return samples


def load_training_log(model_key: str) -> pd.DataFrame:
    return pd.read_csv(MODEL_SPECS[model_key]["log"])


def build_experiment_summary() -> pd.DataFrame:
    rows = []
    for model_key, spec in MODEL_SPECS.items():
        log_df = load_training_log(model_key)
        best_row = log_df.loc[log_df["val_f1_macro"].idxmax()]
        rows.append(
            {
                "model_key": model_key,
                "model": spec["title"],
                "best_epoch": int(best_row["epoch"]),
                "val_accuracy": round(float(best_row["val_acc"]), 4),
                "val_precision": round(float(best_row["val_prec"]), 4),
                "val_recall": round(float(best_row["val_rec"]), 4),
                "val_f1_macro": round(float(best_row["val_f1_macro"]), 4),
                "val_f1_weighted": round(float(best_row["val_f1_weighted"]), 4),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(
        by=["val_f1_macro", "val_accuracy"],
        ascending=False,
    )
    return summary_df


def load_architecture_images() -> list[dict[str, str | Path]]:
    return [image for image in ARCHITECTURE_IMAGES if Path(image["path"]).exists()]


def load_pil_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    return IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0)


class GraphProjectPredictor:
    def __init__(self, model_key: str, device: str | None = None):
        configure_runtime()
        if model_key not in MODEL_SPECS:
            raise KeyError(f"Unknown model key: {model_key}")

        self.model_key = model_key
        self.spec = MODEL_SPECS[model_key]
        self.device = device or get_device()

        with suppress_noisy_output():
            self.model = self.spec["model_class"](device=self.device).to(self.device)
            checkpoint = torch.load(
                self.spec["checkpoint"],
                map_location=self.device,
                weights_only=False,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

        self.checkpoint = checkpoint

    def predict(self, caption: str, image: Image.Image) -> dict[str, object]:
        caption = caption.strip()
        if not caption:
            raise ValueError("Caption cannot be empty.")

        image_tensor = preprocess_image(image).to(self.device)

        with torch.no_grad():
            logits = self.model([caption], image_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

        predicted_index = int(probabilities.argmax().item())
        probability_map = {
            LABEL_NAMES[idx]: float(probabilities[idx].item())
            for idx in range(len(LABEL_NAMES))
        }

        return {
            "predicted_index": predicted_index,
            "predicted_label": LABEL_NAMES[predicted_index],
            "confidence": probability_map[LABEL_NAMES[predicted_index]],
            "probabilities": probability_map,
        }
