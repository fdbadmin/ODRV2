from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference.service import create_app, load_checkpoint


def _resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_sample_metadata(labels_csv: Path, image_root: Path, eye: str) -> tuple[Path, float, int]:
    df = pd.read_csv(labels_csv)
    row = df.iloc[0]
    image_column = f"{eye}-Fundus"
    image_path = image_root / row[image_column]
    age = float(row["Patient Age"])
    sex = 0 if str(row["Patient Sex"]).lower().startswith("m") else 1
    return image_path, age, sex


def smoke_test(checkpoint: Path, config: Path, labels_csv: Path, image_root: Path, eye: str) -> None:
    device = _resolve_device()
    model, training_cfg = load_checkpoint(checkpoint, config, device)
    app = create_app(model, training_cfg, device)
    client = TestClient(app)

    image_path, age, sex = _load_sample_metadata(labels_csv, image_root, eye)

    with image_path.open("rb") as handle:
        response = client.post(
            "/predict",
            files={"file": (image_path.name, handle, "image/jpeg")},
            params={"age": age, "sex": sex},
        )

    response.raise_for_status()
    payload = response.json()
    probabilities = payload.get("probabilities", [])

    print("Smoke test response")
    print(f"Checkpoint: {checkpoint}")
    print(f"Sample image: {image_path}")
    print(f"Age: {age}, Sex code: {sex}")
    print("Probabilities: " + ", ".join(f"{prob:.4f}" for prob in probabilities))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test the FastAPI inference service")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained checkpoint")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"), help="Path to the training config")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("data/processed/odir_eye_labels.csv"),
        help="Processed labels CSV used for training",
    )
    parser.add_argument("--image-root", type=Path, default=Path("RAW DATA FULL"), help="Fundus image root directory")
    parser.add_argument("--eye", choices=("Left", "Right"), default="Right", help="Which eye to sample")
    args = parser.parse_args()
    smoke_test(args.checkpoint, args.config, args.labels_csv, args.image_root, args.eye)
