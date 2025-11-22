from __future__ import annotations

from pathlib import Path

import torch  # type: ignore[import]
from fastapi import FastAPI  # type: ignore[import]

from src.inference.service import create_app, load_ensemble

MODEL_DIR = Path("models")
CONFIG_PATH = Path("configs/training.yaml")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def init_app() -> FastAPI:
    models, config = load_ensemble(MODEL_DIR, CONFIG_PATH, DEVICE)
    return create_app(models, config, DEVICE)


app = init_app()
