from __future__ import annotations

from pathlib import Path

import torch  # type: ignore[import]
from fastapi import FastAPI  # type: ignore[import]

from src.inference.service import create_app, load_checkpoint

MODEL_PATH = Path("models/model.ckpt")
CONFIG_PATH = Path("configs/training.yaml")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_app() -> FastAPI:
    model = load_checkpoint(MODEL_PATH, CONFIG_PATH, DEVICE)
    return create_app(model, DEVICE)


app = init_app()
