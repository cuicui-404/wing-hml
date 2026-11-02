from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for key, value in b.items():
        out[key] = merge(out.get(key, {}), value) if isinstance(value, dict) else value
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "extends" in cfg:
        base = load_config(path.parent / cfg.pop("extends"))
        cfg = merge(base, cfg)
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value: {item}")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(text: str, seed: int = 0) -> int:
    # 64-bit seeds avoid birthday collisions in a 100,000-task corpus.
    return int(hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()[:16], 16)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as f:
        temporary = Path(f.name)
        f.write(text + "\n")
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int, threads: int = 1):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Exact reproducibility is guaranteed on the same software/hardware stack,
    # not between CPU, MPS and different CUDA architectures.
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(name: str = "auto"):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check the PyTorch wheel and driver.")
    return device
