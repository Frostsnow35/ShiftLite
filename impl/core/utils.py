"""图像读写、预处理与计时工具。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 256


def load_image(path: str | Path, size: int = SIZE) -> Image.Image:
    """读图 → 居中裁剪成正方形 → 缩放到 size×size（与官方 transforms 一致）。"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.BICUBIC)


def to_tensor(img: Image.Image) -> np.ndarray:
    """HWC uint8 → NCHW float32，归一化到 [-1, 1]。"""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.transpose(arr, (2, 0, 1))[None, ...]


def to_image(tensor: np.ndarray) -> Image.Image:
    """NCHW float32 [-1,1] → HWC uint8。"""
    arr = np.asarray(tensor)
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 0.5 + 0.5) * 255.0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def save_image(img: Image.Image, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


class Stopwatch:
    """毫秒级计时器，支持多轮统计。"""

    def __init__(self):
        self.records: list[float] = []

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.records.append((time.perf_counter() - self._t0) * 1000.0)

    @property
    def last(self) -> float:
        return self.records[-1]

    def summary(self, warmup: int = 2) -> dict:
        """丢弃前 warmup 轮，返回均值、标准差与 FPS。"""
        data = np.array(self.records[warmup:] or self.records, dtype=np.float64)
        mean = float(data.mean())
        std = float(data.std(ddof=1)) if data.size > 1 else 0.0
        return {
            "rounds": int(data.size),
            "mean_ms": round(mean, 2),
            "std_ms": round(std, 2),
            "min_ms": round(float(data.min()), 2),
            "max_ms": round(float(data.max()), 2),
            "fps": round(1000.0 / mean, 2) if mean > 0 else 0.0,
            "samples": [round(float(v), 2) for v in self.records],
        }


def file_size_mb(path: str | Path) -> float:
    return round(Path(path).stat().st_size / (1024 * 1024), 2)


def dir_size_mb(root: str | Path) -> float:
    total = 0
    for p in Path(root).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return round(total / (1024 * 1024), 2)
