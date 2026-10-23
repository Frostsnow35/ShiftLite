"""A / B / C 三组推理后端。

A = PyTorch FP32（基线）
B = OpenVINO FP32（隔离"换引擎"的收益）
C = OpenVINO INT8（隔离"压精度"的收益）

三组接收同一个 NCHW float32 输入，返回同形状输出，保证可比。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import SIZE

GROUP_LABEL = {
    "A": "PyTorch FP32",
    "B": "OpenVINO FP32",
    "C": "OpenVINO INT8",
}

# ⚠️ 关键坑：OpenVINO CPU 插件在支持的 CPU 上会默认按 bfloat16 推理。
# 实测（i5-1340P）：默认配置下 B 组与 PyTorch 的最大逐像素差达 0.037（约 2%），
# 显式指定 f32 后降到 6e-6。若不加这一项，B 组根本不是 FP32 基线，
# LPIPS(a,b) 会被误判成"转换有 bug"，整条归因链就断了。
OV_CONFIG_FP32 = {"INFERENCE_PRECISION_HINT": "f32"}


class Runtime:
    def __init__(self, weights: Path, xml_fp32: Path, xml_int8: Path,
                 device: str = "CPU"):
        self.weights = Path(weights)
        self.xml_fp32 = Path(xml_fp32)
        self.xml_int8 = Path(xml_int8)
        self.device = device
        self._cache: dict[str, object] = {}
        self._core = None

    # ---------- 可用性 ----------

    def availability(self) -> dict:
        return {
            "A": self.weights.exists(),
            "B": self.xml_fp32.exists(),
            "C": self.xml_int8.exists(),
        }

    @property
    def core(self):
        if self._core is None:
            import openvino as ov
            self._core = ov.Core()
        return self._core

    # ---------- 加载 ----------

    def _torch(self):
        if "net" not in self._cache:
            from .model import load_generator
            self._cache["net"] = load_generator(str(self.weights), self.device)
        return self._cache["net"]

    def _ov(self, group: str):
        key = f"ov_{group}"
        if key not in self._cache:
            path = self.xml_fp32 if group == "B" else self.xml_int8
            if not path.exists():
                raise FileNotFoundError(f"{group} 组模型不存在：{path}")
            # B 组必须锁死 f32；C 组模型已量化为 INT8，配置只影响未量化部分
            compiled = self.core.compile_model(str(path), self.device, OV_CONFIG_FP32)
            self._cache[key] = compiled
        return self._cache[key]

    def precision(self, group: str) -> str:
        """返回该组编译后实际的推理精度，便于在报告里如实记录。"""
        if group == "A":
            return "float32"
        try:
            return str(self._ov(group).get_property("INFERENCE_PRECISION_HINT")).split("'")[-2]
        except Exception:
            return "unknown"

    # ---------- 推理 ----------

    def infer(self, x: np.ndarray, group: str = "A") -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if group == "A":
            import torch
            net = self._torch()
            with torch.no_grad():
                return net(torch.from_numpy(x)).numpy()
        return np.asarray(self._ov(group)(x)[0])

    def warmup(self, group: str, rounds: int = 2) -> None:
        x = np.zeros((1, 3, SIZE, SIZE), dtype=np.float32)
        for _ in range(rounds):
            self.infer(x, group)

    # ---------- 体积 ----------

    def model_size_mb(self, group: str) -> float | None:
        if group == "A":
            if not self.weights.exists():
                return None
            return round(self.weights.stat().st_size / 1024 / 1024, 2)
        path = self.xml_fp32 if group == "B" else self.xml_int8
        if not path.exists():
            return None
        bin_path = path.with_suffix(".bin")
        total = path.stat().st_size + (bin_path.stat().st_size if bin_path.exists() else 0)
        return round(total / 1024 / 1024, 2)
