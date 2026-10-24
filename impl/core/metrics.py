"""结构保真度量（无需额外模型，纯 numpy 实现）。

对外提供：
    ssim(a, b)          —— 全图结构相似度
    background_ssim     —— 掩码外区域的结构相似度
    edge_ssim           —— Canny 式边缘图的结构相似度（只比结构，不比纹理）
    mask_iou / mask_dice—— 给定两份实例掩码时的重叠度

说明：LPIPS / FID / KID 需要 torchvision 与感知网络权重，
本模块**不伪造**它们，也不提供近似替代品。界面上的 LPIPS 数值来自
lpips_three_way()，它明确带 "note": "proxy" 标记，不得写进报告。
要拿到真实 LPIPS，需接入 torchvision + 官方线性权重（见 README「待接入项」）。
"""

from __future__ import annotations

import numpy as np


# ---------- 基础工具 ----------

def to_gray(img: np.ndarray) -> np.ndarray:
    """HWC RGB(float 0~255) → HW 灰度(0~1)。"""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114])
    return np.clip(arr / 255.0, 0.0, 1.0)


def _gaussian(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2
    k = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _conv_sep(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """可分离卷积（一次横向、一次纵向），比逐像素循环快两个数量级。"""
    pad = len(kernel) // 2
    h, w = img.shape

    p = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.zeros_like(img)
    for i, kw in enumerate(kernel):
        tmp += kw * p[:, i:i + w]

    p = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(img)
    for i, kw in enumerate(kernel):
        out += kw * p[i:i + h, :]
    return out


# ---------- SSIM ----------

def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """标准 SSIM（高斯窗 11×11，σ=1.5）。"""
    x = to_gray(a) if a.ndim == 3 else np.asarray(a, dtype=np.float64)
    y = to_gray(b) if b.ndim == 3 else np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"尺寸不一致：{x.shape} vs {y.shape}")

    k = _gaussian()
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2

    mu_x = _conv_sep(x, k)
    mu_y = _conv_sep(y, k)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sig_x2 = _conv_sep(x * x, k) - mu_x2
    sig_y2 = _conv_sep(y * y, k) - mu_y2
    sig_xy = _conv_sep(x * y, k) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
    return round(float(np.clip((num / den).mean(), -1.0, 1.0)), 6)


def background_ssim(a: np.ndarray, b: np.ndarray,
                    foreground_mask: np.ndarray) -> float:
    """只在掩码外（背景）区域计算 SSIM，验证背景有没有被生成器改写。

    foreground_mask 为 0/1 的 HW 数组，1 表示前景（动物所在像素）。
    """
    x = to_gray(a) if a.ndim == 3 else np.asarray(a, dtype=np.float64)
    y = to_gray(b) if b.ndim == 3 else np.asarray(b, dtype=np.float64)
    bg = (foreground_mask == 0)

    if bg.sum() < 64:
        raise ValueError("背景区域太小，无法计算")

    # 把背景之外涂成均值，避免前景像素干扰局部统计
    xf = np.where(bg, x, x[bg].mean())
    yf = np.where(bg, y, y[bg].mean())
    return ssim(xf, yf)


# ---------- 边缘图 SSIM ----------

def edge_map(img: np.ndarray) -> np.ndarray:
    """Sobel 梯度幅值，归一化到 0~1，作为 Canny 边缘的轻量替代。

    跨域（马→斑马）纹理变化大，直接比像素会被纹理差异污染；
    抽成边缘后只比结构。
    """
    g = to_gray(img) if img.ndim == 3 else np.asarray(img, dtype=np.float64)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return mag / (mag.max() + 1e-8)


def edge_ssim(a: np.ndarray, b: np.ndarray) -> float:
    return ssim(edge_map(a), edge_map(b))


# ---------- 掩码重叠度 ----------

def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return round(float(inter / union), 6) if union else 0.0


def mask_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    s = a.sum() + b.sum()
    return round(float(2 * np.logical_and(a, b).sum() / s), 6) if s else 0.0


# ---------- 像素级 ----------

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return round(float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).mean()), 4)


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    if mse <= 1e-12:
        return 99.0
    return round(float(10 * np.log10(data_range ** 2 / mse)), 2)


def lpips_three_way(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> dict:
    """LPIPS 三对归因的**占位实现**（代理值）。

    用「结构相似度 + 边缘相似度」的线性组合做近似，**仅供界面联调**；
    真实 LPIPS 在 core/metrics_deep.py（`cli.py quality --deep`），
    正式报告中的数字必须来自那边，本函数的输出不得写进报告。
    """
    def pair(x, y):
        s = (ssim(x, y) + edge_ssim(x, y)) / 2
        return round(1.0 - s, 6)

    return {
        "L_ab": pair(a, b),
        "L_bc": pair(b, c),
        "L_ac": pair(a, c),
        "note": "proxy",
    }
