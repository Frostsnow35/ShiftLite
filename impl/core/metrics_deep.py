"""需要外部感知网络的深度指标：真实 LPIPS / FID / KID / 实例掩码 / 斑马判别。

与 `core/metrics.py` 的分工
--------------------------
`metrics.py` 是纯 numpy 的结构指标（不依赖模型权重），这里是需要预训练感知
网络的指标。**本模块不做任何近似替代**：依赖缺失就抛错，绝不返回代理值——
`metrics.lpips_three_way()` 那份带 `"note": "proxy"` 的结果只能用于界面联调，
正式报告里的数字必须来自本模块。

方案口径（开题报告 §四）
----------------------
- LPIPS：AlexNet 版本，官方线性权重，三个方向 L_ab / L_bc / L_ac。
- FID / KID：InceptionV3 的 2048 维池化特征，与真实目标域比较。
- 结构保真：**Canny** 边缘 SSIM（不是梯度幅值）+ **Mask R-CNN** 实例掩码 IoU
  + 掩码外背景 SSIM + **ResNet-50** 斑马判别准确率。
- 异常样本单独登记，不混在全量均值里。

用法：
    python -m core.metrics_deep --self-test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# ImageNet 里包含 "zebra" 的类通常不止一个（多个斑马亚种/プリント），
# 判别任务按类名匹配后求和，避免漏掉变体。
_TARGET_LABEL_SUBSTRINGS = ("zebra",)
_MASK_LABEL = "zebra"


# ---------------- 通用工具 ----------------

def to_batch(images: list[np.ndarray]) -> torch.Tensor:
    """HWC uint8/float RGB(0~255) 列表 → NCHW float32 [-1,1]，LPIPS 的输入约定。"""
    arrs = [np.asarray(im, dtype=np.float32) for im in images]
    arrs = [np.transpose(a, (2, 0, 1)) for a in arrs]
    return torch.from_numpy(np.stack(arrs)) / 127.5 - 1.0


def _require(module: str, pip_name: str | None = None) -> None:
    raise SystemExit(
        f"缺依赖 {module}；装：pip install {pip_name or module}")


def _cache(name: str):
    """简易惰性单例，避免每次调用都重新下载/加载权重。"""
    store = {}

    def deco(fn):
        def wrapper(*args, **kwargs):
            if name not in store:
                store[name] = fn(*args, **kwargs)
            return store[name]
        return wrapper
    return deco


# ---------------- LPIPS（真实值） ----------------

@_cache("lpips_alex")
def lpips_model(net: str = "alex"):
    try:
        import lpips
    except ImportError:
        _require("lpips")
    return lpips.LPIPS(net=net).eval()


def lpips_pair(a: list[np.ndarray], b: list[np.ndarray],
               net: str = "alex") -> list[float]:
    """两批图之间的真实 LPIPS（逐张）。"""
    model = lpips_model(net)
    xa, xb = to_batch(a), to_batch(b)
    if xa.shape != xb.shape:
        raise ValueError(f"批量不一致：{xa.shape} vs {xb.shape}")
    with torch.no_grad():
        d = model(xa, xb)
    return [round(float(v), 6) for v in d.flatten()]


def lpips_three_way_real(a: list[np.ndarray], b: list[np.ndarray],
                         c: list[np.ndarray],
                         net: str = "alex") -> dict:
    """LPIPS 三对归因的真实值。

    a=A 组 PyTorch FP32 输出，b=B 组 OV FP32 输出，c=C 组 OV INT8 输出。
    返回逐张均值与每对的距离列表。
    """
    pairs = {"L_ab": (a, b), "L_bc": (b, c), "L_ac": (a, c)}
    out = {"net": net, "n_images": len(a)}
    for key, (x, y) in pairs.items():
        vals = lpips_pair(list(x), list(y), net)
        out[key] = round(float(np.mean(vals)), 6)
        out[key + "_per_image"] = vals
    return out


# ---------------- FID / KID ----------------

@_cache("inception")
def inception_model():
    from torchvision.models import Inception_V3_Weights, inception_v3
    try:
        m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                         aux_logits=True)
    except TypeError:  # torchvision 新版不再需要 aux_logits
        m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    return m.eval()


def inception_features(images: list[np.ndarray], batch_size: int = 16,
                       size: int = 299) -> np.ndarray:
    """一批图 → InceptionV3 的 2048 维池化特征（N, 2048）。

    FID 官方实现用 299×299 输入，这里按模型要求重采样，不用生成任务的 256。
    """
    try:
        from PIL import Image
    except ImportError:
        _require("Pillow")

    m = inception_model()
    feats = []
    for i in range(0, len(images), batch_size):
        batch = []
        for im in images[i:i + batch_size]:
            arr = np.asarray(im)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr).resize((size, size), Image.BICUBIC)
            arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
            batch.append(np.transpose(arr, (2, 0, 1)))
        x = torch.from_numpy(np.stack(batch))
        with torch.no_grad():
            feats.append(m(x).numpy())
    return np.concatenate(feats, axis=0)


def _sqrtm_product(sigma1: np.ndarray, sigma2: np.ndarray) -> float:
    """tr((Σ1·Σ2)^(1/2))，纯 numpy：把乘积化为对称形式后取特征根。

    scipy.linalg.sqrtm 在 Windows CPU 环境常碰到缺 BLAS 的麻烦，这里用
    eigvalsh 直接算迹，数值上够用（协方差矩阵都是对称半正定的）。
    """
    s1, u1 = np.linalg.eigh(sigma1)
    # u1 @ diag(sqrt(s1)) @ u1.T
    sqrt_s1 = (u1 * np.sqrt(np.clip(s1, 0, None))) @ u1.T
    prod = sqrt_s1 @ sigma2 @ sqrt_s1
    ev = np.linalg.eigvalsh((prod + prod.T) / 2.0)
    return float(np.sqrt(np.clip(ev, 0, None)).sum())


def fid(feat_a: np.ndarray, feat_b: np.ndarray, eps: float = 1e-6) -> float:
    """两个特征集合之间的 Frechet Inception Distance。

    要求 n >= 2；特征维数太小（< 2048）时协方差奇异，用 eps 补对角线。
    """
    a = np.asarray(feat_a, dtype=np.float64)
    b = np.asarray(feat_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("特征必须是 (N, D)")
    if min(len(a), len(b)) < 2:
        raise ValueError("FID 至少需要每侧 2 个样本")

    mu1, mu2 = a.mean(axis=0), b.mean(axis=0)
    sigma1 = np.cov(a, rowvar=False) + eps * np.eye(a.shape[1])
    sigma2 = np.cov(b, rowvar=False) + eps * np.eye(b.shape[1])

    diff = mu1 - mu2
    return round(float(diff @ diff
                       + np.trace(sigma1) + np.trace(sigma2)
                       - 2 * _sqrtm_product(sigma1, sigma2)), 4)


def _poly_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3,
                 gamma: float | None = None, coef0: float = 1.0) -> np.ndarray:
    d = x.shape[1]
    g = 1.0 / d if gamma is None else gamma
    return (g * (x @ y.T) + coef0) ** degree


def kid(feat_a: np.ndarray, feat_b: np.ndarray, degree: int = 3,
        n_subsets: int = 50, subset_size: int = 100,
        seed: int = 20261014) -> dict:
    """MMD^2 形式的有偏估计（多项式核），返回均值与标准差。"""
    a = np.asarray(feat_a, dtype=np.float64)
    b = np.asarray(feat_b, dtype=np.float64)
    m = min(subset_size, len(a), len(b))
    if m < 2:
        raise ValueError("KID 子集至少需要 2 个样本")

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_subsets):
        ia = rng.choice(len(a), m, replace=False)
        ib = rng.choice(len(b), m, replace=False)
        x, y = a[ia], b[ib]
        kxx = _poly_kernel(x, x, degree)
        kyy = _poly_kernel(y, y, degree)
        kxy = _poly_kernel(x, y, degree)
        # 去掉对角线上的自比较项
        scale = 1.0 / (m * (m - 1))
        mmd = (kxx.sum() - np.trace(kxx)) * scale \
            + (kyy.sum() - np.trace(kyy)) * scale \
            - 2 * (kxy.sum()) / (m * m)
        vals.append(float(mmd))

    vals = np.asarray(vals)
    return {
        "mean": round(float(vals.mean()), 6),
        "std": round(float(vals.std(ddof=1)) if vals.size > 1 else 0.0, 6),
        "degree": degree,
        "subset_size": m,
        "n_subsets": n_subsets,
    }


# ---------------- 实例掩码（Mask R-CNN） ----------------

@_cache("maskrcnn")
def mask_rcnn_model(conf: float = 0.5):
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )
    try:
        w = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        m = maskrcnn_resnet50_fpn_v2(weights=w)
    except (ImportError, RuntimeError, OSError):
        _require("torchvision", "torchvision")
    m.eval()
    return m, w


def coco_label_index(label: str) -> int:
    """按类名取 COCO 类别下标，避免硬编码索引出错。"""
    _, w = mask_rcnn_model()
    cats = list(w.meta["categories"])
    if label not in cats:
        raise KeyError(f"COCO 类别里没有 {label}；可用样例 {cats[:8]}")
    return cats.index(label)


def instance_masks(images: list[np.ndarray], label: str = _MASK_LABEL,
                   score_thresh: float = 0.5, size: int = 256) -> list[dict]:
    """检测目标类别的实例，返回每图的掩码与置信度。

    返回结构：[{"mask": HW bool, "score": float, "box": [x0,y0,x1,y1]}]
    取置信度最高的那个实例（方案里比较的是主体动物的掩码）。
    """
    model, _ = mask_rcnn_model()
    cat_id = coco_label_index(label)
    # torchvision 检测模型收的是**逐个 0~1 的张量列表**，不是拼好的 batch；
    # 传 batch 会报 "expected np.ndarray (got Tensor)" 之类的错。
    # 但整批塞进去会让检测头一次性分配巨量中间特征（120 张实测分配 4.9 GB
    # 失败，2026-10-24），必须分块前向。
    outs = []
    with torch.no_grad():
        for i in range(0, len(images), 4):
            batch = (to_batch(images[i:i + 4]) + 1.0) / 2.0
            outs.extend(model([t for t in batch]))

    results = []
    for out in outs:
        keep = (out["labels"] == cat_id) & (out["scores"] >= score_thresh)
        if not bool(keep.any()):
            results.append({"mask": np.zeros((size, size), bool),
                            "score": 0.0, "box": None, "found": False})
            continue
        best = int(torch.argmax(out["scores"][keep]))
        m = (out["masks"][keep][best, 0].numpy() >= 0.5)
        if m.shape != (size, size):
            from PIL import Image
            m = np.asarray(Image.fromarray(m).resize((size, size),
                                                     Image.NEAREST)) > 0
        results.append({"mask": m.astype(bool),
                        "score": round(float(out["scores"][keep][best]), 4),
                        "box": [round(float(v), 1)
                                for v in out["boxes"][keep][best].tolist()],
                        "found": True})
    return results


# ---------------- 斑马判别（ResNet-50 / ImageNet） ----------------

@_cache("resnet50")
def resnet50_model():
    from torchvision.models import ResNet50_Weights, resnet50
    try:
        w = ResNet50_Weights.IMAGENET1K_V1
        m = resnet50(weights=w)
    except (ImportError, RuntimeError, OSError):
        _require("torchvision", "torchvision")
    m.eval()
    return m, w


def label_indices(substrings=_TARGET_LABEL_SUBSTRINGS) -> list[int]:
    _, w = resnet50_model()
    cats = list(w.meta["categories"])
    hits = [i for i, c in enumerate(cats)
            if any(s in c.lower() for s in substrings)]
    if not hits:
        raise KeyError(f"ImageNet 类别里找不到 {substrings}")
    return hits


def target_probabilities(images: list[np.ndarray],
                         batch_size: int = 16) -> list[dict]:
    """每图给出：目标类（斑马）概率之和、top1 类别与置信度。"""
    m, w = resnet50_model()
    cats = list(w.meta["categories"])
    hits = label_indices()

    out = []
    x = (to_batch(images) + 1.0) / 2.0  # 分类网络输入 0~1
    for i in range(0, len(images), batch_size):
        with torch.no_grad():
            logits = m(x[i:i + batch_size])
            probs = torch.softmax(logits, dim=1)
        for row in probs.numpy():
            top = int(np.argmax(row))
            out.append({
                "target_prob": round(float(row[hits].sum()), 6),
                "top1": cats[top],
                "top1_prob": round(float(row[top]), 6),
                "is_target": cats[top] in {cats[h] for h in hits},
            })
    return out


# ---------------- Canny 边缘 ----------------

def canny_edges(img: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """方案指定的 Canny：双阈值 + 迟滞连接，不是简单的梯度幅值。"""
    try:
        import cv2
    except ImportError:
        _require("opencv-python", "opencv-python")

    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.Canny(arr, low, high) / 255.0


def canny_edge_ssim(a: np.ndarray, b: np.ndarray, **kw) -> float:
    from .metrics import ssim
    return ssim(canny_edges(a, **kw), canny_edges(b, **kw))


# ---------------- 异常样本登记 ----------------

def register_anomalies(records: list[dict], out_dir: Path) -> Path:
    """把异常样例写盘，供 12 月的异常拼图使用。

    records 每项：{"name": str, "image": HWC ndarray, "reason": str,
                   "metrics": dict}。

    注意本函数**只做写盘登记，不改任何统计量**：异常样本照常计入 quality.json
    的 testA 全集均值（实测 mean.ssim_bc = 全 120 张均值，剔除 3 例后由 0.885447
    升到 0.888369）。这与开题报告「逐图计算后取 testA 全集均值，异常图单独登记」
    一致——单独成目录只是方便复查，不是把它们从统计里剔除。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    index = []
    for r in records:
        arr = np.asarray(r["image"])
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        name = r.get("name") or f"anomaly_{len(index):03d}"
        p = out_dir / f"{name}.png"
        Image.fromarray(arr).save(p)
        index.append({**{k: v for k, v in r.items() if k != "image"},
                      "file": p.name})
    (out_dir / "anomalies.json").write_text(
        __import__("json").dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return out_dir / "anomalies.json"


# ---------------- 自检 ----------------

def self_test() -> None:
    """只测不依赖外部权重的部分（FID / KID 的数学正确性）。

    KID 用的是 MMD^2 的**无偏**估计量，单次数值可以为负，不必惊慌；但分布
    不同时必须明显为正。另外别拿同一个样本池跟自己比——两侧子集都从里面抽，
    必然重叠，交叉项被系统性拉负（实测 -0.23 量级），那是抽样的性质不是 bug。
    """
    rng = np.random.default_rng(0)
    same = rng.standard_normal((256, 32))
    other = rng.standard_normal((256, 32))
    shifted = rng.standard_normal((256, 32)) * 2 + 5

    print("FID(自身)     =", fid(same, same), "← 应为 0")
    print("FID(同分布)   =", fid(same, other), "← 应接近 0（有限样本下有偏）")
    print("FID(平移+缩放)=", fid(same, shifted), "← 应远大于 0")

    print("KID(同分布)   =", kid(same, other, n_subsets=20)["mean"],
          "← 应接近 0（无偏估计，可为负）")
    print("KID(平移+缩放)=", kid(same, shifted, n_subsets=20)["mean"],
          "← 应明显为正")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="深度指标自检")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        ap.print_help()
