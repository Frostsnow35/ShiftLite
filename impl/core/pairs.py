"""迁移对（pair）配置的加载与路径解析。

设计目标：把「换权重即换应用」这件事从代码里挪到数据里。
一个 pair = 一份 checkpoint + 一套数据集 + 一组前处理/指标口径。

路径策略（#24 方案 §2.4 选项 A）：
    - 默认 pair（horse2zebra）沿用既有的平铺路径，已入库的正式口径证据零改动；
    - 其余 pair 落在 artifacts/<pair-id>/ 下，互不覆盖。

这样做的理由：artifacts/results/benchmark.json 等文件是已入库的证据，
被 README 与开题/进展报告引用；统一改成子目录会让这些引用全部失效。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
PAIRS_DIR = ROOT / "pairs"

DEFAULT_PAIR = "horse2zebra"

# 迁移类型：用于 E4 跨类型层间对比的分组
TYPES = {
    "instance_texture": "实例纹理型（改属性，结构不变）",
    "domain": "全局域型（整体色调/季节）",
    "style": "风格型（笔触/画风）",
    "reverse": "同域反向（方向依赖性对照）",
}


@dataclass(frozen=True)
class Paths:
    models: Path
    samples: Path
    calib: Path
    # 评测集按「角色」命名，不按 zip 里的 testA/testB 命名。
    # 原因：zebra2horse / style_monet 的输入域来自 zip 的 testB，
    # 若目录仍叫 testA 就会出现「testA 里装着斑马」这种自相矛盾。
    eval_in: Path      # 生成批输入（pair.input_dir）
    eval_ref: Path     # FID/KID 参照批（pair.ref_dir）
    results: Path


@dataclass(frozen=True)
class Pair:
    id: str
    label: str
    type: str
    # 权重
    ckpt_file: str
    ckpt_url: str
    ckpt_bytes: int
    # 数据集
    dataset_zip: str
    dataset_url: str
    dataset_mb: float
    # 域映射：哪个子目录是生成批输入、哪个是参照批、哪个给校准
    input_prefix: str
    ref_prefix: str
    input_dir: str
    ref_dir: str
    calib_dir: str
    # 指标开关
    metrics: dict
    paths: Paths

    @property
    def weights(self) -> Path:
        return self.paths.models / self.ckpt_file

    @property
    def xml(self) -> Path:
        return self.paths.models / f"{self.id}.xml"

    @property
    def int8_xml(self) -> Path:
        return self.paths.models / f"{self.id}_int8.xml"

    @property
    def type_label(self) -> str:
        return TYPES.get(self.type, self.type)

    def metric_enabled(self, name: str) -> bool:
        m = self.metrics.get(name)
        if isinstance(m, dict):
            return bool(m.get("enabled", False))
        return bool(m)

    def metric_reason(self, name: str) -> str:
        m = self.metrics.get(name)
        return m.get("reason", "") if isinstance(m, dict) else ""

    def classifier_label(self) -> str:
        """类别判别的目标 ImageNet 类（随迁移对方向取）。

        horse2zebra → zebra；zebra2horse → horse。缺省回退到 ref_prefix。
        """
        m = self.metrics.get("class_discrimination")
        if isinstance(m, dict) and m.get("imagenet_class"):
            return m["imagenet_class"]
        return self.ref_prefix


def _paths(pair_id: str, ckpt_file: str) -> Paths:
    if pair_id == DEFAULT_PAIR:
        return Paths(
            models=ART / "models",
            samples=ART / "samples",
            calib=ART / "calib",
            eval_in=ART / "eval" / "input",
            eval_ref=ART / "eval" / "ref",
            results=ART / "results",
        )
    base = ART / pair_id
    return Paths(
        models=base / "models",
        samples=base / "samples",
        calib=base / "calib",
        eval_in=base / "eval" / "input",
        eval_ref=base / "eval" / "ref",
        results=base / "results",
    )


def list_pairs() -> list[str]:
    if not PAIRS_DIR.exists():
        return [DEFAULT_PAIR]
    return sorted(p.stem for p in PAIRS_DIR.glob("*.json"))


def load_pair(pair_id: str = DEFAULT_PAIR) -> Pair:
    path = PAIRS_DIR / f"{pair_id}.json"
    if not path.exists():
        raise SystemExit(
            f"未知的迁移对：{pair_id}\n"
            f"  可用：{', '.join(list_pairs())}\n"
            f"  manifest 目录：{PAIRS_DIR}"
        )
    cfg = json.loads(path.read_text(encoding="utf-8"))

    ds = cfg["dataset"]
    paths = _paths(cfg["id"], cfg["checkpoint"]["file"])

    return Pair(
        id=cfg["id"],
        label=cfg["label"],
        type=cfg["type"],
        ckpt_file=cfg["checkpoint"]["file"],
        ckpt_url=cfg["checkpoint"]["url"],
        ckpt_bytes=int(cfg["checkpoint"]["bytes"]),
        dataset_zip=ds["zip"],
        dataset_url=ds["url"],
        dataset_mb=float(ds.get("size_mb", 0)),
        input_prefix=ds["input_prefix"],
        ref_prefix=ds["ref_prefix"],
        input_dir=ds["input_dir"],
        ref_dir=ds["ref_dir"],
        calib_dir=ds["calib_dir"],
        metrics=cfg.get("metrics", {}),
        paths=paths,
    )
