"""E3 目标类别判别的共享只读解析与格式化契约。

仅依赖 Python 标准库与轻量 `core.pairs`，供 `server.py`（实验卡片与 `/api/state`）
及 `core/figures.py`（离线 `quality_table.md`）共用同一份选择优先级与数值校验规则。
"""

from __future__ import annotations

import math
from typing import Any

from .pairs import DEFAULT_PAIR, Pair, load_pair

REAL_OR_PARTIAL = ("real", "partial")


def _resolve_pair(pair: Pair | str | None) -> Pair:
    if pair is None:
        return load_pair(DEFAULT_PAIR)
    if isinstance(pair, str):
        return load_pair(pair)
    return pair


def is_valid_accuracy(value: Any) -> bool:
    """有效 top1_accuracy 为 [0, 1] 内的有限 int/float（排除 bool）。"""
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def is_valid_sample_count(value: Any) -> bool:
    """样本量 n 仅接受正整数（严格排除 bool、0、负数、浮点数等）。"""
    return type(value) is int and value > 0


def format_accuracy_pct(value: float | int) -> str:
    """按两位小数格式化原始比例（沿用 Python 银行家舍入，如 0.03125 -> 3.12%）。"""
    return f"{value:.2%}"


def _parse_batch(raw_batch: Any) -> dict[str, Any] | None:
    if not isinstance(raw_batch, dict):
        return None
    acc = raw_batch.get("top1_accuracy")
    if not is_valid_accuracy(acc):
        return None
    n_raw = raw_batch.get("n")
    n_val = n_raw if is_valid_sample_count(n_raw) else None
    return {"top1_accuracy": acc, "n": n_val}


def _parse_schema_batches(raw_batches: Any) -> dict[str, dict[str, Any] | None]:
    if not isinstance(raw_batches, dict):
        return {"b": None, "c": None}
    return {
        "b": _parse_batch(raw_batches.get("b")),
        "c": _parse_batch(raw_batches.get("c")),
    }


def _result(
    *,
    enabled: bool,
    status: str,
    target: str,
    reason: str = "",
    source: str | None = None,
    b: dict[str, Any] | None = None,
    c: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "status": status,
        "reason": reason,
        "target": target,
        "source": source,
        "batches": {"b": b, "c": c},
    }


def resolve_classification(pair: Pair | str | None, quality: Any) -> dict[str, Any]:
    """解析当前迁移对在给定 quality 记录下的目标类别判别状态。

    优先级：
    1. manifest 关闭 -> `disabled`（附配置原因，优先于遗留数据）
    2. 无质量记录 -> `untested`
    3. 非 `real`/`partial`（含 `proxy` 或未知来源） -> `proxy`（不升级来源不明数据）
    4. 目标域匹配的候选中按「完整新格式 -> 完整旧格式 -> 部分新格式 -> 部分旧格式」择一，
       不跨来源拼凑 B/C；均无有效批次时返回 `missing`。
    """
    p = _resolve_pair(pair)
    target = p.ref_prefix

    if not p.metric_enabled("class_discrimination"):
        reason = p.metric_reason("class_discrimination") or f"{p.id} 未启用目标类别判别"
        return _result(enabled=False, status="disabled", target=target, reason=reason)

    if not isinstance(quality, dict) or not quality:
        return _result(enabled=True, status="untested", target=target)

    if quality.get("metrics_source") not in REAL_OR_PARTIAL:
        return _result(enabled=True, status="proxy", target=target)

    deep = quality.get("deep")
    if not isinstance(deep, dict):
        return _result(enabled=True, status="missing", target=target)

    candidates: list[tuple[str, dict[str, dict[str, Any] | None]]] = []

    cd = deep.get("class_discrimination")
    if isinstance(cd, dict):
        raw_target = cd.get("target")
        if raw_target is None or raw_target == "":
            target_ok = True
        elif isinstance(raw_target, str):
            target_ok = raw_target == target
        else:
            target_ok = False
        if target_ok:
            candidates.append(("class_discrimination", _parse_schema_batches(cd.get("by_batch"))))

    if target == "zebra" and "zebra_discrimination" in deep:
        candidates.append(("zebra_discrimination", _parse_schema_batches(deep.get("zebra_discrimination"))))

    for source_name, batches in candidates:
        if batches["b"] is not None and batches["c"] is not None:
            return _result(
                enabled=True,
                status="complete",
                target=target,
                source=source_name,
                b=batches["b"],
                c=batches["c"],
            )

    for source_name, batches in candidates:
        if batches["b"] is not None or batches["c"] is not None:
            return _result(
                enabled=True,
                status="partial",
                target=target,
                source=source_name,
                b=batches["b"],
                c=batches["c"],
            )

    return _result(enabled=True, status="missing", target=target)


def _batch_n_clause(label: str, batch: dict[str, Any]) -> str:
    return f"{label} 批 n={batch['n']}" if batch.get("n") is not None else f"{label} 批样本量未记录"


def format_classification_card(cls: dict[str, Any], *, has_errors: bool = False) -> str | None:
    """将 `resolve_classification()` 结果格式化为 E3 实验卡片的子项说明。"""
    status = cls.get("status")
    batches = cls.get("batches") or {}
    b = batches.get("b")
    c = batches.get("c")

    if status == "disabled":
        reason = cls.get("reason") or "当前迁移对未启用目标类别判别"
        return f"类别判别不适用（{reason}）"
    if status == "complete" and b is not None and c is not None:
        b_pct = format_accuracy_pct(b["top1_accuracy"])
        c_pct = format_accuracy_pct(c["top1_accuracy"])
        return (
            f"目标类别判别（B {b_pct} / C {c_pct}），"
            f"{_batch_n_clause('B', b)}，{_batch_n_clause('C', c)}"
        )
    if status == "partial":
        if b is not None:
            b_pct = format_accuracy_pct(b["top1_accuracy"])
            return f"目标类别判别（B 批 {b_pct}，尚缺 C 批），{_batch_n_clause('B', b)}"
        if c is not None:
            c_pct = format_accuracy_pct(c["top1_accuracy"])
            return f"目标类别判别（C 批 {c_pct}，尚缺 B 批），{_batch_n_clause('C', c)}"
    if status == "missing":
        suffix = "（请查 deep.errors）" if has_errors else ""
        return f"此次未取得有效类别判别结果{suffix}"
    return None
