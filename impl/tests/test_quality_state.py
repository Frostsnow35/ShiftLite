import copy
import json
from pathlib import Path
import unittest

from core.pairs import load_pair
from core.quality_state import (
    format_accuracy_pct,
    format_classification_card,
    resolve_classification,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "e3_partial_b_only.json"
REQUIRED_KEYS = {"enabled", "status", "reason", "target", "source", "batches"}


class QualityStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = {
            pid: load_pair(pid)
            for pid in ("horse2zebra", "zebra2horse", "summer2winter", "style_monet")
        }
        cls.partial_b_fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def assertContractShape(self, result):
        self.assertEqual(set(result.keys()), REQUIRED_KEYS)
        self.assertIsInstance(result["enabled"], bool)
        self.assertIsInstance(result["status"], str)
        self.assertIsInstance(result["reason"], str)
        self.assertIsInstance(result["target"], str)
        self.assertIn(result["source"], (None, "class_discrimination", "zebra_discrimination"))
        self.assertEqual(set(result["batches"].keys()), {"b", "c"})
        # 保证可直接 JSON 序列化
        roundtrip = json.loads(json.dumps(result, ensure_ascii=False))
        self.assertEqual(roundtrip, result)

    def test_purity_and_partial_b_fixture(self):
        raw = copy.deepcopy(self.partial_b_fixture)
        before = copy.deepcopy(raw)
        first = resolve_classification(self.pairs["horse2zebra"], raw)
        second = resolve_classification(self.pairs["horse2zebra"], raw)
        self.assertEqual(raw, before)
        self.assertEqual(first, second)
        self.assertContractShape(first)
        self.assertEqual(
            first,
            {
                "enabled": True,
                "status": "partial",
                "reason": "",
                "target": "zebra",
                "source": "class_discrimination",
                "batches": {"b": {"top1_accuracy": 0.75, "n": 12}, "c": None},
            },
        )
        self.assertEqual(
            format_classification_card(first, has_errors=True),
            "目标类别判别（B 批 75.00%，尚缺 C 批），B 批 n=12",
        )

    def test_six_statuses_and_disabled_override(self):
        # 1) disabled：即使带有有效 zebra 结果也优先按 manifest 关闭
        for pid in ("summer2winter", "style_monet"):
            with self.subTest(pair=pid):
                pair = self.pairs[pid]
                res = resolve_classification(pair, copy.deepcopy(self.partial_b_fixture))
                self.assertContractShape(res)
                self.assertFalse(res["enabled"])
                self.assertEqual(res["status"], "disabled")
                self.assertEqual(res["reason"], pair.metric_reason("class_discrimination"))
                self.assertIsNone(res["source"])
                self.assertEqual(res["batches"], {"b": None, "c": None})

        h2z = self.pairs["horse2zebra"]
        # 2) untested
        for empty in (None, {}, []):
            with self.subTest(empty=empty):
                res = resolve_classification(h2z, empty)
                self.assertContractShape(res)
                self.assertEqual((res["enabled"], res["status"], res["source"]), (True, "untested", None))
                self.assertEqual(res["batches"], {"b": None, "c": None})

        # 3) proxy 及未知 metrics_source 不得升级为真实评测
        for bad_source in ("proxy", None, "legacy_unknown", 123):
            with self.subTest(metrics_source=bad_source):
                q = {
                    "metrics_source": bad_source,
                    "n_samples": 4,
                    "deep": {"zebra_discrimination": {"b": {"top1_accuracy": 0.8, "n": 4},
                                                      "c": {"top1_accuracy": 0.8, "n": 4}}},
                }
                res = resolve_classification(h2z, q)
                self.assertContractShape(res)
                self.assertEqual((res["status"], res["source"]), ("proxy", None))
                self.assertEqual(res["batches"], {"b": None, "c": None})

        # 4) complete（含 overall partial 但分类完整，以及 0 / 1 边界和独立分母）
        q_complete = {
            "metrics_source": "partial",
            "deep": {
                "class_discrimination": {
                    "target": "zebra",
                    "by_batch": {
                        "b": {"top1_accuracy": 0, "n": 10},
                        "c": {"top1_accuracy": 1.0, "n": 15},
                    },
                }
            },
        }
        res_complete = resolve_classification(h2z, q_complete)
        self.assertContractShape(res_complete)
        self.assertEqual(res_complete["status"], "complete")
        self.assertEqual(res_complete["batches"]["b"], {"top1_accuracy": 0, "n": 10})
        self.assertEqual(res_complete["batches"]["c"], {"top1_accuracy": 1.0, "n": 15})
        self.assertEqual(
            format_classification_card(res_complete),
            "目标类别判别（B 0.00% / C 100.00%），B 批 n=10，C 批 n=15",
        )

        # 5) partial（overall real 但仅 C 批有效）
        q_real_c_only = {
            "metrics_source": "real",
            "deep": {
                "class_discrimination": {
                    "target": "zebra",
                    "by_batch": {"c": {"top1_accuracy": 0.5, "n": 8}},
                }
            },
        }
        res_c_only = resolve_classification(h2z, q_real_c_only)
        self.assertContractShape(res_c_only)
        self.assertEqual(res_c_only["status"], "partial")
        self.assertEqual(res_c_only["batches"], {"b": None, "c": {"top1_accuracy": 0.5, "n": 8}})
        self.assertEqual(
            format_classification_card(res_c_only),
            "目标类别判别（C 批 50.00%，尚缺 B 批），C 批 n=8",
        )

        # 6) missing（已有 real/partial，但无有效匹配批次）
        for deep_val in (None, [], {}, {"class_discrimination": {"target": "zebra", "by_batch": {}}}):
            with self.subTest(deep=deep_val):
                res_missing = resolve_classification(h2z, {"metrics_source": "real", "deep": deep_val})
                self.assertContractShape(res_missing)
                self.assertEqual((res_missing["status"], res_missing["source"]), ("missing", None))
                self.assertEqual(res_missing["batches"], {"b": None, "c": None})

    def test_schema_priority_and_target_domain_matching(self):
        h2z = self.pairs["horse2zebra"]
        z2h = self.pairs["zebra2horse"]
        new_full = {"b": {"top1_accuracy": 0.61, "n": 12}, "c": {"top1_accuracy": 0.62, "n": 12}}
        old_full = {"b": {"top1_accuracy": 0.85, "n": 120}, "c": {"top1_accuracy": 0.86, "n": 120}}

        cases = (
            (
                "new_full_beats_old_full",
                h2z,
                {"class_discrimination": {"target": "zebra", "by_batch": new_full},
                 "zebra_discrimination": old_full},
                "complete",
                "class_discrimination",
                0.61,
                0.62,
            ),
            (
                "new_empty_falls_back_to_old_full",
                h2z,
                {"class_discrimination": {"target": "zebra", "by_batch": {}},
                 "zebra_discrimination": old_full},
                "complete",
                "zebra_discrimination",
                0.85,
                0.86,
            ),
            (
                "new_partial_falls_back_to_old_full",
                h2z,
                {"class_discrimination": {"target": "zebra", "by_batch": {"b": new_full["b"]}},
                 "zebra_discrimination": old_full},
                "complete",
                "zebra_discrimination",
                0.85,
                0.86,
            ),
            (
                "new_b_and_old_c_never_mixed_prefers_partial_new",
                h2z,
                {"class_discrimination": {"target": "zebra", "by_batch": {"b": new_full["b"]}},
                 "zebra_discrimination": {"c": old_full["c"]}},
                "partial",
                "class_discrimination",
                0.61,
                None,
            ),
            (
                "new_missing_target_defaults_to_pair_target",
                z2h,
                {"class_discrimination": {"by_batch": new_full}},
                "complete",
                "class_discrimination",
                0.61,
                0.62,
            ),
            (
                "explicit_target_conflict_skips_new_candidate",
                h2z,
                {"class_discrimination": {"target": "horse", "by_batch": new_full}},
                "missing",
                None,
                None,
                None,
            ),
            (
                "reverse_pair_rejects_legacy_zebra_only",
                z2h,
                {"zebra_discrimination": old_full},
                "missing",
                None,
                None,
                None,
            ),
        )
        for name, pair, deep, exp_status, exp_source, exp_b, exp_c in cases:
            with self.subTest(case=name, pair=pair.id):
                res = resolve_classification(pair, {"metrics_source": "real", "deep": deep})
                self.assertContractShape(res)
                self.assertEqual(res["status"], exp_status)
                self.assertEqual(res["source"], exp_source)
                got_b = res["batches"]["b"]["top1_accuracy"] if res["batches"]["b"] else None
                got_c = res["batches"]["c"]["top1_accuracy"] if res["batches"]["c"] else None
                self.assertEqual((got_b, got_c), (exp_b, exp_c))

    def test_accuracy_n_and_rounding_boundaries(self):
        h2z = self.pairs["horse2zebra"]
        for bad_acc in (None, True, False, -0.01, 1.01, "0.75", float("nan"), float("inf"), -float("inf"), []):
            with self.subTest(bad_accuracy=bad_acc):
                q = {
                    "metrics_source": "real",
                    "deep": {
                        "class_discrimination": {
                            "target": "zebra",
                            "by_batch": {
                                "b": {"top1_accuracy": bad_acc, "n": 10},
                                "c": {"top1_accuracy": 0.5, "n": 10},
                            },
                        }
                    },
                }
                res = resolve_classification(h2z, q)
                self.assertEqual(res["status"], "partial")
                self.assertIsNone(res["batches"]["b"])
                self.assertEqual(res["batches"]["c"], {"top1_accuracy": 0.5, "n": 10})

        for bad_n in (None, True, False, 0, -3, 12.0, 12.5, "12", []):
            with self.subTest(bad_n=bad_n):
                q = {
                    "metrics_source": "partial",
                    "deep": {
                        "class_discrimination": {
                            "target": "zebra",
                            "by_batch": {"b": {"top1_accuracy": 0.75, "n": bad_n}},
                        }
                    },
                }
                res = resolve_classification(h2z, q)
                self.assertEqual(res["status"], "partial")
                self.assertEqual(res["batches"]["b"], {"top1_accuracy": 0.75, "n": None})
                self.assertIn("样本量未记录", format_classification_card(res))

        # 半分位舍入契约：0.03125 -> 3.12%，0.09375 -> 9.38%
        self.assertEqual(format_accuracy_pct(0.03125), "3.12%")
        self.assertEqual(format_accuracy_pct(0.09375), "9.38%")


if __name__ == "__main__":
    unittest.main()
