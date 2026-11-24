import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import cli
import server
from core import metrics_deep as md
from core import pairs
from core.quality_state import resolve_classification


FIXTURE_PATH = Path(__file__).resolve().parent / "e3_partial_b_only.json"


class QualityContractTests(unittest.TestCase):
    def test_real_canny_is_per_image_and_partial_is_not_complete(self):
        image = np.full((32, 32, 3), 90, np.uint8)
        mask = np.zeros((32, 32), bool)
        mask[12:20, 12:20] = True
        def masks(batch, label):
            return [dict(mask=mask, found=True, score=.9, box=[12,12,20,20],
                         candidate_count=1, label=label, reason=None) for _ in batch]
        def lpips(a,b,c):
            return {'net': 'test', 'n_images': 1, 'L_ab': 0., 'L_bc': 0., 'L_ac': 0.,
                    'L_ab_per_image': [0.], 'L_bc_per_image': [0.], 'L_ac_per_image': [0.]}
        rows = [{'sample': 'test', 'ssim_bc': 1., 'edge_ssim_bc': .7}]
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(cli, 'RESULTS', Path(tmp)), \
             patch.object(cli, 'eval_zebra_paths', return_value=[]), \
             patch.object(md, 'lpips_three_way_real', side_effect=lpips), \
             patch.object(md, 'canny_edge_ssim', return_value=.123), \
             patch.object(md, 'instance_masks', side_effect=masks), \
             patch.object(md, 'target_probabilities', return_value=[dict(is_target=True, target_prob=.9)]):
            result = cli._deep_metrics([image], [image], [image], rows, Path(tmp),
                                       SimpleNamespace(anomaly_threshold=.8), [image])
        self.assertFalse(result['complete'])  # 缺真实斑马参考批不能标完整
        self.assertEqual(rows[0]['canny_edge_ssim_bc'], .123)
        self.assertEqual(rows[0]['edge_ssim_bc'], .7)
        self.assertEqual(result['canny_n_valid']['xc'], 1)
        self.assertEqual(rows[0]['structure']['pairs']['x_c']['dice'], 1.)


class QualityCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = {pid: pairs.load_pair(pid) for pid in pairs.list_pairs()}
        cls.partial_b_fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def card(self, source, deep, pair_id="horse2zebra"):
        p = self.pairs[pair_id]
        quality = {"metrics_source": source, "deep": deep} if source else None
        cls = resolve_classification(p, quality)
        cards = server._experiments(None, None, None, quality, {}, pair=p, classification=cls)
        return next(card for card in cards if card["code"] == "E3"), cls

    def test_classification_batch_formats_and_target_domains(self):
        complete = {
            "b": {"top1_accuracy": 0.75, "n": 12},
            "c": {"top1_accuracy": 0.7417, "n": 120},
        }
        legacy = {
            "zebra_discrimination": {
                "b": {"top1_accuracy": 0.03125, "n": 16},
                "c": {"top1_accuracy": 0.09375, "n": 32},
            }
        }

        # 1) 完整双批（新格式 horse 在 zebra2horse 下成功，旧格式及回退在 horse2zebra 下成功）
        card_h, cls_h = self.card(
            "real",
            {"class_discrimination": {"target": "horse", "by_batch": complete}},
            pair_id="zebra2horse",
        )
        self.assertEqual((card_h["status"], cls_h["status"], cls_h["source"]),
                         ("done", "complete", "class_discrimination"))
        self.assertIn("目标类别判别（B 75.00% / C 74.17%），B 批 n=12，C 批 n=120", card_h["note"])

        for name, deep in (
            ("legacy", legacy),
            ("fallback_from_empty_new", {"class_discrimination": {"target": "zebra", "by_batch": {}}, **legacy}),
            ("fallback_from_malformed_new", {"class_discrimination": [], **legacy}),
        ):
            with self.subTest(name=name):
                card_z, cls_z = self.card("real", deep, pair_id="horse2zebra")
                self.assertEqual((card_z["status"], cls_z["status"], cls_z["source"]),
                                 ("done", "complete", "zebra_discrimination"))
                self.assertIn("目标类别判别（B 3.12% / C 9.38%），B 批 n=16，C 批 n=32", card_z["note"])

        # 2) 新 B + 旧 C 跨格式不拼凑，按部分新格式保留 B 批并指出缺 C 批
        split_deep = {
            "class_discrimination": {"target": "zebra", "by_batch": {"b": complete["b"]}},
            "zebra_discrimination": {"c": complete["c"]},
        }
        card_split, cls_split = self.card("real", split_deep, pair_id="horse2zebra")
        self.assertEqual((card_split["status"], cls_split["status"], cls_split["source"]),
                         ("done", "partial", "class_discrimination"))
        self.assertIn("目标类别判别（B 批 75.00%，尚缺 C 批），B 批 n=12", card_split["note"])

        # 3) 显式目标冲突与反向对遗留 zebra 数据均视为 missing，不展示伪造判别率
        for name, pid, deep in (
            ("target_conflict_on_h2z", "horse2zebra",
             {"class_discrimination": {"target": "horse", "by_batch": complete}}),
            ("legacy_zebra_on_z2h", "zebra2horse", legacy),
            ("target_only", "zebra2horse", {"class_discrimination": {"target": "horse"}}),
            ("invalid_batches", "horse2zebra", {"class_discrimination": {"target": "zebra", "by_batch": []}}),
            ("missing", "horse2zebra", {}),
        ):
            with self.subTest(name=name):
                card_m, cls_m = self.card("real", deep, pair_id=pid)
                self.assertEqual(cls_m["status"], "missing")
                self.assertNotIn("目标类别判别", card_m["note"])
                self.assertIn("此次未取得有效类别判别结果", card_m["note"])

        # 4) 单侧无效 accuracy 时，保留有效对侧并如实报缺批（real 与 partial 均如此）
        for invalid in (True, float("nan"), float("inf"), -0.1, 1.1, None, "0"):
            with self.subTest(accuracy=invalid):
                deep = {
                    "class_discrimination": {
                        "target": "zebra",
                        "by_batch": {"b": {"top1_accuracy": invalid, "n": 10}, "c": complete["c"]},
                    }
                }
                card_one, cls_one = self.card("real", deep, pair_id="horse2zebra")
                self.assertEqual(cls_one["status"], "partial")
                self.assertIn("目标类别判别（C 批 74.17%，尚缺 B 批），C 批 n=120", card_one["note"])

    def test_single_batch_real_and_partial_and_missing_n(self):
        for overall_source, exp_card_status in (("real", "done"), ("partial", "partial")):
            for measured, pct, other in (("b", "0.00%", "C"), ("c", "100.00%", "B")):
                acc = 0.0 if measured == "b" else 1.0
                with self.subTest(source=overall_source, measured=measured):
                    deep = {
                        "lpips": {"L_bc": 0.1},
                        "class_discrimination": {
                            "target": "zebra",
                            "by_batch": {measured: {"top1_accuracy": acc}},
                        },
                    }
                    card, cls = self.card(overall_source, deep, pair_id="horse2zebra")
                    self.assertEqual(card["status"], exp_card_status)
                    self.assertEqual(cls["status"], "partial")
                    self.assertIn("真实 LPIPS", card["note"])
                    self.assertIn(f"{measured.upper()} 批 {pct}，尚缺 {other} 批", card["note"])
                    self.assertIn(f"{measured.upper()} 批样本量未记录", card["note"])
                    self.assertNotIn("代理", card["note"])

        # 整体 partial 但 B/C 分类完整
        both = {
            "target": "zebra",
            "by_batch": {
                "b": {"top1_accuracy": 0.75, "n": 12},
                "c": {"top1_accuracy": 0.5, "n": 12},
            },
        }
        card_both, cls_both = self.card(
            "partial",
            {"class_discrimination": both, "errors": ["fid"]},
            pair_id="horse2zebra",
        )
        self.assertEqual((card_both["status"], cls_both["status"]), ("partial", "complete"))
        self.assertIn("深度评测部分完成", card_both["note"])
        self.assertIn("目标类别判别（B 75.00% / C 50.00%），B 批 n=12，C 批 n=12", card_both["note"])

        for deep in ({"errors": ["lpips"]}, {"anomalies": {"count": 0}}, []):
            with self.subTest(deep=deep):
                card_err, cls_err = self.card("partial", deep, pair_id="horse2zebra")
                self.assertEqual(cls_err["status"], "missing")
                self.assertIn("deep.errors", card_err["note"])
                self.assertIn("此次未取得有效类别判别结果", card_err["note"])

    def test_proxy_pending_and_build_state_api_contract(self):
        proxy, cls_proxy = self.card("proxy", {}, pair_id="horse2zebra")
        self.assertEqual((proxy["status"], cls_proxy["status"]), ("partial", "proxy"))
        self.assertIn("代理", proxy["note"])
        self.assertIn("尚未运行真实类别判别", proxy["note"])
        self.assertIn("适用", proxy["note"])
        self.assertNotIn("斑马判别", proxy["note"])
        self.assertNotIn("掩码 IoU", proxy["note"])

        pending, cls_pending = self.card(None, {}, pair_id="horse2zebra")
        self.assertEqual((pending["status"], pending["note"], cls_pending["status"]),
                         ("todo", "待测", "untested"))

        # 通过真实 build_state() 验证 E3-01 与 E3-05（同一份合成 JSON 放入启用对与禁用对）
        raw_text = FIXTURE_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as root, \
                patch.object(pairs, "ART", Path(root)), \
                patch.object(server, "ART", Path(root)), \
                patch.object(server, "PAIRS", server._pair_registry()), \
                patch.object(server, "_runtime", None), \
                patch.object(server, "_runtime_pair", None):
            for pid in ("horse2zebra", "summer2winter", "style_monet"):
                q_path = server.PAIRS[pid].paths.results / "quality.json"
                q_path.parent.mkdir(parents=True, exist_ok=True)
                q_path.write_text(raw_text, encoding="utf-8")

            st_h2z = server.build_state("horse2zebra")
            self.assertEqual(st_h2z["quality"], self.partial_b_fixture)
            self.assertEqual(
                st_h2z["quality_classification"],
                {
                    "enabled": True,
                    "status": "partial",
                    "reason": "",
                    "target": "zebra",
                    "source": "class_discrimination",
                    "batches": {"b": {"top1_accuracy": 0.75, "n": 12}, "c": None},
                },
            )
            e3_h2z = next(e for e in st_h2z["experiments"] if e["code"] == "E3")
            self.assertEqual(e3_h2z["status"], "partial")
            self.assertIn("目标类别判别（B 批 75.00%，尚缺 C 批），B 批 n=12", e3_h2z["note"])

            for disabled_pid in ("summer2winter", "style_monet"):
                with self.subTest(disabled_pair=disabled_pid):
                    st_dis = server.build_state(disabled_pid)
                    reason = server.PAIRS[disabled_pid].metric_reason("class_discrimination")
                    self.assertEqual(
                        st_dis["quality_classification"],
                        {
                            "enabled": False,
                            "status": "disabled",
                            "reason": reason,
                            "target": server.PAIRS[disabled_pid].ref_prefix,
                            "source": None,
                            "batches": {"b": None, "c": None},
                        },
                    )
                    e3_dis = next(e for e in st_dis["experiments"] if e["code"] == "E3")
                    self.assertIn(f"类别判别不适用（{reason}）", e3_dis["note"])
                    self.assertNotIn("75.00%", e3_dis["note"])


if __name__ == '__main__':
    unittest.main()
