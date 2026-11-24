import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import cli
import server
from core import metrics_deep as md


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
    def card(self, source, deep):
        quality = {"metrics_source": source, "deep": deep} if source else None
        cards = server._experiments(None, None, None, quality, {})
        return next(card for card in cards if card["code"] == "E3")

    def test_classification_batch_formats(self):
        complete = {"b": {"top1_accuracy": 0.0}, "c": {"top1_accuracy": 1}}

        def current(batches):
            return {"class_discrimination": {"target": "horse", "by_batch": batches}}

        legacy = {"zebra_discrimination": complete}
        for name, deep, shown in (
            ("legacy", legacy, True),
            ("current", current(complete), True),
            ("fallback", {**current({"b": {}}), **legacy}, True),
            ("separate schemas", {**current({"b": complete["b"]}),
                                  "zebra_discrimination": {"c": complete["c"]}}, False),
            ("malformed new container", {"class_discrimination": [], **legacy}, True),
            ("target only", {"class_discrimination": {"target": "horse"}}, False),
            ("invalid batches", current([]), False),
            ("empty batch", current({"b": {}, "c": complete["c"]}), False),
            ("single batch", current({"b": complete["b"]}), False),
            ("missing", {}, False),
        ):
            with self.subTest(name=name):
                card = self.card("real", deep)
                self.assertEqual(card["status"], "done")
                self.assertEqual("目标类别判别" in card["note"], shown)

        for invalid in (True, float("nan"), float("inf"), None, "0"):
            with self.subTest(accuracy=invalid):
                deep = current({"b": {"top1_accuracy": invalid}, "c": complete["c"]})
                self.assertNotIn("目标类别判别", self.card("real", deep)["note"])

    def test_partial_reports_observed_batches_and_other_metrics(self):
        batch = {"top1_accuracy": 0.0}
        for measured, missing in (("b", "C"), ("c", "B")):
            with self.subTest(measured=measured):
                deep = {"lpips": {"L_bc": 0.1},
                        "class_discrimination": {"by_batch": {measured: batch}}}
                card = self.card("partial", deep)
                self.assertEqual(card["status"], "partial")
                self.assertIn("真实 LPIPS", card["note"])
                self.assertIn(f"尚缺 {missing} 批", card["note"])
                self.assertNotIn("代理", card["note"])

        both = {"by_batch": {"b": batch, "c": batch}}
        card = self.card("partial", {"class_discrimination": both, "errors": ["fid"]})
        self.assertIn("部分完成", card["note"])
        self.assertIn("目标类别判别", card["note"])
        for deep in ({"errors": ["lpips"]}, {"anomalies": {"count": 0}}, []):
            with self.subTest(deep=deep):
                self.assertIn("deep.errors", self.card("partial", deep)["note"])

    def test_proxy_and_pending(self):
        proxy = self.card("proxy", {})
        self.assertEqual(proxy["status"], "partial")
        self.assertIn("代理", proxy["note"])
        self.assertIn("适用", proxy["note"])
        self.assertNotIn("斑马判别", proxy["note"])
        self.assertNotIn("掩码 IoU", proxy["note"])
        pending = self.card(None, {})
        self.assertEqual((pending["status"], pending["note"]), ("todo", "待测"))


if __name__ == '__main__':
    unittest.main()
