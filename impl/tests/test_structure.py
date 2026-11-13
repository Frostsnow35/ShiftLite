"""不下载权重：验证几何/缺失分母/预处理/逐图字段等实验契约。"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from core.metrics import background_ssim_valid, mask_dice, mask_iou
from core.structure import measure_structure, write_review_materials


class StructureTests(unittest.TestCase):
    def test_geometry_known_overlap(self):
        a = np.array([[1, 1], [0, 0]], bool)
        b = np.array([[1, 0], [1, 0]], bool)
        self.assertAlmostEqual(mask_iou(a, b), 1/3, places=6)
        self.assertEqual(mask_dice(a, b), .5)

    def test_background_excludes_entire_windows(self):
        a = np.full((32, 32, 3), 90, np.uint8)
        b = a.copy()
        m = np.zeros((32, 32), bool)
        m[12:20, 12:20] = True
        b[m] = 255
        value, count = background_ssim_valid(a, b, m, m)
        self.assertEqual(value, 1.)
        self.assertEqual(count, 32*32-18*18)
        self.assertEqual(background_ssim_valid(a, b, np.ones_like(m), m), (None, 0))

    def test_missing_side_and_denominators(self):
        image = np.full((32, 32, 3), 90, np.uint8)
        m = np.zeros((32, 32), bool)
        m[12:20, 12:20] = True
        good = dict(mask=m, found=True, score=.9, box=[12,12,20,20],
                    candidate_count=1, label='zebra', reason=None)
        images = {side: [image, image] for side in 'XABC'}
        ds = {side: [copy.deepcopy(good), copy.deepcopy(good)] for side in 'XABC'}
        ds['C'][1].update(found=False, score=.3, candidate_count=0, reason='below_score_threshold')
        rows = [{'sample': 'first'}, {'sample': 'second'}]
        report = measure_structure(rows, images, ds)
        self.assertEqual(report['pairs']['x_c']['dice'], {'mean': 1., 'n_valid': 1, 'n_total': 2})
        self.assertEqual(report['pairs']['x_b']['dice']['n_valid'], 2)
        self.assertIsNone(rows[1]['structure']['pairs']['x_c']['iou'])
        self.assertEqual(report['failures'][0]['side'], 'C')
        self.assertEqual(report['failures'][0]['score'], .3)
        self.assertFalse(report['human_review_complete'])
        with tempfile.TemporaryDirectory() as tmp:
            write_review_materials(report, rows, images, ds, Path(tmp))
            self.assertTrue((Path(tmp) / 'second.png').exists())
            self.assertTrue((Path(tmp) / 'review_queue.json').exists())

    def test_classifier_uses_weight_transform(self):
        from core import metrics_deep as md
        calls = []
        class Weights:
            meta = {'categories': ['other', 'zebra']}
            def transforms(self):
                def transform(x):
                    calls.append(tuple(x.shape))
                    return x * 0 + 42
                return transform
        def model(x):
            self.assertTrue(torch.all(x == 42))
            return torch.tensor([[0., 1.]]).repeat(len(x), 1)
        with patch.object(md, 'resnet50_model', return_value=(model, Weights())):
            results = md.target_probabilities([np.zeros((32,32,3), np.uint8)] * 2)
        self.assertEqual(calls, [(2, 3, 32, 32)])
        self.assertTrue(all(r['is_target'] for r in results))

    def test_detector_preserves_failure_confidence(self):
        from core import metrics_deep as md
        def model(batch):
            return [dict(labels=torch.tensor([1]), scores=torch.tensor([.3]),
                         masks=torch.zeros((1,1,32,32)), boxes=torch.zeros((1,4))) for _ in batch]
        with patch.object(md, 'mask_rcnn_model', return_value=(model, None)), \
             patch.object(md, 'coco_label_index', return_value=1):
            result = md.instance_masks([np.zeros((32,32,3),np.uint8)], label='horse', size=32)[0]
        self.assertFalse(result['found'])
        self.assertEqual(result['score'], .3)
        self.assertEqual(result['reason'], 'below_score_threshold')
        self.assertEqual(result['label'], 'horse')

    def test_empty_anomaly_index_replaces_previous(self):
        from core.metrics_deep import register_anomalies
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'anomalies.json'
            p.write_text('[{"name":"old"}]')
            register_anomalies([], Path(tmp))
            self.assertEqual(p.read_text(), '[]')


if __name__ == '__main__':
    unittest.main()
