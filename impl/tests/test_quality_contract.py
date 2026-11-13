import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import cli
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


if __name__ == '__main__':
    unittest.main()
