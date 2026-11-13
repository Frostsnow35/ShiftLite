import json
import tempfile
import unittest
from pathlib import Path

from core.figures import quality_table


class QualityTableTests(unittest.TestCase):
    def test_dice_is_parallel_with_iou_and_old_json_is_missing(self):
        kid = {'mean': .1, 'std': .01, 'subset_size': 2, 'n_subsets': 2}
        q = {'metrics_source': 'real', 'mean': {'ssim_bc': .9, 'psnr_bc': 29}, 'deep': {
            'lpips': {'n_images': 2, 'L_ab': 0, 'L_bc': .1, 'L_ac': .1},
            'fid': {'n_real': 2, 'b_vs_real_zebra': 1, 'c_vs_real_zebra': 2},
            'kid': {'b_vs_real_zebra': kid, 'c_vs_real_zebra': kid},
            'mask': {'mean_iou_ac': .6, 'mean_dice_ac': .75, 'detected_pairs': 1,
                     'mean_background_ssim_ac': .9}, 'canny_edge_ssim_bc': .7,
            'zebra_discrimination': {'b': {'top1_accuracy': .5, 'n': 2}, 'c': {'top1_accuracy': .5, 'n': 2}},
            'anomalies': {'count': 0}}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for value in (.75, None):
                if value is None:
                    del q['deep']['mask']['mean_dice_ac']
                (root / 'quality.json').write_text(json.dumps(q))
                text = quality_table(root, root).read_text(encoding='utf-8')
                self.assertIn('掩码 IoU(a,c)', text)
                expected = '0.7500' if value is not None else '缺失'
                denominator = '1' if value is not None else '未记录'
                self.assertIn(f'掩码 Dice(a,c) | {expected} | 有效 {denominator}/2 对', text)


if __name__ == '__main__':
    unittest.main()
