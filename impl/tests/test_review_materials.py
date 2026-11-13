import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from core.review import build_review_materials, sha256


class ReviewMaterialsTests(unittest.TestCase):
    def fixture(self, root):
        results = root / 'results'
        (results / 'structure_review').mkdir(parents=True)
        rows, queue = [], []
        for name in ('first', 'second'):
            Image.new('RGB', (32, 32), 'red').save(results / 'structure_review' / (name + '.png'))
            rows.append({'sample': name, 'structure': {'detections': {}}})
            queue.append({'sample': name, 'reasons': ['detection_failure']})
        q = {'protocol_version': 'quality-v2', 'rows': rows,
             'deep': {'structure': {'protocol': {}, 'review_queue': queue, 'failures': []}}}
        path = results / 'quality.json'
        path.write_text(json.dumps(q), encoding='utf-8')
        return path

    def test_order_hashes_and_existing_human_work_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            q = self.fixture(root)
            selected = root / 'selected.json'
            selected.write_text('{"samples":["second","first"]}')
            out = root / 'export'
            report = build_review_materials(q, out, root, selected)
            self.assertEqual(report['sample_order'], ['second', 'first'])
            self.assertEqual(report['quality_sha256'], sha256(q))
            self.assertEqual(report['samples'][0]['comparison_sha256'], sha256(out / 'images/second.png'))
            with (out / 'human_review_template.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertTrue(all(r['human_status'] == 'pending' for r in csv.DictReader(stream)))
            (out / 'human_review.csv').write_bytes(b'actual human work')
            (out / 'human_review_template.csv').write_bytes(b'kept template')
            build_review_materials(q, out, root, selected)
            self.assertEqual((out / 'human_review.csv').read_bytes(), b'actual human work')
            self.assertEqual((out / 'human_review_template.csv').read_bytes(), b'kept template')
            Image.new('RGB', (32, 32), 'blue').save(q.parent / 'structure_review/second.png')
            previous = (out / 'images/second.png').read_bytes()
            with self.assertRaises(ValueError):
                build_review_materials(q, out, root, selected)
            self.assertEqual((out / 'images/second.png').read_bytes(), previous)
            q.write_text(q.read_text() + '\n')
            with self.assertRaises(ValueError):
                build_review_materials(q, out, root, selected)

    def test_unknown_or_unsafe_samples_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            q = self.fixture(root)
            selected = root / 'selected.json'
            for samples in (['missing'], ['../escape'], ['first', 'first']):
                selected.write_text(json.dumps({'samples': samples}))
                with self.assertRaises(ValueError):
                    build_review_materials(q, root / 'out', root, selected)
                self.assertFalse((root / 'out').exists())

    def test_fallback_is_explicitly_originals_only_and_missing_images_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            q = self.fixture(root)
            (q.parent / 'structure_review/first.png').unlink()
            data = json.loads(q.read_text())
            data['rows'][0]['input'] = 'input.png'
            q.write_text(json.dumps(data))
            selected = root / 'selected.json'
            selected.write_text('{"samples":["first"]}')
            with self.assertRaises(FileNotFoundError):
                build_review_materials(q, root / 'out', root, selected)
            self.assertFalse((root / 'out').exists())
            (q.parent / 'quality').mkdir()
            for path in [root / 'input.png', *[q.parent / 'quality' / f'first_{g}.png' for g in 'ABC']]:
                Image.new('RGB', (256, 256), 'gray').save(path)
            report = build_review_materials(q, root / 'out', root, selected)
            self.assertEqual(report['samples'][0]['comparison_kind'], 'originals_only_no_masks')
            with Image.open(root / 'out/images/first.png') as picture:
                self.assertEqual(picture.size, (1024, 280))


if __name__ == '__main__':
    unittest.main()
