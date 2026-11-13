"""结构协议 v2：输入保真和输出一致性分开；机器登记不代替人工复核。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import background_ssim_valid, mask_dice, mask_iou

PAIRS = {'x_a': ('X', 'A'), 'x_b': ('X', 'B'),
         'x_c': ('X', 'C'), 'a_c': ('A', 'C')}
PROTOCOL = {
    'version': 'structure-v2',
    'detector': 'MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1',
    'labels': {'X': 'horse', 'A': 'zebra', 'B': 'zebra', 'C': 'zebra'},
    'score_threshold': 0.5, 'mask_threshold': 0.5,
    'selection': 'highest confidence target instance per side; multiple candidates require review',
    'background': '11x11 SSIM windows fully outside union of both foreground masks; reflect border; minimum 64 windows',
    'missing': 'null; excluded only from corresponding metric mean; denominator recorded',
    'interpretation': 'detector-derived proxy masks, not ground truth or verified instance correspondence',
}


def measure_structure(rows, images, detections):
    """填充逐图记录与有效样本数；所有输入必须按同一 sample 顺序对齐。"""
    n = len(rows)
    if set(images) != {'X', 'A', 'B', 'C'} or set(detections) != set(images):
        raise ValueError('结构评测需要 X/A/B/C 四侧')
    if any(len(v) != n for v in [*images.values(), *detections.values()]):
        raise ValueError('图像/检测/样本数量不一致')
    failures, review = [], []
    for i, row in enumerate(rows):
        info = {}
        for side, ds in detections.items():
            d = ds[i]
            info[side] = {k: d.get(k) for k in
                          ('found', 'score', 'box', 'candidate_count', 'label', 'reason')}
            if not d['found']:
                failures.append({'sample': row['sample'], 'side': side, **info[side],
                                 'human_status': 'pending'})
        pairs = {}
        for name, (left, right) in PAIRS.items():
            da, db = detections[left][i], detections[right][i]
            missing = [s for s, d in ((left, da), (right, db)) if not d['found']]
            item = {'iou': None, 'dice': None, 'background_ssim': None,
                    'background_windows': 0, 'missing_sides': missing,
                    'background_reason': 'missing_detection' if missing else None}
            if not missing:
                item['iou'] = mask_iou(da['mask'], db['mask'])
                item['dice'] = mask_dice(da['mask'], db['mask'])
                value, count = background_ssim_valid(images[left][i], images[right][i],
                                                       da['mask'], db['mask'])
                item.update(background_ssim=value, background_windows=count,
                            background_reason=None if value is not None else 'insufficient_background')
            pairs[name] = item
        row['structure'] = {'detections': info, 'pairs': pairs}
        # 兼容既有读者的 A/C 输出字段；背景旧口径在 CLI 中单独保留。
        row['mask_iou_ac'] = pairs['a_c']['iou']
        row['mask_dice_ac'] = pairs['a_c']['dice']
        reasons = []
        if any(not d['found'] for d in info.values()):
            reasons.append('detection_failure')
        if any((d.get('candidate_count') or 0) > 1 for d in info.values()):
            reasons.append('multiple_instances_correspondence_unverified')
        if any(p['background_reason'] == 'insufficient_background' for p in pairs.values()):
            reasons.append('insufficient_background')
        if reasons:
            review.append({'sample': row['sample'], 'reasons': reasons,
                           'detections': info, 'human_status': 'pending',
                           'reviewer': None, 'conclusion': None})
    aggregates = {}
    for name in PAIRS:
        aggregates[name] = {}
        for metric in ('iou', 'dice', 'background_ssim'):
            vals = [r['structure']['pairs'][name][metric] for r in rows
                    if r['structure']['pairs'][name][metric] is not None]
            aggregates[name][metric] = {
                'mean': round(float(np.mean(vals)), 6) if vals else None,
                'n_valid': len(vals), 'n_total': n}
    return {'protocol': PROTOCOL, 'pairs': aggregates, 'failures': failures,
            'n_failure_samples': len({f['sample'] for f in failures}),
            'review_queue': review, 'human_review_complete': False}


def write_review_materials(report, rows, images, detections, out_dir: Path):
    """生成带置信度的四侧原图/掩码叠加图；永不填写真人意见。"""
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    lookup = {r['sample']: i for i, r in enumerate(rows)}
    for record in report['review_queue']:
        i = lookup[record['sample']]
        canvas = Image.new('RGB', (1024, 556), 'white')
        draw = ImageDraw.Draw(canvas)
        for col, side in enumerate(('X', 'A', 'B', 'C')):
            array = np.asarray(images[side][i], np.uint8)
            overlay = array.copy()
            mask = detections[side][i]['mask']
            overlay[mask] = (array[mask] * .55 + np.array([255, 50, 50]) * .45).astype(np.uint8)
            canvas.paste(Image.fromarray(array), (col * 256, 22))
            canvas.paste(Image.fromarray(overlay), (col * 256, 300))
            d = detections[side][i]
            draw.text((col * 256 + 4, 3), f"{side} {d.get('label')} found={d['found']}", fill='black')
            draw.text((col * 256 + 4, 281), f"score={d.get('score')} count={d.get('candidate_count')}", fill='black')
        filename = f"{record['sample']}.png"
        canvas.save(out_dir / filename)
        record['comparison_file'] = filename
    (out_dir / 'review_queue.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
