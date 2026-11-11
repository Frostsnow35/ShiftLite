"""从已有评测结果导出可移交的复核材料；不推断或覆盖人工意见。"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import io
from pathlib import Path

FIELDS = ['sample', 'comparison_file', 'machine_reasons', 'human_status',
          'reviewer', 'date', 'conclusion', 'notes']


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_review_materials(quality_path: Path, output: Path, image_root: Path,
                           sample_list: Path | None = None) -> dict:
    """默认导出风险队列；sample-list 可选取历史样本，保持给定顺序。

    有同次评测叠加图时原样复制；否则只生成原图对照，不伪造分割掩码。
    不同实验/样本集合使用不同目录；人工 CSV 与已有模板均不覆盖。
    """
    quality_path, output, image_root = map(Path, (quality_path, output, image_root))
    q = json.loads(quality_path.read_text(encoding='utf-8'))
    if q.get('protocol_version') != 'quality-v2' or not q.get('deep', {}).get('structure'):
        raise ValueError('需要带 deep.structure 的 quality-v2 结果；先完成 quality --deep')
    report = q['deep']['structure']
    rows = {row['sample']: row for row in q['rows']}
    queue = {row['sample']: row for row in report['review_queue']}
    samples = (json.loads(Path(sample_list).read_text(encoding='utf-8'))['samples']
               if sample_list else list(queue))
    if (not isinstance(samples, list) or any(not isinstance(s, str) or
            not re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in samples) or
            len(set(samples)) != len(samples)):
        raise ValueError('sample-list 必须是无重复的安全样本 ID 数组')
    missing = [s for s in samples if s not in rows]
    if missing:
        raise ValueError(f'本次 quality.json 不含这些样本：{missing}')
    identity = {'quality_sha256': sha256(quality_path), 'sample_order': samples}
    manifest_path = output / 'review_manifest.json'
    old = {}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if any(old.get(k) != v for k, v in identity.items()):
            raise ValueError('输出目录已有其他实验或样本集合，请使用新目录以免混合证据')
    sources = []
    for sample in samples:
        overlay = quality_path.parent / 'structure_review' / f'{sample}.png'
        if overlay.exists():
            sources.append(('original_and_mask_overlay', [overlay]))
        else:
            row = rows[sample]
            paths = [image_root / row['input']] + [quality_path.parent / 'quality' /
                    f'{sample}_{group}.png' for group in 'ABC']
            if any(not p.is_file() for p in paths):
                raise FileNotFoundError(f'{sample} 缺对照图来源，请保留同次评测图片及输入图')
            sources.append(('originals_only_no_masks', paths))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir(exist_ok=True)
    records = []
    previous_images = {r['sample']: r['comparison_sha256'] for r in old.get('samples', [])}
    for sample, (kind, paths) in zip(samples, sources):
        dest = output / 'images' / f'{sample}.png'
        if kind == 'original_and_mask_overlay':
            data = paths[0].read_bytes()
        else:
            from PIL import Image, ImageDraw
            from .utils import load_image
            canvas = Image.new('RGB', (1024, 280), 'white')
            draw = ImageDraw.Draw(canvas)
            pictures = [load_image(paths[0])] + [Image.open(p).convert('RGB') for p in paths[1:]]
            for col, (side, picture) in enumerate(zip('XABC', pictures)):
                canvas.paste(picture.resize((256, 256)), (col * 256, 24))
                draw.text((col * 256 + 4, 4), f'{side} (no mask overlay)', fill='black')
            buffer = io.BytesIO()
            canvas.save(buffer, format='PNG')
            data = buffer.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        if sample in previous_images and previous_images[sample] != digest:
            raise ValueError(f'{sample} 图片来源发生变化，请使用新目录，不能覆盖已关联的人工证据')
        dest.write_bytes(data)
        records.append({'sample': sample, 'comparison_file': f'images/{sample}.png',
                        'comparison_kind': kind, 'comparison_sha256': digest,
                        'reasons': queue.get(sample, {}).get('reasons', ['selected_for_review']),
                        'detections': rows[sample]['structure']['detections'],
                        'human_status': 'pending'})
    manifest = {'schema': 'review-materials-v1', **identity, 'protocol': report['protocol'],
                'generated_at': q.get('generated_at'), 'samples': records,
                'human_review_complete': False}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    template = output / 'human_review_template.csv'
    if not template.exists():
        with template.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for row in records:
                writer.writerow({'sample': row['sample'], 'comparison_file': row['comparison_file'],
                                 'machine_reasons': ';'.join(row['reasons']), 'human_status': 'pending'})
    with (output / 'detection_failures.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample', 'side', 'label', 'score', 'reason'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(row for row in report['failures'] if row['sample'] in samples)
    lines = ['# 人工复核材料', '',
             '1. 将 human_review_template.csv 另存为 human_review.csv，再填写实际复核人、日期、结论和 notes。',
             '2. 查看 images 中 X/A/B/C 对照。叠加图上排是原图、下排是检测掩码；originals_only_no_masks 表示没有掩码叠加。',
             '3. 判断主体是否对应、是否漏检、生成外观与背景是否异常；不确定可保留 needs_discussion。',
             '4. reviewed 表示已查看并记录，不等于质量通过。命令不代填人工结论，也不读取或改写 human_review.csv。',
             '5. 共享时发送整个材料目录；不同实验使用新目录。人工意见只适用于对应图片，不能套用到另一轮输出。', '',
             f"质量 JSON SHA-256：`{identity['quality_sha256']}`；样本数：{len(samples)}。", '',
             '| 样本 | 对照图 | 机器原因 |', '|---|---|---|']
    lines += [f"| {row['sample']} | [查看]({row['comparison_file']}) | {'; '.join(row['reasons'])} |" for row in records]
    (output / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return manifest
