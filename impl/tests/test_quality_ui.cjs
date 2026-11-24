const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, 'e3_partial_b_only.json'), 'utf8'));

function sliceBetween(startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  assert.notEqual(start, -1, `missing start marker: ${startMarker}`);
  const end = source.indexOf(endMarker, start + startMarker.length);
  assert.notEqual(end, -1, `missing end marker: ${endMarker}`);
  return source.slice(start, end);
}

function element(tag, props = {}, children = []) {
  const node = {
    tag,
    className: props.class || '',
    children: [],
    attrs: {},
    set innerHTML(val) {
      this._html = String(val);
      this.children = val ? [String(val)] : [];
    },
    get innerHTML() {
      return this._html || '';
    },
    set textContent(val) {
      this._html = '';
      this.children = val === '' || val == null ? [] : [String(val)];
    },
    get textContent() {
      return flattenText(this);
    },
    append(...items) {
      for (const item of items) {
        if (item !== null && item !== undefined && item !== false) {
          this.children.push(item);
        }
      }
    },
    setAttribute(k, v) {
      this.attrs[k] = String(v);
    },
  };
  if (props.text !== undefined && props.text !== null && props.text !== false) {
    node.children.push(String(props.text));
  }
  for (const c of [].concat(children)) {
    if (c !== null && c !== undefined && c !== false) {
      node.children.push(c);
    }
  }
  return node;
}

function flattenText(node) {
  if (node == null) return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  return (node.children || []).map(flattenText).join('');
}

function createSandbox(initialState = null) {
  const hosts = {
    qualityNotice: element('div'),
    lpipsChart: element('div'),
    structChart: element('div'),
    qualityTable: element('table'),
    qualitySub: element('div'),
    deepTiles: element('ul'),
  };
  const context = {
    STATE: initialState,
    $: (id) => hosts[id],
    el: element,
    BigInt,
    Number,
    String,
    Math,
    Array,
    Object,
    bars(host, items, title) {
      host.textContent = `${title}:${items.map((i) => `${i.label}=${i.value}`).join(',')}`;
    },
    emptyChart(host, msg) {
      host.textContent = msg;
    },
    bindTips() {},
    tipHtml(label, rows) { return JSON.stringify(rows); },
  };
  vm.createContext(context);
  const helpersCode = sliceBetween('function fmt(v, digits = 1)', 'function relTime(ts)');
  const qualityPanelCode = sliceBetween('function qualityClassificationState(q)', 'function bars(host, items, title, unit)');
  const qualityTableCode = sliceBetween('function renderQualityTable(q)', '\nfunction renderExperiments()');
  vm.runInContext(`${helpersCode}\n${qualityPanelCode}\n${qualityTableCode}`, context);
  return { context, hosts };
}

function manifest(id) {
  const cfg = JSON.parse(fs.readFileSync(path.join(__dirname, '../pairs', id + '.json'), 'utf8'));
  return { ...cfg, ref_prefix: cfg.dataset.ref_prefix };
}
const h2zPair = manifest('horse2zebra');
const winterPair = manifest('summer2winter');
const winterReason = winterPair.metrics.class_discrimination.reason;

/* 1. 保留原有逐图表格契约测试 */
function renderRow(row) {
  const { context, hosts } = createSandbox({ pair: h2zPair });
  context.renderQualityTable({ metrics_source: 'real', n_samples: 1, rows: [row] });
  return hosts.qualityTable.children[1].children[0].children;
}
const cells = renderRow({
  sample: 's', canny_edge_ssim_ab: 0.1111, edge_ssim_ab: 0.9999,
  canny_edge_ssim_bc: 0.2222, edge_ssim_bc: 0.8888, mask_dice_ac: 0.5,
  structure: { pairs: { x_c: { iou: 0.2, dice: 0.3, background_ssim: null } } },
});
assert.equal(flattenText(cells[2]), '0.1111');
assert.equal(flattenText(cells[4]), '0.2222');
assert.equal(flattenText(cells[7]), '0.500');
assert.equal(flattenText(cells[8]), '0.200');
assert.equal(flattenText(cells[10]), '—');
assert.equal(flattenText(renderRow({ sample: 'old', edge_ssim_ab: 0.9999 })[2]), '0.9999');
assert.equal(flattenText(renderRow({ sample: 'old', edge_ssim_bc: 0.8888 })[4]), '0.8888');

/* 2. 半分位舍入对齐 Python :.2%（0.03125 -> 3.12%，0.09375 -> 9.38%） */
{
  const { context } = createSandbox();
  assert.equal(context.fmtClassPct(0.03125), '3.12%');
  assert.equal(context.fmtClassPct(0.09375), '9.38%');
  assert.equal(context.fmtClassPct(0), '0.00%');
  assert.equal(context.fmtClassPct(1), '100.00%');
  assert.equal(context.fmtClassPct(null), '—');
  for (const [v, pct] of [[0.38835, '38.83%'], [0.61165, '61.17%'], [-0, '-0.00%'], [Number.MIN_VALUE, '0.00%']]) {
    assert.equal(context.fmtClassPct(v), pct);
  }
}

/* 3. 六种 quality_classification 状态、分母、单侧 FID/KID 缺失及跨迁移对切换 */
function assertClassificationShape(cls) {
  assert.deepEqual(
    Object.keys(cls).sort(),
    ['batches', 'enabled', 'reason', 'source', 'status', 'target'],
  );
  assert.deepEqual(Object.keys(cls.batches).sort(), ['b', 'c']);
}

function tileMap(hosts) {
  const out = {};
  for (const li of hosts.deepTiles.children) {
    if (li.tag !== 'li') continue;
    const label = flattenText(li.children[0]);
    const value = flattenText(li.children[1]);
    const sub = flattenText(li.children[2]);
    out[label] = { value, sub, full: `${label} ${value} ${sub}` };
  }
  return out;
}

function assertNoBadTokens(hosts) {
  const all = flattenText(hosts.qualityNotice) + ' ' + flattenText(hosts.deepTiles);
  assert.ok(!all.includes('undefined'), `unexpected undefined in UI: ${all}`);
  assert.ok(!all.includes('NaN'), `unexpected NaN in UI: ${all}`);
}

const { context, hosts } = createSandbox();

// (a) untested：无质量记录保持空态
const clsUntested = {
  enabled: true, status: 'untested', reason: '', target: 'zebra', source: null,
  batches: { b: null, c: null },
};
assertClassificationShape(clsUntested);
context.STATE = { pair: h2zPair, quality: null, quality_classification: clsUntested };
context.renderQuality();
assert.match(flattenText(hosts.qualityNotice), /还没有质量数据/);
assert.equal(hosts.deepTiles.children.length, 0);

// (b) proxy：不指向不存在的 deep.errors，提示尚未运行真实类别判别
const clsProxy = {
  enabled: true, status: 'proxy', reason: '', target: 'zebra', source: null,
  batches: { b: null, c: null },
};
assertClassificationShape(clsProxy);
context.STATE = {
  pair: h2zPair,
  quality: { metrics_source: 'proxy', n_samples: 6, mean: { ssim_bc: 0.89, psnr_bc: 29.8 }, rows: [] },
  quality_classification: clsProxy,
};
context.renderQuality();
assertNoBadTokens(hosts);
assert.match(flattenText(hosts.qualityNotice), /尚未运行真实类别判别/);
assert.doesNotMatch(flattenText(hosts.qualityNotice), /deep\.errors/);
assert.equal(tileMap(hosts)['类别判别'].sub, '尚未运行真实类别判别（需 quality --deep）');

// (c) E3-01 + E3-05：horse2zebra（仅 B=75%, n=12） -> summer2winter（禁用但带同一份遗留 JSON） -> 切回 horse2zebra
const clsPartialB = {
  enabled: true, status: 'partial', reason: '', target: 'zebra', source: 'class_discrimination',
  batches: { b: { top1_accuracy: 0.75, n: 12 }, c: null },
};
const clsDisabled = {
  enabled: false, status: 'disabled', reason: winterReason, target: 'winter', source: null,
  batches: { b: null, c: null },
};
assertClassificationShape(clsPartialB);
assertClassificationShape(clsDisabled);

context.STATE = { pair: h2zPair, quality: fixture, quality_classification: clsPartialB };
context.renderQuality();
assertNoBadTokens(hosts);
let tiles = tileMap(hosts);
assert.equal(tiles['判别 · B 批（zebra）'].value, '75.00%（n=12）');
assert.equal(tiles['判别 · C 批（zebra）'].value, '缺少 C 批结果');
assert.match(flattenText(hosts.qualityNotice), /B 批 75\.00%，尚缺 C 批，B 批 n=12/);
assert.match(flattenText(hosts.qualityNotice), /输入样本量 20 张/);

// 切到 summer2winter（故意保留同一份含 zebra 75% 的 quality）
context.STATE = { pair: winterPair, quality: fixture, quality_classification: clsDisabled };
context.renderQuality();
assertNoBadTokens(hosts);
tiles = tileMap(hosts);
assert.equal(tiles['类别判别'].value, '类别判别不适用');
assert.equal(tiles['类别判别'].sub, winterReason);
assert.ok(!('判别 · B 批（zebra）' in tiles));
assert.doesNotMatch(flattenText(hosts.qualityNotice) + flattenText(hosts.deepTiles), /75\.00%/);
assert.match(flattenText(hosts.qualityNotice), new RegExp(winterReason));
assert.doesNotMatch(flattenText(hosts.qualityNotice), /人工待复核/);

// 切回 horse2zebra：B 批 75.00%（n=12）恢复，前一对原因消失
context.STATE = { pair: h2zPair, quality: fixture, quality_classification: clsPartialB };
context.renderQuality();
assertNoBadTokens(hosts);
tiles = tileMap(hosts);
assert.equal(tiles['判别 · B 批（zebra）'].value, '75.00%（n=12）');
assert.equal(tiles['判别 · C 批（zebra）'].value, '缺少 C 批结果');
assert.doesNotMatch(flattenText(hosts.qualityNotice) + flattenText(hosts.deepTiles), new RegExp(winterReason));

// (d) complete（含半分位 0.03125 / 0.09375 与某批 n=null 样本量未记录）及单侧 FID/KID 缺失
const clsComplete = {
  enabled: true, status: 'complete', reason: '', target: 'zebra', source: 'zebra_discrimination',
  batches: { b: { top1_accuracy: 0.03125, n: 12 }, c: { top1_accuracy: 0.09375, n: null } },
};
assertClassificationShape(clsComplete);
context.STATE = {
  pair: h2zPair,
  quality: {
    metrics_source: 'partial',
    n_samples: 12,
    mean: { ssim_bc: 0.9, psnr_bc: 28.5 },
    rows: [],
    deep: {
      fid: { n_generated: 12, n_real: 14, ref_domain: 'zebra', b_vs_real_zebra: 80.5 },
      kid: { b_vs_real_zebra: { mean: 0.02, std: 0.005, subset_size: 10, n_subsets: 20 } },
    },
  },
  quality_classification: clsComplete,
};
context.renderQuality();
assertNoBadTokens(hosts);
tiles = tileMap(hosts);
assert.equal(tiles['判别 · B 批（zebra）'].value, '3.12%（n=12）');
assert.equal(tiles['判别 · C 批（zebra）'].value, '9.38%（样本量未记录）');
assert.equal(tiles['FID · B 批'].value, '80.50');
assert.equal(tiles['FID · C 批'].value, '缺失');
assert.equal(tiles['KID · B 批'].value, '0.0200');
assert.equal(tiles['KID · C 批'].value, '缺失');

// (e) missing：真实或部分深评无可用分类批次
const clsMissing = {
  enabled: true, status: 'missing', reason: '', target: 'zebra', source: null,
  batches: { b: null, c: null },
};
assertClassificationShape(clsMissing);
context.STATE = {
  pair: h2zPair,
  quality: { metrics_source: 'real', n_samples: 10, mean: {}, rows: [], deep: { errors: ['cls'] } },
  quality_classification: clsMissing,
};
context.renderQuality();
assertNoBadTokens(hosts);
tiles = tileMap(hosts);
assert.equal(tiles['类别判别'].value, '缺失');
assert.match(tiles['类别判别'].sub, /此次未取得有效类别判别结果/);

// Sparse structure and review materials must not interrupt the classification tiles.
const partial = structuredClone(fixture);
partial.deep.structure = { pairs: {} };
context.STATE = { pair: h2zPair, quality: partial, quality_classification: clsPartialB };
context.renderQuality();
assert.equal(tileMap(hosts)['判别 · B 批（zebra）'].value, '75.00%（n=12）');
assert.equal(tileMap(hosts)['输出一致性 a_c · IoU'].value, '缺失');
assert.doesNotMatch(flattenText(hosts.qualityNotice), /待复核/);
partial.deep.structure.review_queue = [];
context.renderQuality();
assert.match(flattenText(hosts.qualityNotice), /待复核 0 项/);
partial.deep.structure.review_queue.push({sample:'example'});
context.renderQuality();
assert.ok(!hosts.deepTiles.children.some(el => el.tag === 'a'));
context.STATE.structure_review_url = '/artifacts/results/structure_review/review_queue.json';
context.renderQuality();
assert.ok(hosts.deepTiles.children.some(el => el.tag === 'a'));

const history = JSON.parse(fs.readFileSync(path.join(__dirname, '../artifacts/results/quality.json'), 'utf8'));
context.STATE = { pair: winterPair, quality: history, quality_classification: clsDisabled };
context.renderQuality();
assert.equal(tileMap(hosts)['FID · B 批'].value, '缺失');
assert.match(tileMap(hosts)['FID · B 批'].sub, /winter/);
assert.equal(tileMap(hosts)['KID · B 批'].value, '缺失');
const row = hosts.qualityTable.children[1].children[0];
assert.equal(flattenText(row.children[6]), '—');
assert.ok(row.attrs['data-tip']);
assert.doesNotMatch(row.attrs['data-tip'], /"掩码 IoU .a,c.","[0-9]/);
context.STATE.pair = {...h2zPair, metrics:{...h2zPair.metrics, background_ssim:false}};
context.STATE.quality_classification = clsComplete;
for (const [value, n] of [[undefined,'未记录'],[null,'未记录'],[0,110],[0.7,110]]) {
  history.deep.mask.mean_dice_ac = value;
  context.renderQuality();
  assert.equal(tileMap(hosts)['输出一致性 Dice (a,c)'].sub, `有效 ${n}/120`);
  assert.ok(!tileMap(hosts)['旧背景 SSIM (a,c)']);
}
history.metrics_source = 'proxy';
context.STATE.quality_classification = clsProxy;
context.renderQuality();
assert.ok(!tileMap(hosts)['FID · B 批']);
assert.ok(!tileMap(hosts)['输出一致性 IoU (a,c)']);
assert.equal(flattenText(hosts.qualityTable.children[1].children[0].children[6]), '—');
context.STATE = {pair:h2zPair, quality:partial, quality_classification:clsPartialB,
  quality_measurements:{pixels:[], deep:['FID（仅 B 批）', 'KID（仅 C 批）']}};
partial.deep.fid = { b_vs_real_zebra:0 };
partial.deep.kid = { c_vs_real_zebra:{mean:0} };
context.renderQuality();
assert.match(flattenText(hosts.qualityNotice), /FID（仅 B 批）、KID（仅 C 批）/);
assert.equal(tileMap(hosts)['FID · B 批'].value, '0.00');
assert.equal(tileMap(hosts)['KID · C 批'].value, '0.0000');
partial.deep.fid = {n_real:140};
partial.deep.kid = {};
context.STATE.quality_measurements.deep = [];
context.renderQuality();
assert.doesNotMatch(flattenText(hosts.qualityNotice), /FID|KID|SSIM|PSNR|MAE/);
assert.equal(tileMap(hosts)['FID · B 批'].value, '缺失');
console.log('PASS: E3 quality panel, target domains, partial structure, proxy/disabled isolation, Dice counts and percentage rounding');
