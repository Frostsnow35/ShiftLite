/* ShiftLite · 前端逻辑（无外部依赖） */

const NS = 'http://www.w3.org/2000/svg';
const GROUP_COLOR = { A: '#64748b', B: '#2563eb', C: '#d97706' };
const GROUP_ROLE = { A: '基线', B: '运行时收益', C: '量化收益' };

let STATE = null;
let SEL = { sample: null, source: 'samples', group: 'A', mode: 'side' };

/* ══════════ 工具 ══════════ */

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    // null / undefined / false 表示"不要这个属性"。若不跳过，
    // setAttribute('disabled', null) 会写出 disabled="null"，按钮照样被禁用。
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c != null) node.append(c.nodeType ? c : document.createTextNode(c));
  }
  return node;
}

function svg(tag, attrs = {}) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

function fmt(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}

/* ══════════ 启动 ══════════ */

async function init() {
  bindTabs();
  bindViewerTools();
  bindRun();
  bindUpload();
  bindSlider();

  try {
    const r = await fetch('/api/state');
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || '未知错误');
    STATE = j.state;

    // 支持 ?sample=horse_01&group=C 直接看某张图的结果（便于分享与截图）
    const q = new URLSearchParams(location.search);
    const autoSample = q.get('sample');
    const autoGroup = (q.get('group') || '').toUpperCase();
    if (autoSample) SEL.sample = autoSample;
    if (['A', 'B', 'C'].includes(autoGroup)) SEL.group = autoGroup;

    render();
    $('statusChip').className = 'chip status ok';
    $('statusChip').innerHTML = '<i></i>服务正常';

    if (autoSample) {
      const s = (STATE.samples || []).find((x) => x.id === autoSample);
      if (s) {
        preview(s.url);
        runInference();
      }
    }
  } catch (e) {
    $('statusChip').className = 'chip status bad';
    $('statusChip').innerHTML = '<i></i>连接失败';
    $('footTime').textContent = String(e.message || e);
  }
  $('footTime').textContent = new Date().toLocaleString('zh-CN');
}

function render() {
  const s = STATE;
  $('deviceChip').textContent = s.device && s.device !== '-' ? s.device : '未采集设备信息';
  renderSamples();
  renderGroups();
  renderKpis();
  renderProtocol();
  renderQuality();
  renderExperiments();
  renderHonesty();
  applyHash();
}

/* ══════════ 标签页 ══════════ */

function bindTabs() {
  document.querySelectorAll('.tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      activatePanel(btn.dataset.panel);
      history.replaceState(null, '', '#' + btn.dataset.panel);
    });
  });
}

function activatePanel(name) {
  document.querySelectorAll('.tab').forEach((b) => {
    b.classList.toggle('active', b.dataset.panel === name);
  });
  document.querySelectorAll('.panel-page').forEach((p) => {
    p.classList.toggle('active', p.id === 'page-' + name);
  });
}

function applyHash() {
  const name = (location.hash || '').replace('#', '');
  if (['demo', 'perf', 'quality', 'plan'].includes(name)) activatePanel(name);
}

function bindViewerTools() {
  document.querySelectorAll('.viewer-tools .ghost').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.viewer-tools .ghost').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      SEL.mode = btn.dataset.mode;
      applyMode();
    });
  });
}

function applyMode() {
  const compare = $('compare');
  const wrap = $('sliderWrap');
  const hasResult = compare.classList.contains('has-result');
  if (!hasResult) return;

  compare.classList.add('show');
  if (SEL.mode === 'slide') {
    compare.classList.remove('side');
    compare.classList.add('overlay');
    compare.style.display = 'none';
    wrap.classList.add('show');
  } else {
    compare.classList.add('side');
    compare.classList.remove('overlay');
    compare.style.display = '';
    wrap.classList.remove('show');
  }
}

function bindSlider() {
  const range = $('slideRange');
  range.addEventListener('input', () => {
    const v = range.value;
    $('slideOut').style.clipPath = `inset(0 0 0 ${v}%)`;
    $('slideHandle').style.left = v + '%';
  });
}

/* ══════════ 样例与后端 ══════════ */

function renderSamples() {
  const box = $('samples');
  box.innerHTML = '';
  const list = (STATE.samples || []);
  if (!list.length) {
    box.append(el('p', { class: 'hint', text: '样例图未就绪，可先上传' }));
    return;
  }
  list.forEach((s) => {
    const btn = el('button', {
      class: 'samp' + (SEL.sample === s.id && SEL.source === 'samples' ? ' active' : ''),
      title: s.name,
      onclick: () => {
        SEL.sample = s.id;
        SEL.source = 'samples';
        document.querySelectorAll('.samp').forEach((n) => n.classList.remove('active'));
        btn.classList.add('active');
        preview(s.url);
        $('runBtn').disabled = false;
        $('runNote').textContent = '';
      },
    }, [
      el('img', { src: s.url, alt: s.name, loading: 'lazy' }),
      s.kind === 'zebra' ? el('span', { class: 'tag', text: '斑马' }) : null,
    ]);
    box.append(btn);
  });
}

function renderGroups() {
  const box = $('groupSegments');
  box.innerHTML = '';
  (STATE.groups || []).forEach((g) => {
    const ok = g.available;
    const btn = el('button', {
      class: 'seg' + (SEL.group === g.id ? ' active' : ''),
      disabled: ok ? null : 'disabled',
      onclick: () => {
        SEL.group = g.id;
        document.querySelectorAll('.seg').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        $('groupHint').textContent = `${g.id} 组 · ${g.name} — ${
          g.id === 'A' ? '原生 PyTorch，作为参照系'
          : g.id === 'B' ? '只换推理引擎，隔离运行时收益'
          : '在 B 组基础上压精度，隔离量化收益'}`;
      },
    }, [
      el('span', { class: 'dot', style: `background:${GROUP_COLOR[g.id]}` }),
      el('span', { class: 'seg-name', text: `${g.id} · ${g.name}` }),
      el('span', { class: 'seg-role', text: ok ? GROUP_ROLE[g.id] : '未就绪' }),
    ]);
    box.append(btn);
  });

  const anyOk = (STATE.groups || []).some((g) => g.available);
  $('runBtn').disabled = !anyOk || !SEL.sample;
  if (!anyOk) $('runNote').textContent = '三组模型都还没生成，先跑 python cli.py setup';
}

function preview(url) {
  $('imgIn').src = url;
  $('slideIn').src = url;
  $('stageEmpty').style.display = 'none';
  $('compare').classList.add('show', 'has-result');
  applyMode();
  $('viewerBadge').textContent = '已载入输入';
  $('viewerBadge').className = 'badge';
}

/* ══════════ 推理 ══════════ */

function bindRun() {
  $('runBtn').addEventListener('click', runInference);
}

async function runInference() {
  if (!SEL.sample) return;
  $('stageLoading').classList.add('show');
  $('runBtn').disabled = true;
  $('viewerBadge').textContent = '推理中';
  $('viewerBadge').className = 'badge run';

  try {
    const r = await fetch('/api/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample: SEL.sample, group: SEL.group, source: SEL.source }),
    });
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || '推理失败');

    $('imgIn').src = j.input_url;
    $('imgOut').src = j.output_url;
    $('slideIn').src = j.input_url;
    $('slideOut').src = j.output_url;
    $('capOut').textContent = `输出 · ${j.group} 组`;
    $('stageEmpty').style.display = 'none';
    $('compare').classList.add('show', 'has-result');
    applyMode();

    $('lmLatency').textContent = fmt(j.latency_ms, 1) + ' ms';
    $('lmFps').textContent = fmt(j.fps, 1);
    $('lmGroup').textContent = `${j.group} · ${GROUP_ROLE[j.group]}`;
    $('lmSize').textContent = '256 × 256';

    $('viewerBadge').textContent = '完成';
    $('viewerBadge').className = 'badge ok';
    $('runNote').textContent = '';
  } catch (e) {
    $('viewerBadge').textContent = '失败';
    $('viewerBadge').className = 'badge';
    $('runNote').textContent = String(e.message || e);
  } finally {
    $('stageLoading').classList.remove('show');
    $('runBtn').disabled = false;
  }
}

function bindUpload() {
  $('fileInput').addEventListener('change', async (ev) => {
    const f = ev.target.files && ev.target.files[0];
    if (!f) return;
    if (f.size > 32 * 1024 * 1024) {
      $('runNote').textContent = '文件超过 32MB';
      return;
    }
    $('runNote').textContent = '上传中…';
    try {
      const buf = await f.arrayBuffer();
      const r = await fetch('/api/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: buf,
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || '上传失败');
      SEL.sample = j.id;
      SEL.source = 'uploads';
      document.querySelectorAll('.samp').forEach((n) => n.classList.remove('active'));
      preview(j.url);
      $('runBtn').disabled = false;
      $('runNote').textContent = '已上传自己的图片';
    } catch (e) {
      $('runNote').textContent = String(e.message || e);
    }
  });
}

/* ══════════ 性能面板 ══════════ */

function renderKpis() {
  const box = $('kpiRow');
  box.innerHTML = '';
  (STATE.groups || []).forEach((g) => {
    const has = g.available && g.mean_ms;
    const card = el('div', { class: 'kpi', style: `--k-color:${GROUP_COLOR[g.id]}` }, [
      el('div', { class: 'kpi-top' }, [
        el('span', { class: 'kpi-dot' }),
        el('span', { class: 'kpi-name', text: `${g.id} 组 · ${g.name}` }),
      ]),
      el('div', { class: 'kpi-main' }, [
        has ? fmt(g.mean_ms, 1) : '—',
        el('small', { text: has ? ' ms' : '' }),
      ]),
      el('div', {
        class: 'kpi-sub',
        text: has
          ? `${fmt(g.fps, 1)} FPS · ${fmt(g.size_mb, 2)} MB · 相对 A ${fmt(g.speedup_vs_A, 2)}×`
          : '尚未测速',
      }),
      has && g.precision
        ? el('div', { class: 'kpi-sub', style: 'margin-top:2px', text: `实际推理精度 ${g.precision}` })
        : null,
    ]);
    box.append(card);
  });

  const groups = (STATE.groups || []).filter((g) => g.available && g.mean_ms);
  latencyChart($('latencyChart'), groups);
  sizeChart($('sizeChart'), (STATE.groups || []).filter((g) => g.available));
}

function emptyChart(el, msg) {
  el.innerHTML = '';
  el.append(el_node('div', msg));
}

function el_node(tag, msg) {
  const d = document.createElement(tag);
  d.textContent = msg;
  d.style.cssText = 'padding:56px 0;text-align:center;color:#8a94a6;font-size:12.5px';
  return d;
}

function latencyChart(host, groups) {
  host.innerHTML = '';
  if (!groups.length) return host.append(el_node('div', '还没有测速数据，运行 python cli.py bench'));

  const W = 560, H = 260, pad = { t: 26, r: 20, b: 46, l: 52 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const max = Math.max(...groups.map((g) => (g.mean_ms || 0) + (g.std_ms || 0))) * 1.18 || 1;

  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });

  // 网格与 y 轴
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + ih - (ih * i) / 4;
    s.append(svg('line', {
      x1: pad.l, x2: W - pad.r, y1: y, y2: y,
      stroke: '#eef0f3', 'stroke-width': 1,
    }));
    const tx = svg('text', {
      x: pad.l - 10, y: y + 4, 'text-anchor': 'end',
      'font-size': 11, fill: '#8a94a6',
    });
    tx.textContent = Math.round((max * i) / 4);
    s.append(tx);
  }

  const bw = Math.min(64, (iw / groups.length) * 0.44);
  groups.forEach((g, i) => {
    const cx = pad.l + (iw / groups.length) * (i + 0.5);
    const h = (g.mean_ms / max) * ih;
    const x = cx - bw / 2;
    const y = pad.t + ih - h;

    s.append(svg('rect', {
      x, y, width: bw, height: h, rx: 6,
      fill: GROUP_COLOR[g.id], opacity: 0.9,
    }));

    // 误差棒
    if (g.std_ms) {
      const eh = (g.std_ms / max) * ih;
      s.append(svg('line', {
        x1: cx, x2: cx, y1: y - eh, y2: y + eh,
        stroke: '#14181f', 'stroke-width': 1.2, opacity: 0.55,
      }));
      s.append(svg('line', {
        x1: cx - 7, x2: cx + 7, y1: y - eh, y2: y - eh,
        stroke: '#14181f', 'stroke-width': 1.2, opacity: 0.55,
      }));
      s.append(svg('line', {
        x1: cx - 7, x2: cx + 7, y1: y + eh, y2: y + eh,
        stroke: '#14181f', 'stroke-width': 1.2, opacity: 0.55,
      }));
    }

    const val = svg('text', {
      x: cx, y: y - (g.std_ms ? (g.std_ms / max) * ih + 14 : 10),
      'text-anchor': 'middle', 'font-size': 12.5, 'font-weight': 600, fill: '#14181f',
    });
    val.textContent = fmt(g.mean_ms, 1) + ' ms';
    s.append(val);

    const lab = svg('text', {
      x: cx, y: H - pad.b + 20, 'text-anchor': 'middle', 'font-size': 12, fill: '#475467',
    });
    lab.textContent = `${g.id} · ${g.name}`;
    s.append(lab);

    if (g.speedup_vs_A && g.id !== 'A') {
      const sp = svg('text', {
        x: cx, y: H - pad.b + 36, 'text-anchor': 'middle', 'font-size': 11, fill: '#8a94a6',
      });
      sp.textContent = `${fmt(g.speedup_vs_A, 2)}× vs A`;
      s.append(sp);
    }
  });

  host.append(s);
}

function sizeChart(host, groups) {
  host.innerHTML = '';
  if (!groups.length) return host.append(el_node('div', '还没有模型体积数据'));

  const rows = groups.filter((g) => g.size_mb);
  const W = 560, rowH = 44, pad = { t: 16, r: 90, b: 16, l: 110 };
  const H = pad.t + pad.b + rows.length * rowH;
  const max = Math.max(...rows.map((g) => g.size_mb)) * 1.1 || 1;
  const iw = W - pad.l - pad.r;

  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
  const base = rows.find((g) => g.id === 'A');

  rows.forEach((g, i) => {
    const y = pad.t + i * rowH + 8;
    const w = (g.size_mb / max) * iw;

    const lab = svg('text', {
      x: pad.l - 12, y: y + 14, 'text-anchor': 'end', 'font-size': 12, fill: '#475467',
    });
    lab.textContent = `${g.id} · ${g.name}`;
    s.append(lab);

    s.append(svg('rect', {
      x: pad.l, y, width: iw, height: 20, rx: 5, fill: '#f2f4f7',
    }));
    s.append(svg('rect', {
      x: pad.l, y, width: Math.max(w, 3), height: 20, rx: 5,
      fill: GROUP_COLOR[g.id], opacity: 0.9,
    }));

    const val = svg('text', {
      x: pad.l + Math.max(w, 3) + 10, y: y + 14,
      'font-size': 12, 'font-weight': 600, fill: '#14181f',
    });
    val.textContent = fmt(g.size_mb, 2) + ' MB';
    s.append(val);

    if (base && g.id !== 'A' && base.size_mb) {
      const r = svg('text', {
        x: W - 16, y: y + 14, 'text-anchor': 'end', 'font-size': 11, fill: '#8a94a6',
      });
      r.textContent = `${fmt(g.size_mb / base.size_mb, 2)}× A`;
      s.append(r);
    }
  });

  host.append(s);
}

function renderProtocol() {
  const box = $('protocolBox');
  box.innerHTML = '';
  const p = STATE.protocol;
  const items = p ? [
    ['输入规格', p.input],
    ['batch size', p.batch_size],
    ['预热轮数', p.warmup],
    ['计时轮数', p.rounds],
    ['计入统计', `${p.rounds - p.dropped_rounds} 轮`],
    ['设备', STATE.device],
    ['运行环境', STATE.platform],
  ] : [['状态', '尚未测速，运行 python cli.py bench']];

  items.forEach(([k, v]) => {
    box.append(el('div', { class: 'proto-item' }, [
      el('div', { class: 'proto-k', text: k }),
      el('div', { class: 'proto-v', text: v ?? '—' }),
    ]));
  });
}

/* ══════════ 质量面板 ══════════ */

function renderQuality() {
  const q = STATE.quality;
  const notice = $('qualityNotice');

  if (!q) {
    notice.className = 'notice';
    notice.innerHTML = '还没有质量数据。运行 <b>python cli.py quality</b> 会在样例图上跑三组推理并计算结构指标。';
    $('lpipsChart').innerHTML = '';
    $('lpipsChart').append(el_node('div', '暂无数据'));
    $('structChart').innerHTML = '';
    $('structChart').append(el_node('div', '暂无数据'));
    $('qualityTable').innerHTML = '';
    return;
  }

  notice.className = 'notice';
  notice.innerHTML = `<b>口径说明</b>：这里的 SSIM / 边缘 SSIM 是真实计算结果（纯 numpy）。` +
    `LPIPS / FID / KID 需要感知网络权重，当前环境未安装 torchvision，` +
    `<b>表中 LPIPS 为结构相似度代理值，仅供界面联调，不可写进报告</b>。`;

  const m = q.mean || {};
  lpipsBars($('lpipsChart'), [
    { label: '转换差异 a→b', value: m.ssim_ab != null ? 1 - m.ssim_ab : null, color: '#94a3b8' },
    { label: '量化差异 b→c', value: m.ssim_bc != null ? 1 - m.ssim_bc : null, color: '#d97706' },
    { label: '总差异 a→c', value: m.ssim_ac != null ? 1 - m.ssim_ac : null, color: '#334155' },
  ]);

  structBars($('structChart'), [
    { label: 'SSIM (a vs b)', value: m.ssim_ab, color: '#2563eb' },
    { label: '边缘 SSIM (a vs b)', value: m.edge_ssim_ab, color: '#0ea5e9' },
    { label: 'SSIM (b vs c)', value: m.ssim_bc, color: '#d97706' },
    { label: '边缘 SSIM (b vs c)', value: m.edge_ssim_bc, color: '#f59e0b' },
  ]);

  renderQualityTable(q);
}

function lpipsBars(host, items) {
  host.innerHTML = '';
  const W = 560, rowH = 42, pad = { t: 14, r: 76, b: 14, l: 150 };
  const H = pad.t + pad.b + items.length * rowH;
  const vals = items.map((i) => Math.abs(i.value || 0));
  const max = Math.max(...vals, 0.02) * 1.25;
  const iw = W - pad.l - pad.r;

  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
  items.forEach((it, i) => {
    const y = pad.t + i * rowH + 6;
    const w = (Math.abs(it.value || 0) / max) * iw;

    const lab = svg('text', {
      x: pad.l - 12, y: y + 13, 'text-anchor': 'end', 'font-size': 11.5, fill: '#475467',
    });
    lab.textContent = it.label;
    s.append(lab);

    s.append(svg('rect', { x: pad.l, y, width: iw, height: 18, rx: 5, fill: '#f2f4f7' }));
    s.append(svg('rect', {
      x: pad.l, y, width: Math.max(w, 2), height: 18, rx: 5, fill: it.color, opacity: .9,
    }));

    const v = svg('text', {
      x: pad.l + Math.max(w, 2) + 10, y: y + 13,
      'font-size': 11.5, 'font-weight': 600, fill: '#14181f',
    });
    v.textContent = it.value == null ? '—' : Number(it.value).toFixed(4);
    s.append(v);
  });
  host.append(s);
}

function structBars(host, items) {
  host.innerHTML = '';
  const W = 560, rowH = 42, pad = { t: 14, r: 70, b: 14, l: 150 };
  const H = pad.t + pad.b + items.length * rowH;
  const iw = W - pad.l - pad.r;

  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
  items.forEach((it, i) => {
    const y = pad.t + i * rowH + 6;
    const v = Math.max(0, Math.min(1, it.value || 0));
    const w = v * iw;

    const lab = svg('text', {
      x: pad.l - 12, y: y + 13, 'text-anchor': 'end', 'font-size': 11.5, fill: '#475467',
    });
    lab.textContent = it.label;
    s.append(lab);

    s.append(svg('rect', { x: pad.l, y, width: iw, height: 18, rx: 5, fill: '#f2f4f7' }));
    s.append(svg('rect', {
      x: pad.l, y, width: Math.max(w, 2), height: 18, rx: 5, fill: it.color, opacity: .9,
    }));

    const t = svg('text', {
      x: pad.l + Math.max(w, 2) + 10, y: y + 13,
      'font-size': 11.5, 'font-weight': 600, fill: '#14181f',
    });
    t.textContent = it.value == null ? '—' : Number(it.value).toFixed(4);
    s.append(t);
  });
  host.append(s);
}

function renderQualityTable(q) {
  const t = $('qualityTable');
  t.innerHTML = '';
  $('qualitySub').textContent = `${q.n_samples} 张样例 · ${q.generated_at}`;

  const cols = ['样例', 'SSIM(a,b)', '边缘(a,b)', 'SSIM(b,c)', '边缘(b,c)', 'SSIM(a,c)', 'PSNR(b,c)'];
  const thead = el('thead', {}, [el('tr', {}, cols.map((c) => el('th', { text: c })))]);
  const tbody = el('tbody', {}, (q.rows || []).map((r) => el('tr', {}, [
    el('td', { text: r.sample }),
    el('td', { text: r.ssim_ab != null ? fmt(r.ssim_ab, 4) : '—' }),
    el('td', { text: r.edge_ssim_ab != null ? fmt(r.edge_ssim_ab, 4) : '—' }),
    el('td', { text: r.ssim_bc != null ? fmt(r.ssim_bc, 4) : '—' }),
    el('td', { text: r.edge_ssim_bc != null ? fmt(r.edge_ssim_bc, 4) : '—' }),
    el('td', { text: r.ssim_ac != null ? fmt(r.ssim_ac, 4) : '—' }),
    el('td', { text: r.psnr_bc != null ? fmt(r.psnr_bc, 2) : '—' }),
  ])));
  t.append(thead, tbody);
}

/* ══════════ 实验进度 ══════════ */

function renderExperiments() {
  const box = $('expGrid');
  box.innerHTML = '';
  const flag = { done: '已完成', partial: '部分完成', todo: '未开始' };
  (STATE.experiments || []).forEach((e) => {
    box.append(el('div', { class: `exp ${e.status}` }, [
      el('div', { class: 'exp-head' }, [
        el('span', { class: 'exp-code', text: e.code }),
        el('span', { class: 'exp-title', text: e.title }),
        el('span', { class: 'exp-flag', text: flag[e.status] || e.status }),
      ]),
      el('div', { class: 'exp-note', text: e.note || '' }),
      el('div', { class: 'exp-note', text: e.half || '' }),
    ]));
  });
}

function renderHonesty() {
  const box = $('honestyBox');
  box.innerHTML = '';
  const rows = [
    ['✅', '真实实测', 'A/B/C 三组推理、延迟与 FPS、模型体积、SSIM 与边缘 SSIM —— 全部由本机真实计算。'],
    ['⚠️', '代理指标', 'LPIPS / FID / KID 需要感知网络权重，当前未安装 torchvision，界面中的 LPIPS 为结构相似度代理值。'],
    ['⛔', '尚未接入', 'Mask R-CNN 实例掩码 IoU、ResNet-50 斑马判别、apple2orange 迁移对 —— 代码位置已预留。'],
    ['📌', '必须写进口径', 'FID/KID 衡量的是与真实目标域的特征分布距离，不等于「逼真度」；多接一对只说明系统兼容，不说明模型通用。'],
  ];
  rows.forEach(([mark, title, text]) => {
    box.append(el('div', { class: 'honesty-row' }, [
      el('span', { class: 'h-mark', text: mark }),
      el('span', {}, [el('b', { text: title + '：' }), text]),
    ]));
  });
}

document.addEventListener('DOMContentLoaded', init);
