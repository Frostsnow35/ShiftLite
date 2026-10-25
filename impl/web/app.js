/* ShiftLite · 前端交互层
   零外部依赖：手绘 SVG 图表、原生 Canvas 差异图、CSS 动效，完全离线可用。
   数据纪律：界面只呈现 /api/state 与 /api/infer 返回的内容。
            服务端没有返回的中间量（如逐阶段耗时）不编造，只用真实墙钟计时。 */

const NS = 'http://www.w3.org/2000/svg';
const GROUP_COLOR = { A: '#6b7580', B: '#2c56a3', C: '#b07717' };
const GROUP_ROLE = { A: '基线', B: '运行时收益', C: '量化收益' };
const HISTORY_KEY = 'shiftlite.runs.v1';

const C_GRID = '#e8e6e0';
const C_AXIS = '#bdbab1';
const C_TEXT = '#6d737d';
const C_INK = '#14161a';

let STATE = null;
let SEL = { sample: null, source: 'samples', group: 'A', mode: 'side' };
let TOOL = { loupe: false, grid: false };
let HISTORY = [];
let TIMER = null;

/* ══════════ 工具 ══════════ */

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, String(v));
  }
  for (const c of [].concat(children)) {
    if (c != null) node.append(c.nodeType ? c : document.createTextNode(c));
  }
  return node;
}

function svg(tag, attrs = {}) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    node.setAttribute(k, String(v));
  }
  return node;
}

function fmt(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}

function pctv(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return (Number(v) * 100).toFixed(digits) + '%';
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function relTime(ts) {
  const d = new Date(ts);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
}

/* ══════════ 浮动提示 ══════════ */

const tip = { box: null };

function showTip(ev, html) {
  tip.box.innerHTML = html;
  tip.box.hidden = false;
  requestAnimationFrame(() => tip.box.classList.add('is-on'));
  moveTip(ev);
}

function moveTip(ev) {
  const pad = 14;
  const w = tip.box.offsetWidth;
  const h = tip.box.offsetHeight;
  let x = ev.clientX + pad;
  let y = ev.clientY + pad;
  if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = ev.clientY - h - pad;
  tip.box.style.left = `${Math.max(8, x)}px`;
  tip.box.style.top = `${Math.max(8, y)}px`;
}

function hideTip() {
  tip.box.classList.remove('is-on');
}

function bindTips(host) {
  host.querySelectorAll('[data-tip]').forEach((n) => {
    n.addEventListener('pointerenter', (ev) => showTip(ev, n.getAttribute('data-tip')));
    n.addEventListener('pointermove', moveTip);
    n.addEventListener('pointerleave', hideTip);
  });
}

function tipHtml(title, rows) {
  const body = rows.map(([k, v]) =>
    `<div class="tip__row"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('');
  return `<div class="tip__title">${esc(title)}</div>${body}`;
}

/* ══════════ 启动 ══════════ */

async function init() {
  tip.box = $('tip');
  bindTabs();
  bindViewerTools();
  bindWipe();
  bindLoupe();
  bindUpload();
  bindRun();
  bindInfoNotes();
  HISTORY = loadHistory();
  renderHistory();

  window.addEventListener('resize', moveIndicator);

  try {
    const r = await fetch('/api/state');
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || '未知错误');
    STATE = j.state;
    render();
    setStatus('ok', '服务正常');
    moveIndicator();

    const q = new URLSearchParams(location.search);
    const autoSample = q.get('sample');
    const autoGroup = (q.get('group') || '').toUpperCase();
    if (autoSample) SEL.sample = autoSample;
    if (['A', 'B', 'C'].includes(autoGroup)) SEL.group = autoGroup;

    if (autoSample) {
      const s = (STATE.samples || []).find((x) => x.id === autoSample);
      if (s) {
        selectSample(s.id, 'samples');
        runInference();
      }
    }
  } catch (e) {
    setStatus('bad', '连接失败');
    $('footTime').textContent = String(e.message || e);
  }
  $('footTime').textContent = new Date().toLocaleString('zh-CN');
}

function setStatus(kind, text) {
  const chip = $('statusChip');
  chip.className = 'chip ' + (kind === 'ok' ? 'is-ok' : kind === 'bad' ? 'is-bad' : '');
  chip.innerHTML = `<i></i>${esc(text)}`;
}

function render() {
  const s = STATE;
  $('deviceChip').textContent = s.device && s.device !== '-' ? s.device : '未采集设备信息';
  renderSamples();
  renderGroups();
  renderKpis();
  renderProtocol();
  renderQuality();
  renderLayerChart();
  renderExperiments();
  renderHonesty();
}

/* ══════════ 标签页 ══════════ */

function bindTabs() {
  const tabs = [...document.querySelectorAll('.tab')];
  tabs.forEach((btn, i) => {
    btn.addEventListener('click', () => activatePanel(btn.dataset.panel));
    btn.addEventListener('keydown', (ev) => {
      const map = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 };
      if (!(ev.key in map)) return;
      ev.preventDefault();
      const next = (map[ev.key] + tabs.length) % tabs.length;
      tabs[next].focus();
      activatePanel(tabs[next].dataset.panel);
    });
  });
  // 支持 ?/# 直达某一页
  const name = (location.hash || '').replace('#', '');
  if (['demo', 'perf', 'quality', 'plan'].includes(name)) activatePanel(name, false);
}

function activatePanel(name, focus = true) {
  document.querySelectorAll('.tab').forEach((b) => {
    const on = b.dataset.panel === name;
    b.setAttribute('aria-selected', on ? 'true' : 'false');
    b.tabIndex = on ? 0 : -1;
  });
  document.querySelectorAll('.page').forEach((p) => {
    p.classList.toggle('is-active', p.id === 'page-' + name);
  });
  moveIndicator();
  // 保留地址栏中的页面锚点，便于把某一页直接分享给别人
  history.replaceState(null, '', '#' + name);
  if (focus) document.querySelector(`.tab[data-panel="${name}"]`).focus();
  if (name === 'quality' || name === 'perf') hideTip();
}

function moveIndicator() {
  const active = document.querySelector('.tab[aria-selected="true"]');
  const bar = $('tabIndicator');
  if (!active || !bar) return;
  bar.style.width = `${active.offsetWidth}px`;
  bar.style.transform = `translateX(${active.offsetLeft}px)`;
}

/* ══════════ 查看模式与工具 ══════════ */

function bindViewerTools() {
  document.querySelectorAll('.viewer__head [data-mode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      SEL.mode = btn.dataset.mode;
      document.querySelectorAll('.viewer__head [data-mode]').forEach((b) => {
        const on = b === btn;
        b.classList.toggle('is-active', on);
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      $('stage').dataset.mode = SEL.mode;
      hideTip();
      if (SEL.mode === 'diff') renderDiff();
    });
  });

  document.querySelectorAll('.viewer__head [data-tool]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.tool;
      TOOL[key] = !TOOL[key];
      btn.classList.toggle('is-active', TOOL[key]);
      btn.setAttribute('aria-pressed', TOOL[key] ? 'true' : 'false');
      $('stage').classList.toggle('is-grid', TOOL.grid);
      if (!TOOL.loupe) [$('loupeIn'), $('loupeOut')].forEach((n) => n.classList.remove('is-on'));
    });
  });
}

function bindWipe() {
  const range = $('slideRange');
  range.addEventListener('input', updateWipe);

  const area = $('wipe');
  area.addEventListener('pointerdown', (ev) => {
    if (SEL.mode !== 'slide') return;
    const rect = area.getBoundingClientRect();
    const set = (x) => {
      range.value = String(Math.max(0, Math.min(100, ((x - rect.left) / rect.width) * 100)));
      updateWipe();
    };
    set(ev.clientX);
    const move = (e) => set(e.clientX);
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  });

  // 分界线可用方向键微调（Shift 加速）
  $('stage').addEventListener('keydown', (ev) => {
    if (SEL.mode !== 'slide') return;
    const step = ev.shiftKey ? 5 : 1;
    if (ev.key === 'ArrowLeft') range.value = String(Math.max(0, Number(range.value) - step));
    else if (ev.key === 'ArrowRight') range.value = String(Math.min(100, Number(range.value) + step));
    else if (ev.key === 'Home') range.value = '0';
    else if (ev.key === 'End') range.value = '100';
    else return;
    ev.preventDefault();
    updateWipe();
  });
}

function updateWipe() {
  const v = Number($('slideRange').value);
  $('slideOut').style.clipPath = `inset(0 0 0 ${v}%)`;
  $('slideHandle').style.left = `${v}%`;
}

function bindLoupe() {
  const stage = $('stage');
  [['imgIn', 'loupeIn'], ['imgOut', 'loupeOut']].forEach(([imgId, lensId]) => {
    const img = $(imgId);
    const lens = $(lensId);
    img.addEventListener('mouseenter', () => {
      if (TOOL.loupe && SEL.mode === 'side' && stage.dataset.has === '1') lens.classList.add('is-on');
    });
    img.addEventListener('mouseleave', () => lens.classList.remove('is-on'));
    img.addEventListener('mousemove', (ev) => {
      if (!TOOL.loupe || SEL.mode !== 'side' || stage.dataset.has !== '1') {
        lens.classList.remove('is-on');
        return;
      }
      const zoom = 2.6;
      const R = 66;
      const r = img.getBoundingClientRect();
      const sr = stage.getBoundingClientRect();
      const px = (ev.clientX - r.left) / r.width;
      const py = (ev.clientY - r.top) / r.height;
      lens.style.backgroundImage = `url("${img.currentSrc || img.src}")`;
      lens.style.backgroundSize = `${r.width * zoom}px ${r.height * zoom}px`;
      lens.style.backgroundPosition = `${-(px * r.width * zoom - R)}px ${-(py * r.height * zoom - R)}px`;
      lens.style.left = `${ev.clientX - sr.left - R}px`;
      lens.style.top = `${ev.clientY - sr.top - R}px`;
      lens.classList.add('is-on');
    });
  });
}

/* 差异热力图：Canvas 逐像素计算，均值在图上标注来源 */

function renderDiff() {
  const canvas = $('diffCanvas');
  const size = 256;
  const stat = $('diffStat');
  if ($('stage').dataset.has !== '1') {
    stat.textContent = '先跑一次推理';
    return;
  }
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const a = new Image();
  const b = new Image();
  let pending = 2;
  const next = () => {
    pending -= 1;
    if (pending === 0) paint();
  };
  a.onerror = b.onerror = () => { stat.textContent = '图像载入失败'; };
  a.onload = next;
  b.onload = next;
  a.src = $('imgIn').src;
  b.src = $('imgOut').src;

  function paint() {
    const off = document.createElement('canvas');
    off.width = size;
    off.height = size;
    const octx = off.getContext('2d');
    octx.drawImage(a, 0, 0, size, size);
    const da = octx.getImageData(0, 0, size, size);
    octx.clearRect(0, 0, size, size);
    octx.drawImage(b, 0, 0, size, size);
    const db = octx.getImageData(0, 0, size, size);
    const out = ctx.createImageData(size, size);
    let sum = 0;

    for (let i = 0; i < da.data.length; i += 4) {
      const d = (Math.abs(da.data[i] - db.data[i]) +
        Math.abs(da.data[i + 1] - db.data[i + 1]) +
        Math.abs(da.data[i + 2] - db.data[i + 2])) / 3;
      sum += d;
      const t = d / 255;
      let r, g, bl;
      if (t < 0.25) {
        const u = t / 0.25;
        r = 110 + 40 * u; g = 128 + 22 * u; bl = 156 - 6 * u;
      } else if (t < 0.6) {
        const u = (t - 0.25) / 0.35;
        r = 150 + 80 * u; g = 150 - 40 * u; bl = 150 - 110 * u;
      } else {
        const u = (t - 0.6) / 0.4;
        r = 230 - 38 * u; g = 110 - 58 * u; bl = 40 + 6 * u;
      }
      out.data[i] = r;
      out.data[i + 1] = g;
      out.data[i + 2] = bl;
      out.data[i + 3] = d < 2 ? 0 : Math.min(255, 55 + t * 200);
    }
    ctx.putImageData(out, 0, 0);
    const n = da.data.length / 4;
    stat.textContent = `平均 |Δ| = ${(sum / n).toFixed(2)} / 255 · 由浏览器实时计算`;
  }
}

function develop(img) {
  img.classList.remove('is-loaded');
  void img.offsetWidth;
  img.classList.add('is-loaded');
}

/* ══════════ 样例、上传、后端 ══════════ */

function renderSamples() {
  const box = $('samples');
  box.innerHTML = '';
  const list = STATE.samples || [];
  if (!list.length) {
    box.append(el('p', { class: 'hint', text: '样例图未就绪，可直接上传' }));
    return;
  }
  list.forEach((s) => {
    const active = SEL.sample === s.id && SEL.source === 'samples';
    box.append(el('button', {
      class: 'samp' + (active ? ' is-active' : ''),
      role: 'radio',
      'aria-checked': active ? 'true' : 'false',
      title: s.name,
      onclick: () => selectSample(s.id, 'samples'),
    }, [
      el('img', { src: s.url, alt: s.name, loading: 'lazy' }),
      s.kind === 'zebra' ? el('span', { class: 'tag', text: 'zebra' }) : null,
    ]));
  });
}

function selectSample(id, source) {
  SEL.sample = id;
  SEL.source = source;
  document.querySelectorAll('.samp').forEach((n) => {
    const on = n.title === id.replace(/_/g, ' ');
    n.classList.toggle('is-active', on);
    n.setAttribute('aria-checked', on ? 'true' : 'false');
  });
  const s = (STATE.samples || []).find((x) => x.id === id);
  if (s) {
    $('imgIn').src = s.url;
    $('slideIn').src = s.url;
    $('stage').dataset.has = '1';
    develop($('imgIn'));
    setBadge('已载入输入', '');
  }
  const anyOk = (STATE.groups || []).some((g) => g.available);
  $('runBtn').disabled = !anyOk;
}

function renderGroups() {
  const box = $('groupSegments');
  box.innerHTML = '';
  (STATE.groups || []).forEach((g) => {
    const ok = g.available;
    if (SEL.group === g.id && !ok) SEL.group = 'A';
    const active = SEL.group === g.id;
    box.append(el('button', {
      class: 'seg' + (active ? ' is-active' : ''),
      role: 'radio',
      'data-gid': g.id,
      'aria-checked': active ? 'true' : 'false',
      disabled: ok ? null : 'disabled',
      onclick: () => selectGroup(g.id),
    }, [
      el('span', { class: 'dot', style: `background:${GROUP_COLOR[g.id]}` }),
      el('span', { class: 'seg-name', text: `${g.id} · ${g.name}` }),
      el('span', {
        class: 'seg-role',
        text: ok ? (g.mean_ms ? `${fmt(g.mean_ms, 0)} ms` : GROUP_ROLE[g.id]) : '未就绪',
      }),
    ]));
  });

  const anyOk = (STATE.groups || []).some((g) => g.available);
  $('runBtn').disabled = !anyOk || !SEL.sample;
  if (!anyOk) setNote('三组模型都还没生成，先跑 python cli.py setup');
}

function selectGroup(id) {
  SEL.group = id;
  document.querySelectorAll('.seg').forEach((b) => {
    const on = b.dataset.gid === id;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-checked', on ? 'true' : 'false');
  });
  const g = (STATE.groups || []).find((x) => x.id === id);
  if (g) {
    $('groupHint').textContent = g.id === 'A'
      ? 'A 组 · 原生 PyTorch，作为参照系'
      : g.id === 'B'
        ? `${g.name}：只换推理引擎，隔离运行时收益`
        : `${g.name}：在 B 组基础上压精度，隔离量化收益`;
  }
}

function bindUpload() {
  const input = $('fileInput');
  const zone = $('uploadLabel');

  input.addEventListener('change', (ev) => {
    const f = ev.target.files && ev.target.files[0];
    if (f) upload(f);
  });

  ['dragenter', 'dragover'].forEach((t) => zone.addEventListener(t, (ev) => {
    ev.preventDefault();
    zone.classList.add('is-over');
  }));
  ['dragleave', 'drop'].forEach((t) => zone.addEventListener(t, (ev) => {
    ev.preventDefault();
    zone.classList.remove('is-over');
  }));
  zone.addEventListener('drop', (ev) => {
    const f = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
    if (f) upload(f);
  });
}

async function upload(file) {
  if (file.size > 32 * 1024 * 1024) return setNote('文件超过 32MB', true);
  setNote('上传中…');
  try {
    const buf = await file.arrayBuffer();
    const r = await fetch('/api/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: buf,
    });
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || '上传失败');
    SEL.sample = j.id;
    SEL.source = 'uploads';
    document.querySelectorAll('.samp').forEach((n) => {
      n.classList.remove('is-active');
      n.setAttribute('aria-checked', 'false');
    });
    $('imgIn').src = j.url;
    $('slideIn').src = j.url;
    $('stage').dataset.has = '1';
    develop($('imgIn'));
    setBadge('已载入输入', '');
    $('runBtn').disabled = false;
    setNote('已上传：' + j.id);
  } catch (e) {
    setNote(String(e.message || e), true);
  }
}

function setNote(text, isError = false) {
  const n = $('runNote');
  n.textContent = text;
  n.classList.toggle('is-error', isError);
}

function setBadge(text, cls) {
  const b = $('viewerBadge');
  b.textContent = text;
  b.className = 'badge' + (cls ? ' ' + cls : '');
}

/* ══════════ 推理 ══════════ */

function bindRun() {
  $('runBtn').addEventListener('click', runInference);
}

function startTimer() {
  const t0 = performance.now();
  $('stageLoading').classList.add('is-on');
  const label = $('phaseText');
  label.textContent = `服务端推理中 · ${SEL.group} 组`;
  const tick = () => {
    $('elapsed').textContent = ((performance.now() - t0) / 1000).toFixed(2) + ' s';
    TIMER = requestAnimationFrame(tick);
  };
  tick();
}

function stopTimer() {
  if (TIMER) cancelAnimationFrame(TIMER);
  TIMER = null;
  $('stageLoading').classList.remove('is-on');
}

async function runInference() {
  if (!SEL.sample) return;
  startTimer();
  setBadge('推理中', 'is-run');
  $('runBtn').disabled = true;
  setNote('');

  try {
    const r = await fetch('/api/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample: SEL.sample, group: SEL.group, source: SEL.source }),
    });
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || '推理失败');
    applyResult(j);
    setBadge('完成', 'is-ok');
    pushHistory(j);
  } catch (e) {
    setBadge('失败', 'is-bad');
    setNote(String(e.message || e), true);
  } finally {
    stopTimer();
    $('runBtn').disabled = false;
  }
}

function applyResult(j) {
  $('imgIn').src = j.input_url;
  $('imgOut').src = j.output_url;
  $('slideIn').src = j.input_url;
  $('slideOut').src = j.output_url;
  develop($('imgIn'));
  develop($('imgOut'));
  $('capOut').textContent = `输出 · ${j.group} 组`;
  $('wipeOutLabel').textContent = `输出 · ${j.group} 组`;
  $('stage').dataset.has = '1';

  $('lmLatency').textContent = fmt(j.latency_ms, 1) + ' ms';
  $('lmFps').textContent = fmt(j.fps, 1);
  $('lmGroup').textContent = `${j.group} · ${GROUP_ROLE[j.group]}`;
  $('lmSize').textContent = '256 × 256';

  if (SEL.mode === 'diff') renderDiff();
}

/* ── 运行记录 ── */

function loadHistory() {
  try {
    return JSON.parse(sessionStorage.getItem(HISTORY_KEY)) || [];
  } catch (e) {
    return [];
  }
}

function saveHistory() {
  try {
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(HISTORY.slice(0, 10)));
  } catch (e) { /* 会话存储不可用时仅内存保留 */ }
}

function pushHistory(j) {
  const item = {
    sample: SEL.sample,
    group: j.group,
    latency_ms: j.latency_ms,
    fps: j.fps,
    ts: Date.now(),
    input_url: j.input_url,
    output_url: j.output_url,
  };
  HISTORY.unshift(item);
  HISTORY = HISTORY.slice(0, 10);
  saveHistory();
  renderHistory();
}

function renderHistory() {
  const box = $('runHistory');
  box.innerHTML = '';
  const empty = $('runsEmpty');
  empty.hidden = HISTORY.length > 0;
  HISTORY.forEach((h) => {
    box.append(el('li', {}, [el('button', {
      class: 'run-item',
      onclick: () => {
        applyResult(h);
        setBadge('回看该次运行', 'is-ok');
      },
      title: '点击回看这次结果',
    }, [
      el('span', { class: 'dot', style: `background:${GROUP_COLOR[h.group]}` }),
      el('span', { class: 'name', text: `${h.group} · ${h.sample}` }),
      el('span', { class: 'val', text: `${fmt(h.latency_ms, 1)} ms / ${fmt(h.fps, 1)} FPS` }),
      el('span', { class: 'time', text: relTime(h.ts) }),
    ])]));
  });
}

/* ══════════ 性能面板 ══════════ */

function emptyChart(host, msg) {
  host.innerHTML = '';
  host.append(el('div', { class: 'chart-empty', text: msg }));
}

function renderKpis() {
  const box = $('kpiRow');
  box.innerHTML = '';
  (STATE.groups || []).forEach((g) => {
    const has = g.available && g.mean_ms;
    box.append(el('li', { class: 'tile tile--accent', style: `--accent:${GROUP_COLOR[g.id]}` }, [
      el('span', { class: 'tile__label', text: `${g.id} · ${g.name}` }),
      el('span', { class: 'tile__value' }, [
        has ? fmt(g.mean_ms, 1) : '—',
        el('small', { text: has ? 'ms' : '' }),
      ]),
      el('span', {
        class: 'tile__sub',
        text: has
          ? `${fmt(g.fps, 1)} FPS · ${fmt(g.size_mb, 2)} MB · 相对 A ${fmt(g.speedup_vs_A, 2)}×`
          : '尚未测速',
      }),
      g.precision ? el('span', { class: 'tile__sub', text: '精度 ' + g.precision }) : null,
    ]));
  });

  const groups = (STATE.groups || []).filter((g) => g.available && g.mean_ms);
  latencyChart($('latencyChart'), groups);
  sizeChart($('sizeChart'), (STATE.groups || []).filter((g) => g.available && g.size_mb));
}

function latencyChart(host, groups) {
  host.innerHTML = '';
  if (!groups.length) return emptyChart(host, '还没有测速数据，运行 python cli.py bench');

  const W = 620, H = 300, pad = { t: 30, r: 18, b: 54, l: 56 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const max = Math.max(...groups.map((g) => (g.mean_ms || 0) + (g.std_ms || 0))) * 1.2 || 1;
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '三组单图延迟对比' });

  for (let i = 0; i <= 4; i++) {
    const y = pad.t + ih - (ih * i) / 4;
    s.append(svg('line', { x1: pad.l, x2: W - pad.r, y1: y, y2: y, stroke: C_GRID, 'stroke-width': 1 }));
    const tx = svg('text', { x: pad.l - 10, y: y + 4, 'text-anchor': 'end', 'font-size': 10.5, fill: C_TEXT, 'font-family': 'var(--font-mono)' });
    tx.textContent = Math.round((max * i) / 4);
    s.append(tx);
  }
  s.append(svg('text', { x: pad.l - 10, y: pad.t - 12, 'text-anchor': 'end', 'font-size': 10, fill: C_TEXT, 'font-family': 'var(--font-mono)' }));
  s.lastChild.textContent = 'ms';

  const bw = Math.min(72, (iw / groups.length) * 0.42);
  groups.forEach((g, i) => {
    const cx = pad.l + (iw / groups.length) * (i + 0.5);
    const h = (g.mean_ms / max) * ih;
    const x = cx - bw / 2;
    const y = pad.t + ih - h;

    s.append(svg('rect', {
      x, y, width: bw, height: h, rx: 3, class: 'grow-y',
      fill: GROUP_COLOR[g.id], opacity: 0.92,
    }));

    if (g.std_ms) {
      const eh = (g.std_ms / max) * ih;
      [y - eh, y + eh].forEach((yy) => {
        s.append(svg('line', { x1: cx - 8, x2: cx + 8, y1: yy, y2: yy, stroke: C_INK, 'stroke-width': 1, opacity: 0.5 }));
      });
      s.append(svg('line', { x1: cx, x2: cx, y1: y - eh, y2: y + eh, stroke: C_INK, 'stroke-width': 1, opacity: 0.5 }));
    }

    const val = svg('text', {
      x: cx, y: y - (g.std_ms ? (g.std_ms / max) * ih + 16 : 12),
      'text-anchor': 'middle', 'font-size': 13, 'font-weight': 600, fill: C_INK,
      'font-family': 'var(--font-mono)',
    });
    val.textContent = fmt(g.mean_ms, 1);
    s.append(val);

    const lab = svg('text', { x: cx, y: H - pad.b + 22, 'text-anchor': 'middle', 'font-size': 12, fill: C_INK });
    lab.textContent = `${g.id} · ${g.name}`;
    s.append(lab);

    const sub = svg('text', { x: cx, y: H - pad.b + 38, 'text-anchor': 'middle', 'font-size': 10.5, fill: C_TEXT, 'font-family': 'var(--font-mono)' });
    sub.textContent = g.id === 'A' ? '参照基线' : `${fmt(g.speedup_vs_A, 2)}× vs A`;
    s.append(sub);

    const hot = svg('rect', {
      x: cx - (iw / groups.length) / 2, y: pad.t, width: iw / groups.length, height: ih,
      fill: 'transparent', class: 'hot',
      'data-tip': tipHtml(`${g.id} 组 · ${g.name}`, [
        ['延迟均值', `${fmt(g.mean_ms, 2)} ms`],
        ['标准差', g.std_ms != null ? `± ${fmt(g.std_ms, 2)} ms` : '—'],
        ['吞吐', `${fmt(g.fps, 2)} FPS`],
        ['模型体积', `${fmt(g.size_mb, 2)} MB`],
        ['相对 A', g.speedup_vs_A != null ? `${fmt(g.speedup_vs_A, 2)}×` : '—'],
        ['推理精度', g.precision || 'float32'],
      ]),
    });
    s.append(hot);
  });

  host.append(s);
  bindTips(host);
}

function sizeChart(host, groups) {
  host.innerHTML = '';
  if (!groups.length) return emptyChart(host, '还没有模型体积数据');

  const W = 620, rowH = 46, pad = { t: 18, r: 96, b: 14, l: 116 };
  const H = pad.t + pad.b + groups.length * rowH;
  const max = Math.max(...groups.map((g) => g.size_mb)) * 1.12 || 1;
  const iw = W - pad.l - pad.r;
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '三组模型体积对比' });
  const base = groups.find((g) => g.id === 'A');

  groups.forEach((g, i) => {
    const y = pad.t + i * rowH;
    const w = (g.size_mb / max) * iw;

    const lab = svg('text', { x: pad.l - 14, y: y + 21, 'text-anchor': 'end', 'font-size': 12, fill: C_INK });
    lab.textContent = `${g.id} · ${g.name}`;
    s.append(lab);

    s.append(svg('rect', { x: pad.l, y, width: iw, height: 22, rx: 4, fill: '#efeee9' }));
    s.append(svg('rect', {
      x: pad.l, y, width: Math.max(w, 3), height: 22, rx: 4,
      fill: GROUP_COLOR[g.id], opacity: 0.92, class: 'grow-x',
    }));

    const val = svg('text', {
      x: pad.l + Math.max(w, 3) + 10, y: y + 21, 'font-size': 12, 'font-weight': 600,
      fill: C_INK, 'font-family': 'var(--font-mono)',
    });
    val.textContent = fmt(g.size_mb, 2) + ' MB';
    s.append(val);

    const ratio = base && base.size_mb ? `${fmt(g.size_mb / base.size_mb, 3)}× A` : '';
    const rt = svg('text', { x: W - 14, y: y + 21, 'text-anchor': 'end', 'font-size': 10.5, fill: C_TEXT, 'font-family': 'var(--font-mono)' });
    rt.textContent = ratio;
    s.append(rt);

    s.append(svg('rect', {
      x: pad.l, y: y - 6, width: iw, height: 34, fill: 'transparent', class: 'hot',
      'data-tip': tipHtml(`${g.id} 组 · ${g.name}`, [
        ['体积', `${fmt(g.size_mb, 3)} MB`],
        ['相对 A 体积', base && base.size_mb ? `${fmt(g.size_mb / base.size_mb, 3)}×` : '—'],
        ['推理精度', g.precision || 'float32'],
      ]),
    }));
  });

  host.append(s);
  bindTips(host);
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
    ['测试设备', STATE.device],
    ['运行环境', STATE.platform],
  ] : [['状态', '尚未测速，运行 python cli.py bench']];

  items.forEach(([k, v]) => {
    box.append(el('div', { class: 'tile' }, [
      el('dt', { class: 'tile__label', text: k }),
      el('dd', { class: 'tile__sub mono', style: 'margin-top:2px;font-size:13px;color:var(--ink)', text: v ?? '—' }),
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
    emptyChart($('lpipsChart'), '暂无数据');
    emptyChart($('structChart'), '暂无数据');
    $('qualityTable').innerHTML = '';
    $('deepTiles').innerHTML = '';
    return;
  }

  const m = q.mean || {};
  const d = q.deep || {};
  const real = q.metrics_source === 'real' && d.lpips;

  notice.className = 'notice' + (real ? ' is-ok' : '');
  notice.innerHTML = real
    ? `<b>口径</b>：SSIM / Canny 边缘 SSIM 为纯 numpy 真实计算；LPIPS 为官方 AlexNet 真实感知距离，` +
      `FID / KID 与掩码、判别同步跑通。<b>样本量 ${esc(q.n_samples)} 张</b>，数值在 quality.json 的 deep 字段，可直接进报告。`
    : `<b>口径</b>：当前只有结构代理值（1 − SSIM），<b>不可写进报告</b>。跑 <b>python cli.py quality --deep</b> 得真实 LPIPS / FID / KID。`;

  renderDeepTiles(q, d, real, m);

  if (real) {
    bars($('lpipsChart'), [
      { label: 'a→b 转换', value: d.lpips.L_ab, color: '#6b7580' },
      { label: 'b→c 量化', value: d.lpips.L_bc, color: '#b07717' },
      { label: 'a→c 总计', value: d.lpips.L_ac, color: '#1b3a6b' },
    ], `真实 LPIPS（${d.lpips.net}）· n=${d.lpips.n_images}`, '越小越相似');
  } else {
    bars($('lpipsChart'), [
      { label: '结构差异 a→b', value: m.ssim_ab != null ? 1 - m.ssim_ab : null, color: '#6b7580' },
      { label: '结构差异 b→c', value: m.ssim_bc != null ? 1 - m.ssim_bc : null, color: '#b07717' },
      { label: '结构差异 a→c', value: m.ssim_ac != null ? 1 - m.ssim_ac : null, color: '#1b3a6b' },
    ], '代理值 1 − SSIM（非正式 LPIPS）', '仅供联调');
  }

  bars($('structChart'), [
    { label: 'SSIM (a,b)', value: m.ssim_ab, color: '#2c56a3' },
    { label: 'Canny 边缘 (a,b)', value: d.canny_edge_ssim_ab ?? m.edge_ssim_ab, color: '#6b93c9' },
    { label: 'SSIM (b,c)', value: m.ssim_bc, color: '#b07717' },
    { label: 'Canny 边缘 (b,c)', value: d.canny_edge_ssim_bc ?? m.edge_ssim_bc, color: '#d4a548' },
  ], '结构保真 · 0~1', '越接近 1 越好');

  renderQualityTable(q);
}

function renderDeepTiles(q, d, real, m) {
  const box = $('deepTiles');
  box.innerHTML = '';
  const put = (label, value, sub, accent) => box.append(el('li', {
    class: 'tile' + (accent ? ' tile--accent' : ''),
  }, [
    el('span', { class: 'tile__label', text: label }),
    el('span', { class: 'tile__value' }, [value]),
    el('span', { class: 'tile__sub', text: sub }),
  ]));

  put('样本量', `${q.n_samples}`, `生成批 · 生成于 ${esc(q.generated_at || '—')}`, true);
  put('PSNR (b,c)', fmt(m.psnr_bc, 2), 'dB · 越高越好');

  if (real && d.lpips) {
    put('LPIPS (b,c)', fmt(d.lpips.L_bc, 4), `AlexNet · a→b 为 ${fmt(d.lpips.L_ab, 4)}`, true);
  } else {
    put('LPIPS (b,c)', '—', '需 quality --deep');
  }

  if (d.fid) {
    put('FID · B 批', fmt(d.fid.b_vs_real_zebra, 2), `生成 ${d.fid.n_generated} vs 真实 ${d.fid.n_real}`);
    put('FID · C 批', fmt(d.fid.c_vs_real_zebra, 2), '越小分布越接近目标域');
  }
  if (d.kid) {
    const kb = d.kid.b_vs_real_zebra || {};
    const kc = d.kid.c_vs_real_zebra || {};
    put('KID · B 批', fmt(kb.mean, 4), `± ${fmt(kb.std, 4)} · subset ${kb.subset_size}×${kb.n_subsets}`);
    put('KID · C 批', fmt(kc.mean, 4), `± ${fmt(kc.std, 4)}`);
  }
  if (d.mask) {
    put('掩码 IoU (a,c)', fmt(d.mask.mean_iou_ac, 3), `检出 ${d.mask.detected_pairs} 对 · Mask R-CNN`);
    put('背景 SSIM (a,c)', fmt(d.mask.mean_background_ssim_ac, 3), '掩码外区域未被改写');
  }
  if (d.zebra_discrimination) {
    const zb = d.zebra_discrimination.b || {};
    const zc = d.zebra_discrimination.c || {};
    put('斑马判别 · B 批', pctv(zb.top1_accuracy, 2), `n=${zb.n} · ResNet-50 top1`, true);
    put('斑马判别 · C 批', pctv(zc.top1_accuracy, 2), `n=${zc.n} · 量化前后应接近`);
  }
  if (d.anomalies) {
    put('异常登记', `${d.anomalies.count}`, `阈值 ssim(b,c) < ${d.anomalies.threshold}`);
  }
}

function bars(host, items, title, unit) {
  host.innerHTML = '';
  const W = 620, rowH = 40, pad = { t: 16, r: 88, b: 12, l: 148 };
  const H = pad.t + pad.b + items.length * rowH;
  const iw = W - pad.l - pad.r;
  const vals = items.map((i) => (i.value == null ? 0 : Math.abs(i.value)));
  const max = Math.max(...vals, 0.02) * 1.25;
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': title });

  items.forEach((it, i) => {
    const y = pad.t + i * rowH;
    const w = it.value == null ? 0 : (Math.abs(it.value) / max) * iw;

    const lab = svg('text', { x: pad.l - 12, y: y + 19, 'text-anchor': 'end', 'font-size': 11.5, fill: C_TEXT });
    lab.textContent = it.label;
    s.append(lab);

    s.append(svg('rect', { x: pad.l, y: y + 3, width: iw, height: 18, rx: 4, fill: '#efeee9' }));
    s.append(svg('rect', {
      x: pad.l, y: y + 3, width: Math.max(w, 2), height: 18, rx: 4,
      fill: it.color, opacity: 0.92, class: 'grow-x',
    }));

    const v = svg('text', {
      x: pad.l + Math.max(w, 2) + 10, y: y + 19, 'font-size': 11.5, 'font-weight': 600,
      fill: C_INK, 'font-family': 'var(--font-mono)',
    });
    v.textContent = it.value == null ? '—' : Number(it.value).toFixed(4);
    s.append(v);

    s.append(svg('rect', {
      x: pad.l, y, width: iw, height: rowH - 4, fill: 'transparent', class: 'hot',
      'data-tip': tipHtml(title, [
        [it.label, it.value == null ? '—' : Number(it.value).toFixed(4)],
        ['读法', unit],
      ]),
    }));
  });

  host.append(s);
  bindTips(host);
}

/* ── E4 层间漂移曲线 ── */

function renderLayerChart() {
  const host = $('layerChart');
  const L = STATE.layers;
  const sub = $('layerSub');
  if (!L || !L.rows || !L.rows.length) {
    host.innerHTML = '';
    emptyChart(host, '还没有层间数据，运行 python cli.py layers');
    sub.textContent = '未采集';
    return;
  }

  const order = (L.probe_layers || L.rows.map((r) => r.layer));
  const rows = order.map((name) => L.rows.find((r) => r.layer === name)).filter(Boolean);
  sub.textContent = `${rows.length} 个探针层 · n=${L.n_images} 张 · 校准 ${L.n_calibration} 张`;

  const W = 780, H = 300, pad = { t: 26, r: 58, b: 66, l: 58 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const cosMin = Math.min(...rows.map((r) => r.cosine)) - 0.012;
  const cosMax = 1.002;
  const relMax = Math.max(...rows.map((r) => r.rel_l2)) * 1.15 || 1;

  const x = (i) => pad.l + (iw * i) / (rows.length - 1);
  const yCos = (v) => pad.t + ih - ((v - cosMin) / (cosMax - cosMin)) * ih;
  const yRel = (v) => pad.t + ih - (v / relMax) * ih;

  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '逐层表示漂移曲线' });

  // 残差块区间底色
  const firstRes = rows.findIndex((r) => r.layer.startsWith('resblock'));
  const lastRes = rows.length - 1 - [...rows].reverse().findIndex((r) => r.layer.startsWith('resblock'));
  if (firstRes >= 0 && lastRes > firstRes) {
    s.append(svg('rect', {
      x: x(firstRes) - 10, y: pad.t, width: x(lastRes) - x(firstRes) + 20, height: ih,
      fill: '#f3f2ee', rx: 4,
    }));
    const band = svg('text', { x: (x(firstRes) + x(lastRes)) / 2, y: pad.t + 14, 'text-anchor': 'middle', 'font-size': 10, fill: C_TEXT, 'font-family': 'var(--font-mono)' });
    band.textContent = 'RESBLOCK';
    s.append(band);
  }

  // 网格与双侧刻度
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (ih * i) / 4;
    s.append(svg('line', { x1: pad.l, x2: W - pad.r, y1: y, y2: y, stroke: C_GRID, 'stroke-width': 1 }));
    const lc = svg('text', { x: pad.l - 9, y: y + 4, 'text-anchor': 'end', 'font-size': 10, fill: '#2c56a3', 'font-family': 'var(--font-mono)' });
    lc.textContent = (cosMax - ((cosMax - cosMin) * i) / 4).toFixed(3);
    s.append(lc);
    const lr = svg('text', { x: W - pad.r + 9, y: y + 4, 'font-size': 10, fill: '#b07717', 'font-family': 'var(--font-mono)' });
    lr.textContent = (relMax - (relMax * i) / 4).toFixed(2);
    s.append(lr);
  }

  const line = (pts, color, dash) => s.append(svg('polyline', {
    points: pts, fill: 'none', stroke: color, 'stroke-width': 2,
    'stroke-linejoin': 'round', 'stroke-linecap': 'round',
    'stroke-dasharray': dash || null,
  }));

  line(rows.map((r, i) => `${x(i)},${yCos(r.cosine)}`).join(' '), '#2c56a3');
  line(rows.map((r, i) => `${x(i)},${yRel(r.rel_l2)}`).join(' '), '#b07717');

  // 悬浮引导线
  const guide = svg('line', {
    x1: 0, x2: 0, y1: pad.t, y2: pad.t + ih,
    stroke: C_AXIS, 'stroke-width': 1, 'stroke-dasharray': '3 3', opacity: 0,
  });
  s.append(guide);

  rows.forEach((r, i) => {
    s.append(svg('circle', {
      cx: x(i), cy: yCos(r.cosine), r: 3, fill: '#fff',
      stroke: '#2c56a3', 'stroke-width': 1.6,
    }));
    s.append(svg('circle', {
      cx: x(i), cy: yRel(r.rel_l2), r: 3, fill: '#fff',
      stroke: '#b07717', 'stroke-width': 1.6,
    }));

    const t = svg('text', {
      x: x(i), y: H - pad.b + 16, 'text-anchor': 'end', 'font-size': 10,
      fill: C_TEXT, 'font-family': 'var(--font-mono)',
      transform: `rotate(-38 ${x(i)} ${H - pad.b + 16})`,
    });
    t.textContent = r.layer.replace('resblock_', 'rb').replace('down_relu_', 'down').replace('up_relu_', 'up').replace('head_conv', 'head');
    s.append(t);

    const hot = svg('rect', {
      x: x(i) - iw / (rows.length - 1) / 2, y: pad.t, width: iw / (rows.length - 1), height: ih,
      fill: 'transparent', class: 'hot',
      'data-tip': tipHtml(r.layer, [
        ['余弦相似度', r.cosine.toFixed(6)],
        ['相对 L2', r.rel_l2.toFixed(6)],
        ['平均绝对误差', r.mean_abs_err != null ? Number(r.mean_abs_err).toExponential(2) : '—'],
        ['激活量级', r.activation_mean_abs != null ? Number(r.activation_mean_abs).toExponential(2) : '—'],
      ]),
    });
    hot.addEventListener('pointerenter', () => {
      guide.setAttribute('x1', x(i));
      guide.setAttribute('x2', x(i));
      guide.setAttribute('opacity', '1');
    });
    hot.addEventListener('pointerleave', () => guide.setAttribute('opacity', '0'));
    s.append(hot);
  });

  // 图例
  const lg = [['余弦相似度（左轴）', '#2c56a3'], ['相对 L2（右轴）', '#b07717']];
  lg.forEach(([name, color], i) => {
    const gx = pad.l + i * 150;
    s.append(svg('line', { x1: gx, x2: gx + 16, y1: H - 12, y2: H - 12, stroke: color, 'stroke-width': 2 }));
    const tx = svg('text', { x: gx + 22, y: H - 8, 'font-size': 10.5, fill: C_TEXT });
    tx.textContent = name;
    s.append(tx);
  });

  host.innerHTML = '';
  host.append(s);
  bindTips(host);
}

function renderQualityTable(q) {
  const t = $('qualityTable');
  t.innerHTML = '';
  $('qualitySub').textContent = `${q.n_samples} 张 · ${esc(q.generated_at || '')} · 鼠标悬停看该张明细`;

  const cols = ['样例', 'SSIM(a,b)', 'Canny(a,b)', 'SSIM(b,c)', 'Canny(b,c)', 'PSNR(b,c)', '掩码 IoU(a,c)'];
  const thead = el('thead', {}, [el('tr', {}, cols.map((c) => el('th', { text: c })))]);
  const threshold = (q.deep && q.deep.anomalies && q.deep.anomalies.threshold) || 0.8;

  const tbody = el('tbody', {}, (q.rows || []).map((r) => {
    const lpips = r.lpips && r.lpips.L_bc != null;
    const tr = el('tr', {}, [
      el('td', { text: r.sample + (r.ssim_bc != null && r.ssim_bc < threshold ? ' ·异常' : '') }),
      el('td', { text: r.ssim_ab != null ? fmt(r.ssim_ab, 4) : '—' }),
      el('td', { text: r.edge_ssim_ab != null ? fmt(r.edge_ssim_ab, 4) : '—' }),
      el('td', { text: r.ssim_bc != null ? fmt(r.ssim_bc, 4) : '—' }),
      el('td', { text: r.edge_ssim_bc != null ? fmt(r.edge_ssim_bc, 4) : '—' }),
      el('td', { text: r.psnr_bc != null ? fmt(r.psnr_bc, 2) : '—' }),
      el('td', { text: r.mask_iou_ac != null ? fmt(r.mask_iou_ac, 3) : '—' }),
    ]);
    if (lpips) {
      tr.setAttribute('data-tip', tipHtml(r.sample, [
        ['LPIPS (b,c)', fmt(r.lpips.L_bc, 4)],
        ['SSIM (b,c)', fmt(r.ssim_bc, 4)],
        ['背景 SSIM (a,c)', r.background_ssim_ac != null ? fmt(r.background_ssim_ac, 4) : '—'],
        ['掩码 IoU (a,c)', r.mask_iou_ac != null ? fmt(r.mask_iou_ac, 4) : '—'],
      ]));
    }
    return tr;
  }));

  t.append(thead, tbody);
  bindTips(t);
}

/* ══════════ 实验进度 ══════════ */

function renderExperiments() {
  const box = $('expGrid');
  box.innerHTML = '';
  const flag = { done: '已完成', partial: '部分完成', todo: '未开始' };
  (STATE.experiments || []).forEach((e) => {
    box.append(el('div', { class: `exp is-${e.status}` }, [
      el('div', { class: 'exp__head' }, [
        el('span', { class: 'exp__code', text: e.code }),
        el('span', { class: 'exp__title', text: e.title }),
        el('span', { class: 'exp__flag', text: flag[e.status] || e.status }),
      ]),
      el('p', { class: 'exp__note', text: e.note || '' }),
      e.half ? el('p', { class: 'exp__note', text: e.half }) : null,
    ]));
  });
}

function renderHonesty() {
  const box = $('honestyBox');
  box.innerHTML = '';
  const deep = (STATE.quality && STATE.quality.deep) || {};
  const L = STATE.layers;
  const rows = [
    ['✅', '真实实测', 'A/B/C 三组推理、延迟与 FPS、模型体积、SSIM / Canny 边缘 SSIM / PSNR —— 全部本机真实计算。'],
    deep.lpips
      ? ['✅', '真实感知指标', 'LPIPS（AlexNet）三对归因、FID / KID、Mask R-CNN 掩码 IoU、ResNet-50 斑马判别 —— quality --deep 已跑通，数值在 quality.json 的 deep 字段。']
      : ['⚠️', '感知指标待跑', 'LPIPS / FID / KID / 掩码 / 判别的代码已接入（python cli.py quality --deep），当前 quality.json 仍是结构代理值，不可写进报告。'],
    L && L.rows
      ? ['✅', '层间分工', `覆盖 ${L.probe_layers.length} 个探针层，n=${L.n_images} 张图，逐層余弦与相对 L2 在「质量评测」页可视化。`]
      : ['⚠️', '层间分工待跑', 'E4 尚未产出 layers.json（python cli.py layers），深度支柱暂时只有提纲。'],
    ['📌', '引用口径', 'FID/KID 是与真实目标域的特征分布距离，不等于「逼真度」；延迟与加速比强平台依赖，引用必须标注测试平台。'],
  ];
  rows.forEach(([mark, title, text]) => {
    box.append(el('div', { class: 'honesty__row' }, [
      el('span', { class: 'mark', text: mark }),
      el('span', {}, [el('b', { text: title + '：' }), text]),
    ]));
  });
}

/* ══════════ 口径说明折叠 ══════════ */

function bindInfoNotes() {
  document.querySelectorAll('.info-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = $(btn.dataset.note);
      const open = target.hasAttribute('hidden');
      if (open) target.removeAttribute('hidden');
      else target.setAttribute('hidden', '');
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  });
}

document.addEventListener('DOMContentLoaded', init);
