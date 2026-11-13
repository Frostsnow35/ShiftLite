const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
const start = source.indexOf('function renderQualityTable(q)');
const end = source.indexOf('\nfunction ', start + 1);
function element(tag, props = {}, children = []) {
  return { tag, ...props, children, append(...items) { this.children.push(...items); },
    setAttribute() {} };
}
function render(row) {
  const hosts = { qualityTable: element('table'), qualitySub: element('div') };
  const context = { $: id => hosts[id], el: element, esc: String,
    fmt: (n, d) => n == null ? '—' : Number(n).toFixed(d), bindTips() {}, tipHtml() {} };
  vm.createContext(context);
  vm.runInContext(source.slice(start, end), context);
  context.renderQualityTable({ n_samples: 1, rows: [row] });
  return hosts.qualityTable.children[1].children[0].children;
}
const cells = render({ sample: 's', canny_edge_ssim_ab: .1111, edge_ssim_ab: .9999,
  canny_edge_ssim_bc: .2222, edge_ssim_bc: .8888, mask_dice_ac: .5,
  structure: { pairs: { x_c: { iou: .2, dice: .3, background_ssim: null } } } });
assert.equal(cells[2].text, '0.1111');
assert.equal(cells[4].text, '0.2222');
assert.equal(cells[7].text, '0.500');
assert.equal(cells[8].text, '0.200');
assert.equal(cells[10].text, '—');
assert.equal(render({ sample: 'old', edge_ssim_ab: .9999 })[2].text, '0.9999');
assert.equal(render({ sample: 'old', edge_ssim_bc: .8888 })[4].text, '0.8888');
console.log('PASS: Canny bindings, Dice/input bindings, missing values, legacy JSON fallback');
