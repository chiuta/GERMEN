// 40-charts.js — the charts. Drawn as SVG from the numbers, nothing else: thin lines, a hairline
// grid, a legend whenever there are two series, a cursor that reads every series at once (also
// from the keyboard), a table of the same numbers, a smoothing slider and a log scale.
// Two colours carry data (--s1, --s2); everything else is ink.

function niceTicks(lo, hi, target = 5, whole = false) {
  if (!(hi > lo)) hi = lo + 1;
  const rough = (hi - lo) / target;
  const mag = 10 ** Math.floor(Math.log10(rough));
  const norm = rough / mag;
  let step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  if (whole) step = Math.max(1, Math.round(step)); // a number of steps or threads: no 10.2, 10.4…
  const ticks = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : Number(v.toPrecision(12)));
  return { ticks, step };
}

/**
 * Gridlines for a log axis: powers of ten; with 2 and 5 when the range is short; with every digit when
 * it is shorter still. Always at least three, and never more than `most`.
 */
function logTicks(lo, hi, most = 6) {
  const make = (digits) => {
    const out = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e += 1) {
      for (const m of digits) {
        const v = m * 10 ** e;
        if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
      }
    }
    return out;
  };
  for (const digits of [[1], [1, 2, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9]]) {
    const out = make(digits);
    if (out.length >= 3) {
      const every = Math.ceil(out.length / most);
      return every > 1 ? out.filter((_, i) => i % every === 0) : out;
    }
  }
  return niceTicks(lo, hi, 4).ticks; // narrower than any digit step: ordinary round numbers
}

/** Exponential moving average, with the first value as the start (what the smoothing slider does). */
function smoothed(ys, amount) {
  if (!(amount > 0)) return ys;
  const alpha = 1 - amount;
  const out = [];
  let acc = null;
  for (const y of ys) {
    if (!isNum(y)) {
      out.push(y);
      continue;
    }
    acc = acc === null ? y : acc + alpha * (y - acc);
    out.push(acc);
  }
  return out;
}

/** Index of the value of `xs` (ascending) nearest to `x`. */
function nearest(xs, x) {
  let lo = 0;
  let hi = xs.length - 1;
  if (hi < 0) return -1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < x) lo = mid;
    else hi = mid;
  }
  return Math.abs(xs[lo] - x) <= Math.abs(xs[hi] - x) ? lo : hi;
}

/**
 * options: label (what a screen reader hears), series [{key, name, color, dash, markers, area, smooth}],
 * xTitle, yTitle, height, canLog, canSmooth, yFormat, xFormat, valueFormat (for the tooltip).
 * `update({key: {x: [...], y: [...]}})` redraws.
 */
function lineChart(options) {
  const o = { height: 280, canLog: false, canSmooth: false, ...options };
  const state = { data: {}, log: false, smoothing: o.canSmooth ? 0.6 : 0, view: 'chart', cursor: null, width: 600 };
  const margin = { l: 54, r: 16, t: 14, b: 40 };

  const svg = s('svg', { role: 'presentation' });
  const tip = h('div.tip', { hidden: true, role: 'presentation' });
  const live = h('div.sr', { 'aria-live': 'polite' });
  const surface = h('div.chart', { tabindex: '0', role: 'group', 'aria-roledescription': t('chart.role'), 'aria-label': o.label }, svg, tip, live);
  const tableBox = tableWrap(o.yTitle || o.label);
  tableBox.hidden = true;
  const wrap = h('div.stack');

  // --- tools
  const tools = h('div.chart-tools');
  let smoothLabel = null;
  if (o.canSmooth) {
    const slider = h('input', { type: 'range', min: 0, max: 0.95, step: 0.05, value: state.smoothing, 'aria-label': t('chart.smoothing') });
    slider.addEventListener('input', () => {
      state.smoothing = Number(slider.value);
      draw();
    });
    smoothLabel = h('label', t('chart.smoothing'), slider);
    tools.appendChild(smoothLabel);
  }
  if (o.canLog) {
    const box = h('input', { type: 'checkbox' });
    box.addEventListener('change', () => {
      state.log = box.checked;
      draw();
    });
    tools.appendChild(h('label', box, t('chart.log')));
  }
  const toggle = h('button.btn.btn-quiet.btn-sm', { type: 'button', 'aria-pressed': 'false' }, icon('table'), t('chart.table'));
  toggle.addEventListener('click', () => {
    state.view = state.view === 'chart' ? 'table' : 'chart';
    toggle.setAttribute('aria-pressed', String(state.view === 'table'));
    draw();
  });
  tools.appendChild(toggle);

  const legend =
    o.series.length >= 2
      ? h(
          'div.legend',
          o.series.map((sr) => h('span', h('i', { class: sr.dash ? 'dash' : sr.markers === 'dot' ? 'dot' : '', style: { color: sr.color, borderColor: sr.color } }), sr.name)),
        )
      : null;
  wrap.append(...[tools, legend, surface, tableBox].filter(Boolean));

  const xfmt = o.xFormat || ((v) => (Math.abs(v) >= 10000 ? fmt.compact(v) : fmt.int(v)));
  // `extra` asks for more digits: the labels grow more precise until no two gridlines read the same.
  const yfmt = (v, step, extra = 0, big = false) => {
    if (o.yFormat) return o.yFormat(v, extra);
    if (big) return fmt.compact(v, extra);
    const d = clamp(-Math.floor(Math.log10(step || 1)) + extra, 0, 8);
    return nf({ minimumFractionDigits: d, maximumFractionDigits: d }).format(v);
  };
  function yLabels(ticks, lo, hi, step) {
    const shown = ticks.filter((v) => v >= lo - 1e-9 && v <= hi + 1e-9);
    // One style for the whole axis: "9.5K, 10K, 10.5K", never "9,500, 10K".
    const big = shown.some((v) => Math.abs(v) >= 10000);
    for (let extra = 0; extra < 6; extra += 1) {
      const labels = shown.map((v) => yfmt(v, step, extra, big));
      if (new Set(labels).size === labels.length || extra === 5) return shown.map((v, i) => [v, labels[i]]);
    }
    return [];
  }
  const vfmt = o.valueFormat || ((v) => (Math.abs(v) >= 1000 ? fmt.int(v) : fmt.sig(v, 4)));

  function visible(sr) {
    const d = state.data[sr.key];
    return d && d.x && d.x.length ? d : null;
  }

  function plotted(sr) {
    const d = visible(sr);
    if (!d) return null;
    const ys = sr.smooth && state.smoothing > 0 ? smoothed(d.y, state.smoothing) : d.y;
    return { x: d.x, y: ys, raw: d.y };
  }

  let geom = null;

  function draw() {
    const W = Math.max(280, Math.round(surface.clientWidth || state.width));
    state.width = W;
    const H = o.height;
    const pw = W - margin.l - margin.r;
    const ph = H - margin.t - margin.b;
    const all = o.series.map((sr) => ({ sr, p: plotted(sr) })).filter((e) => e.p);
    toggle.hidden = all.length === 0;
    surface.hidden = state.view === 'table';
    tableBox.hidden = state.view !== 'table';
    if (smoothLabel) smoothLabel.hidden = state.view === 'table';
    if (state.view === 'table') {
      drawTable();
      return;
    }
    // domains
    let xlo = Infinity;
    let xhi = -Infinity;
    let ylo = Infinity;
    let yhi = -Infinity;
    for (const { p } of all) {
      for (let i = 0; i < p.x.length; i += 1) {
        const y = p.y[i];
        const raw = p.raw[i];
        if (!isNum(y) || (state.log && !(y > 0))) continue;
        xlo = Math.min(xlo, p.x[i]);
        xhi = Math.max(xhi, p.x[i]);
        ylo = Math.min(ylo, y, isNum(raw) && (!state.log || raw > 0) ? raw : y);
        yhi = Math.max(yhi, y, isNum(raw) && (!state.log || raw > 0) ? raw : y);
      }
    }
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('width', String(W));
    svg.setAttribute('height', String(H));
    svg.replaceChildren();
    if (!(xlo <= xhi)) {
      geom = null;
      return;
    }
    if (xhi === xlo) xhi = xlo + 1;
    if (o.yMin !== undefined) ylo = Math.min(ylo, o.yMin);
    let yTicks;
    let ylo2 = ylo;
    let yhi2 = yhi;
    if (state.log) {
      ylo2 = ylo * 0.9;
      yhi2 = yhi * 1.1;
      yTicks = logTicks(ylo2, yhi2, Math.max(3, Math.round(ph / 40)));
    } else {
      const pad = (yhi - ylo || Math.abs(yhi) || 1) * 0.08;
      ylo2 = o.yMin !== undefined ? Math.min(ylo, o.yMin) : ylo - pad;
      yhi2 = yhi + pad;
      if (ylo2 < 0 && ylo >= 0) ylo2 = 0;
      yTicks = niceTicks(ylo2, yhi2, Math.max(3, Math.round(ph / 56))).ticks;
    }
    const step = yTicks.length > 1 ? Math.min(...yTicks.slice(1).map((v, i) => Math.abs(v - yTicks[i]))) || 1 : 1;
    const X = (x) => margin.l + ((x - xlo) / (xhi - xlo)) * pw;
    const Y = state.log
      ? (y) => margin.t + ph - ((Math.log10(y) - Math.log10(ylo2)) / (Math.log10(yhi2) - Math.log10(ylo2))) * ph
      : (y) => margin.t + ph - ((y - ylo2) / (yhi2 - ylo2)) * ph;
    geom = { X, Y, xlo, xhi, margin, pw, ph, W, H, all };

    const grid = s('g', { class: 'grid' });
    const axis = s('g', { class: 'axis' });
    for (const [v, label] of yLabels(yTicks, ylo2, yhi2, step)) {
      const y = Y(v);
      grid.appendChild(s('line', { x1: margin.l, x2: W - margin.r, y1: y, y2: y }));
      axis.appendChild(s('text', { x: margin.l - 8, y: y + 4, 'text-anchor': 'end' }, label));
    }
    const xt = niceTicks(xlo, xhi, Math.max(2, Math.round(pw / 90)), o.xWhole !== false);
    for (const v of xt.ticks) {
      const x = X(v);
      axis.appendChild(s('line', { x1: x, x2: x, y1: margin.t + ph, y2: margin.t + ph + 5 }));
      axis.appendChild(s('text', { x, y: margin.t + ph + 20, 'text-anchor': 'middle' }, xfmt(v)));
    }
    axis.appendChild(s('line', { x1: margin.l, x2: W - margin.r, y1: margin.t + ph, y2: margin.t + ph }));
    if (o.xTitle) axis.appendChild(s('text', { class: 'axis-title', x: W - margin.r, y: H - 4, 'text-anchor': 'end' }, o.xTitle));
    if (o.yTitle) axis.appendChild(s('text', { class: 'axis-title', x: margin.l, y: 9 }, o.yTitle));
    svg.append(grid, axis);

    for (const { sr, p } of all) {
      const pts = [];
      for (let i = 0; i < p.x.length; i += 1) {
        if (isNum(p.y[i]) && (!state.log || p.y[i] > 0)) pts.push([X(p.x[i]), Y(p.y[i])]);
      }
      if (!pts.length) continue;
      const d = pts.map((q, i) => (i ? 'L' : 'M') + q[0].toFixed(1) + ' ' + q[1].toFixed(1)).join('');
      if (sr.area && pts.length > 1) {
        const area = s('path', { class: 'area', d: d + `L${pts[pts.length - 1][0].toFixed(1)} ${margin.t + ph}L${pts[0][0].toFixed(1)} ${margin.t + ph}Z` });
        area.style.fill = sr.color;
        area.style.opacity = '0.1';
        svg.appendChild(area);
      }
      const smoothing = sr.smooth && state.smoothing > 0;
      if (smoothing) {
        const rawPts = [];
        for (let i = 0; i < p.x.length; i += 1) if (isNum(p.raw[i]) && (!state.log || p.raw[i] > 0)) rawPts.push([X(p.x[i]), Y(p.raw[i])]);
        const rd = rawPts.map((q, i) => (i ? 'L' : 'M') + q[0].toFixed(1) + ' ' + q[1].toFixed(1)).join('');
        const raw = s('path', { class: 'ser raw', d: rd });
        raw.style.stroke = sr.color;
        svg.appendChild(raw);
      }
      if (pts.length > 1) {
        const line = s('path', { class: 'ser' + (sr.dash ? ' dash' : ''), d });
        line.style.stroke = sr.color;
        svg.appendChild(line);
      }
      // a dot on every point only while the points are far enough apart to read as dots, not as a bead chain
      const showMarkers = sr.markers === 'dot' || (sr.markers === 'auto' && pts.length <= Math.min(60, pw / 13));
      const marks = showMarkers ? pts : [pts[pts.length - 1]];
      for (const q of marks) {
        const m = s('circle', { class: 'mk', cx: q[0], cy: q[1], r: 4.5 });
        m.style.fill = sr.color;
        svg.appendChild(m);
      }
    }
    // cursor
    const cross = s('line', { class: 'cross', y1: margin.t, y2: margin.t + ph });
    cross.style.display = 'none';
    const dots = s('g');
    svg.append(cross, dots);
    const hit = s('rect', { class: 'hit', x: margin.l, y: margin.t, width: pw, height: ph });
    svg.appendChild(hit);
    geom.cross = cross;
    geom.dots = dots;
    if (state.cursor !== null) showCursorAtX(state.cursor, false);
    else tip.hidden = true;
  }

  function seriesAt(x) {
    const rows = [];
    for (const { sr, p } of geom.all) {
      const i = nearest(p.x, x);
      if (i < 0) continue;
      rows.push({ sr, x: p.x[i], y: p.raw[i], shown: p.y[i] });
    }
    return rows;
  }

  function showCursorAtX(x, speak) {
    if (!geom) return;
    const rows = seriesAt(x).filter((r) => isNum(r.y));
    if (!rows.length) return;
    // the main series decides where the cursor stands
    const lead = rows.reduce((a, b) => (Math.abs(a.x - x) <= Math.abs(b.x - x) ? a : b));
    state.cursor = lead.x;
    const cx = geom.X(lead.x);
    geom.cross.style.display = '';
    geom.cross.setAttribute('x1', String(cx));
    geom.cross.setAttribute('x2', String(cx));
    geom.dots.replaceChildren();
    for (const r of rows) {
      const c = s('circle', { class: 'mk', cx: geom.X(r.x), cy: geom.Y(r.y), r: 5.5 });
      c.style.fill = r.sr.color;
      geom.dots.appendChild(c);
    }
    const xt = o.xTitle ? o.xTitle + ' ' : '';
    // A series with no point at this x shows its nearest one, and says where that is.
    const where = (r) => (r.x !== lead.x ? ` (${xt}${fmt.int(r.x)})` : '');
    mount(
      tip,
      h('div.tip-h', `${xt}${fmt.int(lead.x)}`),
      rows.map((r) => h('div.tip-r', h('span', h('i.sw', { style: { background: r.sr.color } }), r.sr.name), h('b', vfmt(r.y) + where(r)))),
    );
    tip.hidden = false;
    const tw = tip.offsetWidth || 150;
    const left = cx + 14 + tw > geom.W ? cx - tw - 14 : cx + 14;
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${margin.t + 4}px`;
    if (speak) live.textContent = `${xt}${fmt.int(lead.x)}: ` + rows.map((r) => `${r.sr.name} ${vfmt(r.y)}${where(r)}`).join(', ');
  }

  function hideCursor() {
    state.cursor = null;
    tip.hidden = true;
    if (geom?.cross) geom.cross.style.display = 'none';
    geom?.dots?.replaceChildren();
  }

  svg.addEventListener('pointermove', (e) => {
    if (!geom) return;
    const box = svg.getBoundingClientRect();
    const x = geom.xlo + ((e.clientX - box.left - margin.l) / geom.pw) * (geom.xhi - geom.xlo);
    showCursorAtX(clamp(x, geom.xlo, geom.xhi), false);
  });
  svg.addEventListener('pointerleave', () => {
    if (document.activeElement !== surface) hideCursor();
  });
  surface.addEventListener('blur', hideCursor);
  surface.addEventListener('keydown', (e) => {
    if (!geom) return;
    const primary = geom.all.reduce((a, b) => (a.p.x.length >= b.p.x.length ? a : b)).p;
    let i = state.cursor === null ? primary.x.length - 1 : nearest(primary.x, state.cursor);
    const jump = Math.max(1, Math.round(primary.x.length / 10));
    if (e.key === 'ArrowLeft') i -= e.shiftKey ? jump : 1;
    else if (e.key === 'ArrowRight') i += e.shiftKey ? jump : 1;
    else if (e.key === 'Home') i = 0;
    else if (e.key === 'End') i = primary.x.length - 1;
    else if (e.key === 'Escape') return hideCursor();
    else return;
    e.preventDefault();
    showCursorAtX(primary.x[clamp(i, 0, primary.x.length - 1)], true);
  });

  function drawTable() {
    const xs = new Set();
    for (const sr of o.series) for (const x of visible(sr)?.x || []) xs.add(x);
    let rows = [...xs].sort((a, b) => a - b);
    if (rows.length > 200) {
      const keep = [];
      for (let i = 0; i < 200; i += 1) keep.push(rows[Math.round((i * (rows.length - 1)) / 199)]);
      rows = [...new Set(keep)];
    }
    const lookup = o.series.map((sr) => {
      const d = visible(sr);
      const m = new Map();
      if (d) d.x.forEach((x, i) => m.set(x, d.y[i]));
      return m;
    });
    mount(
      tableBox,
      h(
        'table.tbl',
        h('caption', o.label),
        h('thead', h('tr', h('th', o.xTitle || 'x'), o.series.map((sr) => h('th.r', sr.name)))),
        h(
          'tbody',
          rows.map((x) => h('tr', h('td', fmt.int(x)), lookup.map((m) => h('td.r', m.has(x) ? vfmt(m.get(x)) : '–')))),
        ),
      ),
    );
  }

  const redraw = frameBatched(draw);
  if (typeof ResizeObserver !== 'undefined') {
    let lastWidth = 0;
    new ResizeObserver(() => {
      const w = surface.clientWidth;
      if (w && Math.abs(w - lastWidth) > 1) {
        lastWidth = w;
        redraw();
      }
    }).observe(surface);
  }

  return {
    el: wrap,
    update(data) {
      state.data = data;
      redraw();
    },
    redrawNow: draw,
  };
}

/** A figure: the chart, and under it a caption "Fig. N, text". */
function figure(number, caption, content) {
  return h('figure.fig', content, h('figcaption.figcap', h('b', t('chart.fig', { n: number })), ' ', caption));
}
