// 30-ui.js — the parts every page is made of: icons, the sprout, toasts, dialogs, the file picker
// and the form controls.

const ICONS = {
  train: ['M12 21v-9', 'M12 13c0-5-3.5-7-8-6.5 0 5 3.5 7.5 8 6.5z', 'M12 11c0-4 3-6.5 8-6.5 0 4.5-3.5 7-8 6.5z'],
  models: ['M12 3 3 7.5 12 12l9-4.5L12 3z', 'M3 12l9 4.5 9-4.5', 'M3 16.5 12 21l9-4.5'],
  write: ['M4 20l1-4L17 4a2.1 2.1 0 013 3L8 19l-4 1z', 'M14 7l3 3'],
  speed: ['M4.5 17a8.5 8.5 0 1115 0', 'M12 17l4-6', ['circle', { cx: 12, cy: 17, r: 1.2 }]],
  settings: ['M4 7h9', 'M17 7h3', 'M4 17h3', 'M11 17h9', ['circle', { cx: 15, cy: 7, r: 2 }], ['circle', { cx: 9, cy: 17, r: 2 }]],
  about: [['circle', { cx: 12, cy: 12, r: 9 }], 'M12 11v5', 'M12 7.8h.01'],
  play: [['path', { d: 'M8 5.5v13l11-6.5z', fill: 'currentColor' }]],
  stop: [['rect', { x: 6.5, y: 6.5, width: 11, height: 11, rx: 2, fill: 'currentColor' }]],
  folder: ['M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V7z'],
  file: ['M6 3h8l4 4v14H6V3z', 'M14 3v4h4'],
  check: ['M5 12.5l4.5 4.5L19 7.5'],
  warn: ['M12 3.5l9.5 16.5h-19L12 3.5z', 'M12 10v4.5', 'M12 17.4h.01'],
  error: [['circle', { cx: 12, cy: 12, r: 9 }], 'M9 9l6 6', 'M15 9l-6 6'],
  copy: [['rect', { x: 9, y: 9, width: 11, height: 11, rx: 2 }], 'M5 15V6a2 2 0 012-2h9'],
  close: ['M6 6l12 12', 'M18 6L6 18'],
  chevron: ['M9 6l6 6-6 6'],
  down: ['M12 5v14', 'M6 13l6 6 6-6'],
  up: ['M12 19V5', 'M6 11l6-6 6 6'],
  refresh: ['M20 11a8 8 0 10-2.3 5.7', 'M20 5v6h-6'],
  plus: ['M12 5v14', 'M5 12h14'],
  upfolder: ['M9 14l-4-4 4-4', 'M5 10h9a5 5 0 015 5v3'],
  dice: [['rect', { x: 4, y: 4, width: 16, height: 16, rx: 3 }], 'M8.5 8.5h.01', 'M15.5 8.5h.01', 'M12 12h.01', 'M8.5 15.5h.01', 'M15.5 15.5h.01'],
  power: ['M12 3v9', 'M6.4 7a8 8 0 1011.2 0'],
  save: ['M5 4h11l3 3v13H5V4z', 'M8 4v5h7V4', 'M8 20v-6h8v6'],
  clock: [['circle', { cx: 12, cy: 12, r: 9 }], 'M12 7v5l3 2'],
  table: [['rect', { x: 3.5, y: 5, width: 17, height: 14, rx: 2 }], 'M3.5 10h17', 'M9.5 10v9'],
  chart: ['M4 19V5', 'M4 19h16', 'M7 15l4-5 3 3 5-7'],
  info: [['circle', { cx: 12, cy: 12, r: 9 }], 'M12 11v5', 'M12 7.8h.01'],
  seed: [['ellipse', { cx: 12, cy: 13, rx: 4.5, ry: 6.5, transform: 'rotate(25 12 13)' }], 'M10.5 8.5c1-1.8 2.8-2.8 4.6-2.8'],
  help: [['circle', { cx: 12, cy: 12, r: 9 }], 'M9.6 9.4a2.5 2.5 0 114 2c-.9.6-1.6 1.1-1.6 2.3', 'M12 16.8h.01'],
};

function icon(name, cls = '') {
  const svg = s('svg', {
    viewBox: '0 0 24 24',
    class: cls,
    'aria-hidden': 'true',
    focusable: 'false',
    fill: 'none',
    stroke: 'currentColor',
    'stroke-width': 1.8,
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round',
  });
  for (const part of ICONS[name] || []) {
    if (typeof part === 'string') svg.appendChild(s('path', { d: part }));
    else if (part[0] === 'path') svg.appendChild(s('path', part[1]));
    else svg.appendChild(s(part[0], part[1]));
  }
  return svg;
}

// ---------------------------------------------------------------------------- the sprout

/** How far the sprout has grown: 0 a seed, 1 a root, 2 a seedling, 3 true leaves, 4 in bloom. */
function stageOf(progress, state) {
  if (state === 'done') return 4;
  if (!isNum(progress)) return 0;
  if (progress < 0.04) return 0;
  if (progress < 0.3) return 1;
  if (progress < 0.62) return 2;
  return 3;
}

function sprout(stage = 3, { breathing = false } = {}) {
  const svg = s('svg', { viewBox: '0 0 64 64', class: 'sprout' + (breathing ? ' breathing' : ''), 'aria-hidden': 'true', focusable: 'false' });
  const part = (from, el) => {
    el.classList.add('grow-part');
    el.dataset.from = String(from);
    svg.appendChild(el);
    return el;
  };
  svg.appendChild(s('path', { class: 'ground', d: 'M9 53h46', 'stroke-width': 2.4 }));
  const seed = part(0, s('ellipse', { class: 'seed', cx: 32, cy: 49, rx: 5.5, ry: 3.6 }));
  seed.dataset.until = '2';
  part(1, s('path', { class: 'stem', d: 'M32 53c0 4-2 6-4.5 8M32 53c1 3 3 5 5.5 6.5', 'stroke-width': 2 }));
  const stem = s('path', { class: 'stem', d: 'M32 53V19', 'stroke-width': 3.2, pathLength: 100 });
  stem.dataset.stem = '1';
  svg.appendChild(stem);
  part(2, s('path', { class: 'leaf-b', d: 'M32 42C26 42 22.5 38.5 22 33c6 0 9.5 3 10 9z' }));
  part(2, s('path', { class: 'leaf-a', d: 'M32 40c1-6 5-8.5 10.5-8.5 0 5.5-3.5 8.5-10.5 8.5z' }));
  part(3, s('path', { class: 'leaf-a', d: 'M32 31C23 31 17 26 16 18c9 0 15 5 16 13z' }));
  part(3, s('path', { class: 'leaf-b', d: 'M32 27c1-9 8-13.5 17-13.5 0 9-7 14.5-17 13.5z' }));
  const bloom = part(4, s('g'));
  bloom.appendChild(s('circle', { class: 'sun', cx: 32, cy: 13, r: 5.2 }));
  bloom.appendChild(s('circle', { cx: 32, cy: 13, r: 9, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 1.6, opacity: 0.7 }));
  setSprout(svg, stage);
  return svg;
}

function setSprout(svg, stage) {
  svg.dataset.stage = String(stage);
  for (const el of $$('.grow-part', svg)) {
    const from = Number(el.dataset.from);
    const until = el.dataset.until ? Number(el.dataset.until) : 99;
    const on = stage >= from && stage < until;
    el.style.transform = on ? 'scale(1)' : 'scale(0.05)';
    el.style.opacity = on ? '1' : '0';
  }
  const stem = $('[data-stem]', svg);
  if (stem) {
    const shown = [0, 22, 44, 70, 100][clamp(stage, 0, 4)];
    stem.setAttribute('stroke-dasharray', '100');
    stem.setAttribute('stroke-dashoffset', String(100 - shown));
    stem.style.opacity = stage >= 1 ? '1' : '0';
  }
}

// ---------------------------------------------------------------------------- toasts, announcements

function toast(message, kind = 'info', { timeout } = {}) {
  let host = $('.toasts');
  if (!host) {
    host = h('div.toasts');
    document.body.appendChild(host);
  }
  const el = h('div.toast', { class: kind }, icon(kind === 'ok' ? 'check' : kind === 'bad' ? 'error' : 'info'), h('div.t-body', rich(message)));
  const close = h('button.btn.btn-quiet.btn-icon.btn-sm', { type: 'button', 'aria-label': t('common.close'), onclick: () => el.remove() }, icon('close'));
  el.appendChild(close);
  host.appendChild(el);
  while (host.children.length > 4) host.firstChild.remove();
  announce(typeof message === 'string' ? message.replaceAll('`', '') : message, kind === 'bad');
  setTimeout(() => el.remove(), timeout ?? (kind === 'bad' ? 12000 : 6000));
  return el;
}

// ---------------------------------------------------------------------------- dialogs

/**
 * A modal dialog. `body` is a node (or a function of `close`); `actions` are buttons, each with the
 * value the promise resolves with. Escape and a click outside resolve with `null`.
 */
function dialog({ title, body, actions = [], onOpen, wide = false }) {
  return new Promise((resolve) => {
    const opener = document.activeElement;
    const titleId = uid('dlg');
    let result = null;
    const dlg = h('dialog.dlg', { aria: { labelledby: titleId } });
    const close = (value = null) => {
      result = value;
      dlg.close();
    };
    const content = typeof body === 'function' ? body(close) : body;
    const buttons = actions.map((a) =>
      h(
        'button.btn',
        {
          type: 'button',
          class: a.kind === 'primary' ? 'btn-primary' : a.kind === 'danger' ? 'btn-danger' : '',
          onclick: () => close(a.value),
          autofocus: a.autofocus,
        },
        a.label,
      ),
    );
    dlg.appendChild(
      h(
        'div.dlg-in',
        h('div.dlg-h', h('h2', { id: titleId }, title), h('button.btn.btn-quiet.btn-icon', { type: 'button', 'aria-label': t('common.close'), onclick: () => close(null) }, icon('close'))),
        h('div.dlg-b', content),
        buttons.length ? h('div.dlg-f', buttons) : null,
      ),
    );
    if (wide) dlg.style.width = 'min(60rem, calc(100vw - 1.5rem))';
    dlg.addEventListener('close', () => {
      dlg.remove();
      if (opener && document.contains(opener)) opener.focus();
      resolve(result);
    });
    dlg.addEventListener('click', (e) => {
      if (e.target === dlg) close(null);
    });
    document.body.appendChild(dlg);
    dlg.showModal();
    onOpen?.(dlg, close);
  });
}

async function confirmBox({ title, text, confirm, cancel, danger = false }) {
  const ok = await dialog({
    title,
    body: h('p', text),
    actions: [
      { label: cancel || t('common.cancel'), value: false },
      { label: confirm, value: true, kind: danger ? 'danger' : 'primary', autofocus: true },
    ],
  });
  return ok === true;
}

// ---------------------------------------------------------------------------- form controls

function field({ label, hint, error, control, optional = false, id }) {
  const controlId = id || uid('f');
  const hintId = hint ? controlId + '-hint' : null;
  const errId = error ? controlId + '-err' : null;
  if (control?.setAttribute && !control.id) control.id = controlId;
  const described = [hintId, errId].filter(Boolean).join(' ');
  if (control?.setAttribute && described) control.setAttribute('aria-describedby', described);
  if (control?.setAttribute && error) control.setAttribute('aria-invalid', 'true');
  return h(
    'div.field',
    h('label.label', { for: control?.id || controlId }, label, optional ? h('span.opt', ' ' + t('common.optional')) : null),
    control,
    hint ? h('div.hint', { id: hintId }, hint) : null,
    error ? h('div.error-text', { id: errId, role: 'alert' }, icon('error'), error) : null,
  );
}

/** A group of radio "cards"; `onchange(value)` when another is chosen. */
function choiceGroup({ name, label, options, value, onchange }) {
  const group = h('div.choices', { role: 'radiogroup', 'aria-label': label });
  const cards = options.map((o) => {
    const input = h('input', { type: 'radio', name, value: o.value, checked: o.value === value });
    const card = h(
      'label.choice',
      { dataset: { checked: String(o.value === value) } },
      input,
      h('span.c-tick', icon('check')),
      h('span.c-title', o.title),
      o.num ? h('span.c-num', o.num) : null,
      o.sub ? h('span.c-sub', o.sub) : null,
    );
    input.addEventListener('change', () => {
      for (const c of cards) c.dataset.checked = String($('input', c).checked);
      onchange?.(o.value);
    });
    return card;
  });
  group.append(...cards);
  group.setValue = (v) => {
    for (const c of cards) {
      const i = $('input', c);
      i.checked = i.value === v;
      c.dataset.checked = String(i.checked);
    }
  };
  return group;
}

/** A row of radio buttons that look like one control. */
function segmented({ name, label, options, value, onchange }) {
  const box = h('div.seg', { role: 'radiogroup', 'aria-label': label });
  for (const o of options) {
    const input = h('input', { type: 'radio', name, value: o.value, checked: o.value === value });
    input.addEventListener('change', () => onchange?.(o.value));
    box.appendChild(h('label', input, h('span', o.icon ? icon(o.icon) : null, o.label)));
  }
  box.setValue = (v) => {
    for (const i of $$('input', box)) i.checked = i.value === v;
  };
  return box;
}

function switchBox({ title, hint, checked, onchange }) {
  const input = h('input', { type: 'checkbox', role: 'switch', checked: !!checked });
  input.addEventListener('change', () => onchange?.(input.checked));
  const box = h('label.switch', input, h('span.track'), h('span.s-text', h('span.s-title', title), hint ? h('span.hint', hint) : null));
  box.input = input;
  return box;
}

function numberInput({ value, min, max, step = 1, onchange, oncommit, placeholder, inputmode = 'numeric', label }) {
  const input = h('input.input.num', { type: 'number', min, max, step, inputmode, value: value ?? '', placeholder, 'aria-label': label });
  const read = () => (input.value === '' ? null : Number(input.value));
  input.addEventListener('input', () => onchange?.(read(), input));
  // `change` comes when the person is done with the box (leaves it, or presses Enter), not at every key.
  input.addEventListener('change', () => oncommit?.(read(), input));
  return input;
}

/**
 * Holds a table that may be wider than the screen. When it is, the box can be scrolled with the keyboard:
 * it becomes a labelled, focusable region (and only then, so that wide screens get no extra tab stop or landmark).
 */
function tableWrap(label, ...kids) {
  const el = h('div.tbl-wrap', ...kids);
  const check = () => {
    if (el.scrollWidth > el.clientWidth + 1) {
      el.tabIndex = 0;
      el.setAttribute('role', 'region');
      el.setAttribute('aria-label', label);
    } else {
      el.removeAttribute('tabindex');
      el.removeAttribute('role');
      el.removeAttribute('aria-label');
    }
  };
  new ResizeObserver(check).observe(el);
  return el;
}

function banner(kind, title, body, actions) {
  return h(
    'div.banner',
    { class: kind, role: kind === 'bad' ? 'alert' : null },
    icon(kind === 'ok' ? 'check' : kind === 'bad' ? 'error' : kind === 'warn' ? 'warn' : 'info'),
    h('div.b-body', title ? h('b', title) : null, body ? h('div', rich(body)) : null, actions ? h('div.row', actions) : null),
  );
}

function chip(kind, text, iconName) {
  return h('span.chip', { class: kind }, iconName ? icon(iconName) : null, text);
}

function bar(progress, { thick = false, label } = {}) {
  const el = h('div.bar', { class: thick ? 'thick' : '', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-label': label }, h('i'));
  el.set = (p) => {
    const v = clamp(p || 0, 0, 1);
    el.firstChild.style.setProperty('--p', String(v));
    el.setAttribute('aria-valuenow', String(Math.round(v * 100)));
  };
  el.set(progress);
  return el;
}

function copyButton(getText, label) {
  return h(
    'button.btn.btn-quiet.btn-sm',
    {
      type: 'button',
      onclick: async (e) => {
        const button = e.currentTarget; // gone once the event has finished, so take it before waiting
        try {
          await copyText(getText());
          toast(t('common.copied'), 'ok', { timeout: 2500 });
        } catch {
          toast(t('common.copyFailed'), 'bad');
        }
        button.focus();
      },
    },
    icon('copy'),
    label || t('common.copy'),
  );
}

// ---------------------------------------------------------------------------- the file picker

/** The segments of a path as [label, path] pairs, for the breadcrumbs. */
function pathCrumbs(path) {
  const win = /^[A-Za-z]:[\\/]/.test(path);
  const sep = win ? '\\' : '/';
  const parts = path.split(/[\\/]+/).filter(Boolean);
  const out = [];
  let acc = win ? parts.shift() + sep : sep;
  out.push([win ? acc : sep, acc]);
  for (const p of parts) {
    acc = acc.endsWith(sep) ? acc + p : acc + sep + p;
    out.push([p, acc]);
  }
  return out;
}

/**
 * Lets the person pick a file, a folder, or a model folder on this computer; resolves with the
 * path, or `null`. The server lists the folders (the page cannot see the disk by itself).
 */
async function pickPath({ mode = 'file', start = '', title }) {
  const state = { path: start, hidden: false, selected: null, data: null };
  const crumbs = h('nav.crumbs', { 'aria-label': t('pick.where') });
  const places = h('div.row');
  const list = h('ul.picker-list');
  const note = h('div');
  const chosen = h('div.hint', { 'aria-live': 'polite' });
  let closeDialog = () => {};
  let chooseButton;

  const choose = () => {
    const p = state.selected || (mode === 'folder' ? state.data?.path : null);
    if (p) closeDialog(p);
  };
  const select = (path) => {
    state.selected = path;
    chosen.textContent = path ? t('pick.selected', { path }) : '';
    for (const b of $$('.pick', list)) b.setAttribute('aria-selected', String(b.dataset.path === path));
    if (chooseButton) chooseButton.disabled = !(path || (mode === 'folder' && state.data));
  };

  async function load(path) {
    try {
      const data = await api.get('/api/fs?path=' + encodeURIComponent(path) + (state.hidden ? '&hidden=1' : ''));
      state.data = data;
      state.path = data.path;
      state.selected = null;
      render();
      mount(note);
    } catch (e) {
      mount(note, banner('bad', null, explain(e)));
    }
  }

  function render() {
    const d = state.data;
    mount(
      crumbs,
      pathCrumbs(d.path).map(([label, p], i) => [i ? h('span.sep', { 'aria-hidden': 'true' }, '›') : null, h('button', { type: 'button', onclick: () => load(p) }, label)]),
    );
    mount(
      places,
      d.roots.map((r) =>
        h('button.btn.btn-sm', { type: 'button', onclick: () => load(r.path) }, icon(r.kind === 'drive' ? 'folder' : r.kind === 'models' ? 'models' : r.kind === 'home' ? 'folder' : 'file'), r.kind === 'drive' ? r.path : t('pick.place.' + r.kind)),
      ),
    );
    const items = [];
    if (d.parent) {
      items.push(h('li', h('button.pick', { type: 'button', onclick: () => load(d.parent) }, icon('upfolder'), h('span.p-name', t('pick.up')), h('span'))));
    }
    for (const e of d.entries) {
      const full = d.path.replace(/[\\/]+$/, '') + (d.path.includes('\\') && !d.path.startsWith('/') ? '\\' : '/') + e.name;
      if (e.dir) {
        const isModel = e.model;
        const pickIt = mode === 'model' && isModel;
        items.push(
          h(
            'li',
            h(
              'button.pick',
              {
                type: 'button',
                dataset: { path: full },
                'aria-selected': 'false',
                onclick: () => (pickIt ? select(full) : load(full)),
                ondblclick: () => (pickIt ? closeDialog(full) : null),
              },
              icon(isModel ? 'models' : 'folder'),
              h('span.p-name', e.name),
              h('span.p-meta', isModel ? t('pick.model') : ''),
            ),
          ),
        );
      } else if (mode === 'file') {
        items.push(
          h(
            'li',
            h(
              'button.pick',
              { type: 'button', dataset: { path: full }, 'aria-selected': 'false', onclick: () => select(full), ondblclick: () => closeDialog(full) },
              icon('file'),
              h('span.p-name', e.name),
              h('span.p-meta', fmt.bytes(e.size)),
            ),
          ),
        );
      }
    }
    if (d.truncated) items.push(h('li', h('div.hint', { style: { padding: '0.6rem 0.8rem' } }, t('pick.truncated'))));
    if (!items.length || (items.length === 1 && d.parent)) items.push(h('li', h('div.hint', { style: { padding: '0.6rem 0.8rem' } }, t('pick.empty'))));
    mount(list, items);
    select(null);
  }

  const hiddenSwitch = switchBox({
    title: t('pick.hidden'),
    checked: false,
    onchange: (v) => {
      state.hidden = v;
      load(state.path);
    },
  });
  const modeText = mode === 'folder' ? t('pick.hintFolder') : mode === 'model' ? t('pick.hintModel') : t('pick.hintFile');

  const result = dialog({
    title: title || (mode === 'file' ? t('pick.titleFile') : mode === 'model' ? t('pick.titleModel') : t('pick.titleFolder')),
    wide: true,
    body: () => h('div.stack', h('p.hint', modeText), places, crumbs, note, list, chosen, hiddenSwitch),
    actions: [
      { label: t('common.cancel'), value: null },
    ],
    onOpen: (dlg, close) => {
      closeDialog = close;
      chooseButton = h('button.btn.btn-primary', { type: 'button', disabled: true, onclick: choose }, icon('check'), mode === 'folder' ? t('pick.chooseFolder') : t('pick.choose'));
      $('.dlg-f', dlg).appendChild(chooseButton);
      load(start || '');
    },
  });
  return result;
}
