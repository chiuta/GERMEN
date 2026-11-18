// 00-core.js — what every other part uses: building elements, state, small helpers.
// The page never turns a string into markup: text goes in with textContent / text nodes, and
// elements are made with `h` and `s` below (the policy of the page would refuse anything else).

const MODE = document.querySelector('meta[name="germen-mode"]')?.getAttribute('content') === 'demo' ? 'demo' : 'app';
const SVGNS = 'http://www.w3.org/2000/svg';
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

/** A text a model wrote, with its control characters (a byte model writes some) shown as the replacement sign: no empty boxes on the page. */
const printable = (text) => String(text ?? '').replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/g, '\uFFFD');

const PROPS = new Set(['value', 'checked', 'disabled', 'selected', 'hidden', 'open', 'readOnly', 'indeterminate']);

function addKids(el, kids) {
  for (const kid of kids) {
    if (kid === null || kid === undefined || kid === false || kid === true) continue;
    if (Array.isArray(kid)) addKids(el, kid);
    else if (kid instanceof Node) el.appendChild(kid);
    else el.appendChild(document.createTextNode(String(kid)));
  }
}

function applyAttrs(el, attrs, svg) {
  if (!attrs) return;
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'class') {
      if (svg) el.setAttribute('class', String(value));
      else el.classList.add(...String(value).split(/\s+/).filter(Boolean));
    } else if (key === 'text') {
      el.textContent = value;
    } else if (key === 'dataset') {
      Object.assign(el.dataset, value);
    } else if (key === 'style') {
      for (const [prop, x] of Object.entries(value)) {
        if (prop.startsWith('--')) el.style.setProperty(prop, x);
        else el.style[prop] = x;
      }
    } else if (key === 'aria') {
      for (const [name, x] of Object.entries(value)) {
        if (x !== undefined && x !== null && x !== false) el.setAttribute('aria-' + name, x === true ? 'true' : String(x));
      }
    } else if (key.startsWith('on') && typeof value === 'function') {
      el.addEventListener(key.slice(2), value);
    } else if (!svg && PROPS.has(key)) {
      el[key] = value;
    } else {
      el.setAttribute(key, value === true ? '' : String(value));
    }
  }
}

/** `h('button.btn.btn-primary#go', { onclick }, 'Text', child, [more])` makes an element. */
function h(spec, attrs, ...kids) {
  const m = /^([a-z][a-z0-9-]*)?((?:[.#][\w-]+)*)$/i.exec(spec);
  const el = document.createElement(m?.[1] || 'div');
  for (const part of (m?.[2] || '').match(/[.#][\w-]+/g) || []) {
    if (part[0] === '.') el.classList.add(part.slice(1));
    else el.id = part.slice(1);
  }
  if (attrs && (typeof attrs !== 'object' || attrs instanceof Node || Array.isArray(attrs))) {
    kids.unshift(attrs);
    attrs = null;
  }
  applyAttrs(el, attrs, false);
  addKids(el, kids);
  return el;
}

/** The same for SVG elements. */
function s(tag, attrs, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  if (attrs && (typeof attrs !== 'object' || attrs instanceof Node || Array.isArray(attrs))) {
    kids.unshift(attrs);
    attrs = null;
  }
  applyAttrs(el, attrs, true);
  addKids(el, kids);
  return el;
}

function mount(el, ...kids) {
  el.replaceChildren();
  addKids(el, kids);
  return el;
}

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const last = (a) => (a.length ? a[a.length - 1] : undefined);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
/** Text with `code` spans: what sits between backticks becomes <code>. A node, or text without backticks, passes through. */
function rich(text) {
  if (typeof text !== 'string' || !text.includes('`')) return text;
  return text.split('`').map((part, i) => (i % 2 ? h('code', part) : part));
}

// Brings an element into view. It does not animate when the person asked for less motion.
function reveal(el, block = 'nearest') {
  const calm = document.documentElement.dataset.motion === 'reduce' || matchMedia('(prefers-reduced-motion: reduce)').matches;
  el?.scrollIntoView?.({ block, behavior: calm ? 'auto' : 'smooth' });
}
let uidCounter = 0;
const uid = (prefix = 'u') => `${prefix}${++uidCounter}`;

function debounce(fn, ms) {
  let timer = 0;
  const wrapped = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
  wrapped.cancel = () => clearTimeout(timer);
  wrapped.flush = (...args) => {
    clearTimeout(timer);
    fn(...args);
  };
  return wrapped;
}

/** Calls `fn` at most once per animation frame, with the arguments of the last call. */
function frameBatched(fn) {
  let queued = false;
  let lastArgs = [];
  return (...args) => {
    lastArgs = args;
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      fn(...lastArgs);
    });
  };
}

const bus = {
  map: new Map(),
  on(name, fn) {
    if (!this.map.has(name)) this.map.set(name, new Set());
    this.map.get(name).add(fn);
    return () => this.map.get(name)?.delete(fn);
  },
  emit(name, data) {
    for (const fn of [...(this.map.get(name) || [])]) {
      try {
        fn(data);
      } catch (e) {
        console.error(e);
      }
    }
  },
};

/** What the whole page knows. Pages read it, the boot code and the event loops write it. */
const S = {
  token: '',
  settings: {},
  system: null,
  models: [],
  jobs: [],
  online: true,
  quitting: false,
  run: null, // the training run that is shown (see 60-train.js)
};

function announce(message, urgent = false) {
  const el = $(urgent ? '#alert' : '#live');
  if (!el) return;
  el.textContent = '';
  setTimeout(() => {
    el.textContent = message;
  }, 60);
}

function copyText(text) {
  if (navigator.clipboard?.writeText) return navigator.clipboard.writeText(text);
  return Promise.reject(new Error('no clipboard'));
}

function basename(path) {
  const parts = String(path).split(/[\\/]+/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : String(path);
}

/** Same thinning as the server's curves: at most `cap` points, evenly spaced. */
class Curve {
  constructor(names, cap = 2400) {
    this.names = names;
    this.cap = cap;
    this.cols = names.map(() => []);
    this.stride = 1;
    this.seen = 0;
  }
  static from(names, columns, cap = 2400) {
    const c = new Curve(names, cap);
    c.cols = names.map((n) => (columns?.[n] || []).slice());
    c.seen = c.cols[0].length;
    return c;
  }
  push(row) {
    const keep = this.seen % this.stride === 0;
    this.seen += 1;
    if (!keep) return;
    this.cols.forEach((col, i) => col.push(row[i]));
    if (this.cols[0].length > this.cap) {
      this.cols = this.cols.map((col) => col.filter((_, i) => i % 2 === 0));
      this.stride *= 2;
    }
  }
  col(name) {
    return this.cols[this.names.indexOf(name)];
  }
  get length() {
    return this.cols[0].length;
  }
}
