// 60-train.js — training. First the run itself, which is followed in the background and so outlives
// whichever page shows it; then the form that starts one; then the view of one that is going on.

const RUN_COLS = ['step', 'loss', 'lr', 'tps', 'grad_norm'];
const EVAL_COLS = ['step', 'val_loss', 'bpb', 'best'];
const PRESET_NAMES = ['nano', 'micro', 'mini', 'small', 'base'];

const run = {
  id: null,
  state: 'idle', // idle | starting | running | stopping | done | stopped | failed
  label: '',
  request: null,
  stage: { name: 'queued' },
  start: null,
  last: null,
  lastEval: null,
  steps: new Curve(RUN_COLS),
  evals: new Curve(EVAL_COLS),
  samples: [],
  log: [],
  result: null,
  error: null,
  started: 0,
  ctl: null,
  tail: [],
  recent: null,
  firstLoss: null,
  milestone: 0,
  get active() {
    return this.state === 'starting' || this.state === 'running' || this.state === 'stopping';
  },
  get progress() {
    if (this.state === 'done') return 1;
    const l = this.last;
    return l && isNum(l.total) && l.total > 0 ? clamp(l.step / l.total, 0, 1) : 0;
  },
};
S.run = run;

// The big number is the mean of the last few logged losses: it follows the curve closely, but one noisy
// batch does not make it jump. (A longer average would trail far behind a loss that is still falling fast.)
const LOSS_TAIL = 3;
function noteLoss(x) {
  run.tail.push(x);
  if (run.tail.length > LOSS_TAIL) run.tail.shift();
  run.recent = run.tail.reduce((a, b) => a + b, 0) / run.tail.length;
}

const emitRun = frameBatched(() => bus.emit('run'));
const runClock = () => Math.max(0, Date.now() / 1000 - (run.started || Date.now() / 1000));

function runReset() {
  run.ctl?.abort();
  Object.assign(run, {
    id: null,
    state: 'idle',
    label: '',
    request: null,
    stage: { name: 'queued' },
    start: null,
    last: null,
    lastEval: null,
    steps: new Curve(RUN_COLS),
    evals: new Curve(EVAL_COLS),
    samples: [],
    log: [],
    result: null,
    error: null,
    started: 0,
    ctl: null,
    tail: [],
    recent: null,
    firstLoss: null,
    milestone: 0,
  });
  bus.emit('run');
}

function runFromSnapshot(snap) {
  const tr = snap.train || {};
  run.id = snap.id;
  run.label = snap.label;
  run.request = snap.request;
  run.state = snap.state;
  run.started = snap.started;
  run.stage = tr.stage || { name: 'queued' };
  run.start = tr.start?.type ? tr.start : null;
  run.last = tr.last?.type ? tr.last : null;
  run.steps = Curve.from(RUN_COLS, tr.steps);
  run.evals = Curve.from(EVAL_COLS, tr.evals);
  run.samples = (tr.samples || []).map((x) => ({ ...x, text: printable(x.text) }));
  run.log = (tr.log || []).slice();
  run.result = snap.result && typeof snap.result === 'object' && Object.keys(snap.result).length ? snap.result : null;
  run.error = snap.error?.code ? snap.error : null;
  const losses = run.steps.col('loss').filter(isNum);
  run.firstLoss = losses.length ? losses[0] : null;
  run.tail = [];
  run.recent = null;
  for (const x of losses.slice(-LOSS_TAIL)) noteLoss(x);
  const evals = run.evals;
  run.lastEval = evals.length ? { step: last(evals.col('step')), val_loss: last(evals.col('val_loss')), bpb: last(evals.col('bpb')) } : null;
  run.milestone = Math.floor((run.progress * 100) / 25) * 25;
  if (run.milestone >= 100) run.milestone = 75;
}

function runApply(e) {
  switch (e.type) {
    case 'stage':
      run.stage = { name: e.name, done: e.done, total: e.total };
      if (e.total === null || e.total === undefined) run.log.push({ t: runClock(), type: 'stage', name: e.name });
      break;
    case 'start':
      run.start = e;
      run.stage = { name: 'training' };
      if (run.state === 'starting') run.state = 'running';
      run.log.push({ t: runClock(), type: 'start' });
      break;
    case 'step':
      run.last = e;
      run.steps.push([e.step, e.loss, e.lr, e.tps, e.grad_norm]);
      if (run.state === 'starting') run.state = 'running';
      if (isNum(e.loss)) {
        noteLoss(e.loss);
        if (run.firstLoss === null) run.firstLoss = e.loss;
      }
      announceMilestone();
      break;
    case 'eval':
      run.evals.push([e.step, e.val_loss, e.bpb, e.best ? 1 : 0]);
      run.lastEval = e;
      break;
    case 'sample':
      run.samples.push({ step: e.step, text: printable(e.text) });
      if (run.samples.length > 16) run.samples.shift();
      break;
    case 'saved':
      run.log.push({ t: runClock(), type: 'saved', step: e.step });
      break;
    case 'recovered':
      run.log.push({ t: runClock(), type: 'recovered', kind: e.kind, files: e.files });
      break;
    case 'stopping':
      run.state = 'stopping';
      run.log.push({ t: runClock(), type: 'stopping' });
      break;
    case 'end':
      runEnded(e);
      break;
    default:
      break;
  }
  if (run.log.length > 300) run.log.splice(0, run.log.length - 300);
  emitRun();
}

function announceMilestone() {
  const pct = run.progress * 100;
  if (pct >= run.milestone + 25 && pct < 100) {
    run.milestone = Math.floor(pct / 25) * 25;
    announce(t('run.milestone', { pct: run.milestone, loss: fmt.loss(run.recent) }));
  }
}

function runEnded(e) {
  run.state = e.state;
  run.result = e.result && typeof e.result === 'object' && Object.keys(e.result).length ? e.result : null;
  run.error = e.error?.code ? e.error : null;
  run.log.push({ t: runClock(), type: 'end', state: e.state });
  refreshModels();
  notifyEnd();
}

function notifyEnd() {
  const name = run.label;
  const loss = fmt.loss(run.result?.best_val_loss ?? run.result?.last_train_loss);
  const text = run.state === 'done' ? t('notify.done', { name, loss }) : run.state === 'stopped' ? t('notify.stopped', { name }) : t('notify.failed', { name });
  announce(text, run.state === 'failed');
  if (S.settings.sound && run.state === 'done') playChime();
  if (S.settings.notify && typeof Notification !== 'undefined' && Notification.permission === 'granted' && (document.hidden || !document.hasFocus())) {
    try {
      new Notification('GERMEN', { body: text, tag: 'germen-run' });
    } catch {
      /* some browsers only allow notifications from a service worker: nothing to do */
    }
  }
}

/** Starts following the training job `id` (its snapshot is fetched unless one is given). */
async function runAttach(id, snap) {
  run.ctl?.abort();
  const snapshot = snap || (await api.get(`/api/jobs/${id}`));
  runFromSnapshot(snapshot);
  run.ctl = new AbortController();
  bus.emit('run');
  if (FINAL.has(run.state)) return;
  follow(id, {
    after: snapshot.seq,
    signal: run.ctl.signal,
    onEvent: runApply,
    onSnapshot: (again) => {
      runFromSnapshot(again);
      bus.emit('run');
    },
  }).catch((e) => {
    if (e instanceof ApiFailure && e.status === 404 && run.id === id && run.active) {
      run.state = 'failed';
      run.error = { code: 'lost', message: e.message };
      bus.emit('run');
    }
  });
}

async function runStop() {
  if (!run.id || !run.active) return;
  try {
    await api.post(`/api/jobs/${run.id}/stop`);
  } catch (e) {
    toast(explain(e), 'bad');
  }
}

let pageTitle = 'GERMEN Studio';
function updateTitle() {
  document.title = (run.active ? `${fmt.pct(run.progress)} · ` : '') + pageTitle;
}

// ---------------------------------------------------------------------------- the form

const FORM_DEFAULTS = {
  name: '',
  source: 'demo',
  mb: 2,
  path: '',
  preset: 'mini',
  tokenizer: 'byte',
  vocab: 2048,
  bpe_mb: 64,
  steps: 2000,
  batch: 16,
  lr: 6e-4,
  warmup: 100,
  seq: null,
  accum: 1,
  replicas: 0,
  min_lr: 0.1,
  wd: 0.01,
  clip: 1,
  eval_every: 100,
  eval_batches: 8,
  save_every: 500,
  log_every: 10,
  samples: true,
  sample_every: 0,
  seed: 1,
  val: 0.05,
};

// the keys under which the form is kept in the settings (the server allows 48 short keys)
const FORM_KEYS = {
  name: 't_name',
  source: 't_source',
  mb: 't_mb',
  path: 't_path',
  preset: 't_preset',
  tokenizer: 't_tok',
  vocab: 't_vocab',
  bpe_mb: 't_bpemb',
  steps: 't_steps',
  batch: 't_batch',
  lr: 't_lr',
  warmup: 't_warm',
  seq: 't_seq',
  accum: 't_accum',
  replicas: 't_repl',
  min_lr: 't_minlr',
  wd: 't_wd',
  clip: 't_clip',
  eval_every: 't_evalev',
  eval_batches: 't_evalb',
  save_every: 't_save',
  log_every: 't_log',
  samples: 't_samples',
  sample_every: 't_sample',
  seed: 't_seed',
  val: 't_val',
};

function loadForm() {
  const saved = S.settings.form || {};
  const f = { ...FORM_DEFAULTS, mode: 'resume' };
  for (const [key, flat] of Object.entries(FORM_KEYS)) {
    const v = saved[flat];
    if (v === undefined || v === null) continue;
    if (typeof FORM_DEFAULTS[key] === 'number' && isNum(v)) f[key] = v;
    else if (typeof FORM_DEFAULTS[key] === 'string' && typeof v === 'string') f[key] = v;
    else if (typeof FORM_DEFAULTS[key] === 'boolean' && typeof v === 'boolean') f[key] = v;
    else if (key === 'seq' && isNum(v) && v > 0) f.seq = v;
  }
  if (!['demo', 'file', 'upload'].includes(f.source)) f.source = 'demo';
  if (f.source === 'upload') f.source = 'file'; // an upload does not survive a reload: its file does
  if (!PRESET_NAMES.includes(f.preset)) f.preset = 'mini';
  if (f.tokenizer !== 'bpe') f.tokenizer = 'byte';
  return f;
}

function formForSettings(f) {
  const out = { ...(S.settings.form || {}) };
  for (const [key, flat] of Object.entries(FORM_KEYS)) {
    const v = f[key];
    if (typeof v === 'number' && Number.isFinite(v)) out[flat] = v;
    else if (typeof v === 'boolean') out[flat] = v;
    else if (typeof v === 'string' && v.length <= 400) out[flat] = v;
    else delete out[flat];
  }
  return out;
}

const inRange = (x, lo, hi) => isNum(x) && x >= lo && x <= hi;
const isWhole = (x, lo, hi) => Number.isInteger(x) && x >= lo && x <= hi;

/** Checks the form like the server will; returns the messages by field and, if all is well, the request. */
function buildRequest(f, ctx) {
  const errors = {};
  const whole = (key, lo, hi) => {
    if (!isWhole(f[key], lo, hi)) errors[key] = t('val.int', { min: fmt.int(lo), max: fmt.int(hi) });
  };
  const real = (key, lo, hi, openLow = false) => {
    if (!inRange(f[key], lo, hi) || (openLow && f[key] <= lo)) errors[key] = t(openLow ? 'val.numOpen' : 'val.num', { min: fmt.sig(lo, 6), max: fmt.sig(hi, 8) });
  };
  const name = f.name.trim();
  if (!name) errors.name = t('val.name.empty');
  else if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name) || name.endsWith('.') || /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$/i.test(name)) errors.name = t('val.name.bad');

  let data = null;
  if (f.source === 'demo') {
    real('mb', 0, 1024, true);
    data = { kind: 'demo', mb: f.mb, seed: isWhole(f.seed, 0, Number.MAX_SAFE_INTEGER) ? f.seed : 1 };
  } else if (f.source === 'file') {
    if (!f.path) errors.path = t('val.data.file');
    data = { kind: 'file', path: f.path };
  } else {
    if (ctx.upload?.state === 'sending') errors.upload = t('val.data.uploading');
    else if (ctx.upload?.state !== 'done') errors.upload = t('val.data.upload');
    data = { kind: 'file', path: ctx.upload?.path || '' };
  }

  whole('steps', 1, 10_000_000);
  whole('batch', 1, 4096);
  whole('accum', 1, 4096);
  real('lr', 0, 1, true);
  if (!errors.steps && !isWhole(f.warmup, 0, f.steps)) errors.warmup = t('val.warmup');
  else if (errors.steps && !isWhole(f.warmup, 0, 10_000_000)) errors.warmup = t('val.int', { min: 0, max: fmt.int(10_000_000) });
  if (f.seq !== null && f.seq !== undefined) whole('seq', 8, 16384);
  whole('replicas', 0, 1024);
  real('min_lr', 0, 1);
  real('wd', 0, 1);
  real('clip', 0, 1000);
  whole('eval_every', 0, 10_000_000);
  whole('eval_batches', 1, 1000);
  whole('save_every', 0, 10_000_000);
  whole('log_every', 1, 10_000_000);
  whole('sample_every', 0, 10_000_000);
  whole('seed', 0, Number.MAX_SAFE_INTEGER);
  if (!(f.val >= 0 && f.val < 0.5)) errors.val = t('val.num', { min: 0, max: '0.5' });
  if (ctx.mode === 'new' || ctx.mode === 'overwrite') {
    if (f.tokenizer === 'bpe') {
      whole('vocab', 257, 65536);
      real('bpe_mb', 0, 4096, true);
    }
  } else if (ctx.existing && isWhole(f.steps, 1, 10_000_000) && f.steps <= ctx.existing.step) {
    errors.steps = t('val.steps.resume', { step: fmt.int(ctx.existing.step) });
  }
  if (Object.keys(errors).length) return { errors, request: null };

  const req = {
    name,
    mode: ctx.mode,
    data,
    steps: f.steps,
    batch: f.batch,
    accum: f.accum,
    replicas: f.replicas,
    lr: f.lr,
    min_lr_ratio: f.min_lr,
    warmup: f.warmup,
    weight_decay: f.wd,
    grad_clip: f.clip,
    eval_every: f.eval_every,
    eval_batches: f.eval_batches,
    save_every: f.save_every,
    log_every: f.log_every,
    sample_every: f.samples ? f.sample_every || Math.max(1, Math.round(f.steps / 10)) : 0,
    seed: f.seed,
    val_fraction: f.val,
  };
  if (f.seq) req.seq = f.seq;
  if (ctx.mode !== 'resume') {
    req.preset = f.preset;
    req.tokenizer = f.tokenizer;
    if (f.tokenizer === 'bpe') {
      req.vocab = f.vocab;
      req.bpe_sample_mb = f.bpe_mb;
    }
  }
  return { errors, request: req };
}

/** The size of a file, if its folder can be listed. */
async function fileSizeOf(path) {
  const cut = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
  if (cut < 0) return null;
  let dir = path.slice(0, cut);
  if (!dir || /^[A-Za-z]:$/.test(dir)) dir += path[cut];
  try {
    const listing = await api.get('/api/fs?path=' + encodeURIComponent(dir) + '&hidden=1');
    const entry = listing.entries.find((x) => !x.dir && x.name === path.slice(cut + 1));
    return entry ? entry.size : null;
  } catch {
    return null;
  }
}

function freeModelName() {
  const taken = new Set(S.models.map((m) => m.name));
  for (let n = 1; n < 1000; n += 1) {
    if (!taken.has(`germen-${n}`)) return `germen-${n}`;
  }
  return 'germen';
}

function TrainForm({ onStarted }) {
  const f = loadForm();
  const prefill = S.prefill;
  S.prefill = null;
  if (prefill?.name) {
    f.name = prefill.name;
    f.mode = prefill.mode || 'resume';
  } else if (!f.name || S.models.some((m) => m.name === f.name)) {
    f.name = freeModelName();
  }
  const ui = { slots: {}, showErrors: false, upload: null, uploadCtl: null, fileSize: null, estimate: null, estimateTicket: 0, starting: false, resumedFor: null };
  const root = h('div.page');
  const offs = [];

  // ---- what the form means right now
  const existing = () => S.models.find((m) => m.name === f.name.trim()) || null;
  const modeNow = () => {
    const ex = existing();
    if (!ex) return 'new';
    if (ex.error) return 'overwrite';
    return f.mode === 'overwrite' ? 'overwrite' : 'resume';
  };
  const dataBytes = () => (f.source === 'demo' ? (isNum(f.mb) ? f.mb * 1048576 : null) : f.source === 'file' ? ui.fileSize : ui.upload?.state === 'done' ? ui.upload.size : null);
  const check = () => buildRequest(f, { mode: modeNow(), existing: existing(), upload: ui.upload });

  const persist = debounce(() => {
    const next = formForSettings(f);
    S.settings.form = next;
    saveSettings({ form: next });
  }, 700);

  function changed() {
    persist();
    if (ui.showErrors) showErrors(check().errors);
    drawEstimate();
    fetchEstimate();
  }

  // ---- fields that can show their own error
  function tf(key, { label, hint, control, optional }) {
    const id = uid('tf');
    control.id = id;
    const errId = id + '-err';
    const hintId = hint ? id + '-hint' : null;
    const err = h('div.error-text', { id: errId, role: 'alert', hidden: true }, icon('error'), h('span'));
    ui.slots[key] = { control, err, hintId, errId };
    if (hintId) control.setAttribute('aria-describedby', hintId);
    return h(
      'div.field',
      { dataset: { field: key } },
      h('label.label', { for: id }, label, optional ? h('span.opt', ' ' + t('common.optional')) : null),
      control,
      hint ? h('div.hint', { id: hintId }, hint) : null,
      err,
    );
  }

  function showErrors(errors) {
    for (const [key, slot] of Object.entries(ui.slots)) {
      const message = errors[key];
      slot.err.hidden = !message;
      if (message) {
        slot.err.lastChild.textContent = message;
        slot.control.setAttribute('aria-invalid', 'true');
        slot.control.setAttribute('aria-describedby', [slot.hintId, slot.errId].filter(Boolean).join(' '));
      } else {
        slot.control.removeAttribute('aria-invalid');
        if (slot.hintId) slot.control.setAttribute('aria-describedby', slot.hintId);
        else slot.control.removeAttribute('aria-describedby');
      }
    }
  }

  const num = (key, { min, max, step = 1, int = true, placeholder }) =>
    numberInput({
      value: f[key],
      min,
      max,
      step: int ? step : 'any',
      placeholder,
      inputmode: int ? 'numeric' : 'decimal',
      onchange: (v) => {
        f[key] = v === null ? (key === 'seq' ? null : NaN) : v;
        changed();
      },
      // Fewer steps than the warm-up: the warm-up follows (a tenth of the steps) once the number is final.
      oncommit:
        key === 'steps'
          ? () => {
              if (isWhole(f.steps, 1, 10_000_000) && isWhole(f.warmup, 0, 10_000_000) && f.warmup > f.steps) {
                f.warmup = Math.floor(f.steps / 10);
                const box = ui.slots.warmup?.control;
                if (box) box.value = String(f.warmup);
                changed();
              }
            }
          : undefined,
    });

  // ---- 1. name
  const nameInput = h('input.input', { type: 'text', value: f.name, maxlength: 64, autocomplete: 'off', spellcheck: 'false', autocapitalize: 'off' });
  const modeBox = h('div.stack');
  nameInput.addEventListener('input', () => {
    f.name = nameInput.value;
    updateMode();
    changed();
  });
  const namePlate = h(
    'section.plate',
    { 'aria-labelledby': 'tr-name-h' },
    h('div.plate-h', h('h2', { id: 'tr-name-h' }, t('train.name.title'))),
    tf('name', { label: t('train.name.label'), hint: t('train.name.hint'), control: nameInput }),
    modeBox,
  );

  function updateMode() {
    const ex = existing();
    const m = modeNow();
    if (ex && m === 'resume' && ui.resumedFor !== ex.name && isNum(ex.step)) {
      ui.resumedFor = ex.name;
      f.steps = ex.step + 1000;
      const stepsInput = ui.slots.steps?.control;
      if (stepsInput) stepsInput.value = String(f.steps);
      if (f.warmup > f.steps) f.warmup = f.steps;
    }
    if (!ex) ui.resumedFor = null;
    modelPlate.hidden = m === 'resume';
    if (!ex) {
      mount(modeBox);
      return;
    }
    const choose = ex.error
      ? null
      : choiceGroup({
          name: 'tr-mode',
          label: t('train.mode.label'),
          value: m,
          options: [
            { value: 'resume', title: t('train.mode.resume'), sub: t('train.mode.resumeSub', { step: fmt.int(ex.step) }) },
            { value: 'overwrite', title: t('train.mode.overwrite'), sub: t('train.mode.overwriteSub') },
          ],
          onchange: (v) => {
            f.mode = v;
            updateMode();
            changed();
          },
        });
    mount(
      modeBox,
      ex.error ? banner('warn', t('train.mode.unreadable', { name: ex.name }), t('train.mode.unreadableText')) : banner('info', t('train.mode.exists', { name: ex.name }), t('train.mode.existsText', { step: fmt.int(ex.step), params: fmt.compact(ex.params) })),
      choose,
      m === 'overwrite' ? banner('warn', null, t('train.mode.overwriteWarn')) : null,
    );
  }

  // ---- 2. text
  const sourceBody = h('div.stack');
  const sourceChoice = choiceGroup({
    name: 'tr-source',
    label: t('train.data.title'),
    value: f.source,
    options: [
      { value: 'demo', title: t('train.data.demo'), sub: t('train.data.demoSub') },
      { value: 'file', title: t('train.data.file'), sub: t('train.data.fileSub') },
      { value: 'upload', title: t('train.data.upload'), sub: t('train.data.uploadSub') },
    ],
    onchange: (v) => {
      f.source = v;
      drawSource();
      changed();
    },
  });
  const dataPlate = h('section.plate', { 'aria-labelledby': 'tr-data-h' }, h('div.plate-h', h('h2', { id: 'tr-data-h' }, t('train.data.title'))), sourceChoice, sourceBody);

  async function pickFile() {
    const path = await pickPath({ mode: 'file', start: f.path ? f.path.replace(/[\\/][^\\/]*$/, '') || '/' : S.system?.data_dir || '' });
    if (!path) return;
    f.path = path;
    ui.fileSize = null;
    drawSource();
    changed();
    ui.fileSize = await fileSizeOf(path);
    if (f.path === path) {
      drawSource();
      changed();
    }
  }

  async function chooseUpload(file) {
    if (!file) return;
    const limit = S.system?.limits.upload_bytes || Infinity;
    if (file.size > limit) {
      ui.upload = { state: 'failed', name: file.name, error: t('err.too_large', { limit: fmt.bytes(limit) }) };
      drawSource();
      changed();
      return;
    }
    ui.uploadCtl?.abort();
    ui.uploadCtl = new AbortController();
    ui.upload = { state: 'sending', name: file.name, size: file.size, progress: 0 };
    drawSource();
    changed();
    const mine = ui.upload;
    try {
      const r = await api.upload(file, (p) => {
        mine.progress = p;
        ui.uploadBar?.set(p);
      }, ui.uploadCtl.signal);
      if (ui.upload !== mine) return;
      ui.upload = { state: 'done', name: r.name, size: r.bytes, path: r.path };
      f.path = r.path;
    } catch (e) {
      if (ui.upload !== mine || e?.name === 'AbortError') return;
      ui.upload = { state: 'failed', name: file.name, error: explain(e) };
    }
    drawSource();
    changed();
  }

  function drawSource() {
    ui.uploadBar = null;
    delete ui.slots.mb;
    delete ui.slots.path;
    delete ui.slots.upload;
    if (f.source === 'demo') {
      mount(
        sourceBody,
        h('div.grid-fields', tf('mb', { label: t('train.data.size'), hint: t('train.data.sizeHint'), control: num('mb', { min: 0.1, max: 1024, int: false, step: 'any' }) })),
      );
    } else if (f.source === 'file') {
      const pathInput = h('input.input', { type: 'text', readOnly: true, value: f.path, placeholder: t('train.data.noFile') });
      mount(
        sourceBody,
        tf('path', {
          label: t('train.data.path'),
          hint: ui.fileSize !== null && f.path ? t('train.data.fileSize', { size: fmt.bytes(ui.fileSize) }) : t('train.data.pathHint'),
          control: pathInput,
        }),
        h('div.row', h('button.btn', { type: 'button', onclick: pickFile }, icon('folder'), f.path ? t('train.data.change') : t('train.data.choose'))),
      );
    } else {
      const input = h('input.sr', { type: 'file', accept: '.txt,.md,.text,.csv,.tsv,.json,.log,text/*' });
      input.addEventListener('change', () => chooseUpload(input.files?.[0]));
      const zone = h(
        'div.dropzone',
        h('label.btn', input, icon('plus'), ui.upload ? t('train.data.otherFile') : t('train.data.pickUpload')),
        h('span.hint', t('train.data.drop', { limit: S.system ? fmt.bytes(S.system.limits.upload_bytes) : '' })),
      );
      zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.dataset.over = 'true';
      });
      zone.addEventListener('dragleave', () => {
        zone.dataset.over = 'false';
      });
      zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.dataset.over = 'false';
        chooseUpload(e.dataTransfer?.files?.[0]);
      });
      const up = ui.upload;
      let status = null;
      if (up?.state === 'sending') {
        ui.uploadBar = bar(up.progress, { thick: true, label: t('train.data.sending', { name: up.name }) });
        status = h('div.stack', { role: 'status' }, h('div.row-between.sans', h('span', t('train.data.sending', { name: up.name })), h('button.btn.btn-sm', { type: 'button', onclick: () => { ui.uploadCtl?.abort(); ui.upload = null; drawSource(); changed(); } }, t('common.cancel'))), ui.uploadBar);
      } else if (up?.state === 'done') {
        status = banner('ok', t('train.data.uploaded', { name: up.name }), t('train.data.uploadedText', { size: fmt.bytes(up.size) }));
      } else if (up?.state === 'failed') {
        status = banner('bad', t('train.data.uploadFailed', { name: up.name }), up.error);
      }
      const slotHolder = h('div', { id: uid('up') });
      ui.slots.upload = { control: slotHolder, err: h('div.error-text', { role: 'alert', hidden: true }, icon('error'), h('span')), hintId: null, errId: null };
      mount(sourceBody, zone, status, ui.slots.upload.err);
    }
    if (ui.showErrors) showErrors(check().errors);
  }

  // ---- 3. model size and tokenizer
  const presetData = (S.system?.presets || []).reduce((acc, p) => ({ ...acc, [p.name]: p }), {});
  const speedOf = (name) => S.settings.bench?.[name]?.tps;
  const presetChoice = choiceGroup({
    name: 'tr-preset',
    label: t('train.size.title'),
    value: f.preset,
    options: PRESET_NAMES.map((name) => {
      const p = presetData[name];
      const tps = speedOf(name);
      return {
        value: name,
        title: t('train.size.' + name),
        num: p ? t('train.size.params', { n: fmt.compact(p.params) }) : '',
        sub: p ? [t('train.size.shape', { layers: p.layers, d: fmt.int(p.d_model) }), tps ? t('train.size.speed', { n: fmt.compact(tps) }) : null].filter(Boolean).join(' · ') : '',
      };
    }),
    onchange: (v) => {
      f.preset = v;
      changed();
    },
  });
  const vocabControl = num('vocab', { min: 257, max: 65536 });
  const bpeMbControl = num('bpe_mb', { min: 1, max: 4096, int: false, step: 'any' });
  const bpeFields = h(
    'div.grid-fields',
    { hidden: f.tokenizer !== 'bpe' },
    tf('vocab', { label: t('train.tok.vocab'), hint: t('train.tok.vocabHint'), control: vocabControl }),
    tf('bpe_mb', { label: t('train.tok.sample'), hint: t('train.tok.sampleHint'), control: bpeMbControl }),
  );
  const tokenizerSeg = segmented({
    name: 'tr-tok',
    label: t('train.tok.title'),
    value: f.tokenizer,
    options: [
      { value: 'byte', label: t('train.tok.byte') },
      { value: 'bpe', label: t('train.tok.bpe') },
    ],
    onchange: (v) => {
      f.tokenizer = v;
      bpeFields.hidden = v !== 'bpe';
      tokHint.textContent = t('train.tok.' + v + 'Hint');
      changed();
    },
  });
  const tokHint = h('p.hint', t('train.tok.' + f.tokenizer + 'Hint'));
  const modelPlate = h(
    'section.plate',
    { 'aria-labelledby': 'tr-size-h' },
    h('div.plate-h', h('h2', { id: 'tr-size-h' }, t('train.size.title'))),
    h('p.plate-note', t('train.size.lede')),
    presetChoice,
    h('div.field', h('span.label', t('train.tok.title')), tokenizerSeg, tokHint),
    bpeFields,
  );

  // ---- 4. schedule
  const lrInput = h('input.input.num', { type: 'text', inputmode: 'decimal', value: String(f.lr), autocomplete: 'off', spellcheck: 'false' });
  lrInput.addEventListener('input', () => {
    const text = lrInput.value.trim().replace(',', '.');
    f.lr = text === '' ? NaN : Number(text);
    changed();
  });
  const samplesSwitch = switchBox({
    title: t('train.sched.samples'),
    hint: t('train.sched.samplesHint'),
    checked: f.samples,
    onchange: (v) => {
      f.samples = v;
      changed();
    },
  });
  const adv = h(
    'details.adv',
    h('summary', icon('chevron'), t('train.adv.title')),
    h(
      'div.adv-body',
      h('p.hint', t('train.adv.lede')),
      h(
        'div.grid-fields',
        tf('seq', { label: t('train.adv.seq'), hint: t('train.adv.seqHint'), control: num('seq', { min: 8, max: 16384, placeholder: '256' }), optional: true }),
        tf('accum', { label: t('train.adv.accum'), hint: t('train.adv.accumHint'), control: num('accum', { min: 1, max: 4096 }) }),
        tf('replicas', { label: t('train.adv.replicas'), hint: t('train.adv.replicasHint', { n: S.system?.workers || '?' }), control: num('replicas', { min: 0, max: 1024 }) }),
        tf('min_lr', { label: t('train.adv.minlr'), hint: t('train.adv.minlrHint'), control: num('min_lr', { min: 0, max: 1, int: false }) }),
        tf('wd', { label: t('train.adv.wd'), hint: t('train.adv.wdHint'), control: num('wd', { min: 0, max: 1, int: false }) }),
        tf('clip', { label: t('train.adv.clip'), hint: t('train.adv.clipHint'), control: num('clip', { min: 0, max: 1000, int: false }) }),
        tf('val', { label: t('train.adv.val'), hint: t('train.adv.valHint'), control: num('val', { min: 0, max: 0.49, int: false }) }),
        tf('eval_every', { label: t('train.adv.evalEvery'), hint: t('train.adv.evalEveryHint'), control: num('eval_every', { min: 0, max: 10_000_000 }) }),
        tf('eval_batches', { label: t('train.adv.evalBatches'), hint: t('train.adv.evalBatchesHint'), control: num('eval_batches', { min: 1, max: 1000 }) }),
        tf('save_every', { label: t('train.adv.saveEvery'), hint: t('train.adv.saveEveryHint'), control: num('save_every', { min: 0, max: 10_000_000 }) }),
        tf('log_every', { label: t('train.adv.logEvery'), hint: t('train.adv.logEveryHint'), control: num('log_every', { min: 1, max: 10_000_000 }) }),
        tf('sample_every', { label: t('train.adv.sampleEvery'), hint: t('train.adv.sampleEveryHint'), control: num('sample_every', { min: 0, max: 10_000_000 }) }),
        tf('seed', { label: t('train.adv.seed'), hint: t('train.adv.seedHint'), control: num('seed', { min: 0, max: Number.MAX_SAFE_INTEGER }) }),
      ),
    ),
  );
  const stepsHint = h('span', t('train.sched.stepsHint'));
  const schedPlate = h(
    'section.plate',
    { 'aria-labelledby': 'tr-sched-h' },
    h('div.plate-h', h('h2', { id: 'tr-sched-h' }, t('train.sched.title'))),
    h(
      'div.grid-fields',
      tf('steps', { label: t('train.sched.steps'), hint: stepsHint, control: num('steps', { min: 1, max: 10_000_000 }) }),
      tf('batch', { label: t('train.sched.batch'), hint: t('train.sched.batchHint'), control: num('batch', { min: 1, max: 4096 }) }),
      tf('lr', { label: t('train.sched.lr'), hint: t('train.sched.lrHint'), control: lrInput }),
      tf('warmup', { label: t('train.sched.warmup'), hint: t('train.sched.warmupHint'), control: num('warmup', { min: 0, max: 10_000_000 }) }),
    ),
    samplesSwitch,
    adv,
  );

  // ---- the estimate and the button
  const estimateBox = h('div.estimate', { 'aria-live': 'off' });
  const problems = h('div', { role: 'alert' });
  const startButton = h('button.btn.btn-primary.btn-lg', { type: 'button', onclick: () => start() });
  const side = h(
    'aside.plate.sticky',
    { 'aria-labelledby': 'tr-est-h' },
    h('div.plate-h', h('h2', { id: 'tr-est-h' }, t('train.est.title'))),
    estimateBox,
    problems,
    h('div.stack', startButton, h('p.hint', h('kbd', 'Ctrl'), ' + ', h('kbd', 'Enter'))),
  );

  const fetchEstimate = debounce(async () => {
    if (modeNow() === 'resume') {
      ui.estimate = null;
      return drawEstimate();
    }
    const vocab = f.tokenizer === 'bpe' ? f.vocab : 256;
    const seq = f.seq || 256;
    const ok = isWhole(f.batch, 1, 4096) && isWhole(seq, 8, 16384) && isWhole(vocab, 256, 65536) && isWhole(f.replicas, 0, 1024);
    if (!ok) return;
    const query = new URLSearchParams({ preset: f.preset, batch: String(f.batch), seq: String(seq), vocab: String(vocab), replicas: String(f.replicas) });
    const bytes = dataBytes();
    if (isNum(bytes) && bytes > 0) {
      query.set('data_bytes', String(Math.round(bytes)));
      query.set('tokenizer', f.tokenizer);
    }
    const ticket = ++ui.estimateTicket;
    try {
      const r = await api.get('/api/estimate?' + query);
      if (ticket === ui.estimateTicket) ui.estimate = r;
    } catch {
      if (ticket === ui.estimateTicket) ui.estimate = null;
    }
    if (ticket === ui.estimateTicket) drawEstimate();
  }, 220);

  function drawEstimate() {
    const m = modeNow();
    const ex = existing();
    const rows = [];
    const row = (label, value) => rows.push(h('div.est-row', h('span', label), h('b', value)));
    const est = ui.estimate;
    if (m === 'resume' && ex && !ex.error) {
      row(t('train.est.model'), t('train.est.modelValue', { params: fmt.compact(ex.params), layers: ex.layers, d: fmt.int(ex.d_model) }));
      row(t('train.est.from'), t('train.est.fromValue', { step: fmt.int(ex.step) }));
      if (isWhole(f.steps, 1, 10_000_000)) row(t('train.est.more'), fmt.int(Math.max(0, f.steps - ex.step)));
    } else if (est) {
      row(t('train.est.model'), t('train.est.modelValue', { params: fmt.compact(est.params), layers: est.layers, d: fmt.int(est.d_model) }));
    }
    if (est && m !== 'resume') {
      const limit = est.memory_limit_bytes;
      row(t('train.est.memory'), isNum(limit) ? t('train.est.memoryOf', { need: fmt.bytes(est.train_bytes), have: fmt.bytes(limit) }) : fmt.bytes(est.train_bytes));
      if (isNum(limit) && limit > 0) rows.push(h('div', bar(est.train_bytes / limit, { label: t('train.est.memory') })));
      if (est.replicas > 1) row(t('train.est.replicas'), fmt.int(est.replicas));
    }
    const seq = f.seq || (ex && m === 'resume' ? ex.context : 256);
    if (isWhole(f.batch, 1, 4096) && isWhole(f.accum, 1, 4096) && isWhole(f.steps, 1, 10_000_000) && isWhole(seq, 8, 16384)) {
      const perStep = f.batch * f.accum * seq;
      const stepsToRun = m === 'resume' && ex ? Math.max(0, f.steps - ex.step) : f.steps;
      const total = perStep * stepsToRun;
      row(t('train.est.perStep'), fmt.compact(perStep));
      row(t('train.est.tokens'), fmt.compact(total));
      const bytes = dataBytes();
      const tokenizerNow = m === 'resume' && ex ? ex.tokenizer : f.tokenizer;
      if (tokenizerNow === 'byte' && isNum(bytes) && bytes > 0) {
        const passes = total / (bytes * (1 - (f.val || 0)));
        row(t('train.est.passes'), passes < 1 ? t('train.est.partOf', { pct: fmt.pct(passes) }) : fmt.num(passes, passes < 10 ? 1 : 0));
      }
      const tps = speedOf(m === 'resume' ? '' : f.preset);
      if (tps) row(t('train.est.time'), '≈ ' + fmt.dur(total / tps));
      else rows.push(h('p.hint', t('train.est.noSpeed'), ' ', h('a', { href: '#/speed' }, t('train.est.measure'))));
    }
    const notes = [];
    if (est && m !== 'resume' && est.fits === false) notes.push(banner('bad', t('train.est.noFit'), est.problem || ''));
    if (est && est.corpus_fits === false) notes.push(banner('bad', t('train.est.noCorpus'), est.corpus_problem || ''));
    if (S.system && !S.system.optimised) notes.push(banner('warn', t('train.est.debug'), t('train.est.debugText')));
    mount(estimateBox, rows, notes);
    drawStartButton();
  }

  function drawStartButton() {
    const m = modeNow();
    const busyJob = run.active ? 'train' : bench.run && !FINAL.has(bench.run.state) ? 'bench' : null;
    startButton.disabled = ui.starting || !!busyJob;
    mount(startButton, icon(ui.starting ? 'clock' : 'play'), ui.starting ? t('train.starting') : m === 'resume' ? t('train.go.resume') : m === 'overwrite' ? t('train.go.overwrite') : t('train.go.new'));
    mount(problems, busyJob ? banner('info', null, t(busyJob === 'train' ? 'train.busy.train' : 'train.busy.bench')) : null);
  }

  async function start() {
    if (ui.starting) return;
    ui.showErrors = true;
    const { errors, request } = check();
    showErrors(errors);
    const keys = Object.keys(errors);
    if (keys.length) {
      const first = ui.slots[keys[0]]?.control || nameInput;
      first.closest('details')?.setAttribute('open', '');
      first.focus?.();
      reveal(first, 'center');
      announce(t('train.errors', { n: keys.length }), true);
      return;
    }
    if (request.mode === 'overwrite') {
      const ok = await confirmBox({ title: t('train.replace.title'), text: t('train.replace.text', { name: request.name }), confirm: t('train.replace.confirm'), danger: true });
      if (!ok) return;
    }
    ui.starting = true;
    drawStartButton();
    persist.flush();
    try {
      const r = await api.post('/api/train', request);
      runReset();
      await runAttach(r.job);
      refreshModels();
      onStarted();
    } catch (e) {
      ui.starting = false;
      drawStartButton();
      if (e instanceof ApiFailure && e.code === 'invalid_field' && e.params?.field) {
        const key = { 'data.path': 'path', 'data.mb': 'mb', min_lr_ratio: 'min_lr', weight_decay: 'wd', grad_clip: 'clip', val_fraction: 'val' }[e.params.field] || e.params.field;
        showErrors({ [key]: t('val.server') });
      }
      toast(explain(e), 'bad');
    }
  }

  // ---- put it together
  const left = h('div.stack-lg', namePlate, dataPlate, modelPlate, schedPlate);
  const grid = h('div.train-grid', left, side);
  const draw = () => {
    drawSource();
    updateMode();
    drawEstimate();
  };
  mount(root, pageHead(t('train.title'), t('train.lede')), grid);
  draw();
  fetchEstimate();
  if (f.source === 'file' && f.path) fileSizeOf(f.path).then((size) => { ui.fileSize = size; drawSource(); drawEstimate(); fetchEstimate(); });
  offs.push(bus.on('models', () => { updateMode(); drawEstimate(); }));
  offs.push(bus.on('bench', drawStartButton));
  offs.push(bus.on('run', drawStartButton));
  if (!S.models.length) refreshModels();
  primaryAction = () => start();
  return {
    el: root,
    destroy() {
      for (const off of offs) off();
      ui.uploadCtl?.abort();
      persist.flush();
    },
  };
}

// ---------------------------------------------------------------------------- the view of a run

const STATE_CHIP = {
  starting: ['run', 'clock'],
  running: ['run', 'train'],
  stopping: ['warn', 'stop'],
  done: ['ok', 'check'],
  stopped: ['warn', 'stop'],
  failed: ['bad', 'error'],
};

function TrainRun({ onNew, onContinue }) {
  const root = h('div.page');
  const offs = [];
  let ui = null;
  let shownSample = null; // the step of the sample the person picked, or null to follow the latest

  function build() {
    const sproutSvg = sprout(stageOf(run.progress, run.state), { breathing: run.active });
    const refs = { sprout: sproutSvg };
    const name = h('h1', { tabindex: '-1', id: 'page-title' }, run.label);
    refs.sub = h('div.sub');
    refs.heroNum = h('div.hero-num', '–');
    refs.heroCap = h('div.hero-cap');
    refs.pct = h('span.pb-pct');
    refs.eta = h('span.pb-eta');
    refs.bar = bar(0, { thick: true, label: t('run.progress') });
    refs.facts = h('div.hint.sans');
    refs.actions = h('div.row.run-actions');
    refs.notice = h('div.stack');
    refs.stats = h('dl.mini-stats');
    refs.timeline = h('div.timeline', { role: 'group', 'aria-label': t('run.timeline') });
    refs.specText = h('pre.spec-text', { 'aria-live': 'off' });
    refs.specHead = h('div.row-between');
    refs.log = h('ul.log');

    const loss = lineChart({
      label: t('run.fig1.label'),
      series: [
        { key: 'train', name: t('run.train'), color: 'var(--s1)', smooth: true, markers: 'auto' },
        { key: 'val', name: t('run.val'), color: 'var(--s2)', markers: 'dot' },
      ],
      xTitle: t('chart.step'),
      yTitle: t('chart.loss'),
      height: 300,
      canLog: true,
      canSmooth: true,
      valueFormat: (v) => fmt.loss(v),
      yFormat: (v, extra = 0) => fmt.num(v, (Math.abs(v) >= 10 ? 1 : 2) + extra),
    });
    const speed = lineChart({
      label: t('run.fig2.label'),
      series: [{ key: 'tps', name: t('run.speed'), color: 'var(--ink-2)', markers: 'auto' }],
      xTitle: t('chart.step'),
      yTitle: t('run.tokensPerSec'),
      height: 190,
      valueFormat: (v) => fmt.int(v),
    });
    const rate = lineChart({
      label: t('run.fig3.label'),
      series: [{ key: 'lr', name: t('run.lr'), color: 'var(--ink-2)', markers: 'auto' }],
      xTitle: t('chart.step'),
      yTitle: t('run.lr'),
      height: 190,
      valueFormat: (v) => fmt.lr(v),
      yFormat: (v, extra = 0) => fmt.lr(v, extra),
    });
    const grad = lineChart({
      label: t('run.fig4.label'),
      series: [{ key: 'grad', name: t('run.grad'), color: 'var(--ink-2)', markers: 'auto' }],
      xTitle: t('chart.step'),
      yTitle: t('run.grad'),
      height: 190,
      canLog: true,
      valueFormat: (v) => fmt.sig(v, 3),
    });
    refs.charts = { loss, speed, rate, grad };

    mount(
      root,
      h(
        'section.plate',
        { 'aria-labelledby': 'page-title' },
        h(
          'div.run-top',
          h('div.run-id', sproutSvg, h('div', name, refs.sub)),
          h('div.hero', refs.heroNum, refs.heroCap),
          h('div.progress-box', h('div.pb-top', refs.pct, refs.eta), refs.bar, refs.facts),
        ),
        refs.actions,
        refs.notice,
      ),
      h(
        'div.run-figs',
        h('section.plate', h('div.plate-h', h('h2', t('run.fig1.title'))), figure(1, t('run.fig1.caption'), loss.el)),
        h('section.plate.fill', h('div.plate-h', h('h2', t('run.stats'))), refs.stats),
      ),
      h('section.spec-sheet', { 'aria-labelledby': 'run-spec-h' }, refs.specHead, refs.specText, refs.timeline),
      h(
        'section.plate',
        h('div.plate-h', h('h2', t('run.more'))),
        h('div.grid-3', figure(2, t('run.fig2.caption'), speed.el), figure(3, t('run.fig3.caption'), rate.el), figure(4, t('run.fig4.caption'), grad.el)),
      ),
      h('section.plate', h('div.plate-h', h('h2', t('run.log'))), refs.log),
    );
    ui = refs;
    ui.seen = { samples: -1, log: -1, state: '', pick: undefined };
  }

  function stageLine() {
    const st = run.stage || {};
    const label = t('run.stage.' + (st.name || 'queued'));
    return isNum(st.total) && st.total > 0 ? t('run.stage.progress', { stage: label, done: fmt.int(st.done || 0), total: fmt.int(st.total) }) : label;
  }

  function update() {
    if (!ui) return;
    const [kind, ic] = STATE_CHIP[run.state] || STATE_CHIP.starting;
    const l = run.last;
    const total = l?.total || run.start?.total_steps || run.request?.steps || 0;
    const training = !!run.start;

    // the sprout grows with the run
    setSprout(ui.sprout, stageOf(run.progress, run.state));
    ui.sprout.classList.toggle('breathing', run.active);

    mount(
      ui.sub,
      chip(kind, t('run.state.' + run.state), ic),
      training ? h('span', t('run.stepOf', { step: fmt.int(l?.step || run.start.from_step || 0), total: fmt.int(total) })) : h('span', stageLine()),
      run.start ? h('span.muted', t('run.facts', { params: fmt.compact(run.start.params), layers: run.start.layers, d: fmt.int(run.start.d_model) })) : null,
    );

    // the big number: the loss, smoothed a little
    if (isNum(run.recent)) {
      ui.heroNum.textContent = fmt.loss(run.recent);
      const delta = isNum(run.firstLoss) ? run.recent - run.firstLoss : null;
      const parts = [h('b', t('run.hero.loss'))];
      if (isNum(delta) && Math.abs(delta) >= 0.005) {
        parts.push(' ', h('span.delta', { class: delta < 0 ? 'down' : 'up' }, icon(delta < 0 ? 'down' : 'up'), fmt.num(Math.abs(delta), 2)), ' ', t('run.hero.sinceStart'));
      }
      if (run.lastEval && isNum(run.lastEval.val_loss)) {
        parts.push(h('div', t('run.hero.val', { loss: fmt.loss(run.lastEval.val_loss) }), isNum(run.lastEval.bpb) ? ' · ' + t('run.hero.bpb', { bpb: fmt.num(run.lastEval.bpb, 2) }) : ''));
      } else if (run.active) {
        parts.push(h('div', t('run.hero.noVal')));
      }
      mount(ui.heroCap, parts);
    } else {
      ui.heroNum.textContent = '–';
      mount(ui.heroCap, run.active ? stageLine() : run.state === 'failed' ? t('run.hero.failed') : '');
    }

    // progress
    const stageTotal = run.stage?.total;
    const showStage = !training && isNum(stageTotal) && stageTotal > 0;
    const p = showStage ? (run.stage.done || 0) / stageTotal : run.progress;
    ui.bar.set(p);
    ui.pct.textContent = training || run.state === 'done' ? fmt.pct(run.progress) : showStage ? fmt.pct(p) : '…';
    ui.eta.textContent = run.active && training && isNum(l?.eta) && l.eta > 0 ? t('run.eta', { time: fmt.dur(l.eta) }) : run.state === 'done' ? t('run.took', { time: fmt.dur(run.result?.elapsed_secs) }) : '';
    mount(
      ui.facts,
      training && l ? [t('run.speedNow', { n: fmt.int(l.tps) }), ' · ', t('run.elapsed', { time: fmt.dur(l.elapsed) })] : null,
    );

    // actions and the notice that goes with the end
    const actions = [];
    if (run.active) {
      actions.push(h('button.btn.btn-lg', { type: 'button', disabled: run.state === 'stopping', onclick: runStop }, icon('stop'), run.state === 'stopping' ? t('run.stopping') : t('run.stop')));
      actions.push(h('span.hint', t('run.stopHint')));
    } else {
      if (run.state !== 'failed' || run.result) actions.push(h('a.btn.btn-primary.btn-lg', { href: '#/generate/' + encodeURIComponent(run.label) }, icon('write'), t('run.write')));
      if (run.state !== 'failed') actions.push(h('button.btn', { type: 'button', onclick: onContinue }, icon('train'), t('run.continue')));
      actions.push(h('button.btn', { type: 'button', onclick: onNew }, icon('plus'), t(run.state === 'failed' ? 'run.backToForm' : 'run.another')));
    }
    mount(ui.actions, actions);
    const notice = [];
    if (run.state === 'done') {
      notice.push(banner('ok', t('run.done.title'), t('run.done.text', { step: fmt.int(run.result?.final_step), best: fmt.loss(run.result?.best_val_loss ?? run.result?.last_train_loss), time: fmt.dur(run.result?.elapsed_secs) })));
    } else if (run.state === 'stopped') {
      notice.push(banner('warn', t('run.stopped.title'), run.result?.before_training ? t('run.stopped.early') : t('run.stopped.text', { step: fmt.int(run.result?.final_step) })));
    } else if (run.state === 'failed') {
      notice.push(banner('bad', t('run.failed.title'), run.error ? explain(new ApiFailure(422, run.error.code, run.error.message, run.error.params)) : t('err.unknown')));
    }
    if (run.start && run.start.resume && run.start.optimizer_restored === false) notice.push(banner('info', null, t('run.optimizerFresh')));
    mount(ui.notice, notice);

    // numbers on the side
    const stat = (label, value) => h('div', h('dt', label), h('dd', value));
    const evalBpb = run.lastEval?.bpb;
    const tokensSeen = run.start && l ? (l.step - 0) * run.start.tokens_per_step : null;
    mount(
      ui.stats,
      stat(t('run.s.params'), run.start ? fmt.compact(run.start.params) : '–'),
      stat(t('run.s.tokens'), isNum(tokensSeen) ? fmt.compact(tokensSeen) : '–'),
      stat(t('run.s.perStep'), run.start ? fmt.compact(run.start.tokens_per_step) : '–'),
      stat(t('run.s.corpus'), run.start ? fmt.bytes(run.start.corpus_bytes) : '–'),
      stat(t('run.s.vocab'), run.start ? (run.start.tokenizer === 'bpe' ? t('models.vocabBpe', { n: fmt.int(run.start.vocab_size) }) : t('models.vocabByte')) : '–'),
      stat(t('run.s.memory'), run.start ? fmt.bytes(run.start.est_memory_bytes) : '–'),
      stat(t('run.s.threads'), run.start ? t('run.s.threadsValue', { w: run.start.workers, r: run.start.replicas }) : '–'),
      stat(t('run.s.bpb'), isNum(evalBpb) ? fmt.num(evalBpb, 2) : '–'),
    );

    // charts
    ui.charts.loss.update({
      train: { x: run.steps.col('step'), y: run.steps.col('loss') },
      val: { x: run.evals.col('step'), y: run.evals.col('val_loss') },
    });
    ui.charts.speed.update({ tps: { x: run.steps.col('step'), y: run.steps.col('tps') } });
    ui.charts.rate.update({ lr: { x: run.steps.col('step'), y: run.steps.col('lr') } });
    ui.charts.grad.update({ grad: { x: run.steps.col('step'), y: run.steps.col('grad_norm') } });

    drawSpecimen();
    drawLog();
  }

  function drawSpecimen() {
    const samples = run.samples;
    const latest = samples.length ? samples[samples.length - 1] : null;
    const pick = shownSample === null ? latest : samples.find((x) => x.step === shownSample) || latest;
    const sig = `${samples.length}:${latest?.step}:${pick?.step}:${run.state}`;
    if (ui.seen.sig === sig) return;
    ui.seen.sig = sig;
    mount(
      ui.specHead,
      h('div', h('h2', { id: 'run-spec-h' }, t('run.spec.title')), h('p.hint', pick ? t('run.spec.at', { step: fmt.int(pick.step) }) : t('run.spec.none'))),
      pick ? copyButton(() => pick.text, t('common.copy')) : null,
    );
    mount(ui.specText, pick ? pick.text : run.active ? t('run.spec.waiting') : t('run.spec.empty'));
    mount(
      ui.timeline,
      samples.length > 1
        ? samples.map((x) =>
            h('button.tick', { type: 'button', 'aria-pressed': String(pick && x.step === pick.step), onclick: () => { shownSample = x === latest ? null : x.step; ui.seen.sig = ''; drawSpecimen(); } }, fmt.int(x.step)),
          )
        : null,
    );
  }

  function drawLog() {
    const sig = run.log.length + ':' + run.state;
    if (ui.seen.log === sig) return;
    ui.seen.log = sig;
    const text = (entry) => {
      switch (entry.type) {
        case 'start':
          return ['play', t('run.log.start')];
        case 'saved':
          return ['save', t('run.log.saved', { step: fmt.int(entry.step) })];
        case 'stage':
          return ['clock', t('run.stage.' + entry.name)];
        case 'stopping':
          return ['stop', t('run.log.stopping')];
        case 'recovered':
          return ['refresh', t('run.log.recovered.' + entry.kind, { n: entry.files })];
        case 'end':
          return [entry.state === 'done' ? 'check' : entry.state === 'failed' ? 'error' : 'stop', t('run.log.end.' + entry.state)];
        default:
          return ['info', entry.type];
      }
    };
    mount(
      ui.log,
      run.log
        .slice()
        .reverse()
        .map((entry) => {
          const [ic, label] = text(entry);
          return h('li', h('time', fmt.clock(entry.t)), icon(ic), h('span', label));
        }),
    );
  }

  build();
  update();
  offs.push(bus.on('run', update));
  primaryAction = null;
  return {
    el: root,
    destroy() {
      for (const off of offs) off();
    },
  };
}

// ---------------------------------------------------------------------------- the page

function PageTrain() {
  const root = h('div');
  root.style.display = 'contents';
  let view = null;
  let which = null;

  function show(next) {
    view?.destroy?.();
    which = next;
    if (next === 'run') {
      view = TrainRun({
        onNew() {
          runReset();
          show('form');
        },
        onContinue() {
          S.prefill = { name: run.label, mode: 'resume' };
          runReset();
          show('form');
        },
      });
    } else {
      view = TrainForm({
        onStarted(again) {
          show(again === true ? 'form' : 'run');
        },
      });
    }
    mount(root, view.el);
  }

  show(run.active || (run.id && !S.prefill) ? 'run' : 'form');
  const off = bus.on('run', () => {
    // a run that was started elsewhere (or found after a reload) takes the page over
    if (which === 'form' && run.active) show('run');
  });
  return {
    el: root,
    destroy() {
      off();
      view?.destroy?.();
    },
  };
}
