// 50-pages.js — the frame of the application (rail, router, preferences) and the pages that are not
// training: models, writing, speed, settings, about.

const NAV = [
  ['train', 'train'],
  ['models', 'models'],
  ['generate', 'write'],
  ['speed', 'speed'],
  ['settings', 'settings'],
];
let currentPage = null;
let primaryAction = null; // what Ctrl+Enter does on the page that is shown

// ---------------------------------------------------------------------------- preferences

function applyPrefs() {
  const st = S.settings;
  const root = document.documentElement;
  setLang(st.lang || 'auto');
  const dark = st.theme === 'dark' || (st.theme !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches);
  root.dataset.theme = dark ? 'dark' : 'light';
  root.dataset.contrast = st.contrast === 'more' ? 'more' : 'auto';
  root.dataset.motion = st.motion === 'reduce' ? 'reduce' : 'auto';
  root.dataset.density = st.density === 'compact' ? 'compact' : 'comfortable';
  root.style.setProperty('--scale', String(clamp(Number(st.text_scale) || 1, 0.85, 1.6)));
  const shell = $('.shell');
  if (shell) shell.dataset.rail = st.rail === 'icons' ? 'compact' : 'full';
  bus.emit('prefs');
}

let saveQueue = Promise.resolve();
/** Saves some settings (the server checks them) and applies what it answers. */
function saveSettings(patch) {
  Object.assign(S.settings, patch);
  const send = async () => {
    try {
      S.settings = await api.put('/api/settings', patch);
      applyPrefs();
    } catch (e) {
      toast(explain(e), 'bad');
      try {
        S.settings = await api.get('/api/settings');
        applyPrefs();
      } catch {
        /* the next try will show it */
      }
    }
  };
  saveQueue = saveQueue.then(send);
  return saveQueue;
}
const saveFormLater = debounce((form) => saveSettings({ form }), 700);

// ---------------------------------------------------------------------------- shell and router

function pageHead(title, lede) {
  return h('header.page-head', h('h1', { tabindex: '-1', id: 'page-title' }, title), lede ? h('p.lede', lede) : null);
}

function renderShell() {
  const items = NAV.map(([id, ic]) =>
    h('a.rail-item', { href: `#/${id}`, dataset: { page: id, label: t('nav.' + id) }, 'aria-label': t('nav.' + id) }, icon(ic), h('span.rail-label', t('nav.' + id))),
  );
  const job = h('a.rail-job', { href: '#/train', hidden: true, id: 'rail-job' });
  const aboutLink = h('a.rail-item.rail-more', { href: '#/about', dataset: { page: 'about', label: t('nav.about') }, 'aria-label': t('nav.about') }, icon('about'), h('span.rail-label', t('nav.about')));
  const quit = h('button.rail-item.rail-more', { type: 'button', onclick: quitStudio, dataset: { label: t('nav.quit') }, 'aria-label': t('nav.quit') }, icon('power'), h('span.rail-label', t('nav.quit')));
  const rail = h(
    'nav.rail',
    { 'aria-label': t('nav.label') },
    h('a.brand', { href: '#/train', 'aria-label': 'GERMEN ' + t('shell.tagline') }, sprout(3), h('span.brand-text', h('span.brand-name', 'GERMEN'), ' ', h('span.brand-sub', t('shell.tagline')))),
    h('div.rail-nav', items),
    h('div.rail-sp'),
    job,
    aboutLink,
    MODE === 'demo' ? null : quit, // a demonstration has no program behind it to close
  );
  const shell = h('div.shell', { dataset: { rail: S.settings.rail === 'icons' ? 'compact' : 'full' } }, rail, h('main.main', { id: 'main', tabindex: '-1' }));
  const skip = h('a.skip', { href: '#main', onclick: (e) => { e.preventDefault(); $('#main').focus(); } }, t('shell.skip'));
  mount($('#app'), skip, shell);
}

function updateRailJob() {
  const el = $('#rail-job');
  if (!el) return;
  const run = S.run;
  if (!run || !run.active) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  mount(
    el,
    h('span.rail-job-text', h('b', run.label), h('span', t('shell.training', { pct: fmt.pct(run.progress) }))),
    (() => {
      const b = bar(run.progress, { label: run.label });
      return b;
    })(),
  );
  el.setAttribute('aria-label', t('shell.trainingNow', { name: run.label }));
  el.dataset.label = t('shell.trainingNow', { name: run.label });
}

function route() {
  const found = /^#\/([a-z]+)(?:\/(.*))?$/.exec(location.hash);
  const id = found && PAGES[found[1]] ? found[1] : 'train';
  const arg = found?.[2] ? decodeURIComponent(found[2]) : '';
  currentPage?.destroy?.();
  primaryAction = null;
  const main = $('#main');
  const page = PAGES[id]({ arg });
  mount(main, banners(), page.el);
  currentPage = { id, destroy: page.destroy };
  for (const a of $$('.rail-item[data-page]')) {
    if (a.dataset.page === id) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  }
  pageTitle = `${t('nav.' + id)} · GERMEN Studio`;
  updateTitle();
  window.scrollTo(0, 0);
  drawOnlineBanner();
  if (S.refocus) {
    $(S.refocus)?.focus();
    S.refocus = null;
  } else if (S.routed) {
    $('#page-title')?.focus({ preventScroll: true });
  }
  S.routed = true;
}

function banners() {
  return h('div#banners', { style: { display: 'contents' } });
}

function drawOnlineBanner() {
  const box = $('#banners');
  if (!box) return;
  const notes = [];
  if (MODE === 'demo') notes.push(banner('info', t('demo.title'), t('demo.text')));
  if (!S.online) notes.push(banner('warn', t('shell.offline.title'), t('shell.offline.text')));
  mount(box, notes.length ? h('div.page', { style: { paddingBottom: '0' } }, notes) : null);
}

async function quitStudio() {
  const running = S.run?.active;
  const ok = await confirmBox({
    title: t('quit.title'),
    text: running ? t('quit.textRunning') : t('quit.text'),
    confirm: t('quit.confirm'),
    danger: true,
  });
  if (!ok) return;
  try {
    await api.post('/api/quit');
  } catch {
    /* the server may already be gone */
  }
  S.quitting = true;
  mount($('#app'), h('main.farewell', h('div.plate', sprout(4), h('h1', t('quit.done.title')), h('p', t('quit.done.text')))));
  document.title = 'GERMEN Studio';
}

function renderLocked(reason) {
  setLang('auto');
  document.documentElement.dataset.theme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  mount(
    $('#app'),
    h(
      'main.farewell',
      h('div.plate', sprout(2), h('h1', t('lock.title')), h('p', rich(reason === 'offline' ? t('lock.offline') : t('lock.text'))), h('button.btn.btn-primary', { type: 'button', onclick: () => location.reload() }, icon('refresh'), t('common.retry'))),
    ),
  );
}

// ---------------------------------------------------------------------------- models

function modelFacts(m) {
  const fact = (label, value) => h('span', label + ' ', h('b', value));
  if (m.error) return [h('span', h('b', t('models.unreadable')))];
  const out = [fact(t('models.fact.step'), fmt.int(m.step)), fact(t('models.fact.params'), fmt.compact(m.params))];
  out.push(fact(t('models.fact.vocab'), m.tokenizer === 'bpe' ? t('models.vocabBpe', { n: fmt.int(m.vocab_size) }) : t('models.vocabByte')));
  if (isNum(m.best_val_loss)) out.push(fact(t('models.fact.val'), fmt.loss(m.best_val_loss)));
  else if (isNum(m.last_train_loss)) out.push(fact(t('models.fact.loss'), fmt.loss(m.last_train_loss)));
  out.push(fact(t('models.fact.size'), fmt.bytes(m.bytes)));
  return out;
}

async function refreshModels() {
  try {
    S.models = (await api.get('/api/models')).models;
    bus.emit('models');
  } catch (e) {
    toast(explain(e), 'bad');
  }
}

function PageModels({ arg }) {
  let selected = arg || null;
  const checks = new Map(); // name -> {state, result|error}
  const root = h('div.page');
  const unsub = bus.on('models', draw);

  async function check(m) {
    checks.set(m.name, { state: 'busy' });
    draw();
    try {
      const r = await api.post('/api/verify', { model: m.name });
      checks.set(m.name, { state: 'ok', result: r });
      announce(t('models.check.ok', { step: fmt.int(r.step) }));
    } catch (e) {
      checks.set(m.name, { state: 'bad', error: explain(e) });
      announce(explain(e), true);
    }
    draw();
  }
  async function repair(m) {
    try {
      const r = await api.post('/api/recover', { model: m.name });
      toast(t('models.repair.' + r.kind, { n: r.files }), 'ok');
      await refreshModels();
    } catch (e) {
      toast(explain(e), 'bad');
    }
  }
  async function openOther() {
    const path = await pickPath({ mode: 'model' });
    if (path) location.hash = '#/generate/' + encodeURIComponent(path);
  }
  function openTraining(m) {
    S.prefill = { name: m.name, mode: 'resume' };
    location.hash = '#/train';
  }

  function detail(m) {
    const c = checks.get(m.name);
    const facts = [
      [t('models.d.arch'), m.error ? '–' : t('models.d.archValue', { layers: m.layers, heads: m.heads, d: fmt.int(m.d_model) })],
      [t('models.d.context'), m.error ? '–' : t('models.d.contextValue', { n: fmt.int(m.context) })],
      [t('models.d.params'), m.error ? '–' : fmt.int(m.params)],
      [t('models.d.vocab'), m.error ? '–' : m.tokenizer === 'bpe' ? t('models.vocabBpe', { n: fmt.int(m.vocab_size) }) : t('models.vocabByte')],
      [t('models.d.step'), m.error ? '–' : fmt.int(m.step)],
      [t('models.d.tokens'), m.error ? '–' : fmt.int(m.tokens_seen)],
      [t('models.d.trainLoss'), m.error ? '–' : fmt.loss(m.last_train_loss)],
      [t('models.d.valLoss'), m.error ? '–' : fmt.loss(m.last_val_loss)],
      [t('models.d.bestLoss'), m.error ? '–' : fmt.loss(m.best_val_loss)],
      [t('models.d.corpus'), m.error ? '–' : fmt.bytes(m.corpus_bytes)],
      [t('models.d.time'), m.error ? '–' : fmt.dur(m.elapsed_secs)],
      [t('models.d.size'), fmt.bytes(m.bytes)],
      [t('models.d.saved'), fmt.date(m.modified)],
    ];
    return h(
      'section.plate',
      { 'aria-labelledby': 'model-detail-title' },
      h('div.plate-h', h('h2', { id: 'model-detail-title' }, m.name), h('div.row', copyButton(() => m.path, t('models.copyPath')))),
      h('div.path.muted', m.path),
      m.error ? banner('bad', t('models.unreadable'), m.error) : null,
      !m.error && !m.has_manifest ? banner('info', null, t('models.noManifest')) : null,
      h('dl.kv', facts.map(([k, v]) => [h('dt', k), h('dd', v)])),
      h(
        'div.row',
        h('button.btn', { type: 'button', onclick: () => check(m), disabled: c?.state === 'busy' }, icon('check'), c?.state === 'busy' ? t('models.check.busy') : t('models.check.button')),
        h('button.btn', { type: 'button', onclick: () => repair(m) }, icon('refresh'), t('models.repair.button')),
      ),
      c?.state === 'ok'
        ? banner(
            'ok',
            t('models.check.ok', { step: fmt.int(c.result.step) }),
            h('div', t('models.check.files', { n: c.result.files.length, size: fmt.bytes(c.result.files.reduce((a, f) => a + f.bytes, 0)) })),
          )
        : null,
      c?.state === 'bad' ? banner('bad', t('models.check.bad'), c.error) : null,
      h('p.hint', t('models.repair.hint')),
    );
  }

  function draw() {
    const list = S.models;
    if (selected && !list.some((m) => m.name === selected)) selected = null;
    const m = list.find((x) => x.name === selected);
    mount(
      root,
      pageHead(t('models.title'), t('models.lede')),
      h('div.row', h('button.btn', { type: 'button', onclick: openOther }, icon('folder'), t('models.openOther')), h('button.btn.btn-quiet', { type: 'button', onclick: refreshModels }, icon('refresh'), t('common.refresh'))),
      list.length
        ? h(
            'div.model-list',
            list.map((x) =>
              h(
                'div.model-row',
                { 'aria-current': String(x.name === selected) },
                h('div', h('h2', x.name), h('div.when', t('models.saved', { when: fmt.ago(x.modified) }))),
                h('div.model-facts', modelFacts(x)),
                h(
                  'div.row',
                  h('a.btn.btn-sm', { href: '#/generate/' + encodeURIComponent(x.name), class: x.error ? 'btn-quiet' : '', 'aria-disabled': x.error ? 'true' : null }, icon('write'), t('models.write')),
                  h('button.btn.btn-sm', { type: 'button', onclick: () => openTraining(x), disabled: !!x.error }, icon('train'), t('models.continue')),
                  h('button.btn.btn-sm.btn-quiet', { type: 'button', 'aria-expanded': String(x.name === selected), onclick: () => { selected = selected === x.name ? null : x.name; draw(); } }, t('models.details')),
                ),
              ),
            ),
          )
        : h('div.plate', h('div.empty', sprout(0), h('h2', t('models.empty.title')), h('p', t('models.empty.text')), h('a.btn.btn-primary', { href: '#/train' }, icon('train'), t('models.empty.cta')))),
      m ? detail(m) : null,
    );
  }
  refreshModels();
  draw();
  return {
    el: root,
    destroy: unsub,
  };
}

// ---------------------------------------------------------------------------- writing

function PageGenerate({ arg }) {
  const saved = S.settings.form || {};
  const f = {
    model: arg || S.genModel || '',
    prompt: S.genPrompt || '',
    max_new: Number(saved.gen_max_new) || 300,
    temperature: isNum(saved.gen_temperature) ? saved.gen_temperature : 0.8,
    top_k: isNum(saved.gen_top_k) ? saved.gen_top_k : 40,
    window: Number(saved.gen_window) || 128,
    seed: '',
  };
  const gen = { state: 'idle', job: null, ctl: null, info: null, result: null, error: null };
  const root = h('div.page');
  const outPrompt = h('span.prompt');
  const outText = document.createTextNode('');
  const cursor = h('span.cursor', { hidden: true });
  const outBox = h('div.out-box', { 'aria-live': 'off' }, outPrompt, outText, cursor);
  const status = h('div.row.sans', { 'aria-live': 'off' });
  const actions = h('div.row');
  const unsub = bus.on('models', buildForm);
  const saveGen = () => {
    const next = { ...(S.settings.form || {}), gen_max_new: f.max_new, gen_temperature: f.temperature, gen_top_k: f.top_k, gen_window: f.window };
    S.settings.form = next;
    saveFormLater(next);
  };

  function modelInfo() {
    return S.models.find((m) => m.name === f.model);
  }

  function setState(state) {
    gen.state = state;
    cursor.hidden = !(state === 'writing' || state === 'loading');
    drawStatus();
  }

  function drawStatus() {
    const busy = gen.state === 'loading' || gen.state === 'writing' || gen.state === 'stopping';
    mount(
      status,
      gen.state === 'loading' ? chip('run', t('gen.loading'), 'clock') : null,
      gen.state === 'writing' ? chip('run', t('gen.writing'), 'write') : null,
      gen.state === 'stopping' ? chip('warn', t('gen.stopping'), 'clock') : null,
      gen.state === 'done' && gen.result ? [chip('ok', t('gen.done'), 'check'), h('span.muted', t('gen.stats', { tokens: fmt.int(gen.result.tokens), rate: fmt.int(gen.result.tokens_per_sec), seed: gen.result.seed }))] : null,
      gen.state === 'stopped' && gen.result ? [chip('warn', t('gen.stopped'), 'stop'), h('span.muted', t('gen.stats', { tokens: fmt.int(gen.result.tokens), rate: fmt.int(gen.result.tokens_per_sec), seed: gen.result.seed }))] : null,
      gen.state === 'failed' ? chip('bad', gen.error || t('err.unknown'), 'error') : null,
      gen.info ? h('span.muted', t('gen.loaded', { step: fmt.int(gen.info.step), params: fmt.compact(gen.info.params) })) : null,
    );
    const hasText = outText.data.length > 0;
    mount(
      actions,
      busy
        ? h('button.btn.btn-lg', { type: 'button', onclick: stop }, icon('stop'), t('gen.stop'))
        : h('button.btn.btn-primary.btn-lg', { type: 'button', disabled: !f.model, onclick: start }, icon('write'), t('gen.start')),
      hasText && !busy ? copyButton(() => outPrompt.textContent + outText.data, t('gen.copy')) : null,
      hasText && !busy
        ? h('button.btn.btn-quiet.btn-sm', { type: 'button', onclick: () => { f.prompt = outPrompt.textContent + outText.data; S.genPrompt = f.prompt; outPrompt.textContent = ''; outText.data = ''; buildForm(); } }, icon('plus'), t('gen.continueFrom'))
        : null,
    );
  }

  async function start() {
    if (!f.model) return;
    const body = { model: f.model, prompt: f.prompt, max_new: f.max_new, temperature: f.temperature, top_k: f.top_k, window: f.window };
    if (f.seed !== '' && f.seed !== null) body.seed = Number(f.seed);
    gen.info = null;
    gen.result = null;
    gen.error = null;
    outPrompt.textContent = f.prompt;
    outText.data = '';
    setState('loading');
    if (matchMedia('(max-width: 70rem)').matches) reveal(result, 'start');
    try {
      const r = await api.post('/api/generate', body);
      gen.job = r.job;
      gen.ctl = new AbortController();
      follow(r.job, {
        signal: gen.ctl.signal,
        onSnapshot: (snap) => {
          outText.data = printable(snap.generate?.text);
        },
        onEvent: (e) => {
          if (e.type === 'loaded') {
            gen.info = e;
            setState('writing');
          } else if (e.type === 'text') {
            outText.data += printable(e.text);
          } else if (e.type === 'end') {
            gen.result = e.result;
            if (e.state === 'failed') {
              gen.error = e.error?.message || t('err.unknown');
              if (e.error?.code) gen.error = explain(new ApiFailure(422, e.error.code, e.error.message, e.error.params));
              announce(gen.error, true);
            } else {
              announce(t('gen.finished', { tokens: fmt.int(e.result?.tokens || 0) }));
            }
            gen.job = null;
            setState(e.state);
          }
        },
      }).catch((e) => {
        setState('failed');
        gen.error = explain(e);
        drawStatus();
      });
    } catch (e) {
      gen.error = explain(e);
      setState('failed');
      toast(gen.error, 'bad');
    }
  }

  async function stop() {
    if (!gen.job) return;
    setState('stopping');
    try {
      await api.post(`/api/jobs/${gen.job}/stop`);
    } catch (e) {
      toast(explain(e), 'bad');
    }
  }

  function buildForm() {
    const models = S.models;
    const sel = h('select.select');
    const options = models.filter((m) => !m.error).map((m) => h('option', { value: m.name, selected: m.name === f.model }, `${m.name} (${fmt.compact(m.params)})`));
    if (f.model && !models.some((m) => m.name === f.model)) options.unshift(h('option', { value: f.model, selected: true }, basename(f.model)));
    sel.append(...options, h('option', { value: '__other' }, t('gen.otherFolder')));
    if (!f.model && models.length) f.model = models.find((m) => !m.error)?.name || '';
    if (f.model) sel.value = f.model;
    sel.addEventListener('change', async () => {
      if (sel.value === '__other') {
        const path = await pickPath({ mode: 'model' });
        if (path) {
          f.model = path;
          S.genModel = path;
        }
        buildForm();
        return;
      }
      f.model = sel.value;
      S.genModel = f.model;
      buildForm();
    });
    const info = modelInfo();
    const training = S.run?.active && S.run.label === f.model;
    const prompt = h('textarea.textarea', { rows: 4, placeholder: t('gen.promptPlaceholder'), maxlength: 20000 });
    prompt.value = f.prompt;
    prompt.addEventListener('input', () => {
      f.prompt = prompt.value;
      S.genPrompt = f.prompt;
    });
    const lengthInput = numberInput({ value: f.max_new, min: 1, max: 20000, onchange: (v) => { if (v) { f.max_new = clamp(Math.round(v), 1, 20000); saveGen(); } } });
    const temp = h('input', { type: 'range', min: 0, max: 2, step: 0.05, value: f.temperature });
    const tempNum = numberInput({ label: t('gen.temperature'), value: f.temperature, min: 0, max: 5, step: 0.05, inputmode: 'decimal', onchange: (v) => { if (v !== null) { f.temperature = clamp(v, 0, 5); temp.value = String(f.temperature); saveGen(); } } });
    temp.addEventListener('input', () => {
      f.temperature = Number(temp.value);
      tempNum.value = String(f.temperature);
      saveGen();
    });
    const topk = numberInput({ value: f.top_k, min: 0, max: 65536, onchange: (v) => { if (v !== null) { f.top_k = clamp(Math.round(v), 0, 65536); saveGen(); } } });
    const maxWindow = info?.context || 4096;
    const win = numberInput({ value: Math.min(f.window, maxWindow), min: 2, max: maxWindow, onchange: (v) => { if (v) { f.window = clamp(Math.round(v), 2, maxWindow); saveGen(); } } });
    const seed = numberInput({ value: f.seed, min: 0, placeholder: t('gen.seedRandom'), onchange: (v) => { f.seed = v === null ? '' : Math.max(0, Math.round(v)); } });
    const dice = h('button.btn.btn-icon', { type: 'button', 'aria-label': t('gen.seedRoll'), onclick: () => { f.seed = Math.floor(Math.random() * 1e9); seed.value = String(f.seed); } }, icon('dice'));
    mount(
      startBox,
      h('div.plate-h', h('h2', t('gen.begin'))),
      field({ label: t('gen.model'), control: sel, hint: info ? t('gen.modelHint', { step: fmt.int(info.step), ctx: fmt.int(info.context) }) : null }),
      training ? banner('info', null, t('gen.beingTrained')) : null,
      field({ label: t('gen.prompt'), control: prompt, hint: t('gen.promptHint'), optional: true }),
    );
    mount(
      paramsBox,
      h('div.plate-h', h('h2', t('gen.settings'))),
      field({ label: t('gen.length'), control: lengthInput, hint: t('gen.lengthHint') }),
      h('div.field', h('label.label', { for: 'g-temp' }, t('gen.temperature')), h('div.range-row', Object.assign(temp, { id: 'g-temp' }), tempNum), h('div.hint', t('gen.temperatureHint'))),
      h('div.grid-fields',
        field({ label: t('gen.topk'), control: topk, hint: t('gen.topkHint') }),
        field({ label: t('gen.window'), control: win, hint: t('gen.windowHint', { n: fmt.int(maxWindow) }) }),
      ),
      field({ label: t('gen.seed'), control: h('div.input-row', seed, dice), hint: t('gen.seedHint'), id: 'g-seed' }),
    );
    drawStatus();
  }

  const startBox = h('section.plate.g-start');
  const paramsBox = h('section.plate.g-params');
  const head = pageHead(t('gen.title'), t('gen.lede'));
  const result = h('section.plate.g-result', { 'aria-labelledby': 'gen-result-title' }, h('div.plate-h', h('h2', { id: 'gen-result-title' }, t('gen.result'))), actions, status, outBox);
  const emptyNote = () =>
    h('div.plate', h('div.empty', sprout(0), h('h2', t('gen.empty.title')), h('p', t('gen.empty.text')), h('a.btn.btn-primary', { href: '#/train' }, icon('train'), t('models.empty.cta'))));

  function draw() {
    mount(root, head, S.models.length || f.model ? h('div.gen-grid', startBox, result, paramsBox) : emptyNote());
    buildForm();
  }
  primaryAction = () => (gen.state === 'idle' || gen.state === 'done' || gen.state === 'stopped' || gen.state === 'failed' ? start() : null);
  refreshModels().then(() => {
    if (!f.model) {
      draw();
    }
  });
  draw();
  return {
    el: root,
    destroy() {
      unsub();
      gen.ctl?.abort();
    },
  };
}

// ---------------------------------------------------------------------------- speed

const bench = { run: null, ctl: null, listeners: new Set() };

function benchChanged() {
  bus.emit('bench');
}

function benchApply(e) {
  const r = bench.run;
  if (!r) return;
  if (e.type === 'bench_plan') r.plan = e;
  else if (e.type === 'bench_running') r.current = e.preset;
  else if (e.type === 'bench_result') {
    r.results.push(e.result);
    r.current = null;
  } else if (e.type === 'scaling_point') {
    r.points.push(e);
    r.current = null;
  } else if (e.type === 'end') {
    r.state = e.state;
    r.result = e.result;
    r.error = e.error;
    r.current = null;
    if (e.state === 'done') rememberBench(r);
  }
  benchChanged();
}

function rememberBench(r) {
  const saved = { ...(S.settings.bench || {}) };
  for (const res of r.results) {
    saved[res.preset] = { tps: res.tokens_per_sec, ms: res.ms_per_step, threads: S.system?.workers || 0, batch: res.batch, seq: res.seq, at: Math.round(Date.now() / 1000) };
  }
  for (const p of r.points) {
    const res = p.point.result;
    const best = saved[p.preset];
    if (!best || res.tokens_per_sec > best.tps) saved[p.preset] = { tps: res.tokens_per_sec, ms: res.ms_per_step, threads: p.point.threads, batch: res.batch, seq: res.seq, at: Math.round(Date.now() / 1000) };
  }
  saveSettings({ bench: saved });
}

function benchAttach(id, snap) {
  bench.ctl?.abort();
  bench.ctl = new AbortController();
  const b = snap?.bench || {};
  bench.run = {
    id,
    state: snap?.state || 'running',
    plan: b.plan?.type ? b.plan : null,
    results: (b.results || []).slice(),
    points: (b.points || []).slice(),
    current: b.current?.preset || null,
    result: snap?.result || null,
    error: snap?.error || null,
  };
  benchChanged();
  if (FINAL.has(bench.run.state)) return;
  follow(id, {
    after: snap?.seq || 0,
    signal: bench.ctl.signal,
    onEvent: benchApply,
    onSnapshot: (s2) => benchAttach(id, s2),
  }).catch(() => {});
}

async function benchStart(body) {
  const r = await api.post('/api/bench', body);
  benchAttach(r.job, null);
}

function PageSpeed() {
  const f = { preset: 'all', batch: 16, seq: 256, steps: 4, warmup: 2, scaling: false, threads: '' };
  const root = h('div.page');
  const unsub = bus.on('bench', drawResults);
  const resultsBox = h('div.stack-lg', { 'aria-live': 'off' });
  let error = null;

  function parseThreads(text) {
    const parts = text.split(/[\s,;]+/).filter(Boolean).map(Number);
    if (parts.some((n) => !Number.isInteger(n) || n < 1 || n > (S.system?.limits.max_threads || 1024))) return null;
    return parts;
  }

  async function start() {
    error = null;
    const body = { preset: f.preset, batch: f.batch, seq: f.seq, vocab: 256, steps: f.steps, warmup: f.warmup };
    if (f.scaling) {
      const list = parseThreads(f.threads);
      if (!list) {
        error = t('speed.threadsBad');
        draw();
        return;
      }
      body.scaling = list;
    }
    try {
      await benchStart(body);
    } catch (e) {
      error = explain(e);
      draw();
    }
  }

  async function stop() {
    if (!bench.run?.id) return;
    try {
      await api.post(`/api/jobs/${bench.run.id}/stop`);
    } catch (e) {
      toast(explain(e), 'bad');
    }
  }

  function running() {
    return bench.run && !FINAL.has(bench.run.state);
  }

  function drawResults() {
    const r = bench.run;
    const sections = [];
    if (r) {
      const status = running()
        ? banner('info', r.current ? t('speed.measuring', { preset: r.current }) : t('speed.preparing'), t('speed.measuringText'), [h('button.btn.btn-sm', { type: 'button', onclick: stop }, icon('stop'), t('common.stop'))])
        : r.state === 'failed'
          ? banner('bad', t('speed.failed'), r.error?.message || '')
          : r.state === 'stopped'
            ? banner('warn', t('speed.stopped'))
            : null;
      if (status) sections.push(status);
      if (r.results.length) sections.push(resultsPlate(r.results));
      if (r.points.length) sections.push(scalingPlate(r.points));
    }
    const saved = S.settings.bench || {};
    if (!r?.results.length && !r?.points.length && Object.keys(saved).length) {
      sections.push(
        h('section.plate', h('div.plate-h', h('h2', t('speed.saved'))), h('p.plate-note', t('speed.savedText')),
          tableWrap(t('speed.saved'), h('table.tbl', h('thead', h('tr', h('th', t('speed.col.size')), h('th.r', t('speed.col.tps')), h('th.r', t('speed.col.threads')), h('th', t('speed.col.when')))),
            h('tbody', Object.entries(saved).map(([name, v]) => h('tr', h('td', name), h('td.r', fmt.int(v.tps)), h('td.r', fmt.int(v.threads)), h('td.nowrap', fmt.ago(v.at, 'short')))))))),
      );
    }
    mount(resultsBox, sections);
    const startBtn = $('#speed-start');
    if (startBtn) startBtn.disabled = !!running();
  }

  function resultsPlate(results) {
    const top = Math.max(...results.map((x) => x.tokens_per_sec), 1);
    return h(
      'section.plate',
      h('div.plate-h', h('h2', t('speed.results'))),
      figure(1, t('speed.fig1'),
        h('div.bars', results.map((x) => h('div.bar-row', h('span', x.preset), h('div.bar-track', h('div.bar-fill', { style: { '--p': String(x.tokens_per_sec / top) } })), h('span.v', t('speed.tps', { n: fmt.int(x.tokens_per_sec) })))))),
      tableWrap(t('speed.results'), h('table.tbl',
        h('caption', t('speed.tableCaption', { batch: results[0].batch, seq: results[0].seq })),
        h('thead', h('tr', h('th', t('speed.col.size')), h('th.r', t('speed.col.params')), h('th.r', t('speed.col.ms')), h('th.r', t('speed.col.tps')), h('th.r', t('speed.col.gflops')), h('th.r', t('speed.col.mem')))),
        h('tbody', results.map((x) => h('tr', h('td', x.preset), h('td.r', fmt.compact(x.params)), h('td.r', fmt.num(x.ms_per_step, 1)), h('td.r', fmt.int(x.tokens_per_sec)), h('td.r', fmt.num(x.gflops, 1)), h('td.r', x.peak_rss_mib ? fmt.bytes(x.peak_rss_mib * 1048576) : '–')))))),
      h('p.hint', t('speed.estimateHint')),
    );
  }

  function scalingPlate(points) {
    const byPreset = new Map();
    for (const p of points) {
      if (!byPreset.has(p.preset)) byPreset.set(p.preset, []);
      byPreset.get(p.preset).push(p.point);
    }
    return [...byPreset.entries()].map(([name, list], idx) => {
      const x = list.map((p) => p.threads);
      const y = list.map((p) => p.result.tokens_per_sec);
      const ideal = list.map((p) => (list[0].result.tokens_per_sec / list[0].threads) * p.threads);
      const chart = lineChart({
        label: t('speed.scalingLabel', { preset: name }),
        series: [
          { key: 'ideal', name: t('speed.ideal'), color: 'var(--ink-3)', dash: true, markers: 'none' },
          { key: 'real', name: t('speed.measured'), color: 'var(--s2)', markers: 'dot', area: true },
        ],
        xTitle: t('speed.col.threads'),
        yTitle: t('speed.col.tps'),
        height: 260,
        xFormat: fmt.int,
      });
      chart.update({ real: { x, y }, ideal: { x, y: ideal } });
      return h(
        'section.plate',
        h('div.plate-h', h('h2', t('speed.scalingTitle', { preset: name }))),
        figure(2 + idx, t('speed.fig2'), chart.el),
        tableWrap(t('speed.scalingTitle', { preset: name }), h('table.tbl',
          h('thead', h('tr', h('th.r', t('speed.col.threads')), h('th.r', t('speed.col.tps')), h('th.r', t('speed.col.speedup')), h('th.r', t('speed.col.efficiency')))),
          h('tbody', list.map((p) => h('tr', h('td.r', fmt.int(p.threads)), h('td.r', fmt.int(p.result.tokens_per_sec)), h('td.r', `${fmt.num(p.speedup, 2)}×`), h('td.r', fmt.pct(p.efficiency))))))),
      );
    });
  }

  function draw() {
    const maxSeq = 16384;
    const presetSel = h('select.select', ['all', 'nano', 'micro', 'mini', 'small', 'base'].map((p) => h('option', { value: p, selected: p === f.preset }, p === 'all' ? t('speed.allSizes') : p)));
    presetSel.addEventListener('change', () => { f.preset = presetSel.value; });
    const num = (key, min, max) => numberInput({ value: f[key], min, max, onchange: (v) => { if (v !== null) f[key] = clamp(Math.round(v), min, max); } });
    const threads = h('input.input', { type: 'text', value: f.threads, placeholder: t('speed.threadsPlaceholder'), disabled: !f.scaling, inputmode: 'numeric' });
    threads.addEventListener('input', () => { f.threads = threads.value; });
    const scalingSwitch = switchBox({ title: t('speed.scaling'), hint: t('speed.scalingHint'), checked: f.scaling, onchange: (v) => { f.scaling = v; threads.disabled = !v; } });
    mount(
      root,
      pageHead(t('speed.title'), t('speed.lede')),
      h('section.plate',
        h('div.plate-h', h('h2', t('speed.setup'))),
        h('div.grid-fields',
          field({ label: t('speed.size'), control: presetSel }),
          field({ label: t('speed.batch'), control: num('batch', 1, 4096), hint: t('speed.batchHint') }),
          field({ label: t('speed.seq'), control: num('seq', 8, maxSeq) }),
          field({ label: t('speed.steps'), control: num('steps', 1, 1000), hint: t('speed.stepsHint') }),
        ),
        scalingSwitch,
        field({ label: t('speed.threads'), control: threads, hint: t('speed.threadsHint', { n: S.system?.logical_cpus || '?' }) }),
        error ? banner('bad', null, error) : null,
        S.system && !S.system.optimised ? banner('warn', t('speed.debug.title'), t('speed.debug.text')) : null,
        h('div.row', h('button.btn.btn-primary.btn-lg', { type: 'button', id: 'speed-start', onclick: start, disabled: !!running() }, icon('speed'), t('speed.start'))),
      ),
      resultsBox,
    );
    drawResults();
  }
  primaryAction = () => (running() ? null : start());
  // pick up a measurement that is still running (or finished) from before this page was shown
  if (!bench.run) {
    api.get('/api/jobs').then(async (r) => {
      const job = r.jobs.find((j) => j.kind === 'bench');
      if (job && !bench.run) benchAttach(job.id, await api.get(`/api/jobs/${job.id}`));
    }).catch(() => {});
  }
  draw();
  return {
    el: root,
    destroy: unsub,
  };
}

// ---------------------------------------------------------------------------- settings

function playChime() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const now = ctx.currentTime;
    [523.25, 783.99].forEach((hz, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = hz;
      gain.gain.setValueAtTime(0.0001, now + i * 0.18);
      gain.gain.exponentialRampToValueAtTime(0.18, now + i * 0.18 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.18 + 0.7);
      osc.connect(gain).connect(ctx.destination);
      osc.start(now + i * 0.18);
      osc.stop(now + i * 0.18 + 0.75);
    });
    setTimeout(() => ctx.close(), 1500);
  } catch {
    /* no audio: nothing to do */
  }
}

function PageSettings() {
  const root = h('div.page');
  const st = () => S.settings;
  function radio(name, label, options, key) {
    return segmented({ name, label, options, value: st()[key], onchange: (v) => saveSettings({ [key]: v }) });
  }
  function draw() {
    const langSel = h('select.select', [h('option', { value: 'auto', selected: st().lang === 'auto' }, t('settings.langAuto', { name: LANG_NAMES[pickLang('auto')] })), ...LANGS.map((l) => h('option', { value: l, selected: st().lang === l }, LANG_NAMES[l]))]);
    langSel.id = 's-lang';
    langSel.addEventListener('change', () => {
      S.refocus = '#s-lang';
      saveSettings({ lang: langSel.value });
    });
    const scale = h('input', { type: 'range', min: 0.85, max: 1.6, step: 0.05, value: st().text_scale || 1, id: 's-scale' });
    const scaleOut = h('output.num', { for: 's-scale' }, fmt.pct(st().text_scale || 1));
    scale.addEventListener('input', () => { scaleOut.textContent = fmt.pct(Number(scale.value)); });
    scale.addEventListener('change', () => saveSettings({ text_scale: Number(scale.value) }));
    const notify = switchBox({
      title: t('settings.notify'),
      hint: t('settings.notifyHint'),
      checked: st().notify,
      onchange: async (on) => {
        if (on && typeof Notification !== 'undefined' && Notification.permission !== 'granted') {
          const p = await Notification.requestPermission();
          if (p !== 'granted') {
            notify.input.checked = false;
            toast(t('settings.notifyDenied'), 'bad');
            return;
          }
        }
        if (on && typeof Notification === 'undefined') {
          notify.input.checked = false;
          toast(t('settings.notifyNone'), 'bad');
          return;
        }
        saveSettings({ notify: on });
      },
    });
    const sound = switchBox({ title: t('settings.sound'), hint: t('settings.soundHint'), checked: st().sound, onchange: (on) => saveSettings({ sound: on }) });
    const sys = S.system;
    mount(
      root,
      pageHead(t('settings.title'), t('settings.lede')),
      h('div.grid-2.top',
        h('div.stack',
        h('section.plate', h('div.plate-h', h('h2', t('settings.language'))),
          field({ label: t('settings.languageLabel'), control: langSel, hint: t('settings.languageHint'), id: 's-lang' })),
        h('section.plate', h('div.plate-h', h('h2', t('settings.notifications'))), notify, sound,
          h('div.row', h('button.btn.btn-sm', { type: 'button', onclick: playChime }, icon('play'), t('settings.soundTest')))),
        h('section.plate', h('div.plate-h', h('h2', t('settings.workspace'))), h('p.plate-note', t('settings.workspaceText')),
          h('dl.kv',
            h('dt', t('settings.ws.root')), h('dd.with-action', h('span.path', sys?.workspace || '–'), copyButton(() => sys?.workspace || '', t('common.copy'))),
            h('dt', t('settings.ws.models')), h('dd', h('span.path', sys?.models_dir || '–')),
            h('dt', t('settings.ws.data')), h('dd', h('span.path', sys?.data_dir || '–')))),
        ),
        h('section.plate', h('div.plate-h', h('h2', t('settings.appearance'))),
          h('div.field', h('span.label', t('settings.theme')), radio('s-theme', t('settings.theme'), [{ value: 'auto', label: t('settings.auto') }, { value: 'light', label: t('settings.light') }, { value: 'dark', label: t('settings.dark') }], 'theme')),
          h('div.field', h('span.label', t('settings.contrast')), radio('s-contrast', t('settings.contrast'), [{ value: 'auto', label: t('settings.auto') }, { value: 'more', label: t('settings.contrastMore') }], 'contrast')),
          h('div.field', h('span.label', t('settings.motion')), radio('s-motion', t('settings.motion'), [{ value: 'auto', label: t('settings.auto') }, { value: 'reduce', label: t('settings.motionReduce') }], 'motion')),
          h('div.field', h('span.label', t('settings.density')), radio('s-density', t('settings.density'), [{ value: 'comfortable', label: t('settings.comfortable') }, { value: 'compact', label: t('settings.compact') }], 'density')),
          h('div.field', h('span.label', t('settings.rail')), radio('s-rail', t('settings.rail'), [{ value: 'full', label: t('settings.railFull') }, { value: 'icons', label: t('settings.railIcons') }], 'rail')),
          h('div.field', h('label.label', { for: 's-scale' }, t('settings.textSize')), h('div.range-row', scale, scaleOut), h('p.hint', t('settings.textSizeHint'))),
        ),
      ),
      h('section.plate', h('div.plate-h', h('h2', t('settings.session'))), h('p.plate-note', t('settings.sessionText')),
        h('div.row', h('button.btn.btn-danger', { type: 'button', onclick: quitStudio }, icon('power'), t('nav.quit')), h('a.btn', { href: '#/about' }, icon('about'), t('nav.about')))),
    );
  }
  draw();
  return { el: root };
}

// ---------------------------------------------------------------------------- about

const SHORTCUTS = [
  ['1 – 5', 'keys.pages'],
  ['Ctrl + Enter', 'keys.primary'],
  ['?', 'keys.help'],
  ['Esc', 'keys.close'],
  ['← →', 'keys.chart'],
  ['Shift + ← →', 'keys.chartFast'],
  ['Home End', 'keys.chartEnds'],
];

function shortcutsTable() {
  return tableWrap(t('keys.title'), h('table.tbl', h('caption', t('keys.title')), h('tbody', SHORTCUTS.map(([k, label]) => h('tr', h('td', k.split(' ').map((part) => (/^[+–]$/.test(part) ? ` ${part} ` : h('kbd', part)))), h('td', t(label)))))));
}

function PageAbout() {
  const root = h('div.page');
  function draw() {
    const sys = S.system || {};
    mount(
      root,
      pageHead('GERMEN Studio', t('about.lede')),
      h('div.grid-2',
        h('section.plate', h('div.plate-h', h('h2', t('about.private'))), h('p', t('about.privateText')), h('p', t('about.windowText'))),
        h('section.plate', h('div.plate-h', h('h2', t('about.system'))),
          h('dl.kv',
            h('dt', t('about.version')), h('dd', sys.version || '–'),
            h('dt', t('about.platform')), h('dd', sys.platform || '–'),
            h('dt', t('about.cpus')), h('dd', t('about.cpusValue', { n: fmt.int(sys.logical_cpus || 0), w: fmt.int(sys.workers || 0) })),
            h('dt', t('about.simd')), h('dd', sys.simd || '–'),
            h('dt', t('about.memory')), h('dd', sys.memory?.available ? t('about.memoryValue', { free: fmt.bytes(sys.memory.available), limit: fmt.bytes(sys.memory.limit) }) : '–'),
            h('dt', t('about.build')), h('dd', sys.optimised ? t('about.buildRelease') : t('about.buildDebug')))),
      ),
      h('section.plate', shortcutsTable()),
      h('section.plate', h('div.plate-h', h('h2', t('about.credits'))),
        h('p', t('about.creditsText')),
        h('p.sans.muted', 'MIT · Alexio · ', h('a', { href: 'https://trom.tf', target: '_blank', rel: 'noopener noreferrer' }, 'trom.tf'))),
    );
  }
  draw();
  return { el: root };
}

async function showShortcuts() {
  await dialog({ title: t('keys.title'), body: shortcutsTable(), actions: [{ label: t('common.close'), value: null, kind: 'primary' }] });
}

const PAGES = {
  train: (ctx) => PageTrain(ctx),
  models: PageModels,
  generate: PageGenerate,
  speed: PageSpeed,
  settings: PageSettings,
  about: PageAbout,
};

function installKeys() {
  document.addEventListener('keydown', (e) => {
    const target = e.target;
    const typing = target instanceof HTMLElement && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && primaryAction && !document.querySelector('dialog[open]')) {
      e.preventDefault();
      primaryAction();
      return;
    }
    if (typing || e.ctrlKey || e.metaKey || e.altKey || document.querySelector('dialog[open]')) return;
    if (e.key === '?') {
      e.preventDefault();
      showShortcuts();
    } else if (/^[1-5]$/.test(e.key)) {
      location.hash = '#/' + NAV[Number(e.key) - 1][0];
    }
  });
}
