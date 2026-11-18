// 99-boot.js — the start: the token, what the server knows, the first page.

function renderCrash(error) {
  mount(
    $('#app'),
    h(
      'main.page',
      { style: { maxWidth: '40rem', marginTop: '10vh' } },
      h('div.plate', h('h1', 'GERMEN Studio'), h('p', String(error?.message || error)), h('button.btn', { type: 'button', onclick: () => location.reload() }, 'Reload')),
    ),
  );
}

function wire() {
  window.addEventListener('hashchange', () => {
    if (!S.quitting) route();
  });
  bus.on('online', drawOnlineBanner);
  bus.on('run', () => {
    updateRailJob();
    updateTitle();
  });
  // a new language rebuilds the frame and the page (the texts are built, not translated in place)
  bus.on('lang', () => {
    if (S.quitting || !$('.shell')) return;
    renderShell();
    route();
    updateRailJob();
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (S.settings.theme !== 'light' && S.settings.theme !== 'dark') applyPrefs();
  });
  installKeys();
  if (MODE !== 'demo') {
    const beat = () => {
      if (!S.quitting && !document.hidden) api.get('/api/ping').catch(() => {});
    };
    setInterval(beat, 8000);
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && !S.quitting) {
        beat();
        refreshModels();
      }
    });
  }
}

async function boot() {
  setLang('auto');
  document.documentElement.dataset.theme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  if (MODE === 'demo' && typeof installDemo === 'function') installDemo();
  S.token = MODE === 'demo' ? 'demo' : readToken();
  if (!S.token) return renderLocked('token');
  bus.on('unauthorized', () => {
    if (!S.quitting) renderLocked('token');
  });
  let first;
  try {
    first = await Promise.all([api.get('/api/system'), api.get('/api/settings'), api.get('/api/models'), api.get('/api/jobs')]);
  } catch (e) {
    return renderLocked(e instanceof ApiFailure && e.status === 401 ? 'token' : 'offline');
  }
  S.system = first[0];
  S.settings = first[1];
  S.models = first[2].models;
  S.jobs = first[3].jobs;
  applyPrefs();
  renderShell();
  wire();
  // a training that was going on before this page was opened (or reloaded) is picked up again
  const train = S.jobs.filter((j) => j.kind === 'train').sort((a, b) => b.id - a.id)[0];
  if (train) {
    try {
      await runAttach(train.id);
    } catch {
      /* the job is gone: the form is shown */
    }
  }
  route();
  updateRailJob();
}

boot().catch(renderCrash);
