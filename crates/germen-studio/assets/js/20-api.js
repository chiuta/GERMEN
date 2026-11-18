// 20-api.js — talking to the server: one request function, one error type, the event loop of a
// job, and the upload of a text. The demo build (70-demo.js) swaps `api.impl` for a replay.

class ApiFailure extends Error {
  constructor(status, code, message, params) {
    super(message);
    this.status = status;
    this.code = code;
    this.params = params || {};
  }
}

function setOnline(online) {
  if (S.online === online) return;
  S.online = online;
  bus.emit('online', online);
}

async function realRequest(method, path, body, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      method,
      headers: {
        Authorization: 'Bearer ' + S.token,
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: options.signal,
      cache: 'no-store',
    });
  } catch (e) {
    if (e?.name === 'AbortError') throw e;
    setOnline(false);
    throw new ApiFailure(0, 'network', String(e?.message || e));
  }
  setOnline(true);
  let data = null;
  const text = await response.text();
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!response.ok) {
    const e = data?.error || {};
    if (response.status === 401) bus.emit('unauthorized');
    throw new ApiFailure(response.status, e.code || 'http_' + response.status, e.message || response.statusText, e.params);
  }
  return data;
}

const api = {
  impl: realRequest,
  request(method, path, body, options) {
    return api.impl(method, path, body, options);
  },
  get: (path, options) => api.request('GET', path, undefined, options),
  post: (path, body, options) => api.request('POST', path, body ?? {}, options),
  put: (path, body, options) => api.request('PUT', path, body ?? {}, options),

  /** Sends a file as the body of a request, reporting how far it got (0 to 1). */
  upload(file, onProgress, signal) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/upload?name=' + encodeURIComponent(file.name));
      xhr.setRequestHeader('Authorization', 'Bearer ' + S.token);
      xhr.responseType = 'json';
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress?.(e.loaded / e.total);
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) return resolve(xhr.response);
        const e = xhr.response?.error || {};
        reject(new ApiFailure(xhr.status, e.code || 'http_' + xhr.status, e.message || xhr.statusText, e.params));
      };
      xhr.onerror = () => reject(new ApiFailure(0, 'network', 'upload failed'));
      xhr.onabort = () => reject(new DOMException('aborted', 'AbortError'));
      signal?.addEventListener('abort', () => xhr.abort());
      xhr.send(file);
    });
  },
};

/** A sentence for the person, in their language, for whatever went wrong. */
function explain(err) {
  if (!(err instanceof ApiFailure)) return String(err?.message || err);
  const key = 'err.' + err.code;
  if (I18N.en[key] === undefined) return err.message || t('err.unknown');
  const params = { ...err.params, message: err.message };
  // The kind of a job ("train", "bench"…) is a word of the server's; say it in the person's language.
  if (typeof params.kind === 'string' && I18N.en['kind.' + params.kind] !== undefined) params.kind = t('kind.' + params.kind);
  return t(key, params);
}

const FINAL = new Set(['done', 'stopped', 'failed']);

/**
 * Follows a job: asks for the events after the last one seen and waits for new ones (a long poll),
 * until the job has ended or `signal` is aborted. If the page fell too far behind, the snapshot is
 * fetched again (`onSnapshot`) and the events go on from the number it names.
 */
async function follow(jobId, { after = 0, onEvent, onSnapshot, signal }) {
  let wait = 400;
  while (!signal?.aborted) {
    try {
      const r = await api.get(`/api/jobs/${jobId}/events?after=${after}&wait=4000`, { signal });
      wait = 400;
      if (r.gap) {
        const snap = await api.get(`/api/jobs/${jobId}`, { signal });
        onSnapshot(snap);
        after = snap.seq;
        continue;
      }
      for (const item of r.events) onEvent(item.e, item.seq);
      after = r.next;
      if (FINAL.has(r.state)) return;
    } catch (e) {
      if (e?.name === 'AbortError' || signal?.aborted) return;
      if (e instanceof ApiFailure && (e.status === 401 || e.status === 404)) throw e;
      await sleep(wait);
      wait = Math.min(wait * 1.7, 4000);
    }
  }
}

/** The token: from the address (after `#k=`), then kept for this tab only so that a reload works. */
function readToken() {
  let token = '';
  const found = /[#&]k=([0-9a-f]{64})/.exec(location.hash);
  try {
    if (found) {
      token = found[1];
      sessionStorage.setItem('germen.token', token);
    } else {
      token = sessionStorage.getItem('germen.token') || '';
    }
  } catch {
    token = found ? found[1] : '';
  }
  if (found) history.replaceState(null, '', location.pathname + location.search + '#/');
  return token;
}
