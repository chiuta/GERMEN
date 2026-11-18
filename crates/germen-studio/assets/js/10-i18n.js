// 10-i18n.js — the eight languages (I18N is filled in by the server from assets/i18n/*.json) and
// the number, size, time and date formats of the chosen one.

const LANGS = ['en', 'ro', 'fr', 'it', 'es', 'pt', 'de', 'hu'];
const LANG_NAMES = { en: 'English', ro: 'Română', fr: 'Français', it: 'Italiano', es: 'Español', pt: 'Português', de: 'Deutsch', hu: 'Magyar' };
let lang = 'en';
const fmtCache = new Map();

function pickLang(preference) {
  if (LANGS.includes(preference)) return preference;
  for (const tag of navigator.languages?.length ? navigator.languages : [navigator.language]) {
    const base = String(tag || '').toLowerCase().split('-')[0];
    if (LANGS.includes(base)) return base;
  }
  return 'en';
}

/** Chooses the language; tells the pages only if it is not the one already shown. */
function setLang(preference) {
  const next = pickLang(preference);
  if (next === lang && document.documentElement.lang === next) return;
  lang = next;
  document.documentElement.lang = lang;
  fmtCache.clear();
  bus.emit('lang', lang);
}

/** The text for `key` in the chosen language (English if it has none), with {name} filled in. */
function t(key, params) {
  let text = I18N[lang]?.[key] ?? I18N.en[key];
  if (text === undefined) {
    console.warn('no text for', key);
    return key;
  }
  if (params) text = text.replace(/\{(\w+)\}/g, (whole, name) => (name in params ? String(params[name]) : whole));
  return text;
}

function nf(options) {
  const key = 'n' + JSON.stringify(options);
  if (!fmtCache.has(key)) fmtCache.set(key, new Intl.NumberFormat(lang, options));
  return fmtCache.get(key);
}

const fmt = {
  int: (n) => (isNum(n) ? nf({ maximumFractionDigits: 0 }).format(n) : '–'),
  num: (n, digits = 2) => (isNum(n) ? nf({ minimumFractionDigits: digits, maximumFractionDigits: digits }).format(n) : '–'),
  sig: (n, digits = 3) => (isNum(n) ? nf({ maximumSignificantDigits: digits }).format(n) : '–'),
  compact: (n, extra = 0) => (isNum(n) ? nf({ notation: 'compact', maximumFractionDigits: 1 + extra }).format(n) : '–'),
  pct: (x, digits = 0) => (isNum(x) ? nf({ style: 'percent', minimumFractionDigits: digits, maximumFractionDigits: digits }).format(x) : '–'),
  loss: (x) => (isNum(x) ? nf({ minimumFractionDigits: 3, maximumFractionDigits: 3 }).format(x) : '–'),
  lr: (x, extra = 0) => (isNum(x) ? nf({ notation: 'scientific', maximumFractionDigits: 1 + extra }).format(x) : '–'),
  bytes(n) {
    if (!isNum(n)) return '–';
    const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
    let v = n;
    let i = 0;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i += 1;
    }
    const digits = i === 0 ? 0 : v < 10 ? 2 : v < 100 ? 1 : 0;
    return `${nf({ minimumFractionDigits: digits, maximumFractionDigits: digits }).format(v)} ${units[i]}`;
  },
  dur(seconds) {
    if (!isNum(seconds) || seconds < 0) return '–';
    const total = Math.round(seconds);
    if (total < 60) return `${fmt.int(total)} s`;
    const m = Math.floor(total / 60) % 60;
    const hours = Math.floor(total / 3600);
    if (hours === 0) return `${fmt.int(m)} min ${fmt.int(total % 60)} s`;
    if (hours < 48) return `${fmt.int(hours)} h ${fmt.int(m)} min`;
    return `${fmt.int(Math.floor(hours / 24))} d ${fmt.int(hours % 24)} h`;
  },
  ago(unixSeconds, style = 'long') {
    if (!isNum(unixSeconds) || unixSeconds <= 0) return '–';
    const diff = unixSeconds - Date.now() / 1000;
    const rtf = new Intl.RelativeTimeFormat(lang, { numeric: 'auto', style });
    const a = Math.abs(diff);
    if (a < 60) return rtf.format(Math.round(diff), 'second');
    if (a < 3600) return rtf.format(Math.round(diff / 60), 'minute');
    if (a < 86400) return rtf.format(Math.round(diff / 3600), 'hour');
    if (a < 86400 * 45) return rtf.format(Math.round(diff / 86400), 'day');
    return new Intl.DateTimeFormat(lang, { dateStyle: 'medium' }).format(new Date(unixSeconds * 1000));
  },
  date: (unixSeconds) => (isNum(unixSeconds) && unixSeconds > 0 ? new Intl.DateTimeFormat(lang, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(unixSeconds * 1000)) : '–'),
  clock: (seconds) => {
    const total = Math.max(0, Math.round(seconds));
    const m = Math.floor(total / 60);
    const sec = total % 60;
    return `${m}:${String(sec).padStart(2, '0')}`;
  },
};
