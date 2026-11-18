# GERMEN Studio

`germen ui` — the graphical interface of GERMEN (version 0.5). A window in which you choose a text,
press **Start** and watch a model learn: the loss curve, the learning rate, the speed, and samples
of what the network writes as it improves. Everything the command line can do is there, with the
same code underneath: the Studio calls the library, it does not run `germen` as a subprocess.

It is a small web server that listens on `127.0.0.1` of your own computer and one page that it
serves. Nothing is downloaded, nothing leaves the machine, and there is nothing to install: the
page is a single HTML document with its style, its script, its icons and its eight languages inside.

1. [Starting it](#1-starting-it)
2. [What is in it](#2-what-is-in-it)
3. [How it is built](#3-how-it-is-built)
4. [Security](#4-security)
5. [The interface of the server (JSON)](#5-the-interface-of-the-server-json)
6. [Accessibility and languages](#6-accessibility-and-languages)
7. [The demonstration page](#7-the-demonstration-page)
8. [Tests, and what was checked by hand](#8-tests-and-what-was-checked-by-hand)
9. [What has not been verified](#9-what-has-not-been-verified)

## 1. Starting it

```bash
germen ui                          # opens the Studio in a window of its own
germen ui --no-open                # only prints the address; open it yourself
germen ui --workspace D:\GERMEN    # another folder for the models, the texts and the settings
```

| option | meaning |
|---|---|
| `--workspace DIR` | the folder with the models, the texts and the settings (default `~/GERMEN`, in the profile folder on Windows; created if missing) |
| `--port PORT` | the port on `127.0.0.1` (default 7870, or the next free one up to 7889, or any free one) |
| `--no-open` | do not open a browser; print the address |
| `--max-upload-mb MIB` | the largest text the page may send (1 … 1,000,000; default 2048) |
| `--threads N` | worker threads for training and writing (a global option, as for every command; default: every logical processor) |

The address that is printed looks like `http://127.0.0.1:7870/#k=…`: the part after `#` is a secret
that exists for this run only (§4). The window opens with it; if you copy the address to another
browser, copy all of it.

Where a Chrome, Edge, Chromium or Brave is installed the Studio opens as an *application window*
(`--app=…`: no address bar, no tabs, it looks like a program); otherwise in the default browser.

**Closing.** The window's **Quit** button, or Ctrl-C in the terminal. A training that is running is
asked to stop and **saves first** (a second Ctrl-C leaves at once). A model that was being trained
is complete on disk either way: the Studio writes checkpoints the way `germen train` does.

## 2. What is in it

| page | what it does | the command-line equivalent |
|---|---|---|
| **Train** | choose a text (a file on disk, or one that you drop on the page: it is copied into the workspace), a model size, the tokenizer and the settings, with a live estimate of the memory and of the time; then watch the loss (training and checking text, linear or logarithmic), the learning rate, tokens per second, the gradient norm and the samples as they are written. A reload or a second window in the middle of a run picks it up again | `germen train` (and `--resume`) |
| **Models** | the models in the workspace and any folder you point to; what each one is (size, step, vocabulary, when it was made); **Check files** and **Repair** | `germen verify`, `verify --recover` |
| **Write** | text from a model, written as it is produced; prompt, length, temperature, top-k, seed; copy; continue from what was written. A model that is being trained can be used while it trains | `germen generate` |
| **Speed** | times real training steps for every model size, or for one, and optionally how the speed grows with the number of threads (a chart of the curve, the ideal line, speed-up and efficiency) | `germen bench`, `bench --scaling` |
| **Settings** | language, theme (light, dark, as the system), more contrast, less motion, density, text size, icons-only side bar, a notification and a sound when a training ends | — |
| **About** | what the Studio is and how it keeps you private; the version and the computer (processors, vector instructions, memory); credits | `germen info` |

Training and measuring speed use every core, so only **one of them runs at a time**; writing is light
and may run beside them. A model that is being trained cannot be checked, repaired or trained again
from another place until it is saved: the page says so and the server refuses (409).

The look is a *cyanotype plate* from a botanist's notebook: Prussian blue and paper, a sun-yellow
for the one thing to press, serif for words and a plain face for numbers, figures numbered like
plates ("Fig. 2"). The two data colours (gold and blue) are told apart by shape and by label too, so
they never carry meaning alone.

## 3. How it is built

```
crates/germen-studio/
  src/lib.rs        Studio::start / serve / shutdown handle; binds 127.0.0.1; connection limits
  src/http.rs       the HTTP/1.1 subset that is spoken (std only), with every limit
  src/auth.rs       the token, the Host / Origin / Sec-Fetch-Site checks
  src/api.rs        the gates in order, then the routes
  src/jobs.rs       jobs, their events (a ring), their snapshots; the heavy slot and the light one
  src/work.rs       what a job does: the same steps as train, generate, bench and verify
  src/workspace.rs  the folder: names become paths here, nowhere else; settings; the file picker
  src/page.rs       builds the one page and its Content-Security-Policy
  src/browser.rs    opens an application window or the default browser
  assets/           index.html, style.css, js/ (nine parts), i18n/ (eight JSON files)
```

* **No framework.** The server is written on the standard library alone: `TcpListener`, one thread
  per connection, one request per connection (`Connection: close`). No runtime, no TLS (it is
  loopback), nothing to download at build time beyond what GERMEN already uses.
* **Jobs.** A job (train, generate, bench) writes numbered **events**. The page asks for the events
  after the last number it has seen and waits (a long poll). Every job also keeps a **snapshot**,
  the sum of its events, taken under one lock together with the number of the next event: a page
  that arrives late, or is reloaded, draws the snapshot and asks for the events from that number
  on, and loses none and counts none twice. The long curves are thinned out as they grow, so a run
  of a million steps takes no more memory than one of a thousand; the last twelve jobs are kept.
* **Training uses the library.** The progress of `germen train` already came through `Event`
  callbacks; the Studio is their other listener. A stop request is checked between steps, and the
  run saves before it ends, as Ctrl-C does on the command line.
* **The page** is assembled when the server starts: `index.html` with the style and the eight
  language files and the nine script parts put inside (`src/page.rs`). The script parts are joined
  into one function, so a name may not be used twice. All text on the page is written with
  `textContent`; there is no `innerHTML` anywhere (a test looks for every way of making markup
  from a string). Texts are in flat dotted keys with `{name}` placeholders; English is the fallback.
* **Charts** are drawn by hand in SVG (no library): linear and logarithmic axes with labels that
  stay distinct, a crosshair and a tooltip, the keyboard (arrows, Home, End, Escape), a table view
  of the same numbers, smoothing, and a text that a screen reader says for the point under the cursor.

## 4. Security

The Studio lets a web page start work on your computer (read a file you name, write a model into a
folder, use every core for hours). That is what it is for, and it is exactly what a hostile page,
another user of the computer, or a stray program must not be able to do. This section is the model
and the evidence. The code that implements it is short on purpose: `auth.rs` (the three checks),
`http.rs` (what is accepted), `workspace.rs` (names to paths), `page.rs` (the policy).

### 4.1 Who is kept out

| who | how they would try | what stops them |
|---|---|---|
| a **web page** open in your browser | send requests to `http://127.0.0.1:7870/api/…` (a browser lets any page do that, even if it cannot read the answer) | the **token** (it is only in the address that you opened, after the `#`, which a browser never sends to any server and a page of another origin cannot read); the **`Origin`** and **`Sec-Fetch-Site`** headers, which a browser adds by itself and a page cannot change; no CORS header is ever sent, so another origin cannot read an answer anyway |
| a page that makes its own **name point at 127.0.0.1** (DNS rebinding) | the browser then thinks the Studio is "same origin" with that name | the **`Host`** header must name this very server (`127.0.0.1:<port>` or `localhost:<port>`): a rebound name sends its own, and gets 421 even with the token |
| **another computer** | connect to the port | the socket is bound to `127.0.0.1` only; and a connection whose peer is not a loopback address is closed unanswered, so a listener that someone changed later would still not serve strangers |
| a **script injected** into the page (through a model's text, a file name, a field) | run in the page and use the token | the page's **Content-Security-Policy** lets exactly one inline script and one inline style run, named by their SHA-256, and nothing else to run or load; **Trusted Types** are required, which refuses the sinks that make markup from strings; and the page never makes markup from text in the first place |
| a **name from the network** | `../../x`, an absolute path, a device name, a very long name | names become paths in one module: a model is 1–64 letters, digits, `-`, `_`, `.`, starting with a letter or digit, not ending in a dot and not a Windows device name; an uploaded text gets a sanitised name and is created with *create new*, so it never replaces a file |
| a **malformed request** | a second `Host`, a body that lies about its length, a head of a gigabyte, a slow client | one request per connection; every limit (§4.3) is checked before the data it guards is read; the headers that decide something may appear only once, so two parties along the way cannot disagree about which of two copies counts; chunked bodies, `Upgrade` and pipelining are not spoken at all; time limits on the head and on the body |

The gates are applied in the same order to every request: **Host → (on `/api`) Origin and
Sec-Fetch-Site → token → route**. The page itself (`GET /`) needs the right `Host` only; it holds no
secret (the token is never written into it; the page reads it from the address).

Every answer carries `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`,
`Referrer-Policy: no-referrer`, `Cross-Origin-Resource-Policy: same-origin`,
`Cross-Origin-Opener-Policy: same-origin`, `X-Frame-Options: DENY` and a `Permissions-Policy` that
turns off the camera, the microphone, the location and payment. The policy of the page is

```
default-src 'none'; script-src 'sha256-…'; style-src 'sha256-…'; img-src 'self' data:;
font-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none';
frame-ancestors 'none'; object-src 'none'; require-trusted-types-for 'script'
```

The token is 64 hexadecimal digits (256 bits), made for every start by mixing, through SHA-256, the
random keys of the standard library's hash maps (drawn from the operating system), `/dev/urandom`
where there is one, the clock, the process id and an address; it is compared in constant time.

### 4.2 Things the page may name on disk

A text to train on is any file that you can read (the file picker lists folders and files; hidden
ones only when asked); a model can be opened from any folder that holds a checkpoint. That is a
feature, and the token is what protects it: whoever has the address can read what GERMEN can read.
Everything that is *written* goes to the workspace (models, uploaded texts, `settings.json`) or, for
a training, to the model folder under it. A path that comes from the page is never used to write
anywhere else.

### 4.3 Limits

| what | limit |
|---|---|
| the head of a request (line and headers) | 16 KiB, at most 64 header lines, a target of at most 4096 bytes, at most 32 query parameters |
| time to send a head / silence inside a body | 10 s / 30 s |
| a JSON body | 1 MiB |
| an uploaded text | 2 GiB by default (`--max-upload-mb`), checked from `Content-Length` before the first byte is stored, and counted as it arrives |
| a prompt | 64 KiB |
| connections served at the same moment | 64 (more get `503` and the page tries again) |
| entries in a folder listing | 3000 |
| `settings.json` | 8 KiB; the keys and the values are checked, an unknown key is refused |
| events kept per job / jobs kept | 4096 / 12; the curves are thinned to 2400 points |
| text of a sample kept / samples kept | 6000 characters / 16 |

### 4.4 What it does not defend against — said plainly

* **Anyone who can run programs as you** can read the address from the process list, the terminal
  or the browser, and can do anything you can do anyway. The address is on the command line of the
  browser process when the Studio starts it, and a multi-user Linux computer usually lets every
  user see every command line. On a shared computer start `germen ui --no-open` and paste the
  address into your browser yourself.
* **Someone who looks over your shoulder** at the terminal sees the secret. The address is not
  written to any file by GERMEN.
* **The token is a bearer secret over plain HTTP on loopback.** Nothing on the network sees it, but
  a program on the same computer that can sniff loopback traffic (it needs administrator rights)
  can. This is the usual position of local development servers.
* **A model file is not trusted less than before.** The Studio opens checkpoints with the same
  checks as `germen generate` ([AUDIT.md](AUDIT.md), §4 and §10); a hostile but
  self-consistent checkpoint meets the reader of Burn, as it does on the command line.
* **No one but its authors has reviewed it** (the author and an AI assistant), as with the rest of
  GERMEN; see [AUDIT.md](AUDIT.md), Part III.

### 4.5 Mutation proofs

For each gate one defect is put back, the tests meant to guard it are run, and at least one must
fail: `python3 scripts/audit_mutations.py T-` (about three minutes; it edits the sources in place
and restores them). The table is in [AUDIT.md](AUDIT.md), Part III.

## 5. The interface of the server (JSON)

Every request to `/api/…` carries `Authorization: Bearer <token>`. Answers are JSON; an error is
`{"error": {"code": "…", "message": "…", "params": {…}}}` with an HTTP status; the page translates
the **code** (every code the server can send has a text in every language, a test checks it) and
keeps the message for the log.

| method and path | what it does |
|---|---|
| `GET /api/ping` | `{ "ok": true }`: the page asks it first, to tell a wrong token from a stopped server |
| `GET /api/system` | version, platform, processors, vector instructions, memory, the workspace paths, the model sizes, the limits |
| `GET /api/settings`, `PUT /api/settings` | the choices made in the interface; `PUT` takes a partial object and answers with the whole |
| `GET /api/models` | the models in the workspace, newest first, with what `meta.json` and `config.json` say |
| `POST /api/verify`, `POST /api/recover` | `{ "model": "<name or absolute path>" }`: `germen verify`, and `verify --recover` |
| `GET /api/estimate` | memory and time for a size, a batch, a context length and a vocabulary (and a text size) before anything starts |
| `GET /api/fs?path=…&hidden=0\|1` | the folders and files under an absolute path, for the picker (an empty path lists `data/` of the workspace); the answer also names the roots — the workspace, the home folder, the drives — for the picker's shortcuts |
| `POST /api/upload?name=…` | the body is a text; it is stored in `data/` under a sanitised, unused name |
| `POST /api/train`, `/api/generate`, `/api/bench` | start a job; `202 { "job": id }`; `409 busy` if the slot is taken |
| `GET /api/jobs` | the jobs that are kept, newest first |
| `GET /api/jobs/:id` | the **snapshot** of a job: request, state, the curves so far, the samples, the result |
| `GET /api/jobs/:id/events?after=N&wait=MS` | the events after number `N`, waiting up to `MS` ms for one; `{ events, next, state, gap }` (`gap` says that `N` fell out of the ring: take a new snapshot) |
| `POST /api/jobs/:id/stop` | asks a job to stop; a training saves first |
| `POST /api/quit` | closes the server (a training in progress saves first) |

States of a job: `starting`, `running`, `stopping`, then one of `done`, `stopped`, `failed`.
Training and measuring speed share one slot; writing has its own.

## 6. Accessibility and languages

* Eight languages — English, Romanian, French, Italian, Spanish, Portuguese, German, Hungarian —
  chosen from the system's or in Settings; about 500 texts in each, written in the informal
  address where the language has one. Numbers, dates and units follow the language.
* Everything can be done from the keyboard; a **Skip to content** link, a visible focus ring, and
  charts that can be read by arrows and as a table.
* The page was checked with **axe-core** 4.14 in Chromium 153 (the WCAG 2.0, 2.1 and 2.2 rules at
  levels A and AA, and axe's best practices): no violation on the seven views — a run, the form,
  Models (with a model's details open), Write, Speed, Settings and About — in light and in dark, at
  1280 × 800 in English and at 390 × 844 in Romanian and German, and with **More contrast** on. For
  the text inside the SVG charts and the key glyphs in `<kbd>` axe answers "cannot tell" (it finds no
  background behind SVG text, and a glyph is not text), so those colours were **computed** from the
  palette: chart text at least 4.77:1 on every surface in light and 5.93:1 in dark; the two series
  colours against the plate 3.99:1 and 5.55:1 in light, 4.77:1 and 4.03:1 in dark (3:1 is asked of
  graphics); the border of a field (`--control`, a colour of its own) 3.84:1 and 3.78:1. **More
  contrast** and the system's `prefers-contrast` raise them.
* **Forced colours** (Windows High Contrast) are honoured: system colours for text, borders and
  selection, the series colours kept in the charts and their legends, an edge on every progress bar,
  a sprout drawn in the system's text colour, a border on the tooltips of the icons-only side bar.
  They were looked at in Chromium's emulation, in a light and a dark palette, on the run, Write and
  Settings pages (§8). axe's contrast rule cannot be used in that mode: it reports the side bar (9
  elements, 1.12:1) from the page's own text colour, while the browser's computed colour for them
  is the system's (dark blue on white, seen in the screenshot). That report is a false positive and
  is not counted; the screenshots are.
* **Less motion** (and the system's setting) stops every animation but the one that shows a state
  change.
* The accessible name of a control contains its visible text (so that voice control can say it).
  Live regions announce the end of a training, the result of a check and the point under a chart's
  cursor.
* It works from 320 px wide (a phone) to a wide screen; the side bar becomes a row of tabs, and the
  page keeps clear of a phone's notch and system bars (`env(safe-area-inset-*)`; zero on a computer).
  Text size from 85 % to 160 % in Settings, or with the browser's zoom.

## 7. The demonstration page

```bash
germen ui --write-demo studio-demo.html     # a hidden option
```

writes **one HTML file**, about 440 KiB, that shows the Studio *without* the program: a real training of
a nano model (240 steps, recorded once), a text it wrote, a speed measurement and a thread-scaling
curve are played back through the same page, with the same charts and the same flows (start, stop,
resume, write, check files, repair). A banner says so, there is no **Quit** button, nothing is
computed and nothing is sent anywhere. It is for looking at the Studio, taking its pictures and
showing it to someone who has not installed anything. (Its script differs from the program's by one
part, `70-demo.js`; a test checks that the program's own page does not contain it.)

The same file is published as `demo.html` at the root of the repository (GitHub Pages serves it at
`germen.chiuta.ro/demo.html`, next to the presentation page `index.html`); regenerate it with the command
above and commit it after any change to the Studio's page.

To record it again — after a change to what training reports, say — start a Studio on an empty
workspace and run the recorder, which drives the real server over its own interface and rewrites
the block between `RECORDING:BEGIN` and `RECORDING:END` in `assets/js/70-demo.js`:

```bash
germen ui --no-open --workspace /tmp/demo-ws --port 7872       # prints the address with the key
python3 scripts/record_demo.py 'http://127.0.0.1:7872/#k=<the key>'
```

## 8. Tests, and what was checked by hand

* **60 unit tests** in the crate: the token, the three checks, the parsing of heads and the limits,
  the names, the settings, the file picker, the jobs and their slots, the estimates, and the page
  (the policy, no markup from text, every language has the texts of English and only those, every
  text the script asks for exists, every error the server can send has a text).
* **21 integration tests** (`crates/germen-studio/tests/http.rs`) that start a real server and
  attack it over a socket: no token, a wrong token, another host, another origin, malformed heads,
  a client that leaves half-way, names that try to leave the workspace, oversized bodies and
  uploads, too many connections, a stuck job; and then the ordinary use, all over HTTP: train from
  zero, check, write, train further, stop, a speed measurement and a thread sweep, quit while
  training, two Studios that share nothing.
* **2 tests of the command** (`crates/germen-cli/tests/cli.rs`): `germen ui` is started as a user
  would, prints its address, serves the page, answers 401, 421 and 200 as it must, and ends with
  exit code 0 when the page says Quit; and the refusals that come before anything starts. They run
  on Ubuntu, Windows and macOS in CI.
* **18 mutation proofs** (§4.5).
* **By hand, in a real browser** (Chromium, driven by Playwright; the scripts are not part of the
  repository): every page and every flow of §2 in light and dark, in all eight languages and at
  1280, 1024, 390, 360 and 320 px (a sweep that looks for horizontal overflow, clipped text and
  texts that are still a key); the charts with the mouse and the keyboard; the lock screen for a
  missing or wrong token; Quit; the demonstration page end to end; **axe-core** on every page; the
  run, Write and Settings pages in the emulation of **forced colours**, light and dark. This is how
  the defects of the next paragraph were found.

Defects found that way and fixed before the release, because they show what the tests above cannot:
a copy button that threw after its `await` (it read `event.currentTarget` too late); y-axis labels
that came out identical ("6E-4" four times, "10.9K" three times) on a narrow range; a logarithmic
axis with a single tick; a tooltip that did not say at which step a curve without a point under the
cursor was read; long labels of buttons that widened the whole page on a phone, and tabs whose text
was clipped in German and Portuguese; an icons-only side bar with no names; a closed-Studio screen
that looked like a crash. In forced colours: progress bars that lost their track (a background is
dropped there), a sprout drawn in the side bar's pale ink that vanished on a white canvas, and the
dot of a chart legend that vanished for the same reason as the track.

## 9. What has not been verified

* **Forced colours on a real Windows theme.** Only Chromium's emulation (its default light and
  dark palettes) was looked at; the other themes (Aquatic, Desert, Dusk, Night sky) were not tried,
  and **More contrast** was checked with axe, not by eye.
* **Windows and macOS beyond CI.** The tests run there, but nobody has used the Studio on them with
  a real browser: how the application window opens, the file picker's drives, how the notification
  is asked for.
* **Browsers other than Chromium** (Firefox, Safari): the page uses nothing that they lack and
  nothing was found in reading, but only Chromium was run. Trusted Types are enforced by Chromium
  only; in the others the policy's other directives and the absence of any `innerHTML` still hold.
* **Screen readers.** The structure passes axe and was read in the order a reader would, but no
  person using NVDA, JAWS, VoiceOver or Orca has used it.
* **Many cores and long runs.** The page was run against trainings of a few minutes on two cores. A
  training of days, and a page left open for days (the thinning of the curves is tested, the browser
  tab's memory over days is not).
* **An independent review** of `http.rs` and `auth.rs`: they are small enough to be read in an
  afternoon, and that is the review most worth asking for.
