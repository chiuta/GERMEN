# GERMEN

**Addestra da zero un piccolo modello linguistico, in puro Rust, su una normale CPU.**

Niente Python, niente CUDA, niente toolchain C o C++, niente account, niente accesso alla rete,
niente telemetria. Un solo eseguibile, un file di testo, un comando. Costruito su
[Burn](https://burn.dev) 0.22 con il suo backend per CPU `flex`, scritto interamente in Rust.

> Lingue: [English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · **Italiano** · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

🌐 **Sito:** <https://germen.chiuta.ro> · [demo di Studio online](https://germen.chiuta.ro/demo.html)

## Che cos'è — e che cosa non è

GERMEN addestra una rete di tipo GPT (RMSNorm, posizioni rotatorie, strato feed-forward con ReLU al
quadrato, senza bias) su **token**. Per impostazione predefinita un token è un **byte**: il
vocabolario sono i 256 valori di un byte, quindi qualsiasi testo — diacritici rumeni inclusi —
funziona senza alcuna preparazione. Con `--tokenizer bpe` GERMEN impara prima, dal vostro testo,
un vocabolario BPE a livello di byte (fino a 65.536 token); la stessa finestra di contesto contiene
allora diverse volte più testo (tipicamente 3–4 volte per prosa comune). Il tokenizer viene salvato
insieme al modello.

È un **laboratorio per imparare e sperimentare**, e una base per la ricerca: in pochi minuti si
vede una rete passare dal rumore casuale all'ortografia, alla punteggiatura e allo stile del
vostro testo. **Non** è una fabbrica di chatbot. Un modello da 1–25 milioni di parametri,
addestrato su alcune centinaia di megabyte, impara la *forma* di un testo (parole, frammenti di
grammatica, impaginazione), non conoscenze sul mondo né la conversazione. Assistenti utili
richiedono ordini di grandezza di calcolo in più di quanto offra una CPU da scrivania.

## Avvio rapido

Serve Rust 1.95 o successivo ([rustup.rs](https://rustup.rs); su Windows l'installazione richiede
anche i Visual Studio Build Tools). Poi:

```bash
cargo build --release
./target/release/germen bench                      # quanto è veloce QUESTO computer?
./target/release/germen bench --scaling            # ...e come cresce con il numero di thread?
./target/release/germen train --demo --preset micro --steps 300
./target/release/germen generate                   # testo dal modello appena addestrato
```

Su Windows usate `target\release\germen.exe`. Gli eseguibili precompilati sono allegati alle
versioni (releases).

Addestrate sul vostro testo — qualsiasi file, in qualsiasi lingua:

```bash
germen train --data il-mio-corpus.txt --preset mini --tokenizer bpe --vocab 4096 --steps 5000 --out il-mio-modello
germen generate --model il-mio-modello --prompt "Era odată " --max-new 400
```

Premete **Ctrl-C** durante l'addestramento: termina il passo in corso, salva ed esce. Riprendete più
tardi con lo stesso comando più `--resume`: pesi, contatore dei passi e stato di AdamW vengono
ripristinati, quindi l'addestramento prosegue senza il rialzo della perdita di un riavvio a freddo
(le finestre di testo vengono estratte di nuovo a caso).

Lo stesso comando `train` rilanciato *senza* `--resume` viene rifiutato, perché non sostituisca per
errore un modello addestrato; `--overwrite` lo consente quando si vuole davvero ricominciare.

## La finestra: GERMEN Studio

Preferite una finestra a una riga di comando?

```bash
germen ui
```

apre **GERMEN Studio**: scegliete un testo, premete **Avvia l’addestramento** e guardate un modello
imparare — la curva della perdita, la velocità e campioni di ciò che la rete scrive man mano che
migliora. La stessa finestra genera testo dai vostri modelli, ne controlla e ne ripara i file e
misura la velocità del vostro computer, con lo stesso codice dei comandi qui sopra. Parla otto
lingue, ha il tema chiaro e quello scuro, sta in una finestra grande quanto un telefono come su uno
schermo largo ed è pensata per essere usata da tastiera e con uno screen reader.

È un piccolo server che ascolta solo su `127.0.0.1`, più una pagina che serve: niente da installare,
niente da scaricare, niente inviato da nessuna parte. L'indirizzo che stampa contiene un segreto
valido solo per questa esecuzione, così solo la finestra aperta con esso (o voi) può dargli ordini.
Modelli e testi stanno in `~/GERMEN` (`--workspace` lo cambia); `--no-open` stampa soltanto
l'indirizzo. Come è protetto e il resto: [docs/STUDIO.md](docs/STUDIO.md).

## Comandi

| comando | che cosa fa |
|---|---|
| `germen info` | processori e livello SIMD rilevati, dimensioni dei modelli e memoria dei pesi |
| `germen bench` | cronometra passi di addestramento reali per ogni dimensione e stima il tempo per 100 M e 1 miliardo di token; `--scaling` ripete la misura con 1, 2, 4, … thread e mostra la curva di accelerazione; `--json FILE` salva i numeri |
| `germen demo-corpus` | scrive un corpus rumeno sintetico generato da una piccola grammatica (`--size-mb`) |
| `germen train` | addestra da zero (`--data FILE` oppure `--demo`), o continua con `--resume` |
| `germen generate` | genera testo da un modello addestrato, con una cache chiave/valore (`--temperature`, `--top-k`, `--prompt`) |
| `germen verify` | controlla un checkpoint: dimensione e SHA-256 di ogni file rispetto a `meta.json` (quello che stampa `sha256sum`); `--recover` completa un salvataggio interrotto, senza bisogno di alcun testo |
| `germen ui` | apre GERMEN Studio, l'interfaccia grafica: un server solo su `127.0.0.1` e una finestra propria (`--workspace`, `--port`, `--no-open`); vedere [docs/STUDIO.md](docs/STUDIO.md) |

Opzioni di addestramento da conoscere: `--tokenizer byte|bpe` e `--vocab N` (la dimensione del BPE);
`--batch` e `--accum` (accumulo del gradiente: il batch effettivo è il loro prodotto, la memoria è
quella del solo `--batch`); `--replicas N` (copie del modello che si dividono ogni micro-batch tra i
thread; 0 = automatico, 1 = disattivato; scegliete un `--batch` almeno pari al numero di thread);
`--grad-clip` (norma globale del gradiente, 1 per impostazione predefinita); `--lr`, `--warmup`,
`--weight-decay`. `germen train --help` le elenca tutte. Aggiungete `--threads N` a qualsiasi
comando per limitare il numero di thread di lavoro (predefinito: tutti i processori logici). I
messaggi della riga di comando sono per ora solo in rumeno.

## Dimensioni dei modelli

| nome | parametri | strati × teste × larghezza | pesi (f32) |
|---|---:|---|---:|
| `nano` | 0,85 M | 4 × 4 × 128 | 3 MiB |
| `micro` | 2,76 M | 6 × 6 × 192 | 11 MiB |
| `mini` | 10,82 M | 6 × 6 × 384 | 41 MiB |
| `small` | 25,44 M | 8 × 8 × 512 | 97 MiB |
| `base` | 85,35 M | 12 × 12 × 768 | 326 MiB |

Le dimensioni valgono per il vocabolario di 256 byte; un vocabolario BPE di V token aggiunge
2·(V − 256)·larghezza parametri (embedding e testa di uscita). L'addestramento conserva anche le
due stime di momento di AdamW (≈ 4× i pesi) e le attivazioni, che crescono con `--batch` e
`--seq`.

## Quanto tempo ci vuole?

Il costo dell'addestramento è circa **6 × parametri × token** operazioni in virgola mobile (con i
byte, 1 MB di testo = 1 milione di token). Dividete per ciò che la vostra CPU fornisce davvero e
avete il tempo. `germen bench` misura esattamente questo sul vostro computer.

Misura di riferimento — una piccola macchina virtuale Linux con **2 core** (AVX-512), batch 16 ×
256, vocabolario di byte, impostazioni predefinite (mediana di cinque esecuzioni alternate su una
macchina virtuale rumorosa, quindi ± 10 %):

| nome | token/s | GFLOPS effettivi | 100 M token | 1 mld. token |
|---|---:|---:|---:|---:|
| `nano` | 8.366 | 54 | 3,3 h | 1,4 giorni |
| `micro` | 3.981 | 79 | 7,0 h | 2,9 giorni |
| `mini` | 1.464 | 104 | 19,0 h | 7,9 giorni |
| `small` | 647 | 106 | 1,8 giorni | 18 giorni |

Passare da 1 a 2 thread ha reso `mini` 2,04× più veloce qui (733 → 1.494 token/s; 1,41× con
`--replicas 1`), quindi non aspettatevi che 20 core diano 20×. In un singolo grafo i prodotti di
matrici scalano bene (≈ 1,8× su due thread), l'attenzione meno (≈ 1,2×), e RMSNorm, gli embedding
rotatori e l'aggiornamento di AdamW quasi per niente; le repliche della versione 0.3 eseguono queste
parti affiancate, ma richiedono almeno tante finestre per batch quanti sono i thread. Fin dove
arrivi questo su molti core non è stato misurato: a questo serve `germen bench --scaling`, che
stampa l'intera curva e il numero di thread oltre il quale non si guadagna più nulla. **Eseguite
`germen bench --scaling --json bench.json` sul vostro computer** — è l'unico modo per saperlo. Una
regola empirica per l'addestramento ottimale in termini di calcolo è di circa 20 token per
parametro; i modelli piccoli vengono spesso addestrati su molto di più.

La generazione usa una cache chiave/valore: 400 token con `mini` richiedono circa 1 secondo invece
di 15 sugli stessi due core.

## Com'è costruito

```
crates/germen-core   la libreria: config, model, tokenizer, bpe, data, train, gradients,
                     generate, checkpoint, bench
crates/germen-studio GERMEN Studio: il server e la pagina dietro `germen ui` (solo libreria standard)
crates/germen-cli    la riga di comando `germen` (clap)
```

* **Backend** — `flex` di Burn: Rust puro, prodotti di matrici tramite il crate `gemm`, kernel SIMD
  (AVX2/AVX-512/SSE, NEON) scelti **a tempo di esecuzione**, multithreading con rayon. Un
  eseguibile x86-64 generico usa quindi AVX2 su una CPU che lo possiede.
* **Modello** — blocchi pre-norm, proiezione QKV fusa, attenzione causale, embedding rotatori,
  testa di uscita non legata all'embedding, inizializzazione in stile GPT-2 (proiezioni di uscita
  scalate di `1/√(2·strati)`). L'addestramento usa un'attenzione causale a blocchi che non calcola
  mai la parte sopra la diagonale (passi interi più veloci dell'8–28 % su due core, a seconda
  della dimensione del modello; stessi valori e stessi gradienti); la valutazione usa il kernel
  fuso di Burn, e la generazione una cache chiave/valore.
* **Tokenizer** — byte, oppure un BPE a livello di byte con un pre-tokenizer in stile GPT-2 (lo
  spazio si attacca alla parola che segue, le fusioni non attraversano il confine di una parola).
  L'apprendimento del vocabolario è deterministico e usa solo la parte di addestramento del
  testo; la codifica è parallela; qualsiasi sequenza di byte viene restituita identica.
* **Addestramento** — AdamW (β₂ = 0,95), riscaldamento lineare poi decadimento coseno, clipping del
  gradiente sulla norma globale, accumulo del gradiente, validazione su finestre fisse (in nat per
  token e bit per byte), checkpoint atomici, avanzamento tramite eventi (`Event`) così che
  un'interfaccia grafica possa riusare lo stesso ciclo.
* **Thread** — l'intero comando gira come un lavoratore di un proprio pool di thread, che avvia i
  cicli paralleli senza passaggi di consegne tra thread (+ 7 % per `mini`, fino a + 25 % per `nano`,
  su due core). Inoltre ogni micro-batch è diviso in **repliche di dati paralleli**: copie del
  modello, ciascuna con un proprio grafo di differenziazione automatica, eseguono avanti e indietro
  su thread diversi, e i loro gradienti vengono sommati in un ordine fisso, così un'esecuzione è
  riproducibile per un dato numero di thread e di repliche (da + 23 % a + 44 % su due core a seconda
  del modello, con un picco di memoria più basso; `--replicas 1` dà il passo ordinario). Distribuire
  i micro-batch su thread che condividevano tutti un solo grafo è stato provato nella 0.2 e
  scartato; vedere [docs/PLAN.md](docs/PLAN.md).
* **Checkpoint** — una cartella con `config.json`, `weights.bpk` (il formato burnpack di Burn, f32),
  `optimizer.bpk` (i momenti di AdamW), `tokenizer.json` e `meta.json`. I checkpoint scritti dalla
  versione 0.1 si caricano ancora. Dalla 0.4 `meta.json` registra anche dimensione e SHA-256 degli
  altri file, controllati a ogni caricamento: un file danneggiato, troncato o di un altro
  salvataggio viene nominato nell'errore; quelli più vecchi si caricano senza controllo.
* **Studio** — `germen ui` avvia un piccolo server HTTP scritto con la sola libreria standard,
  legato a `127.0.0.1`, e serve una pagina autonoma (stile, script e otto lingue dentro; non
  carica nulla). La pagina avvia lo stesso codice di addestramento, generazione e misura della riga
  di comando e lo segue tramite eventi. Chi può parlare al server: un segreto nell'indirizzo (dopo
  il `#`, che un browser non invia mai), le intestazioni `Host` e `Origin` e una
  Content-Security-Policy che lascia eseguire esattamente uno script; ognuna di queste porte è
  attaccata da test su un vero socket e disattivata a turno da 18 prove per mutazione, che i test
  devono notare ([docs/STUDIO.md](docs/STUDIO.md)).
* **Test** — 247 test (unitari della libreria, della riga di comando e dell'interfaccia grafica, più
  esecuzioni complete del binario vero e del server tramite un socket), tra cui: l'attenzione è
  causale, l'attenzione a blocchi è uguale a quella di Burn (valori e gradienti), la generazione con
  cache è uguale al passaggio completo, i micro-batch accumulati equivalgono a un batch grande, le
  repliche danno i gradienti di un solo grande grafo e un'esecuzione è riproducibile bit per bit, un
  passo di ottimizzazione dopo il ricaricamento dello stato di AdamW è uguale a quello di un
  addestramento ininterrotto, un checkpoint non resta mai scritto a metà e i pesi non finiti non
  vengono mai salvati, i file ostili di tokenizer e configurazione vengono rifiutati, ogni sequenza
  di byte sopravvive all'andata e ritorno col BPE. Un bit cambiato in un file del checkpoint viene
  intercettato e il file nominato, e un salvataggio interrotto in qualsiasi momento viene completato
  o scartato. L'audit avversariale è in [docs/AUDIT.md](docs/AUDIT.md).

## Limiti (versione 0.5)

* Le somme di controllo proteggono dagli incidenti (un bit cambiato, un file troncato, file di due
  salvataggi diversi), non da chi può modificare anche `meta.json`; i checkpoint scritti prima della
  0.4 non si possono verificare.
* L'intero corpus viene letto in memoria (come identificatori a 16 bit); sono previsti corpus
  mappati in memoria più grandi della RAM.
* In un singolo grafo alcune operazioni del passo (RMSNorm, embedding rotatori, aggiornamento di
  AdamW) non sfruttano bene i core aggiuntivi; le repliche lo nascondono quando `--batch` è almeno
  pari al numero di thread, ma fin dove arrivi su molti core non è ancora misurato.
* Niente precisione mista e niente backend GPU.
* Il corpus `demo` è sintetico e minuscolo: dimostra che la catena impara, nient'altro.
* Non ancora misurato su Windows, macOS o su una macchina con molti core — sopra è spiegato come
  aiutare.
* GERMEN Studio è stato provato in un vero browser solo su Linux e solo in Chromium; su Windows e
  macOS i suoi test passano in CI, ma nessuno ne ha ancora usato la finestra lì, e nessuno l'ha
  usato con uno screen reader.

Vedere [CHANGELOG.md](CHANGELOG.md), [docs/PLAN.md](docs/PLAN.md), [docs/AUDIT.md](docs/AUDIT.md) e
[docs/STUDIO.md](docs/STUDIO.md). I contributi sono benvenuti ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Licenza

MIT — vedere [LICENSE](LICENSE). Copyright © 2026 Alexandru-Ionuț Chiuță (Alexio).
