# GERMEN

**Addestra da zero un piccolo modello linguistico, in puro Rust, su una normale CPU.**

Niente Python, niente CUDA, niente toolchain C o C++, niente account, niente accesso alla rete,
niente telemetria. Un solo eseguibile, un file di testo, un comando. Costruito su
[Burn](https://burn.dev) 0.22 con il suo backend per CPU `flex`, scritto interamente in Rust.

> Lingue: [English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · **Italiano** · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

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

Premete **Ctrl-C** durante l'addestramento: termina il passo in corso, salva ed esce. Riprendete
più tardi con lo stesso comando più `--resume`: pesi, contatore dei passi e stato di AdamW vengono
ripristinati, quindi l'addestramento prosegue senza il rialzo della perdita di un riavvio a
freddo (le finestre di testo vengono estratte di nuovo a caso).

## Comandi

| comando | che cosa fa |
|---|---|
| `germen info` | processori e livello SIMD rilevati, dimensioni dei modelli e memoria dei pesi |
| `germen bench` | cronometra passi di addestramento reali per ogni dimensione e stima il tempo per 100 M e 1 miliardo di token; `--scaling` ripete la misura con 1, 2, 4, … thread e mostra la curva di accelerazione; `--json FILE` salva i numeri |
| `germen demo-corpus` | scrive un corpus rumeno sintetico generato da una piccola grammatica (`--size-mb`) |
| `germen train` | addestra da zero (`--data FILE` oppure `--demo`), o continua con `--resume` |
| `germen generate` | genera testo da un modello addestrato, con una cache chiave/valore (`--temperature`, `--top-k`, `--prompt`) |

Opzioni di addestramento da conoscere: `--tokenizer byte|bpe` e `--vocab N` (la dimensione del
BPE); `--batch` e `--accum` (accumulo del gradiente: il batch effettivo è il loro prodotto, la
memoria è quella del solo `--batch`); `--grad-clip` (norma globale del gradiente, 1 per
impostazione predefinita); `--lr`, `--warmup`, `--weight-decay`. `germen train --help` le elenca
tutte. Aggiungete `--threads N` a qualsiasi comando per limitare il numero di thread di lavoro
(predefinito: tutti i processori logici). I messaggi della riga di comando sono per ora solo in
rumeno.

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

Misura di riferimento — una piccola macchina virtuale Linux con **2 core** (AVX-512), batch
16 × 256, vocabolario di byte (una sola esecuzione su una macchina virtuale rumorosa, quindi
± 5 %):

| nome | token/s | GFLOPS effettivi | 100 M token | 1 mld. token |
|---|---:|---:|---:|---:|
| `nano` | 8.261 | 54 | 3,4 h | 1,4 giorni |
| `micro` | 3.180 | 63 | 8,7 h | 3,6 giorni |
| `mini` | 1.260 | 90 | 22,0 h | 9,2 giorni |
| `small` | 545 | 90 | 2,1 giorni | 21 giorni |

Passare da 1 a 2 thread ha reso `mini` 1,51× più veloce qui (825 → 1.247 token/s), quindi non
aspettatevi che 20 core diano 20×. I prodotti di matrici scalano bene (≈ 1,8× su due thread);
l'attenzione meno (≈ 1,2×), e RMSNorm, gli embedding rotatori e l'aggiornamento di AdamW quasi per
niente, perciò su una macchina con molti core diventano il collo di bottiglia (legge di Amdahl). È
a questo che serve `germen bench --scaling`: stampa l'intera curva e il numero di thread oltre il
quale non si guadagna più nulla. **Eseguite `germen bench --scaling --json bench.json` sul vostro
computer** — è l'unico modo per saperlo. Una regola empirica per l'addestramento ottimale in
termini di calcolo è di circa 20 token per parametro; i modelli piccoli vengono spesso addestrati
su molto di più.

La generazione usa una cache chiave/valore: 400 token con `mini` richiedono circa 1 secondo invece
di 15 sugli stessi due core.

## Com'è costruito

```
crates/germen-core   la libreria: config, model, tokenizer, bpe, data, train, gradients,
                     generate, checkpoint, bench
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
  cicli paralleli senza passaggi di consegne tra thread (+ 7 % per `mini`, fino a + 25 % per
  `nano`, su due core). Distribuire micro-batch su più thread che pilotano tutti Burn è stato
  provato e scartato; vedere [docs/PLAN.md](docs/PLAN.md).
* **Checkpoint** — una cartella con `config.json`, `weights.bpk` (il formato burnpack di Burn,
  f32), `optimizer.bpk` (i momenti di AdamW), `tokenizer.json` e `meta.json`. I checkpoint
  scritti dalla versione 0.1 si caricano ancora.
* **Test** — 70 test unitari, tra cui: l'attenzione è causale, l'attenzione a blocchi è uguale a
  quella di Burn (valori e gradienti), la generazione con cache è uguale al passaggio completo, i
  micro-batch accumulati equivalgono a un batch grande, un passo di ottimizzazione dopo il
  ricaricamento dello stato di AdamW è uguale a quello di un addestramento ininterrotto, ogni
  sequenza di byte sopravvive all'andata e ritorno col BPE.

## Limiti (versione 0.2)

* L'intero corpus viene letto in memoria (come identificatori a 16 bit); sono previsti corpus
  mappati in memoria più grandi della RAM.
* Alcune operazioni del passo (RMSNorm, embedding rotatori, aggiornamento di AdamW) non sfruttano
  ancora bene i core aggiuntivi; renderle parallele è il principale punto aperto sulle prestazioni.
* Niente precisione mista e niente backend GPU.
* Il corpus `demo` è sintetico e minuscolo: dimostra che la catena impara, nient'altro.
* Non ancora misurato su Windows, macOS o su una macchina con molti core — sopra è spiegato come
  aiutare.

Vedere [CHANGELOG.md](CHANGELOG.md) e [docs/PLAN.md](docs/PLAN.md). I contributi sono benvenuti
([CONTRIBUTING.md](CONTRIBUTING.md)).

## Licenza

MIT — vedere [LICENSE](LICENSE). Copyright © 2026 Alexandru-Ionuț Chiuță (Alexio).
