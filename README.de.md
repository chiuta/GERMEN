# GERMEN

**Ein kleines Sprachmodell von Grund auf trainieren – in reinem Rust, auf einer gewöhnlichen CPU.**

Kein Python, kein CUDA, keine C- oder C++-Toolchain, kein Konto, kein Netzwerkzugriff, keine
Telemetrie. Eine einzige ausführbare Datei, eine Textdatei, ein Befehl. Gebaut auf
[Burn](https://burn.dev) 0.22 mit dessen CPU-Backend `flex`, das vollständig in Rust geschrieben
ist.

> Sprachen: [English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · [Español](README.es.md) · [Português](README.pt.md) · **Deutsch** · [Magyar](README.hu.md)

## Was es ist – und was nicht

GERMEN trainiert ein Netz vom Typ GPT (RMSNorm, rotierende Positionen, Feed-Forward-Schicht mit
quadrierter ReLU, ohne Bias) über **Token**. Standardmäßig ist ein Token ein **Byte**: Das
Vokabular besteht aus den 256 Werten eines Bytes, sodass jeder Text – auch mit den rumänischen
Diakritika – ohne jede Vorbereitung funktioniert. Mit `--tokenizer bpe` lernt GERMEN zuerst aus
Ihrem eigenen Text ein BPE-Vokabular auf Byte-Ebene (bis zu 65.536 Token); dasselbe Kontextfenster
fasst dann mehrmals so viel Text (bei gewöhnlicher Prosa typischerweise das 3- bis 4-Fache). Der
Tokenizer wird zusammen mit dem Modell gespeichert.

Es ist eine **Werkstatt zum Lernen und Experimentieren** und eine Grundlage für die Forschung: In
wenigen Minuten sieht man, wie ein Netz vom Zufallsrauschen zu Rechtschreibung, Zeichensetzung und
dem Stil Ihres Textes findet. Es ist **keine** Chatbot-Fabrik. Ein Modell mit 1–25 Millionen
Parametern, trainiert auf einigen hundert Megabyte, lernt die *Form* eines Textes (Wörter,
Grammatikfragmente, Layout), kein Weltwissen und keine Konversation. Nützliche Assistenten
brauchen um Größenordnungen mehr Rechenleistung, als eine Desktop-CPU bietet.

## Schnellstart

Sie brauchen Rust 1.95 oder neuer ([rustup.rs](https://rustup.rs); unter Windows verlangt die
Installation außerdem die Visual Studio Build Tools). Dann:

```bash
cargo build --release
./target/release/germen bench                      # wie schnell ist DIESER Rechner?
./target/release/germen bench --scaling            # ...und wie wächst die Geschwindigkeit mit der Zahl der Threads?
./target/release/germen train --demo --preset micro --steps 300
./target/release/germen generate                   # Text aus dem eben trainierten Modell
```

Unter Windows verwenden Sie `target\release\germen.exe`. Fertig gebaute Programme hängen an den
Releases.

Trainieren Sie mit Ihrem eigenen Text – jede Datei, jede Sprache:

```bash
germen train --data mein-korpus.txt --preset mini --tokenizer bpe --vocab 4096 --steps 5000 --out mein-modell
germen generate --model mein-modell --prompt "Era odată " --max-new 400
```

Drücken Sie während des Trainings **Strg-C**: Es beendet den laufenden Schritt, speichert und
beendet sich. Setzen Sie später mit demselben Befehl plus `--resume` fort: Gewichte, Schrittzähler
und AdamW-Zustand werden wiederhergestellt, sodass das Training ohne den Verlustsprung eines
Kaltstarts weitergeht (die Textfenster werden neu gezogen).

Derselbe `train`-Befehl, erneut *ohne* `--resume` ausgeführt, wird abgelehnt, damit er nicht
versehentlich ein trainiertes Modell ersetzt; `--overwrite` erlaubt es, wenn man absichtlich neu
beginnen will.

## Das Fenster: GERMEN Studio

Lieber ein Fenster als eine Kommandozeile?

```bash
germen ui
```

öffnet **GERMEN Studio**: Wählen Sie einen Text, drücken Sie **Training starten** und sehen Sie
einem Modell beim Lernen zu – der Verlustkurve, der Geschwindigkeit und Proben dessen, was das Netz
schreibt, während es besser wird. Dasselbe Fenster schreibt Text aus Ihren Modellen, prüft und
repariert deren Dateien und misst die Geschwindigkeit Ihres Rechners, mit demselben Code wie die
Befehle oben. Es gibt es in acht Sprachen, hell und dunkel, es passt in ein Fenster in
Telefongröße ebenso wie auf einen breiten Bildschirm und ist dafür gebaut, mit der Tastatur und
mit einem Screenreader bedient zu werden.

Es ist ein kleiner Server, der nur auf `127.0.0.1` lauscht, und eine Seite, die er ausliefert:
nichts zu installieren, nichts herunterzuladen, nichts, das irgendwohin gesendet wird. Die Adresse,
die er ausgibt, enthält ein Geheimnis, das nur für diesen Lauf gilt; daher kann nur das damit
geöffnete Fenster (oder Sie) ihm Befehle geben. Modelle und Texte liegen in `~/GERMEN`
(`--workspace` ändert das); `--no-open` gibt nur die Adresse aus. Wie er geschützt ist, und alles
Weitere: [docs/STUDIO.md](docs/STUDIO.md).

## Befehle

| Befehl | was er tut |
|---|---|
| `germen info` | erkannte Prozessoren und SIMD-Stufe, Modellgrößen und Speicher der Gewichte |
| `germen bench` | misst echte Trainingsschritte für jede Modellgröße und rechnet die Zeit für 100 Mio. und 1 Mrd. Token hoch; `--scaling` wiederholt die Messung mit 1, 2, 4, … Threads und zeigt die Beschleunigungskurve; `--json DATEI` speichert die Zahlen |
| `germen demo-corpus` | schreibt einen synthetischen rumänischen Korpus, erzeugt aus einer kleinen Grammatik (`--size-mb`) |
| `germen train` | trainiert von Grund auf (`--data DATEI` oder `--demo`) oder macht mit `--resume` weiter |
| `germen generate` | erzeugt Text aus einem trainierten Modell, mit Key-Value-Cache (`--temperature`, `--top-k`, `--prompt`) |
| `germen verify` | prüft einen Checkpoint: Größe und SHA-256 jeder Datei gegen `meta.json` (was `sha256sum` ausgibt); `--recover` schließt ein unterbrochenes Speichern ab, ganz ohne Text |
| `germen ui` | öffnet GERMEN Studio, die grafische Oberfläche: ein Server nur auf `127.0.0.1` und ein eigenes Fenster (`--workspace`, `--port`, `--no-open`); siehe [docs/STUDIO.md](docs/STUDIO.md) |

Wissenswerte Trainingsoptionen: `--tokenizer byte|bpe` und `--vocab N` (die BPE-Größe); `--batch`
und `--accum` (Gradientenakkumulation: Der effektive Batch ist ihr Produkt, der Speicher entspricht
dem von `--batch` allein); `--replicas N` (Kopien des Modells, die sich jeden Mikro-Batch auf die
Threads aufteilen; 0 = automatisch, 1 = aus; wählen Sie einen `--batch`, der mindestens so groß ist
wie die Zahl der Threads); `--grad-clip` (globale Gradientennorm, Standard 1); `--lr`, `--warmup`,
`--weight-decay`. `germen train --help` listet alle auf. Fügen Sie `--threads N` zu jedem Befehl
hinzu, um die Zahl der Arbeits-Threads zu begrenzen (Standard: alle logischen Prozessoren). Die
Meldungen der Kommandozeile sind vorerst nur auf Rumänisch.

## Modellgrößen

| Name | Parameter | Schichten × Köpfe × Breite | Gewichte (f32) |
|---|---:|---|---:|
| `nano` | 0,85 Mio. | 4 × 4 × 128 | 3 MiB |
| `micro` | 2,76 Mio. | 6 × 6 × 192 | 11 MiB |
| `mini` | 10,82 Mio. | 6 × 6 × 384 | 41 MiB |
| `small` | 25,44 Mio. | 8 × 8 × 512 | 97 MiB |
| `base` | 85,35 Mio. | 12 × 12 × 768 | 326 MiB |

Die Größen gelten für das Vokabular mit 256 Bytes; ein BPE-Vokabular mit V Token fügt
2·(V − 256)·Breite Parameter hinzu (Embedding und Ausgabekopf). Beim Training werden außerdem die
beiden Momentschätzungen von AdamW (≈ 4× die Gewichte) und die Aktivierungen vorgehalten, die mit
`--batch` und `--seq` wachsen.

## Wie lange dauert es?

Die Trainingskosten betragen etwa **6 × Parameter × Token** Gleitkommaoperationen (bei Bytes ist
1 MB Text = 1 Million Token). Teilen Sie durch das, was Ihre CPU tatsächlich leistet, und Sie
haben die Zeit. `germen bench` misst genau das auf Ihrem Rechner.

Referenzmessung – eine kleine Linux-VM mit **2 Kernen** (AVX-512), Batch 16 × 256, Byte-Vokabular,
Standardeinstellungen (Median aus fünf abwechselnd ausgeführten Läufen auf einer verrauschten VM,
daher ± 10 %):

| Name | Token/s | effektive GFLOPS | 100 Mio. Token | 1 Mrd. Token |
|---|---:|---:|---:|---:|
| `nano` | 8.366 | 54 | 3,3 Std. | 1,4 Tage |
| `micro` | 3.981 | 79 | 7,0 Std. | 2,9 Tage |
| `mini` | 1.464 | 104 | 19,0 Std. | 7,9 Tage |
| `small` | 647 | 106 | 1,8 Tage | 18 Tage |

Der Wechsel von 1 auf 2 Threads machte `mini` hier 2,04× schneller (733 → 1.494 Token/s; 1,41× mit
`--replicas 1`); erwarten Sie also nicht, dass 20 Kerne das 20-Fache bringen. In einem einzelnen
Graphen skalieren Matrixprodukte gut (≈ 1,8× auf zwei Threads), die Attention weniger (≈ 1,2×), und
RMSNorm, rotierende Embeddings und das AdamW-Update kaum; die Replikate der Version 0.3 führen diese
Teile nebeneinander aus, brauchen dafür aber mindestens so viele Fenster pro Batch wie Threads. Wie
weit das auf vielen Kernen trägt, ist noch nicht gemessen: Dafür gibt es `germen bench --scaling`,
das die ganze Kurve ausgibt und die Thread-Zahl, ab der nichts mehr gewonnen wird. **Führen Sie
`germen bench --scaling --json bench.json` auf Ihrem Rechner aus** – nur so lässt es sich wissen.
Eine Faustregel für rechenoptimales Training sind etwa 20 Token pro Parameter; kleine Modelle werden
oft mit sehr viel mehr trainiert.

Die Generierung nutzt einen Key-Value-Cache: 400 Token mit `mini` dauern etwa 1 Sekunde statt 15
auf denselben zwei Kernen.

## Wie es gebaut ist

```
crates/germen-core   die Bibliothek: config, model, tokenizer, bpe, data, train, gradients,
                     generate, checkpoint, bench
crates/germen-studio GERMEN Studio: der Server und die Seite hinter `germen ui` (nur die Standardbibliothek)
crates/germen-cli    die Kommandozeile `germen` (clap)
```

* **Backend** — `flex` von Burn: reines Rust, Matrixprodukte über das Crate `gemm`, SIMD-Kernel
  (AVX2/AVX-512/SSE, NEON), die **zur Laufzeit** gewählt werden, Multithreading mit rayon. Ein
  generisches x86-64-Programm nutzt also auf einer CPU mit AVX2 auch AVX2.
* **Modell** — Pre-Norm-Blöcke, fusionierte QKV-Projektion, kausale Attention, rotierende
  Embeddings, nicht an das Embedding gebundener Ausgabekopf, Initialisierung nach GPT-2-Art
  (Ausgabeprojektionen skaliert mit `1/√(2·Schichten)`). Das Training nutzt eine blockweise
  kausale Attention, die den Teil oberhalb der Diagonale nie berechnet (ganze Schritte auf zwei
  Kernen je nach Modellgröße 8–28 % schneller; gleiche Werte und Gradienten); die Auswertung nutzt
  Burns fusionierten Kernel, die Generierung einen Key-Value-Cache.
* **Tokenizer** — Bytes oder ein BPE auf Byte-Ebene mit einem Pre-Tokenizer nach GPT-2-Art (das
  Leerzeichen klebt am folgenden Wort, Verschmelzungen überschreiten keine Wortgrenze). Das Lernen
  des Vokabulars ist deterministisch und nutzt nur den Trainingsteil des Textes; das Kodieren
  läuft parallel; jede Bytefolge wird exakt wiederhergestellt.
* **Training** — AdamW (β₂ = 0,95), lineares Warm-up, dann Kosinus-Abfall, Gradient-Clipping nach
  globaler Norm, Gradientenakkumulation, Validierung auf festen Fenstern (in Nat pro Token und Bit
  pro Byte), atomare Checkpoints, Fortschritt über Ereignisse (`Event`), damit eine grafische
  Oberfläche dieselbe Schleife wiederverwenden kann.
* **Threads** — der ganze Befehl läuft als Arbeiter eines eigenen Thread-Pools, der parallele
  Schleifen ohne Übergabe zwischen Threads startet (+ 7 % bei `mini`, bis zu + 25 % bei `nano`, auf
  zwei Kernen). Zusätzlich wird jeder Mikro-Batch in **datenparallele Replikate** aufgeteilt: Kopien
  des Modells, jede mit einem eigenen Autodiff-Graphen, laufen vorwärts und rückwärts auf
  verschiedenen Threads, und ihre Gradienten werden in fester Reihenfolge addiert, sodass ein Lauf
  für eine gegebene Zahl von Threads und Replikaten reproduzierbar ist (+ 23 % bis + 44 % auf zwei
  Kernen, je nach Modell, bei niedrigerem Speicherspitzenwert; `--replicas 1` ergibt den
  gewöhnlichen Schritt). Mikro-Batches auf Threads zu verteilen, die alle einen einzigen Graphen
  teilten, wurde in 0.2 versucht und verworfen; siehe [docs/PLAN.md](docs/PLAN.md).
* **Checkpoint** — ein Ordner mit `config.json`, `weights.bpk` (Burns Burnpack-Format, f32),
  `optimizer.bpk` (die AdamW-Momente), `tokenizer.json` und `meta.json`. Von Version 0.1
  geschriebene Checkpoints lassen sich weiterhin laden. Seit 0.4 hält `meta.json` auch Größe und
  SHA-256 der übrigen Dateien fest, bei jedem Laden geprüft: Eine beschädigte, abgeschnittene oder
  aus einem anderen Speicherstand stammende Datei wird in der Fehlermeldung genannt; ältere
  Checkpoints laden ungeprüft.
* **Studio** — `germen ui` startet einen kleinen HTTP-Server, der nur mit der Standardbibliothek
  geschrieben ist und an `127.0.0.1` gebunden bleibt, und liefert eine eigenständige Seite aus (Stil,
  Skript und acht Sprachen stecken darin; sie lädt nichts nach). Die Seite startet denselben Code
  für Training, Generierung und Messung wie die Kommandozeile und verfolgt ihn über Ereignisse. Wer
  mit dem Server sprechen darf: ein Geheimnis in der Adresse (hinter dem `#`, das ein Browser nie
  sendet), die Header `Host` und `Origin` und eine Content-Security-Policy, die genau ein Skript
  laufen lässt; jedes dieser Tore wird von Tests über einen echten Socket angegriffen und der Reihe
  nach von 18 Mutationsbeweisen abgeschaltet, die die Tests bemerken müssen
  ([docs/STUDIO.md](docs/STUDIO.md)).
* **Tests** — 247 Tests (Unit-Tests der Bibliothek, der Kommandozeile und der grafischen Oberfläche
  sowie vollständige Läufe des echten Binärprogramms und des Servers über einen Socket), darunter:
  Die Attention ist kausal, die blockweise Attention gleicht der von Burn (Werte und Gradienten),
  die Generierung mit Cache gleicht dem vollständigen Durchlauf, akkumulierte Mikro-Batches
  entsprechen einem großen Batch, Replikate liefern die Gradienten eines einzigen großen Graphen und
  ein Lauf ist bitgenau reproduzierbar, ein Optimierungsschritt nach dem Neuladen des AdamW-Zustands
  gleicht dem eines ununterbrochenen Trainings, ein Checkpoint bleibt nie halb geschrieben und nicht
  endliche Gewichte werden nie gespeichert, feindliche Tokenizer- und Konfigurationsdateien werden
  abgelehnt, jede Bytefolge übersteht den Hin-und-Rück-Weg durch das BPE. Ein gekipptes Bit in einer
  Checkpoint-Datei wird erkannt und die Datei benannt, und ein zu jedem Zeitpunkt unterbrochenes
  Speichern wird abgeschlossen oder verworfen. Das adversariale Audit steht in
  [docs/AUDIT.md](docs/AUDIT.md).

## Grenzen (Version 0.5)

* Die Prüfsummen schützen vor Unfällen (ein gekipptes Bit, eine abgeschnittene Datei, Dateien aus
  zwei verschiedenen Speicherständen), nicht vor jemandem, der auch `meta.json` ändern kann; vor 0.4
  geschriebene Checkpoints lassen sich nicht prüfen.
* Der gesamte Korpus wird in den Speicher gelesen (als 16-Bit-Kennungen); speicherabgebildete
  Korpora, die größer als der Arbeitsspeicher sind, sind geplant.
* In einem einzelnen Graphen nutzen einige Operationen des Schritts (RMSNorm, rotierende Embeddings,
  AdamW-Update) zusätzliche Kerne nicht gut; Replikate verbergen das, wenn `--batch` mindestens der
  Thread-Zahl entspricht, wie weit das auf vielen Kernen trägt, ist aber noch nicht gemessen.
* Keine gemischte Präzision und kein GPU-Backend.
* Der `demo`-Korpus ist synthetisch und winzig: Er beweist, dass die Kette lernt, mehr nicht.
* Noch nicht gemessen unter Windows, macOS oder auf einem Rechner mit vielen Kernen – oben steht,
  wie Sie helfen können.
* GERMEN Studio wurde nur unter Linux und nur in Chromium in einem echten Browser ausprobiert; unter
  Windows und macOS laufen seine Tests in der CI durch, aber dort hat noch niemand sein Fenster
  benutzt, und niemand hat es mit einem Screenreader benutzt.

Siehe [CHANGELOG.md](CHANGELOG.md), [docs/PLAN.md](docs/PLAN.md), [docs/AUDIT.md](docs/AUDIT.md) und
[docs/STUDIO.md](docs/STUDIO.md). Beiträge sind willkommen ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Lizenz

MIT – siehe [LICENSE](LICENSE). Copyright © 2026 Alexandru-Ionuț Chiuță (Alexio).
