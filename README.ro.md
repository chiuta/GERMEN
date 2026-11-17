# GERMEN

**Antrenează de la zero un model de limbaj mic, în Rust pur, pe un procesor obișnuit.**

Fără Python, fără CUDA, fără compilator C sau C++, fără cont, fără acces la rețea, fără telemetrie.
Un singur executabil, un fișier text, o comandă. Construit pe [Burn](https://burn.dev) 0.22, cu
backendul său de procesor `flex`, scris integral în Rust.

> Limbi: [English](README.md) · **Română** · [Français](README.fr.md) · [Italiano](README.it.md) · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

## Ce este — și ce nu este

GERMEN antrenează o rețea de tip GPT (RMSNorm, poziții rotative, rețea feed-forward cu ReLU pătrat,
fără termeni de bias) peste **tokeni**. Implicit un token este un **octet**: vocabularul are cele
256 de valori ale unui octet, așa că orice text — inclusiv diacriticele românești — merge fără
nicio pregătire. Cu `--tokenizer bpe`, GERMEN învață mai întâi, din textul vostru, un vocabular BPE
la nivel de octet (până la 65.536 de tokeni), așa că aceeași fereastră de context cuprinde de
câteva ori mai mult text (de obicei de 3–4 ori, la proză obișnuită). Tokenizatorul se salvează
împreună cu modelul.

Este un **atelier de învățare și experiment** și o bază pentru cercetare: poți vedea, în câteva
minute, cum o rețea trece de la zgomot la ortografie, punctuație și stilul textului tău. **Nu**
este o fabrică de chatboți. Un model de 1–25 de milioane de parametri, antrenat pe câteva sute de
megaocteți, învață *forma* unui text (cuvinte, fragmente de gramatică, așezarea în pagină), nu
cunoștințe despre lume și nici conversație. Asistenții utili cer ordine de mărime mai multă putere
de calcul decât oferă un procesor de birou.

## Pornire rapidă

Aveți nevoie de Rust 1.95 sau mai nou ([rustup.rs](https://rustup.rs); pe Windows cere și Visual
Studio Build Tools). Apoi:

```bash
cargo build --release
./target/release/germen bench                      # cât de repede e ACEST calculator?
./target/release/germen bench --scaling            # ...și cum crește viteza cu numărul de fire?
./target/release/germen train --demo --preset micro --steps 300
./target/release/germen generate                   # text din modelul abia antrenat
```

Pe Windows folosiți `target\release\germen.exe`. Executabilele gata făcute sunt atașate la
versiuni (releases).

Antrenați pe textul vostru — orice fișier, în orice limbă:

```bash
germen train --data corpusul-meu.txt --preset mini --tokenizer bpe --vocab 4096 --steps 5000 --out modelul-meu
germen generate --model modelul-meu --prompt "Era odată " --max-new 400
```

Apăsați **Ctrl-C** în timpul antrenării: termină pasul curent, salvează și iese. Reluați mai
târziu cu aceeași comandă plus `--resume`: ponderile, contorul de pași și starea AdamW se
restaurează, deci antrenarea continuă fără urcarea pierderii de la o repornire „la rece”
(ferestrele de text se trag din nou la întâmplare).

## Comenzi

| comandă | ce face |
|---|---|
| `germen info` | procesoarele și nivelul SIMD detectate, dimensiunile de model și memoria ponderilor |
| `germen bench` | cronometrează pași reali de antrenare pentru fiecare dimensiune și estimează timpul pentru 100 M și 1 miliard de tokeni; `--scaling` repetă măsurătoarea cu 1, 2, 4, … fire și afișează curba de accelerare; `--json FIȘIER` salvează cifrele |
| `germen demo-corpus` | scrie un corpus sintetic românesc generat dintr-o gramatică mică (`--size-mb`) |
| `germen train` | antrenează de la zero (`--data FIȘIER` sau `--demo`) sau continuă cu `--resume` |
| `germen generate` | generează text dintr-un model antrenat, cu cache cheie/valoare (`--temperature`, `--top-k`, `--prompt`) |

Opțiuni de antrenare bune de știut: `--tokenizer byte|bpe` și `--vocab N` (mărimea BPE);
`--batch` și `--accum` (acumulare de gradienți: lotul efectiv e produsul lor, memoria e doar cea a
lui `--batch`); `--grad-clip` (norma globală a gradientului, implicit 1); `--lr`, `--warmup`,
`--weight-decay`. `germen train --help` le listează pe toate. Adăugați `--threads N` la orice
comandă pentru a limita numărul de fire de lucru (implicit: toate procesoarele logice). Mesajele
liniei de comandă sunt deocamdată doar în română.

## Dimensiuni de model

| nume | parametri | straturi × capete × lățime | ponderi (f32) |
|---|---:|---|---:|
| `nano` | 0,85 M | 4 × 4 × 128 | 3 MiB |
| `micro` | 2,76 M | 6 × 6 × 192 | 11 MiB |
| `mini` | 10,82 M | 6 × 6 × 384 | 41 MiB |
| `small` | 25,44 M | 8 × 8 × 512 | 97 MiB |
| `base` | 85,35 M | 12 × 12 × 768 | 326 MiB |

Dimensiunile sunt pentru vocabularul de 256 de octeți; un vocabular BPE de V tokeni adaugă
2·(V − 256)·lățime parametri (embedding și cap de ieșire). La antrenare se mai păstrează cele două
estimări de moment ale lui AdamW (≈ 4× ponderile) și activările, care cresc odată cu `--batch` și
`--seq`.

## Cât durează?

Costul antrenării este aproximativ **6 × parametri × tokeni** operații în virgulă mobilă (cu
octeți, 1 MB de text = 1 milion de tokeni). Împărțiți la ce livrează de fapt procesorul vostru și
aveți timpul. `germen bench` măsoară exact asta pe calculatorul vostru.

Măsurătoare de referință — o mașină virtuală Linux mică, cu **2 nuclee** (AVX-512), lot 16 × 256,
vocabular de octeți (o singură rulare pe o mașină virtuală zgomotoasă, deci ± 5 %):

| nume | tokeni/s | GFLOPS efectivi | 100 M tokeni | 1 mld. tokeni |
|---|---:|---:|---:|---:|
| `nano` | 8.261 | 54 | 3,4 h | 1,4 zile |
| `micro` | 3.180 | 63 | 8,7 h | 3,6 zile |
| `mini` | 1.260 | 90 | 22,0 h | 9,2 zile |
| `small` | 545 | 90 | 2,1 zile | 21 de zile |

Trecerea de la 1 la 2 fire a făcut `mini` de 1,51× mai rapid aici (825 → 1.247 tokeni/s), deci nu
vă așteptați ca 20 de nuclee să dea 20×. Produsele de matrice scalează bine (≈ 1,8× pe două fire);
atenția scalează mai puțin (≈ 1,2×), iar RMSNorm, embeddingurile rotative și actualizarea AdamW
aproape deloc, așa că pe un calculator cu multe nuclee ele devin gâtuirea (legea lui Amdahl). Pentru
asta există `germen bench --scaling`: afișează toată curba și numărul de fire de la care nu se mai
câștigă nimic. **Rulați `germen bench --scaling --json bench.json` pe calculatorul vostru** — e
singura cale de a ști. O regulă empirică pentru antrenarea optimă ca resurse este de circa 20 de
tokeni pe parametru; modelele mici se antrenează adesea pe mult mai mult.

Generarea folosește un cache cheie/valoare: 400 de tokeni din `mini` durează circa o secundă, în
loc de 15, pe aceleași două nuclee.

## Cum e construit

```
crates/germen-core   biblioteca: config, model, tokenizer, bpe, date, antrenare, gradienți,
                     generare, salvare, benchmark
crates/germen-cli    linia de comandă `germen` (clap)
```

* **Backend** — `flex` din Burn: Rust pur, produse de matrice prin crate-ul `gemm`, nuclee SIMD
  (AVX2/AVX-512/SSE, NEON) alese **la execuție**, multi-threading cu rayon. Un executabil x86-64
  generic folosește deci AVX2 pe un procesor care îl are.
* **Modelul** — blocuri cu normalizare înainte (pre-norm), proiecție QKV fuzionată, atenție
  cauzală, embeddinguri rotative, cap de ieșire nelegat de embedding, inițializare ca la GPT-2
  (proiecțiile de ieșire scalate cu `1/√(2·straturi)`). La antrenare se folosește o atenție
  cauzală pe blocuri, care nu calculează niciodată partea de deasupra diagonalei (pași întregi cu
  8–28 % mai rapizi pe două nuclee, după mărimea modelului; aceleași valori și gradienți);
  evaluarea folosește nucleul fuzionat al lui Burn, iar generarea un cache cheie/valoare.
* **Tokenizatorul** — octeți sau un BPE la nivel de octet, cu un pre-tokenizator de tip GPT-2
  (spațiul se lipește de cuvântul care urmează, îmbinările nu trec peste un cuvânt). Învățarea
  vocabularului e deterministă și folosește doar partea de antrenare a textului; codarea e
  paralelă; orice șir de octeți se reconstruiește exact.
* **Antrenarea** — AdamW (β₂ = 0,95), încălzire liniară apoi descreștere cosinus, tăierea
  gradientului după norma globală, acumulare de gradienți, validare pe ferestre fixe (în nats pe
  token și biți pe octet), puncte de control atomice, progres prin evenimente (`Event`), astfel
  încât o interfață grafică să poată refolosi aceeași buclă.
* **Firele** — toată comanda rulează ca un lucrător al propriului grup de fire, care pornește
  buclele paralele fără predare între fire (+ 7 % la `mini`, până la + 25 % la `nano`, pe două
  nuclee). Împărțirea micro-loturilor pe mai multe fire care conduc toate Burn a fost încercată și
  respinsă; vedeți [docs/PLAN.md](docs/PLAN.md).
* **Punctul de control** — un director cu `config.json`, `weights.bpk` (formatul burnpack al lui
  Burn, f32), `optimizer.bpk` (momentele AdamW), `tokenizer.json` și `meta.json`. Punctele de
  control scrise de versiunea 0.1 se încarcă în continuare.
* **Teste** — 70 de teste unitare, între care: atenția e cauzală, atenția pe blocuri e egală cu cea
  a lui Burn (valori și gradienți), generarea cu cache dă aceleași rezultate ca trecerea completă,
  micro-loturile acumulate echivalează cu un lot mare, un pas de optimizare după reîncărcarea stării
  AdamW e egal cu cel al unei rulări neîntrerupte, orice șir de octeți supraviețuiește turului
  complet prin BPE.

## Limite (versiunea 0.2)

* Tot corpusul se citește în memorie (ca identificatori de 16 biți); corpusuri mapate în memorie,
  mai mari decât RAM-ul, sunt planificate.
* Câteva operații ale pasului (RMSNorm, embeddingurile rotative, actualizarea AdamW) nu folosesc
  încă bine nucleele în plus; paralelizarea lor e principalul punct deschis de performanță.
* Fără precizie mixtă și fără backend GPU.
* Corpusul `demo` e sintetic și mic: arată că fluxul învață, atât.
* Încă nemăsurat pe Windows, macOS sau pe un calculator cu multe nuclee — mai sus e cum puteți
  ajuta.

Vedeți [CHANGELOG.md](CHANGELOG.md) și [docs/PLAN.md](docs/PLAN.md). Contribuțiile sunt binevenite
([CONTRIBUTING.md](CONTRIBUTING.md)).

## Licență

MIT — vedeți [LICENSE](LICENSE). Copyright © 2026 Alexandru-Ionuț Chiuță (Alexio).
