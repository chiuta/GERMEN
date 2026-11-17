# GERMEN

**Antrenează de la zero un model de limbaj mic, în Rust pur, pe un procesor obișnuit.**

Fără Python, fără CUDA, fără compilator C sau C++, fără cont, fără acces la rețea, fără telemetrie.
Un singur executabil, un fișier text, o comandă. Construit pe [Burn](https://burn.dev) 0.22, cu
backendul său de procesor `flex`, scris integral în Rust.

> Alte limbi: [English](README.md) · urmează și alte traduceri.

## Ce este — și ce nu este

GERMEN antrenează o rețea de tip GPT (RMSNorm, poziții rotative, rețea feed-forward cu ReLU pătrat, fără
termeni de bias) peste **octeți**: vocabularul are cele 256 de valori ale unui octet, așa că orice
text — inclusiv diacriticele românești — merge fără să pregătești vreun tokenizator.

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
./target/release/germen train --demo --preset micro --steps 300
./target/release/germen generate                   # text din modelul abia antrenat
```

Pe Windows folosiți `target\release\germen.exe`. Executabilele gata făcute sunt atașate la
versiuni (releases).

Antrenați pe textul vostru — orice fișier, în orice limbă:

```bash
germen train --data corpusul-meu.txt --preset mini --steps 5000 --out modelul-meu
germen generate --model modelul-meu --prompt "Era odată " --max-new 400
```

Apăsați **Ctrl-C** în timpul antrenării: termină pasul curent, salvează și iese. Reluați mai
târziu cu aceeași comandă plus `--resume`.

## Comenzi

| comandă | ce face |
|---|---|
| `germen info` | procesoarele și nivelul SIMD detectate, dimensiunile de model și memoria ponderilor |
| `germen bench` | cronometrează pași reali de antrenare pentru fiecare dimensiune și estimează timpul pentru 100 M și 1 miliard de tokeni; `--json FIȘIER` salvează cifrele |
| `germen demo-corpus` | scrie un corpus sintetic românesc generat dintr-o gramatică mică (`--size-mb`) |
| `germen train` | antrenează de la zero (`--data FIȘIER` sau `--demo`) sau continuă cu `--resume` |
| `germen generate` | generează text dintr-un model antrenat (`--temperature`, `--top-k`, `--prompt`) |

Adăugați `--threads N` la orice comandă pentru a limita numărul de fire de lucru.
`germen train --help` listează toate opțiunile de antrenare (rata de învățare, încălzirea,
descompunerea ponderilor, lotul, contextul, intervalele de validare și de salvare etc.). Mesajele
liniei de comandă sunt deocamdată doar în română.

## Dimensiuni de model

| nume | parametri | straturi × capete × lățime | ponderi (f32) |
|---|---:|---|---:|
| `nano` | 0,85 M | 4 × 4 × 128 | 3 MiB |
| `micro` | 2,76 M | 6 × 6 × 192 | 11 MiB |
| `mini` | 10,82 M | 6 × 6 × 384 | 41 MiB |
| `small` | 25,44 M | 8 × 8 × 512 | 97 MiB |
| `base` | 85,35 M | 12 × 12 × 768 | 326 MiB |

La antrenare se mai păstrează cele două estimări de moment ale lui AdamW (≈ 4× ponderile) și
activările, care cresc odată cu `--batch` și `--seq`.

## Cât durează?

Costul antrenării este aproximativ **6 × parametri × tokeni** operații în virgulă mobilă (un octet
e un token, deci 1 MB de text = 1 milion de tokeni). Împărțiți la ce livrează de fapt procesorul
vostru și aveți timpul. `germen bench` măsoară exact asta pe calculatorul vostru.

Măsurătoare de referință — o mașină virtuală Linux mică, cu **2 nuclee** (AVX-512), lot 16 × 256:

| nume | tokeni/s | GFLOPS efectivi | 100 M tokeni | 1 mld. tokeni |
|---|---:|---:|---:|---:|
| `nano` | 6.040 | 39 | 4,6 h | 1,9 zile |
| `micro` | 2.362 | 47 | 11,8 h | 4,9 zile |
| `mini` | 1.051 | 75 | 1,0 zile | 11 zile |
| `small` | 522 | 86 | 2,1 zile | 22 de zile |

Trecerea de la 1 la 2 fire a făcut `mini` de 1,44× mai rapid aici (730 → 1.051 tokeni/s), deci nu
vă așteptați ca 20 de nuclee să dea 20×: unele operații ale backendului CPU folosesc un singur fir
(familia `exp`/`erf`, motiv pentru care rețeaua feed-forward folosește ReLU pătrat în loc de GELU).
Un calculator cu mult mai multe nuclee ar trebui totuși să fie de câteva ori mai rapid;
`cargo run --release -p germen-core --example profile_ops -- N` arată ce operații scalează.
**Rulați `germen bench --json bench.json` pe
calculatorul vostru** — e singura cale de a ști. O regulă empirică pentru antrenarea optimă ca
resurse este de circa 20 de tokeni pe parametru; modelele mici se antrenează adesea pe mult mai
mult.

## Cum e construit

```
crates/germen-core   biblioteca: config, model, date, antrenare, generare, salvare, benchmark
crates/germen-cli    linia de comandă `germen` (clap)
```

* **Backend** — `flex` din Burn: Rust pur, produse de matrice prin crate-ul `gemm`, nuclee SIMD
  (AVX2/AVX-512/SSE, NEON) alese **la execuție**, multi-threading cu rayon. Un executabil x86-64
  generic folosește deci AVX2 pe un procesor care îl are.
* **Modelul** — blocuri cu normalizare înainte (pre-norm), proiecție QKV fuzionată, atenție
  cauzală, embeddinguri rotative, cap de ieșire nelegat de embedding, inițializare ca la GPT-2
  (proiecțiile de ieșire scalate cu `1/√(2·straturi)`).
* **Antrenarea** — AdamW (β₂ = 0,95), încălzire liniară apoi descreștere cosinus, validare pe
  ferestre fixe (în nats și biți pe octet), puncte de control atomice, progres prin evenimente
  (`Event`), astfel încât o interfață grafică să poată refolosi aceeași buclă.
* **Punctul de control** — un director cu `config.json`, `weights.bpk` (formatul burnpack al lui
  Burn, f32) și `meta.json`.
* **Teste** — 39 de teste unitare, între care: atenția e cauzală, pierderea inițială e ln 256,
  un punct de control se reîncarcă la logiți identici, antrenarea scade pierderea, `--resume`
  continuă.

## Limite (versiunea 0.1)

* Doar la nivel de octet: secvențele sunt de 3–4 ori mai lungi decât cu un vocabular de
  subcuvinte. Un tokenizator BPE e pasul următor.
* `--resume` reface ponderile și contorul de pași, dar nu momentele lui AdamW (o mică urcare a
  pierderii).
* Fără cache cheie/valoare: generarea recitește fereastra de context la fiecare octet nou.
* Fără acumulare de gradienți, fără precizie mixtă, fără backend GPU; tăierea gradienților se
  face pe fiecare tensor în parte.
* Corpusul `demo` e sintetic și mic: arată că fluxul învață, atât.

Vedeți [CHANGELOG.md](CHANGELOG.md) și [docs/PLAN.md](docs/PLAN.md). Contribuțiile sunt binevenite
([CONTRIBUTING.md](CONTRIBUTING.md)).

## Licență

MIT — vedeți [LICENSE](LICENSE). Copyright © 2026 Alexandru-Ionuț Chiuță (Alexio).
