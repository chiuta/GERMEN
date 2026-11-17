# GERMEN

**Taníts be a nulláról egy kis nyelvi modellt, tiszta Rustban, egy hétköznapi processzoron.**

Nincs Python, nincs CUDA, nincs C- vagy C++-eszköztár, nincs fiók, nincs hálózati hozzáférés,
nincs telemetria. Egyetlen futtatható fájl, egy szövegfájl, egy parancs. A [Burn](https://burn.dev)
0.22-re épül, annak teljes egészében Rustban írt `flex` processzor-háttérrendszerével.

> Nyelvek: [English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · **Magyar**

## Mi ez – és mi nem

A GERMEN egy GPT-típusú hálót tanít (RMSNorm, forgó pozíciókódolás, négyzetre emelt ReLU-t használó
előrecsatolt réteg, eltolás nélkül) **tokenek** felett. Alapértelmezés szerint egy token egy
**bájt**: a szókincs a bájt 256 lehetséges értéke, így bármilyen szöveg – a román diakritikus
jelekkel együtt – előkészítés nélkül működik. A `--tokenizer bpe` kapcsolóval a GERMEN először a
saját szövegedből megtanul egy bájtszintű BPE-szókincset (legfeljebb 65 536 tokent); ugyanabba a
kontextusablakba ekkor többszörösen több szöveg fér (rendes próza esetén jellemzően 3–4-szer
annyi). A tokenizálót a modellel együtt menti el.

Ez egy **műhely tanuláshoz és kísérletezéshez**, és alap a kutatáshoz: percek alatt végignézheted,
hogyan jut el egy háló a véletlen zajtól a helyesíráshoz, az írásjelekhez és a szöveged
stílusához. **Nem** chatbot-gyár. Néhány száz megabájton betanított, 1–25 millió paraméteres
modell a szöveg *formáját* tanulja meg (szavak, nyelvtani töredékek, tördelés), nem világtudást és
nem társalgást. A használható asszisztensekhez nagyságrendekkel több számítás kell, mint amennyit
egy asztali processzor nyújt.

## Gyors kezdés

Rust 1.95 vagy újabb kell hozzá ([rustup.rs](https://rustup.rs); Windowson a telepítő a Visual
Studio Build Tools-t is kéri). Aztán:

```bash
cargo build --release
./target/release/germen bench                      # milyen gyors EZ a számítógép?
./target/release/germen bench --scaling            # ...és hogyan nő a sebesség a szálak számával?
./target/release/germen train --demo --preset micro --steps 300
./target/release/germen generate                   # szöveg az imént betanított modellből
```

Windowson a `target\release\germen.exe` fájlt használd. Az előre lefordított futtatható fájlok a
kiadásokhoz (release) vannak csatolva.

Taníts a saját szövegeden – bármilyen fájl, bármilyen nyelv:

```bash
germen train --data sajat-korpusz.txt --preset mini --tokenizer bpe --vocab 4096 --steps 5000 --out sajat-modell
germen generate --model sajat-modell --prompt "Era odată " --max-new 400
```

Tanítás közben nyomj **Ctrl-C**-t: befejezi az aktuális lépést, ment, és kilép. Később folytasd
ugyanazzal a paranccsal és a `--resume` kapcsolóval: a súlyok, a lépésszámláló és az AdamW állapota
visszaáll, így a tanítás a hidegindítás okozta veszteségugrás nélkül folytatódik (a szövegablakokat
újra véletlenszerűen húzza).

## Parancsok

| parancs | mit csinál |
|---|---|
| `germen info` | az észlelt processzorok és SIMD-szint, a modellméretek és a súlyok memóriája |
| `germen bench` | valódi tanítási lépéseket mér minden modellmérettel, és megbecsüli az időt 100 millió és 1 milliárd tokenhez; a `--scaling` 1, 2, 4, … szállal megismétli a mérést, és kiírja a gyorsulási görbét; a `--json FÁJL` elmenti a számokat |
| `germen demo-corpus` | egy kis nyelvtanból generált szintetikus román korpuszt ír (`--size-mb`) |
| `germen train` | a nulláról tanít (`--data FÁJL` vagy `--demo`), vagy a `--resume` kapcsolóval folytatja |
| `germen generate` | szöveget generál egy betanított modellből, kulcs–érték gyorsítótárral (`--temperature`, `--top-k`, `--prompt`) |

Érdemes ismerni ezeket a tanítási kapcsolókat: `--tokenizer byte|bpe` és `--vocab N` (a BPE
mérete); `--batch` és `--accum` (gradiens-felhalmozás: a tényleges köteg a kettő szorzata, a
memória csak a `--batch`-é); `--grad-clip` (a gradiens globális normája, alapértelmezés 1); `--lr`,
`--warmup`, `--weight-decay`. A `germen train --help` mindet felsorolja. Bármelyik parancshoz
hozzáadhatod a `--threads N` kapcsolót a dolgozó szálak számának korlátozására (alapértelmezés:
minden logikai processzor). A parancssor üzenetei egyelőre csak románul vannak.

## Modellméretek

| név | paraméterek | rétegek × fejek × szélesség | súlyok (f32) |
|---|---:|---|---:|
| `nano` | 0,85 M | 4 × 4 × 128 | 3 MiB |
| `micro` | 2,76 M | 6 × 6 × 192 | 11 MiB |
| `mini` | 10,82 M | 6 × 6 × 384 | 41 MiB |
| `small` | 25,44 M | 8 × 8 × 512 | 97 MiB |
| `base` | 85,35 M | 12 × 12 × 768 | 326 MiB |

A méretek a 256 bájtos szókincsre vonatkoznak; egy V tokenes BPE-szókincs 2·(V − 256)·szélesség
paramétert ad hozzá (beágyazás és kimeneti fej). A tanítás az AdamW két momentumbecslését is
tárolja (≈ a súlyok 4-szerese), valamint az aktivációkat, amelyek a `--batch` és a `--seq`
értékével nőnek.

## Mennyi ideig tart?

A tanítás költsége nagyjából **6 × paraméterek × tokenek** lebegőpontos művelet (bájtokkal 1 MB
szöveg = 1 millió token). Oszd el azzal, amit a processzorod valójában nyújt, és megvan az idő. A
`germen bench` pontosan ezt méri a te gépeden.

Referenciamérés – egy kis, **2 magos** (AVX-512) Linux virtuális gép, köteg 16 × 256, bájtszókincs
(egyetlen futás egy zajos virtuális gépen, tehát ± 5 %):

| név | token/s | tényleges GFLOPS | 100 M token | 1 Mrd token |
|---|---:|---:|---:|---:|
| `nano` | 8 261 | 54 | 3,4 óra | 1,4 nap |
| `micro` | 3 180 | 63 | 8,7 óra | 3,6 nap |
| `mini` | 1 260 | 90 | 22,0 óra | 9,2 nap |
| `small` | 545 | 90 | 2,1 nap | 21 nap |

Az 1-ről 2 szálra váltás itt 1,51-szeresére gyorsította a `mini`-t (825 → 1 247 token/s), ezért ne
várd, hogy 20 mag 20-szoros gyorsulást ad. A mátrixszorzatok jól skálázódnak (≈ 1,8× két szálon); a
figyelem kevésbé (≈ 1,2×), az RMSNorm, a forgó beágyazások és az AdamW-frissítés pedig alig, így
sok magos gépen ezek válnak szűk keresztmetszetté (Amdahl törvénye). Erre való a
`germen bench --scaling`: kiírja a teljes görbét és azt a szálszámot, amely felett már semmit sem
nyerünk. **Futtasd le a `germen bench --scaling --json bench.json` parancsot a saját gépeden** –
csak így derül ki. A számításra optimális tanítás egyik gyakorlati szabálya körülbelül 20 token
paraméterenként; a kis modelleket gyakran sokkal többön tanítják.

A generálás kulcs–érték gyorsítótárat használ: 400 token a `mini` modellel nagyjából 1 másodperc
15 helyett, ugyanazon a két magon.

## Hogyan épül fel

```
crates/germen-core   a könyvtár: config, model, tokenizer, bpe, data, train, gradients,
                     generate, checkpoint, bench
crates/germen-cli    a `germen` parancssor (clap)
```

* **Háttérrendszer** — a Burn `flex`-e: tiszta Rust, mátrixszorzatok a `gemm` crate-en át,
  **futásidőben** választott SIMD-magok (AVX2/AVX-512/SSE, NEON), többszálúság rayonnal. Egy
  általános x86-64 program tehát AVX2-t használ azon a processzoron, amelyiknek van.
* **Modell** — pre-norm blokkok, összevont QKV-vetítés, oksági figyelem, forgó beágyazások, a
  beágyazáshoz nem kötött kimeneti fej, GPT-2-szerű inicializálás (a kimeneti vetítések
  `1/√(2·rétegek)`-szel skálázva). A tanítás blokkonként számolt oksági figyelmet használ, amely
  sosem számolja ki az átló feletti részt (két magon a teljes lépések a modell méretétől függően
  8–28 %-kal gyorsabbak; ugyanazok az értékek és gradiensek); az értékelés a Burn összevont
  magját használja, a generálás pedig kulcs–érték gyorsítótárat.
* **Tokenizáló** — bájtok, vagy bájtszintű BPE GPT-2-szerű előtokenizálóval (a szóköz a következő
  szóhoz tapad, az összevonások nem lépnek át szóhatárt). A szókincs tanulása determinisztikus, és
  csak a szöveg tanítórészét használja; a kódolás párhuzamos; bármilyen bájtsorozat pontosan
  visszaállítható.
* **Tanítás** — AdamW (β₂ = 0,95), lineáris bemelegítés, majd koszinuszos csökkentés, gradiensvágás
  a globális norma szerint, gradiens-felhalmozás, validáció rögzített ablakokon (nat/token és
  bit/bájt egységben), atomi ellenőrzőpontok, előrehaladás eseményeken (`Event`) keresztül, hogy
  egy grafikus felület ugyanazt a ciklust újra felhasználhassa.
* **Szálak** — a teljes parancs egy saját szálkészlet dolgozójaként fut, amely a párhuzamos
  ciklusokat szálak közötti átadás nélkül indítja (+ 7 % a `mini`-nél, akár + 25 % a `nano`-nál,
  két magon). A mikro-kötegek elosztását több, mind a Burnt hajtó szál között kipróbáltuk és
  elvetettük; lásd a [docs/PLAN.md](docs/PLAN.md) fájlt.
* **Ellenőrzőpont** — egy mappa `config.json`, `weights.bpk` (a Burn burnpack formátuma, f32),
  `optimizer.bpk` (az AdamW momentumai), `tokenizer.json` és `meta.json` fájlokkal. A 0.1-es
  verzió által írt ellenőrzőpontok továbbra is betölthetők.
* **Tesztek** — 70 egységteszt, köztük: a figyelem oksági, a blokkos figyelem megegyezik a Burnéval
  (értékek és gradiensek), a gyorsítótáras generálás megegyezik a teljes átfutással, a
  felhalmozott mikro-kötegek egyenértékűek egy nagy köteggel, egy optimalizálási lépés az AdamW
  állapotának újratöltése után megegyezik a megszakítatlan tanítás lépésével, minden bájtsorozat
  átvészeli az oda-vissza utat a BPE-n.

## Korlátok (0.2-es verzió)

* A teljes korpusz beolvasásra kerül a memóriába (16 bites azonosítókként); a RAM-nál nagyobb,
  memóriába leképezett korpuszok tervben vannak.
* A lépés néhány művelete (RMSNorm, forgó beágyazások, AdamW-frissítés) még nem használja jól a
  további magokat; ezek párhuzamosítása a fő nyitott teljesítménykérdés.
* Nincs vegyes pontosság és nincs GPU-háttérrendszer.
* A `demo` korpusz szintetikus és parányi: azt bizonyítja, hogy a lánc tanul, többet nem.
* Még nincs lemérve Windowson, macOS-en vagy sok magos gépen – fent leírtuk, hogyan segíthetsz.

Lásd a [CHANGELOG.md](CHANGELOG.md) és a [docs/PLAN.md](docs/PLAN.md) fájlt. A hozzájárulásokat
szívesen fogadjuk ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Licenc

MIT – lásd a [LICENSE](LICENSE) fájlt. Copyright © 2026 Alexandru-Ionuț Chiuță (Alexio).
