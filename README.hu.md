# GERMEN

**Taníts be a nulláról egy kis nyelvi modellt, tiszta Rustban, egy hétköznapi processzoron.**

Nincs Python, nincs CUDA, nincs C- vagy C++-eszköztár, nincs fiók, nincs hálózati hozzáférés,
nincs telemetria. Egyetlen futtatható fájl, egy szövegfájl, egy parancs. A [Burn](https://burn.dev)
0.22-re épül, annak teljes egészében Rustban írt `flex` processzor-háttérrendszerével.

> Nyelvek: [English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · **Magyar**

🌐 **Weboldal:** <https://germen.chiuta.ro> · [Studio-bemutató online](https://germen.chiuta.ro/demo.html)

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

Ugyanaz a `train` parancs `--resume` *nélkül* újra kiadva elutasításra kerül, hogy ne írjon felül
véletlenül egy betanított modellt; az `--overwrite` megengedi, ha tényleg elölről akarod kezdeni.

## Az ablak: GERMEN Studio

Inkább ablakot szeretnél parancssor helyett?

```bash
germen ui
```

megnyitja a **GERMEN Studio**t: válassz egy szöveget, nyomd meg a **Tanítás indítása** gombot, és
nézd, ahogy a modell tanul – a veszteséggörbét, a sebességet és mintákat arról, amit a hálózat ír,
miközben javul. Ugyanez az ablak szöveget ír a modelljeidből, ellenőrzi és javítja a fájljaikat, és
megméri a géped sebességét, ugyanazzal a kóddal, mint a fenti parancsok. Nyolc nyelven érhető el,
világos és sötét változatban, elfér egy telefonméretű ablakban és egy széles képernyőn is, és úgy
készült, hogy billentyűzettel és képernyőolvasóval is használható legyen.

Egyetlen kis szerver, amely csak a `127.0.0.1` címen figyel, és egy oldal, amelyet kiszolgál: nincs
mit telepíteni, nincs mit letölteni, és semmi sem megy sehová. A kiírt cím egy csak erre a futásra
érvényes titkot tartalmaz, így csak a vele megnyitott ablak (vagy te) adhat neki parancsot. A
modellek és a szövegek a `~/GERMEN` mappában vannak (a `--workspace` ezt változtatja meg); a
`--no-open` csak kiírja a címet. Hogyan védett, és minden más: [docs/STUDIO.md](docs/STUDIO.md).

## Parancsok

| parancs | mit csinál |
|---|---|
| `germen info` | az észlelt processzorok és SIMD-szint, a modellméretek és a súlyok memóriája |
| `germen bench` | valódi tanítási lépéseket mér minden modellmérettel, és megbecsüli az időt 100 millió és 1 milliárd tokenhez; a `--scaling` 1, 2, 4, … szállal megismétli a mérést, és kiírja a gyorsulási görbét; a `--json FÁJL` elmenti a számokat |
| `germen demo-corpus` | egy kis nyelvtanból generált szintetikus román korpuszt ír (`--size-mb`) |
| `germen train` | a nulláról tanít (`--data FÁJL` vagy `--demo`), vagy a `--resume` kapcsolóval folytatja |
| `germen generate` | szöveget generál egy betanított modellből, kulcs–érték gyorsítótárral (`--temperature`, `--top-k`, `--prompt`) |
| `germen verify` | ellenőriz egy ellenőrzőpontot: minden fájl méretét és SHA-256 összegét a `meta.json` alapján (amit a `sha256sum` kiír); a `--recover` befejez egy megszakadt mentést, szöveg nélkül |
| `germen ui` | megnyitja a GERMEN Studiót, a grafikus felületet: egy csak a `127.0.0.1` címen figyelő szerver és egy saját ablak (`--workspace`, `--port`, `--no-open`); lásd a [docs/STUDIO.md](docs/STUDIO.md) fájlt |

Érdemes ismerni ezeket a tanítási kapcsolókat: `--tokenizer byte|bpe` és `--vocab N` (a BPE mérete);
`--batch` és `--accum` (gradiens-felhalmozás: a tényleges köteg a kettő szorzata, a memória csak a
`--batch`-é); `--replicas N` (a modell másolatai, amelyek a szálak között osztják szét az egyes
mikro-kötegeket; 0 = automatikus, 1 = ki; olyan `--batch`-et válassz, amely legalább a szálak
számával egyenlő); `--grad-clip` (a gradiens globális normája, alapértelmezés 1); `--lr`,
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

Referenciamérés – egy kis, **2 magos** (AVX-512) Linux virtuális gép, köteg 16 × 256, bájtszókincs,
alapbeállítások (öt, egymást váltogató futás mediánja egy zajos virtuális gépen, tehát ± 10 %):

| név | token/s | tényleges GFLOPS | 100 M token | 1 Mrd token |
|---|---:|---:|---:|---:|
| `nano` | 8 366 | 54 | 3,3 óra | 1,4 nap |
| `micro` | 3 981 | 79 | 7,0 óra | 2,9 nap |
| `mini` | 1 464 | 104 | 19,0 óra | 7,9 nap |
| `small` | 647 | 106 | 1,8 nap | 18 nap |

Az 1-ről 2 szálra váltás itt 2,04-szeresére gyorsította a `mini`-t (733 → 1 494 token/s; `--replicas
1` mellett 1,41-szeresére), ezért ne várd, hogy 20 mag 20-szoros gyorsulást ad. Egyetlen gráfon
belül a mátrixszorzatok jól skálázódnak (≈ 1,8× két szálon), a figyelem kevésbé (≈ 1,2×), az
RMSNorm, a forgó beágyazások és az AdamW-frissítés pedig alig; a 0.3-as verzió replikái ezeket a
részeket egymás mellett futtatják, de ehhez legalább annyi ablak kell kötegenként, ahány szál van.
Hogy ez sok magon meddig visz, azt még nem mértük: erre való a `germen bench --scaling`, amely
kiírja a teljes görbét és azt a szálszámot, amely felett már semmit sem nyerünk. **Futtasd le a
`germen bench --scaling --json bench.json` parancsot a saját gépeden** – csak így derül ki. A
számításra optimális tanítás egyik gyakorlati szabálya körülbelül 20 token paraméterenként; a kis
modelleket gyakran sokkal többön tanítják.

A generálás kulcs–érték gyorsítótárat használ: 400 token a `mini` modellel nagyjából 1 másodperc
15 helyett, ugyanazon a két magon.

## Hogyan épül fel

```
crates/germen-core   a könyvtár: config, model, tokenizer, bpe, data, train, gradients,
                     generate, checkpoint, bench
crates/germen-studio GERMEN Studio: a `germen ui` mögötti szerver és oldal (csak a szabványos könyvtár)
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
  ciklusokat szálak közötti átadás nélkül indítja (+ 7 % a `mini`-nél, akár + 25 % a `nano`-nál, két
  magon). Ezenfelül minden mikro-köteg **adatpárhuzamos replikákra** oszlik: a modell másolatai,
  mindegyik saját automatikus differenciálási gráffal, különböző szálakon futtatják az előre- és a
  visszafelé menetet, a gradiensek pedig rögzített sorrendben adódnak össze, így egy futás adott
  szál- és replikaszám mellett reprodukálható (+ 23 % és + 44 % között két magon, a modelltől
  függően, alacsonyabb memóriacsúccsal; a `--replicas 1` a hagyományos lépést adja). A mikro-kötegek
  elosztását olyan szálak között, amelyek mind ugyanazt az egyetlen gráfot használták, a 0.2-ben
  kipróbáltuk és elvetettük; lásd a [docs/PLAN.md](docs/PLAN.md) fájlt.
* **Ellenőrzőpont** — egy mappa `config.json`, `weights.bpk` (a Burn burnpack formátuma, f32),
  `optimizer.bpk` (az AdamW momentumai), `tokenizer.json` és `meta.json` fájlokkal. A 0.1-es verzió
  által írt ellenőrzőpontok továbbra is betölthetők. A 0.4 óta a `meta.json` a többi fájl méretét és
  SHA-256 összegét is rögzíti, minden betöltéskor ellenőrizve: a sérült, csonka vagy másik mentésből
  való fájlt a hibaüzenet néven nevezi; a régebbiek ellenőrzés nélkül töltődnek be.
* **Studio** — a `germen ui` elindít egy kis HTTP-szervert, amely csak a szabványos könyvtárral
  készült és a `127.0.0.1` címhez kötött, és kiszolgál egy önálló oldalt (benne a stílus, a
  szkript és a nyolc nyelv; nem tölt be semmit). Az oldal ugyanazt a tanító, generáló és mérő kódot
  indítja el, mint a parancssor, és eseményeken át követi. Ki beszélhet a szerverrel: egy titok a
  címben (a `#` után, amelyet a böngésző sosem küld el), a `Host` és az `Origin` fejléc, valamint
  egy Content-Security-Policy, amely pontosan egy szkriptet enged futni; ezeket a kapukat valódi
  socketen át támadó tesztek vizsgálják, és sorra kikapcsolja 18 mutációs bizonyíték, amelyet a
  teszteknek észre kell venniük ([docs/STUDIO.md](docs/STUDIO.md)).
* **Tesztek** — 247 teszt (a könyvtár, a parancssor és a grafikus felület egységtesztjei, valamint a
  valódi bináris és a szerver teljes futtatásai egy socketen át), köztük: a figyelem oksági, a
  blokkos figyelem megegyezik a Burnéval (értékek és gradiensek), a gyorsítótáras generálás
  megegyezik a teljes átfutással, a felhalmozott mikro-kötegek egyenértékűek egy nagy köteggel, a
  replikák egyetlen nagy gráf gradienseit adják és egy futás bitre pontosan reprodukálható, egy
  optimalizálási lépés az AdamW állapotának újratöltése után megegyezik a megszakítatlan tanítás
  lépésével, egy ellenőrzőpont sosem marad félig megírva és nem véges súlyokat sosem mentünk, az
  ellenséges tokenizer- és konfigurációs fájlok elutasításra kerülnek, minden bájtsorozat átvészeli
  az oda-vissza utat a BPE-n. Egy ellenőrzőpont-fájlban megbillent bitet észrevesz és megnevezi a
  fájlt, a bármikor megszakadt mentést pedig befejezi vagy eldobja. Az ellenséges szemléletű
  auditról a [docs/AUDIT.md](docs/AUDIT.md) szól.

## Korlátok (0.5-ös verzió)

* Az ellenőrző összegek a balesetek ellen védenek (megbillent bit, csonka fájl, két különböző mentés
  fájljai), nem az ellen, aki a `meta.json`-t is át tudja írni; a 0.4 előtt írt ellenőrzőpontok nem
  ellenőrizhetők.
* A teljes korpusz beolvasásra kerül a memóriába (16 bites azonosítókként); a RAM-nál nagyobb,
  memóriába leképezett korpuszok tervben vannak.
* Egyetlen gráfon belül a lépés néhány művelete (RMSNorm, forgó beágyazások, AdamW-frissítés) nem
  használja jól a további magokat; a replikák ezt elfedik, ha a `--batch` legalább a szálak számával
  egyenlő, de hogy ez sok magon meddig visz, azt még nem mértük.
* Nincs vegyes pontosság és nincs GPU-háttérrendszer.
* A `demo` korpusz szintetikus és parányi: azt bizonyítja, hogy a lánc tanul, többet nem.
* Még nincs lemérve Windowson, macOS-en vagy sok magos gépen – fent leírtuk, hogyan segíthetsz.
* A GERMEN Studiót valódi böngészőben csak Linuxon és csak Chromiumban próbálták ki; Windowson és
  macOS-en a tesztjei átmennek a CI-ben, de az ablakát ott még senki nem használta, és
  képernyőolvasóval sem használta még senki.

Lásd a [CHANGELOG.md](CHANGELOG.md), a [docs/PLAN.md](docs/PLAN.md), a
[docs/AUDIT.md](docs/AUDIT.md) és a [docs/STUDIO.md](docs/STUDIO.md) fájlt. A hozzájárulásokat
szívesen fogadjuk ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Licenc

MIT – lásd a [LICENSE](LICENSE) fájlt. Copyright © 2026 Alexandru-Ionuț Chiuță (Alexio).
