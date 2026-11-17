# GERMEN

**Entrena desde cero un pequeño modelo de lenguaje, en Rust puro, en una CPU corriente.**

Sin Python, sin CUDA, sin cadena de herramientas de C o C++, sin cuenta, sin acceso a la red, sin
telemetría. Un solo ejecutable, un archivo de texto, un comando. Construido sobre
[Burn](https://burn.dev) 0.22 con su backend de CPU `flex`, escrito íntegramente en Rust.

> Idiomas: [English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · **Español** · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

## Qué es — y qué no es

GERMEN entrena una red de tipo GPT (RMSNorm, posiciones rotatorias, capa feed-forward con ReLU al
cuadrado, sin sesgos) sobre **tokens**. Por defecto un token es un **byte**: el vocabulario son los
256 valores de un byte, así que cualquier texto — con los diacríticos rumanos incluidos — funciona
sin preparar nada. Con `--tokenizer bpe`, GERMEN aprende primero, a partir de su propio texto, un
vocabulario BPE a nivel de byte (hasta 65.536 tokens); la misma ventana de contexto contiene
entonces varias veces más texto (normalmente de 3 a 4 veces en prosa corriente). El tokenizador se
guarda junto con el modelo.

Es un **taller para aprender y experimentar**, y una base para la investigación: en pocos minutos
se ve cómo una red pasa del ruido aleatorio a la ortografía, la puntuación y el estilo de su
texto. **No** es una fábrica de chatbots. Un modelo de 1 a 25 millones de parámetros entrenado con
unos cientos de megabytes aprende la *forma* de un texto (palabras, fragmentos de gramática,
maquetación), no conocimientos del mundo ni conversación. Los asistentes útiles necesitan órdenes
de magnitud más cómputo del que ofrece una CPU de escritorio.

## Inicio rápido

Necesita Rust 1.95 o posterior ([rustup.rs](https://rustup.rs); en Windows la instalación también
pide las Visual Studio Build Tools). Después:

```bash
cargo build --release
./target/release/germen bench                      # ¿qué tan rápido es ESTE ordenador?
./target/release/germen bench --scaling            # ...¿y cómo crece con el número de hilos?
./target/release/germen train --demo --preset micro --steps 300
./target/release/germen generate                   # texto del modelo recién entrenado
```

En Windows use `target\release\germen.exe`. Los ejecutables ya compilados se adjuntan a las
versiones (releases).

Entrene con su propio texto — cualquier archivo, en cualquier idioma:

```bash
germen train --data mi-corpus.txt --preset mini --tokenizer bpe --vocab 4096 --steps 5000 --out mi-modelo
germen generate --model mi-modelo --prompt "Era odată " --max-new 400
```

Pulse **Ctrl-C** durante el entrenamiento: termina el paso en curso, guarda y sale. Reanude más
tarde con el mismo comando más `--resume`: se restauran los pesos, el contador de pasos y el
estado de AdamW, de modo que el entrenamiento sigue sin el repunte de la pérdida de un reinicio en
frío (las ventanas de texto se vuelven a sortear).

## Comandos

| comando | qué hace |
|---|---|
| `germen info` | procesadores y nivel SIMD detectados, tamaños de modelo y memoria de los pesos |
| `germen bench` | cronometra pasos de entrenamiento reales para cada tamaño y extrapola el tiempo para 100 M y 1000 millones de tokens; `--scaling` repite la medición con 1, 2, 4, … hilos y muestra la curva de aceleración; `--json ARCHIVO` guarda las cifras |
| `germen demo-corpus` | escribe un corpus rumano sintético generado a partir de una gramática pequeña (`--size-mb`) |
| `germen train` | entrena desde cero (`--data ARCHIVO` o `--demo`), o continúa con `--resume` |
| `germen generate` | genera texto con un modelo entrenado, con una caché de claves y valores (`--temperature`, `--top-k`, `--prompt`) |

Opciones de entrenamiento que conviene conocer: `--tokenizer byte|bpe` y `--vocab N` (el tamaño
del BPE); `--batch` y `--accum` (acumulación de gradientes: el lote efectivo es su producto, la
memoria es la de `--batch` solo); `--grad-clip` (norma global del gradiente, 1 por defecto);
`--lr`, `--warmup`, `--weight-decay`. `germen train --help` las lista todas. Añada `--threads N`
a cualquier comando para limitar el número de hilos de trabajo (por defecto, todos los
procesadores lógicos). Los mensajes de la línea de comandos están por ahora solo en rumano.

## Tamaños de modelo

| nombre | parámetros | capas × cabezas × anchura | pesos (f32) |
|---|---:|---|---:|
| `nano` | 0,85 M | 4 × 4 × 128 | 3 MiB |
| `micro` | 2,76 M | 6 × 6 × 192 | 11 MiB |
| `mini` | 10,82 M | 6 × 6 × 384 | 41 MiB |
| `small` | 25,44 M | 8 × 8 × 512 | 97 MiB |
| `base` | 85,35 M | 12 × 12 × 768 | 326 MiB |

Los tamaños corresponden al vocabulario de 256 bytes; un vocabulario BPE de V tokens añade
2·(V − 256)·anchura parámetros (embedding y cabeza de salida). El entrenamiento guarda además las
dos estimaciones de momento de AdamW (≈ 4× los pesos) y las activaciones, que crecen con `--batch`
y `--seq`.

## ¿Cuánto tarda?

El coste del entrenamiento es de unas **6 × parámetros × tokens** operaciones en coma flotante
(con bytes, 1 MB de texto = 1 millón de tokens). Divida por lo que su CPU entrega de verdad y
tendrá el tiempo. `germen bench` mide exactamente eso en su máquina.

Medición de referencia — una pequeña máquina virtual Linux con **2 núcleos** (AVX-512), lote
16 × 256, vocabulario de bytes (una sola ejecución en una máquina virtual ruidosa, así que ± 5 %):

| nombre | tokens/s | GFLOPS efectivos | 100 M tokens | 1000 M tokens |
|---|---:|---:|---:|---:|
| `nano` | 8.261 | 54 | 3,4 h | 1,4 días |
| `micro` | 3.180 | 63 | 8,7 h | 3,6 días |
| `mini` | 1.260 | 90 | 22,0 h | 9,2 días |
| `small` | 545 | 90 | 2,1 días | 21 días |

Pasar de 1 a 2 hilos hizo `mini` 1,51× más rápido aquí (825 → 1.247 tokens/s), así que no espere
que 20 núcleos den 20×. Los productos de matrices escalan bien (≈ 1,8× con dos hilos); la atención
menos (≈ 1,2×), y RMSNorm, los embeddings rotatorios y la actualización de AdamW casi nada, por lo
que en una máquina con muchos núcleos se convierten en el cuello de botella (ley de Amdahl). Para
eso existe `germen bench --scaling`: imprime toda la curva y el número de hilos a partir del cual
ya no se gana nada. **Ejecute `germen bench --scaling --json bench.json` en su máquina** — es la
única forma de saberlo. Una regla empírica para el entrenamiento óptimo en cómputo es de unos 20
tokens por parámetro; los modelos pequeños suelen entrenarse con mucho más.

La generación usa una caché de claves y valores: 400 tokens con `mini` tardan aproximadamente 1
segundo en lugar de 15 en los mismos dos núcleos.

## Cómo está construido

```
crates/germen-core   la biblioteca: config, model, tokenizer, bpe, data, train, gradients,
                     generate, checkpoint, bench
crates/germen-cli    la línea de comandos `germen` (clap)
```

* **Backend** — `flex` de Burn: Rust puro, productos de matrices mediante el crate `gemm`, núcleos
  SIMD (AVX2/AVX-512/SSE, NEON) elegidos **en tiempo de ejecución**, multihilo con rayon. Un
  ejecutable x86-64 genérico usa por tanto AVX2 en una CPU que lo tiene.
* **Modelo** — bloques pre-norm, proyección QKV fusionada, atención causal, embeddings rotatorios,
  cabeza de salida no ligada al embedding, inicialización al estilo GPT-2 (proyecciones de salida
  escaladas por `1/√(2·capas)`). El entrenamiento usa una atención causal por bloques que nunca
  calcula la parte por encima de la diagonal (pasos completos entre un 8 y un 28 % más rápidos con
  dos núcleos, según el tamaño del modelo; mismos valores y mismos gradientes); la evaluación usa
  el núcleo fusionado de Burn, y la generación una caché de claves y valores.
* **Tokenizador** — bytes, o un BPE a nivel de byte con un pretokenizador al estilo GPT-2 (el
  espacio se pega a la palabra siguiente, las fusiones no cruzan el límite de una palabra). El
  aprendizaje del vocabulario es determinista y solo usa la parte de entrenamiento del texto; la
  codificación es paralela; cualquier secuencia de bytes se reconstruye exactamente.
* **Entrenamiento** — AdamW (β₂ = 0,95), calentamiento lineal y después decaimiento coseno,
  recorte del gradiente por norma global, acumulación de gradientes, validación en ventanas fijas
  (en nats por token y bits por byte), puntos de control atómicos, progreso mediante eventos
  (`Event`) para que una interfaz gráfica pueda reutilizar el mismo bucle.
* **Hilos** — todo el comando se ejecuta como un trabajador de su propio grupo de hilos, que lanza
  los bucles paralelos sin traspasos entre hilos (+ 7 % para `mini`, hasta + 25 % para `nano`, con
  dos núcleos). Repartir micro-lotes entre varios hilos que manejan Burn a la vez se probó y se
  descartó; véase [docs/PLAN.md](docs/PLAN.md).
* **Punto de control** — una carpeta con `config.json`, `weights.bpk` (el formato burnpack de
  Burn, f32), `optimizer.bpk` (los momentos de AdamW), `tokenizer.json` y `meta.json`. Los puntos
  de control escritos por la versión 0.1 se siguen cargando.
* **Pruebas** — 70 pruebas unitarias, entre ellas: la atención es causal, la atención por bloques
  iguala la de Burn (valores y gradientes), la generación con caché iguala el paso completo, los
  micro-lotes acumulados equivalen a un lote grande, un paso de optimización tras recargar el
  estado de AdamW iguala el de un entrenamiento ininterrumpido, toda secuencia de bytes sobrevive
  al viaje de ida y vuelta por el BPE.

## Límites (versión 0.2)

* Todo el corpus se lee en memoria (como identificadores de 16 bits); están previstos corpus
  mapeados en memoria, mayores que la RAM.
* Algunas operaciones del paso (RMSNorm, embeddings rotatorios, actualización de AdamW) todavía no
  aprovechan bien los núcleos adicionales; paralelizarlas es la principal tarea abierta de
  rendimiento.
* Sin precisión mixta y sin backend de GPU.
* El corpus `demo` es sintético y diminuto: demuestra que la cadena aprende, nada más.
* Aún sin medir en Windows, macOS ni en una máquina con muchos núcleos — arriba se explica cómo
  ayudar.

Véanse [CHANGELOG.md](CHANGELOG.md) y [docs/PLAN.md](docs/PLAN.md). Las contribuciones son
bienvenidas ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Licencia

MIT — véase [LICENSE](LICENSE). Copyright © 2026 Alexandru-Ionuț Chiuță (Alexio).
