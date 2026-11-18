# GERMEN

**Entrena desde cero un pequeño modelo de lenguaje, en Rust puro, en una CPU corriente.**

Sin Python, sin CUDA, sin cadena de herramientas de C o C++, sin cuenta, sin acceso a la red, sin
telemetría. Un solo ejecutable, un archivo de texto, un comando. Construido sobre
[Burn](https://burn.dev) 0.22 con su backend de CPU `flex`, escrito íntegramente en Rust.

> Idiomas: [English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · **Español** · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

🌐 **Sitio web:** <https://germen.chiuta.ro> · [demo de Studio en línea](https://germen.chiuta.ro/demo.html)

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
tarde con el mismo comando más `--resume`: se restauran los pesos, el contador de pasos y el estado
de AdamW, de modo que el entrenamiento sigue sin el repunte de la pérdida de un reinicio en frío
(las ventanas de texto se vuelven a sortear).

El mismo comando `train` lanzado de nuevo *sin* `--resume` se rechaza, para que no sustituya por
error un modelo entrenado; `--overwrite` lo permite cuando de verdad se quiere empezar de cero.

## La ventana: GERMEN Studio

¿Prefiere una ventana a una línea de comandos?

```bash
germen ui
```

abre **GERMEN Studio**: elija un texto, pulse **Empezar a entrenar** y vea cómo aprende un modelo —
la curva de la pérdida, la velocidad y muestras de lo que escribe la red a medida que mejora. La
misma ventana escribe texto con sus modelos, comprueba y repara sus archivos y mide la velocidad de
su ordenador, con el mismo código que los comandos de arriba. Está en ocho idiomas, en claro y
oscuro, cabe en una ventana del tamaño de un teléfono y en una pantalla ancha, y está pensada para
usarse con el teclado y con un lector de pantalla.

Es un pequeño servidor que solo escucha en `127.0.0.1`, y una página que sirve: nada que instalar,
nada que descargar, nada que se envíe a ningún sitio. La dirección que imprime contiene un secreto
válido solo para esta ejecución, de modo que solo la ventana abierta con ella (o usted) puede darle
órdenes. Los modelos y los textos se guardan en `~/GERMEN` (`--workspace` lo cambia); `--no-open`
solo imprime la dirección. Cómo está protegido, y el resto: [docs/STUDIO.md](docs/STUDIO.md).

## Comandos

| comando | qué hace |
|---|---|
| `germen info` | procesadores y nivel SIMD detectados, tamaños de modelo y memoria de los pesos |
| `germen bench` | cronometra pasos de entrenamiento reales para cada tamaño y extrapola el tiempo para 100 M y 1000 millones de tokens; `--scaling` repite la medición con 1, 2, 4, … hilos y muestra la curva de aceleración; `--json ARCHIVO` guarda las cifras |
| `germen demo-corpus` | escribe un corpus rumano sintético generado a partir de una gramática pequeña (`--size-mb`) |
| `germen train` | entrena desde cero (`--data ARCHIVO` o `--demo`), o continúa con `--resume` |
| `germen generate` | genera texto con un modelo entrenado, con una caché de claves y valores (`--temperature`, `--top-k`, `--prompt`) |
| `germen verify` | comprueba un punto de control: el tamaño y el SHA-256 de cada archivo según `meta.json` (lo que imprime `sha256sum`); `--recover` termina un guardado interrumpido, sin necesitar ningún texto |
| `germen ui` | abre GERMEN Studio, la interfaz gráfica: un servidor solo en `127.0.0.1` y una ventana propia (`--workspace`, `--port`, `--no-open`); véase [docs/STUDIO.md](docs/STUDIO.md) |

Opciones de entrenamiento que conviene conocer: `--tokenizer byte|bpe` y `--vocab N` (el tamaño del
BPE); `--batch` y `--accum` (acumulación de gradientes: el lote efectivo es su producto, la memoria
es la de `--batch` solo); `--replicas N` (copias del modelo que se reparten cada micro-lote entre
los hilos; 0 = automático, 1 = desactivado; elija un `--batch` al menos igual al número de hilos);
`--grad-clip` (norma global del gradiente, 1 por defecto); `--lr`, `--warmup`, `--weight-decay`.
`germen train --help` las lista todas. Añada `--threads N` a cualquier comando para limitar el
número de hilos de trabajo (por defecto, todos los procesadores lógicos). Los mensajes de la línea
de comandos están por ahora solo en rumano.

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

Medición de referencia — una pequeña máquina virtual Linux con **2 núcleos** (AVX-512), lote 16 ×
256, vocabulario de bytes, ajustes por defecto (mediana de cinco ejecuciones alternadas en una
máquina virtual ruidosa, así que ± 10 %):

| nombre | tokens/s | GFLOPS efectivos | 100 M tokens | 1000 M tokens |
|---|---:|---:|---:|---:|
| `nano` | 8.366 | 54 | 3,3 h | 1,4 días |
| `micro` | 3.981 | 79 | 7,0 h | 2,9 días |
| `mini` | 1.464 | 104 | 19,0 h | 7,9 días |
| `small` | 647 | 106 | 1,8 días | 18 días |

Pasar de 1 a 2 hilos hizo `mini` 2,04× más rápido aquí (733 → 1.494 tokens/s; 1,41× con `--replicas
1`), así que no espere que 20 núcleos den 20×. En un solo grafo los productos de matrices escalan
bien (≈ 1,8× con dos hilos), la atención menos (≈ 1,2×), y RMSNorm, los embeddings rotatorios y la
actualización de AdamW casi nada; las réplicas de la versión 0.3 ejecutan esas partes una al lado de
otra, pero necesitan al menos tantas ventanas por lote como hilos. Hasta dónde llega eso con muchos
núcleos no se ha medido: para eso existe `germen bench --scaling`, que imprime toda la curva y el
número de hilos a partir del cual ya no se gana nada. **Ejecute `germen bench --scaling --json
bench.json` en su máquina** — es la única forma de saberlo. Una regla empírica para el entrenamiento
óptimo en cómputo es de unos 20 tokens por parámetro; los modelos pequeños suelen entrenarse con
mucho más.

La generación usa una caché de claves y valores: 400 tokens con `mini` tardan aproximadamente 1
segundo en lugar de 15 en los mismos dos núcleos.

## Cómo está construido

```
crates/germen-core   la biblioteca: config, model, tokenizer, bpe, data, train, gradients,
                     generate, checkpoint, bench
crates/germen-studio GERMEN Studio: el servidor y la página tras `germen ui` (solo la biblioteca estándar)
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
  dos núcleos). Además, cada micro-lote se reparte en **réplicas de datos en paralelo**: copias del
  modelo, cada una con su propio grafo de diferenciación automática, ejecutan la pasada hacia
  delante y hacia atrás en hilos distintos, y sus gradientes se suman en un orden fijo, de modo que
  una ejecución es reproducible para un número dado de hilos y de réplicas (de + 23 % a + 44 % con
  dos núcleos según el modelo, con un pico de memoria más bajo; `--replicas 1` da el paso
  ordinario). Repartir micro-lotes entre hilos que compartían todos un mismo grafo se probó en la
  0.2 y se descartó; véase [docs/PLAN.md](docs/PLAN.md).
* **Punto de control** — una carpeta con `config.json`, `weights.bpk` (el formato burnpack de Burn,
  f32), `optimizer.bpk` (los momentos de AdamW), `tokenizer.json` y `meta.json`. Los puntos de
  control escritos por la versión 0.1 se siguen cargando. Desde la 0.4, `meta.json` registra también
  el tamaño y el SHA-256 de los demás archivos, comprobados en cada carga: un archivo dañado,
  truncado o de otro guardado se nombra en el error; los más antiguos se cargan sin comprobación.
* **Studio** — `germen ui` inicia un pequeño servidor HTTP escrito solo con la biblioteca estándar,
  ligado a `127.0.0.1`, y sirve una página autónoma (estilo, script y ocho idiomas dentro; no carga
  nada). La página inicia el mismo código de entrenamiento, generación y medición que la línea de
  comandos y lo sigue mediante eventos. Quién puede hablar con el servidor: un secreto en la
  dirección (tras el `#`, que un navegador nunca envía), las cabeceras `Host` y `Origin` y una
  Content-Security-Policy que deja ejecutar exactamente un script; cada una de estas puertas la
  atacan pruebas sobre un socket real y la apagan por turno 18 pruebas por mutación, que las
  pruebas deben notar ([docs/STUDIO.md](docs/STUDIO.md)).
* **Pruebas** — 247 pruebas (unitarias de la biblioteca, de la línea de comandos y de la interfaz
  gráfica, más ejecuciones completas del binario real y del servidor por un socket), entre ellas: la
  atención es causal, la atención por bloques iguala la de Burn (valores y gradientes), la
  generación con caché iguala el paso completo, los micro-lotes acumulados equivalen a un lote
  grande, las réplicas dan los gradientes de un solo grafo grande y una ejecución es reproducible
  bit a bit, un paso de optimización tras recargar el estado de AdamW iguala el de un entrenamiento
  ininterrumpido, un punto de control nunca queda a medio escribir y los pesos no finitos nunca se
  guardan, los archivos hostiles de tokenizador y de configuración se rechazan, toda secuencia de
  bytes sobrevive al viaje de ida y vuelta por el BPE. Un bit cambiado en un archivo del punto de
  control se detecta y se nombra el archivo, y un guardado interrumpido en cualquier momento se
  completa o se descarta. La auditoría adversarial está en [docs/AUDIT.md](docs/AUDIT.md).

## Límites (versión 0.5)

* Las sumas de comprobación protegen de los accidentes (un bit cambiado, un archivo truncado,
  archivos de dos guardados distintos), no de quien también pueda editar `meta.json`; los puntos de
  control escritos antes de la 0.4 no se pueden verificar.
* Todo el corpus se lee en memoria (como identificadores de 16 bits); están previstos corpus
  mapeados en memoria, mayores que la RAM.
* En un solo grafo, algunas operaciones del paso (RMSNorm, embeddings rotatorios, actualización de
  AdamW) no aprovechan bien los núcleos adicionales; las réplicas lo ocultan cuando `--batch` es al
  menos igual al número de hilos, pero hasta dónde llega eso con muchos núcleos aún no se ha medido.
* Sin precisión mixta y sin backend de GPU.
* El corpus `demo` es sintético y diminuto: demuestra que la cadena aprende, nada más.
* Aún sin medir en Windows, macOS ni en una máquina con muchos núcleos — arriba se explica cómo
  ayudar.
* GERMEN Studio solo se ha probado en un navegador real en Linux y solo en Chromium; en Windows y
  macOS sus pruebas pasan en CI, pero nadie ha usado aún su ventana allí, y nadie lo ha usado con un
  lector de pantalla.

Véanse [CHANGELOG.md](CHANGELOG.md), [docs/PLAN.md](docs/PLAN.md), [docs/AUDIT.md](docs/AUDIT.md) y
[docs/STUDIO.md](docs/STUDIO.md). Las contribuciones son bienvenidas
([CONTRIBUTING.md](CONTRIBUTING.md)).

## Licencia

MIT — véase [LICENSE](LICENSE). Copyright © 2026 Alexandru-Ionuț Chiuță (Alexio).
