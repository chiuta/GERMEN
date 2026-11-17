# GERMEN

**Entraînez un petit modèle de langage à partir de zéro, en Rust pur, sur un processeur ordinaire.**

Pas de Python, pas de CUDA, pas de chaîne d'outils C ou C++, pas de compte, pas d'accès réseau,
pas de télémétrie. Un seul exécutable, un fichier texte, une commande. Construit sur
[Burn](https://burn.dev) 0.22 avec son backend processeur `flex`, écrit entièrement en Rust.

> Langues : [English](README.md) · [Română](README.ro.md) · **Français** · [Italiano](README.it.md) · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

## Ce que c'est — et ce que ce n'est pas

GERMEN entraîne un réseau de type GPT (RMSNorm, positions rotatives, couche feed-forward avec ReLU
au carré, sans biais) sur des **tokens**. Par défaut, un token est un **octet** : le vocabulaire
compte les 256 valeurs d'un octet, si bien que n'importe quel texte — diacritiques roumains
compris — fonctionne sans aucune préparation. Avec `--tokenizer bpe`, GERMEN apprend d'abord, à
partir de votre propre texte, un vocabulaire BPE au niveau de l'octet (jusqu'à 65 536 tokens) ; la
même fenêtre de contexte contient alors plusieurs fois plus de texte (typiquement 3 à 4 fois pour
de la prose ordinaire). Le tokenizer est enregistré avec le modèle.

C'est un **atelier pour apprendre et expérimenter**, et une base pour la recherche : on voit un
réseau passer du bruit aléatoire à l'orthographe, à la ponctuation et au style de votre texte en
quelques minutes. Ce n'est **pas** une usine à chatbots. Un modèle de 1 à 25 millions de paramètres
entraîné sur quelques centaines de mégaoctets apprend la *forme* d'un texte (mots, fragments de
grammaire, mise en page), pas des connaissances sur le monde ni la conversation. Des assistants
utiles demandent des ordres de grandeur de calcul en plus de ce qu'offre un processeur de bureau.

## Démarrage rapide

Il faut Rust 1.95 ou plus récent ([rustup.rs](https://rustup.rs) ; sous Windows, l'installation
demande aussi les Visual Studio Build Tools). Ensuite :

```bash
cargo build --release
./target/release/germen bench                      # quelle est la vitesse de CET ordinateur ?
./target/release/germen bench --scaling            # ...et comment croît-elle avec le nombre de threads ?
./target/release/germen train --demo --preset micro --steps 300
./target/release/germen generate                   # du texte avec le modèle qui vient d'être entraîné
```

Sous Windows, utilisez `target\release\germen.exe`. Des exécutables précompilés sont joints aux
versions (releases).

Entraînez sur votre propre texte — n'importe quel fichier, dans n'importe quelle langue :

```bash
germen train --data mon-corpus.txt --preset mini --tokenizer bpe --vocab 4096 --steps 5000 --out mon-modele
germen generate --model mon-modele --prompt "Era odată " --max-new 400
```

Appuyez sur **Ctrl-C** pendant l'entraînement : il termine le pas en cours, enregistre et s'arrête.
Reprenez plus tard avec la même commande plus `--resume` : les poids, le compteur de pas et l'état
d'AdamW sont restaurés, l'entraînement continue donc sans la remontée de la perte d'un redémarrage
à froid (les fenêtres de texte sont tirées à nouveau au hasard).

## Commandes

| commande | ce qu'elle fait |
|---|---|
| `germen info` | processeurs et niveau SIMD détectés, tailles de modèle et mémoire des poids |
| `germen bench` | chronomètre de vrais pas d'entraînement pour chaque taille et extrapole la durée pour 100 M et 1 milliard de tokens ; `--scaling` répète la mesure avec 1, 2, 4, … threads et affiche la courbe d'accélération ; `--json FICHIER` enregistre les chiffres |
| `germen demo-corpus` | écrit un corpus roumain synthétique généré à partir d'une petite grammaire (`--size-mb`) |
| `germen train` | entraîne à partir de zéro (`--data FICHIER` ou `--demo`), ou continue avec `--resume` |
| `germen generate` | génère du texte à partir d'un modèle entraîné, avec un cache clé/valeur (`--temperature`, `--top-k`, `--prompt`) |

Options d'entraînement à connaître : `--tokenizer byte|bpe` et `--vocab N` (la taille du BPE) ;
`--batch` et `--accum` (accumulation de gradients : le lot effectif est leur produit, la mémoire est
celle de `--batch` seul) ; `--replicas N` (copies du modèle qui se partagent chaque micro-lot entre
les threads ; 0 = automatique, 1 = désactivé ; choisissez un `--batch` au moins égal au nombre de
threads) ; `--grad-clip` (norme globale du gradient, 1 par défaut) ; `--lr`, `--warmup`,
`--weight-decay`. `germen train --help` les liste toutes. Ajoutez `--threads N` à n'importe quelle
commande pour limiter le nombre de threads de travail (par défaut : tous les processeurs logiques).
Les messages de la ligne de commande sont pour l'instant uniquement en roumain.

## Tailles de modèle

| nom | paramètres | couches × têtes × largeur | poids (f32) |
|---|---:|---|---:|
| `nano` | 0,85 M | 4 × 4 × 128 | 3 Mio |
| `micro` | 2,76 M | 6 × 6 × 192 | 11 Mio |
| `mini` | 10,82 M | 6 × 6 × 384 | 41 Mio |
| `small` | 25,44 M | 8 × 8 × 512 | 97 Mio |
| `base` | 85,35 M | 12 × 12 × 768 | 326 Mio |

Ces tailles valent pour le vocabulaire de 256 octets ; un vocabulaire BPE de V tokens ajoute
2·(V − 256)·largeur paramètres (embedding et tête de sortie). L'entraînement garde aussi les deux
estimations de moment d'AdamW (≈ 4× les poids) et les activations, qui croissent avec `--batch` et
`--seq`.

## Combien de temps cela prend-il ?

Le coût de l'entraînement est d'environ **6 × paramètres × tokens** opérations en virgule
flottante (avec des octets, 1 Mo de texte = 1 million de tokens). Divisez par ce que votre
processeur fournit réellement et vous avez la durée. `germen bench` mesure exactement cela sur
votre machine.

Mesure de référence — une petite machine virtuelle Linux à **2 cœurs** (AVX-512), lot 16 × 256,
vocabulaire d'octets, réglages par défaut (médiane de cinq exécutions entrelacées sur une machine
virtuelle bruitée, donc ± 10 %) :

| nom | tokens/s | GFLOPS effectifs | 100 M tokens | 1 Md tokens |
|---|---:|---:|---:|---:|
| `nano` | 8 366 | 54 | 3,3 h | 1,4 j |
| `micro` | 3 981 | 79 | 7,0 h | 2,9 j |
| `mini` | 1 464 | 104 | 19,0 h | 7,9 j |
| `small` | 647 | 106 | 1,8 j | 18 j |

Passer de 1 à 2 threads a rendu `mini` 2,04× plus rapide ici (733 → 1 494 tokens/s ; 1,41× avec
`--replicas 1`) ; n'attendez donc pas de 20 cœurs un gain de 20×. Dans un seul graphe, les produits
de matrices passent bien à l'échelle (≈ 1,8× sur deux threads), l'attention un peu moins (≈ 1,2×),
et RMSNorm, les embeddings rotatifs et la mise à jour d'AdamW presque pas du tout ; les répliques de
la version 0.3 exécutent ces parties côte à côte, mais il leur faut au moins autant de fenêtres par
lot que de threads. Jusqu'où cela va sur de nombreux cœurs n'a pas été mesuré : c'est à cela que
sert `germen bench --scaling`, qui affiche toute la courbe et le nombre de threads au-delà duquel on
ne gagne plus rien. **Lancez `germen bench --scaling --json bench.json` sur votre machine** — c'est
le seul moyen de savoir. Une règle empirique pour un entraînement optimal en calcul est d'environ 20
tokens par paramètre ; les petits modèles sont souvent entraînés sur beaucoup plus.

La génération utilise un cache clé/valeur : 400 tokens avec `mini` prennent environ 1 seconde au
lieu de 15 sur les deux mêmes cœurs.

## Comment c'est construit

```
crates/germen-core   la bibliothèque : config, model, tokenizer, bpe, data, train, gradients,
                     generate, checkpoint, bench
crates/germen-cli    la ligne de commande `germen` (clap)
```

* **Backend** — `flex` de Burn : Rust pur, produits de matrices via le crate `gemm`, noyaux SIMD
  (AVX2/AVX-512/SSE, NEON) choisis **à l'exécution**, multithreading avec rayon. Un exécutable
  x86-64 générique utilise donc AVX2 sur un processeur qui l'a.
* **Modèle** — blocs pré-norm, projection QKV fusionnée, attention causale, embeddings rotatifs,
  tête de sortie non liée à l'embedding, initialisation de type GPT-2 (projections de sortie
  mises à l'échelle par `1/√(2·couches)`). L'entraînement utilise une attention causale par blocs
  qui ne calcule jamais la partie au-dessus de la diagonale (pas entiers 8 à 28 % plus rapides sur
  deux cœurs selon la taille du modèle ; mêmes valeurs et mêmes gradients) ; l'évaluation utilise
  le noyau fusionné de Burn, et la génération un cache clé/valeur.
* **Tokenizer** — octets, ou un BPE au niveau de l'octet avec un pré-tokenizer de type GPT-2 (une
  espace se colle au mot qui suit, les fusions ne franchissent pas la limite d'un mot).
  L'apprentissage du vocabulaire est déterministe et n'utilise que la partie d'entraînement du
  texte ; l'encodage est parallèle ; toute suite d'octets est restituée à l'identique.
* **Entraînement** — AdamW (β₂ = 0,95), échauffement linéaire puis décroissance cosinus, écrêtage
  du gradient par norme globale, accumulation de gradients, validation sur des fenêtres fixes
  (en nats par token et en bits par octet), points de contrôle atomiques, progression par
  événements (`Event`) pour qu'une interface graphique puisse réutiliser la même boucle.
* **Threads** — toute la commande s'exécute comme un travailleur de son propre pool de threads, ce
  qui lance les boucles parallèles sans passage de relais entre threads (+ 7 % pour `mini`, jusqu'à
  + 25 % pour `nano`, sur deux cœurs). De plus, chaque micro-lot est réparti en **répliques de
  données parallèles** : des copies du modèle, chacune avec son propre graphe de différentiation
  automatique, exécutent l'aller et le retour sur des threads différents, et leurs gradients sont
  additionnés dans un ordre fixe, de sorte qu'une exécution est reproductible pour un nombre donné
  de threads et de répliques (de + 23 % à + 44 % sur deux cœurs selon le modèle, avec un pic de
  mémoire plus bas ; `--replicas 1` donne le pas ordinaire). Répartir des micro-lots sur des threads
  partageant tous un même graphe a été essayé en 0.2 et rejeté ; voir [docs/PLAN.md](docs/PLAN.md).
* **Point de contrôle** — un dossier avec `config.json`, `weights.bpk` (le format burnpack de Burn,
  f32), `optimizer.bpk` (les moments d'AdamW), `tokenizer.json` et `meta.json`. Les points de
  contrôle écrits par la version 0.1 se chargent toujours.
* **Tests** — 117 tests (unitaires de la bibliothèque et de la ligne de commande, plus des
  exécutions complètes du vrai binaire), dont : l'attention est causale, l'attention par blocs égale
  celle de Burn (valeurs et gradients), la génération avec cache égale la passe complète, des
  micro-lots accumulés valent un grand lot, les répliques donnent les gradients d'un seul grand
  graphe et une exécution est reproductible au bit près, un pas d'optimisation après rechargement de
  l'état d'AdamW égale celui d'un entraînement ininterrompu, un point de contrôle n'est jamais
  laissé à moitié écrit et des poids non finis ne sont jamais enregistrés, les fichiers hostiles de
  tokenizer et de configuration sont refusés, toute suite d'octets survit à l'aller-retour par le
  BPE. L'audit adversarial est dans [docs/AUDIT.md](docs/AUDIT.md).

## Limites (version 0.3)

* Tout le corpus est lu en mémoire (sous forme d'identifiants de 16 bits) ; des corpus mappés en
  mémoire, plus grands que la RAM, sont prévus.
* Dans un seul graphe, quelques opérations du pas (RMSNorm, embeddings rotatifs, mise à jour
  d'AdamW) n'exploitent pas bien les cœurs supplémentaires ; les répliques masquent cela quand
  `--batch` est au moins égal au nombre de threads, mais jusqu'où cela va sur de nombreux cœurs
  n'est pas encore mesuré.
* Pas de précision mixte ni de backend GPU.
* Le corpus `demo` est synthétique et minuscule : il prouve que la chaîne apprend, rien de plus.
* Pas encore mesuré sous Windows, macOS ni sur une machine à nombreux cœurs — voir plus haut
  comment aider.

Voir [CHANGELOG.md](CHANGELOG.md), [docs/PLAN.md](docs/PLAN.md) et [docs/AUDIT.md](docs/AUDIT.md).
Les contributions sont les bienvenues ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Licence

MIT — voir [LICENSE](LICENSE). Copyright © 2026 Alexandru-Ionuț Chiuță (Alexio).
