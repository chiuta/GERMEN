# GERMEN

**Treine do zero um pequeno modelo de linguagem, em Rust puro, numa CPU comum.**

Sem Python, sem CUDA, sem toolchain de C ou C++, sem conta, sem acesso à rede, sem telemetria. Um
único executável, um arquivo de texto, um comando. Construído sobre o [Burn](https://burn.dev)
0.22 com o seu backend de CPU `flex`, escrito inteiramente em Rust.

> Idiomas: [English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · [Español](README.es.md) · **Português** · [Deutsch](README.de.md) · [Magyar](README.hu.md)

## O que é — e o que não é

O GERMEN treina uma rede do tipo GPT (RMSNorm, posições rotativas, camada feed-forward com ReLU ao
quadrado, sem vieses) sobre **tokens**. Por padrão, um token é um **byte**: o vocabulário são os
256 valores de um byte, de modo que qualquer texto — inclusive com os diacríticos romenos —
funciona sem preparar nada. Com `--tokenizer bpe`, o GERMEN aprende primeiro, a partir do seu
próprio texto, um vocabulário BPE em nível de byte (até 65.536 tokens); a mesma janela de contexto
passa a conter várias vezes mais texto (tipicamente de 3 a 4 vezes em prosa comum). O tokenizador
é salvo junto com o modelo.

É uma **oficina para aprender e experimentar**, e uma base para pesquisa: em poucos minutos dá para
ver uma rede passar do ruído aleatório à ortografia, à pontuação e ao estilo do seu texto. **Não**
é uma fábrica de chatbots. Um modelo de 1 a 25 milhões de parâmetros, treinado com algumas
centenas de megabytes, aprende a *forma* de um texto (palavras, fragmentos de gramática,
diagramação), não conhecimento do mundo nem conversação. Assistentes úteis exigem ordens de
grandeza mais computação do que uma CPU de mesa oferece.

## Início rápido

Você precisa do Rust 1.95 ou mais recente ([rustup.rs](https://rustup.rs); no Windows a instalação
também pede as Visual Studio Build Tools). Depois:

```bash
cargo build --release
./target/release/germen bench                      # qual é a velocidade DESTE computador?
./target/release/germen bench --scaling            # ...e como ela cresce com o número de threads?
./target/release/germen train --demo --preset micro --steps 300
./target/release/germen generate                   # texto do modelo recém-treinado
```

No Windows, use `target\release\germen.exe`. Os executáveis já compilados são anexados às versões
(releases).

Treine com o seu próprio texto — qualquer arquivo, em qualquer idioma:

```bash
germen train --data meu-corpus.txt --preset mini --tokenizer bpe --vocab 4096 --steps 5000 --out meu-modelo
germen generate --model meu-modelo --prompt "Era odată " --max-new 400
```

Pressione **Ctrl-C** durante o treino: ele termina o passo atual, salva e sai. Retome mais tarde com
o mesmo comando mais `--resume`: os pesos, o contador de passos e o estado do AdamW são restaurados,
de modo que o treino segue sem o salto da perda de uma reinicialização a frio (as janelas de texto
são sorteadas de novo).

O mesmo comando `train` executado de novo *sem* `--resume` é recusado, para que não substitua por
engano um modelo treinado; `--overwrite` o permite quando se quer mesmo recomeçar.

## Comandos

| comando | o que faz |
|---|---|
| `germen info` | processadores e nível SIMD detectados, tamanhos de modelo e memória dos pesos |
| `germen bench` | cronometra passos de treino reais para cada tamanho e extrapola o tempo para 100 M e 1 bilhão de tokens; `--scaling` repete a medição com 1, 2, 4, … threads e mostra a curva de aceleração; `--json ARQUIVO` salva os números |
| `germen demo-corpus` | escreve um corpus romeno sintético gerado a partir de uma gramática pequena (`--size-mb`) |
| `germen train` | treina do zero (`--data ARQUIVO` ou `--demo`), ou continua com `--resume` |
| `germen generate` | gera texto a partir de um modelo treinado, com um cache de chave/valor (`--temperature`, `--top-k`, `--prompt`) |
| `germen verify` | verifica um checkpoint: o tamanho e o SHA-256 de cada arquivo segundo `meta.json` (o que `sha256sum` imprime); `--recover` termina um salvamento interrompido, sem precisar de texto algum |

Opções de treino que vale conhecer: `--tokenizer byte|bpe` e `--vocab N` (o tamanho do BPE);
`--batch` e `--accum` (acúmulo de gradientes: o lote efetivo é o produto dos dois, a memória é a de
`--batch` sozinho); `--replicas N` (cópias do modelo que dividem cada micro-lote entre as threads; 0
= automático, 1 = desligado; escolha um `--batch` pelo menos igual ao número de threads);
`--grad-clip` (norma global do gradiente, 1 por padrão); `--lr`, `--warmup`, `--weight-decay`.
`germen train --help` lista todas. Acrescente `--threads N` a qualquer comando para limitar o número
de threads de trabalho (o padrão é todos os processadores lógicos). As mensagens da linha de comando
estão, por enquanto, apenas em romeno.

## Tamanhos de modelo

| nome | parâmetros | camadas × cabeças × largura | pesos (f32) |
|---|---:|---|---:|
| `nano` | 0,85 M | 4 × 4 × 128 | 3 MiB |
| `micro` | 2,76 M | 6 × 6 × 192 | 11 MiB |
| `mini` | 10,82 M | 6 × 6 × 384 | 41 MiB |
| `small` | 25,44 M | 8 × 8 × 512 | 97 MiB |
| `base` | 85,35 M | 12 × 12 × 768 | 326 MiB |

Os tamanhos valem para o vocabulário de 256 bytes; um vocabulário BPE de V tokens acrescenta
2·(V − 256)·largura parâmetros (embedding e cabeça de saída). O treino também guarda as duas
estimativas de momento do AdamW (≈ 4× os pesos) e as ativações, que crescem com `--batch` e
`--seq`.

## Quanto tempo leva?

O custo do treino é de cerca de **6 × parâmetros × tokens** operações em ponto flutuante (com
bytes, 1 MB de texto = 1 milhão de tokens). Divida pelo que a sua CPU realmente entrega e você tem
o tempo. O `germen bench` mede exatamente isso na sua máquina.

Medição de referência — uma pequena máquina virtual Linux com **2 núcleos** (AVX-512), lote 16 ×
256, vocabulário de bytes, configurações padrão (mediana de cinco execuções alternadas numa máquina
virtual ruidosa, portanto ± 10 %):

| nome | tokens/s | GFLOPS efetivos | 100 M tokens | 1 bi de tokens |
|---|---:|---:|---:|---:|
| `nano` | 8.366 | 54 | 3,3 h | 1,4 dia |
| `micro` | 3.981 | 79 | 7,0 h | 2,9 dias |
| `mini` | 1.464 | 104 | 19,0 h | 7,9 dias |
| `small` | 647 | 106 | 1,8 dia | 18 dias |

Passar de 1 para 2 threads deixou o `mini` 2,04× mais rápido aqui (733 → 1.494 tokens/s; 1,41× com
`--replicas 1`); portanto não espere que 20 núcleos deem 20×. Num único grafo, os produtos de
matrizes escalam bem (≈ 1,8× com duas threads), a atenção menos (≈ 1,2×), e o RMSNorm, os embeddings
rotativos e a atualização do AdamW quase nada; as réplicas da versão 0.3 executam essas partes lado
a lado, mas precisam de pelo menos tantas janelas por lote quanto threads. Até onde isso vai em
muitos núcleos ainda não foi medido: é para isso que existe o `germen bench --scaling`, que mostra a
curva inteira e o número de threads a partir do qual nada mais se ganha. **Rode `germen bench
--scaling --json bench.json` na sua máquina** — é a única forma de saber. Uma regra empírica para o
treino ótimo em computação é de cerca de 20 tokens por parâmetro; os modelos pequenos costumam ser
treinados com bem mais.

A geração usa um cache de chave/valor: 400 tokens com o `mini` levam cerca de 1 segundo em vez de
15 nos mesmos dois núcleos.

## Como é construído

```
crates/germen-core   a biblioteca: config, model, tokenizer, bpe, data, train, gradients,
                     generate, checkpoint, bench
crates/germen-cli    a linha de comando `germen` (clap)
```

* **Backend** — `flex` do Burn: Rust puro, produtos de matrizes pelo crate `gemm`, kernels SIMD
  (AVX2/AVX-512/SSE, NEON) escolhidos **em tempo de execução**, multithreading com rayon. Um
  executável x86-64 genérico usa, portanto, AVX2 numa CPU que o tenha.
* **Modelo** — blocos pre-norm, projeção QKV fundida, atenção causal, embeddings rotativos, cabeça
  de saída não amarrada ao embedding, inicialização no estilo GPT-2 (projeções de saída escaladas
  por `1/√(2·camadas)`). O treino usa uma atenção causal em blocos que nunca calcula a parte acima
  da diagonal (passos inteiros 8 a 28 % mais rápidos em dois núcleos, conforme o tamanho do modelo;
  mesmos valores e mesmos gradientes); a avaliação usa o kernel fundido do Burn, e a geração um
  cache de chave/valor.
* **Tokenizador** — bytes, ou um BPE em nível de byte com um pré-tokenizador no estilo GPT-2 (o
  espaço gruda na palavra seguinte, as fusões não atravessam o limite de uma palavra). O
  aprendizado do vocabulário é determinístico e usa apenas a parte de treino do texto; a
  codificação é paralela; qualquer sequência de bytes é reconstruída exatamente.
* **Treino** — AdamW (β₂ = 0,95), aquecimento linear e depois decaimento cosseno, recorte do
  gradiente pela norma global, acúmulo de gradientes, validação em janelas fixas (em nats por
  token e bits por byte), checkpoints atômicos, progresso por eventos (`Event`) para que uma
  interface gráfica possa reutilizar o mesmo laço.
* **Threads** — o comando inteiro roda como um trabalhador do seu próprio pool de threads, que
  inicia os laços paralelos sem repasses entre threads (+ 7 % para o `mini`, até + 25 % para o
  `nano`, em dois núcleos). Além disso, cada micro-lote é dividido em **réplicas de dados
  paralelas**: cópias do modelo, cada uma com o seu próprio grafo de diferenciação automática,
  executam a passada direta e a reversa em threads diferentes, e seus gradientes são somados numa
  ordem fixa, de modo que uma execução é reproduzível para um dado número de threads e de réplicas
  (de + 23 % a + 44 % em dois núcleos conforme o modelo, com um pico de memória menor; `--replicas
  1` dá o passo comum). Repartir micro-lotes entre threads que compartilhavam todas um único grafo
  foi tentado na 0.2 e descartado; veja [docs/PLAN.md](docs/PLAN.md).
* **Checkpoint** — uma pasta com `config.json`, `weights.bpk` (o formato burnpack do Burn, f32),
  `optimizer.bpk` (os momentos do AdamW), `tokenizer.json` e `meta.json`. Os checkpoints escritos
  pela versão 0.1 continuam carregando. Desde a 0.4, `meta.json` registra também o tamanho e o
  SHA-256 dos outros arquivos, verificados a cada carga: um arquivo danificado, truncado ou de outro
  salvamento é nomeado no erro; os mais antigos carregam sem verificação.
* **Testes** — 156 testes (unitários da biblioteca e da linha de comando, mais execuções completas
  do binário real), entre eles: a atenção é causal, a atenção em blocos é igual à do Burn (valores e
  gradientes), a geração com cache é igual à passagem completa, micro-lotes acumulados equivalem a
  um lote grande, as réplicas dão os gradientes de um único grafo grande e uma execução é
  reproduzível bit a bit, um passo de otimização após recarregar o estado do AdamW é igual ao de um
  treino ininterrupto, um checkpoint nunca fica escrito pela metade e pesos não finitos nunca são
  salvos, arquivos hostis de tokenizador e de configuração são recusados, toda sequência de bytes
  sobrevive à ida e volta pelo BPE. Um bit alterado num arquivo do checkpoint é detectado e o
  arquivo é nomeado, e um salvamento interrompido a qualquer momento é concluído ou descartado. A
  auditoria adversarial está em [docs/AUDIT.md](docs/AUDIT.md).

## Limites (versão 0.4)

* As somas de verificação protegem contra acidentes (um bit alterado, um arquivo truncado, arquivos
  de dois salvamentos diferentes), não contra quem também possa editar o `meta.json`; checkpoints
  escritos antes da 0.4 não podem ser verificados.
* Todo o corpus é lido para a memória (como identificadores de 16 bits); estão previstos corpora
  mapeados em memória, maiores que a RAM.
* Num único grafo, algumas operações do passo (RMSNorm, embeddings rotativos, atualização do AdamW)
  não usam bem os núcleos adicionais; as réplicas escondem isso quando `--batch` é pelo menos igual
  ao número de threads, mas até onde isso vai em muitos núcleos ainda não foi medido.
* Sem precisão mista e sem backend de GPU.
* O corpus `demo` é sintético e minúsculo: prova que a cadeia aprende, nada mais.
* Ainda não medido no Windows, no macOS nem numa máquina com muitos núcleos — acima está como
  ajudar.

Veja [CHANGELOG.md](CHANGELOG.md), [docs/PLAN.md](docs/PLAN.md) e [docs/AUDIT.md](docs/AUDIT.md).
Contribuições são bem-vindas ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Licença

MIT — veja [LICENSE](LICENSE). Copyright © 2026 Alexandru-Ionuț Chiuță (Alexio).
