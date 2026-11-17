//! `germen` — command-line front end of GERMEN.
//!
//! Subcommands: `info`, `bench`, `demo-corpus`, `train`, `generate`.

use clap::{Args, Parser, Subcommand};
use germen_core::bench::{BenchReport, bench_model};
use germen_core::checkpoint;
use germen_core::config::{GptConfig, Preset};
use germen_core::data::{Corpus, Rng, demo_corpus};
use germen_core::generate::{SampleOptions, generate};
use germen_core::sys;
use germen_core::tokenizer::{ByteTokenizer, Utf8Stream};
use germen_core::train::{Event, TrainOptions, train};
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Parser)]
#[command(
    name = "germen",
    version,
    about = "GERMEN — antrenează de la zero un model de limbaj mic, în Rust pur, pe procesor",
    after_help = "Pași recomandați:\n  1. germen bench                     (cât de repede antrenează calculatorul ăsta?)\n  2. germen train --demo --steps 300  (primul model, în câteva minute)\n  3. germen generate                  (text din modelul antrenat)"
)]
struct Cli {
    /// Număr de fire de lucru (implicit: toate procesoarele logice)
    #[arg(long, global = true, value_name = "N")]
    threads: Option<usize>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Arată mașina detectată și dimensiunile de model disponibile
    Info,
    /// Măsoară viteza reală de antrenare pe această mașină
    Bench(BenchArgs),
    /// Scrie un corpus sintetic de demonstrație (proză românească dintr-o gramatică mică)
    DemoCorpus(DemoArgs),
    /// Antrenează un model de la zero (sau continuă unul existent cu --resume)
    Train(TrainArgs),
    /// Generează text cu un model antrenat
    Generate(GenerateArgs),
}

#[derive(Args)]
struct BenchArgs {
    /// Dimensiunea modelului: nano, micro, mini, small, base sau all (= primele patru)
    #[arg(long, default_value = "all")]
    preset: String,
    /// Ferestre de text pe pas
    #[arg(long, default_value_t = 16)]
    batch: usize,
    /// Lungimea contextului în tokeni (octeți)
    #[arg(long, default_value_t = 256)]
    seq: usize,
    /// Pași cronometrați pentru fiecare model
    #[arg(long, default_value_t = 4)]
    steps: usize,
    /// Pași de încălzire, necronometrați
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    /// Salvează rezultatele și în acest fișier JSON (util pentru a le trimite mai departe)
    #[arg(long, value_name = "FIȘIER")]
    json: Option<PathBuf>,
}

#[derive(Args)]
struct DemoArgs {
    /// Fișierul de scris
    #[arg(long, default_value = "demo.txt")]
    out: PathBuf,
    /// Dimensiunea aproximativă în MiB
    #[arg(long, default_value_t = 2.0)]
    size_mb: f64,
    /// Sămânța generatorului
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Args)]
struct TrainArgs {
    /// Fișier text (UTF-8 sau orice octeți) cu care se antrenează modelul
    #[arg(long, value_name = "FIȘIER", conflicts_with = "demo")]
    data: Option<PathBuf>,
    /// Folosește corpusul sintetic încorporat în loc de --data
    #[arg(long)]
    demo: bool,
    /// Dimensiunea corpusului de demonstrație, în MiB
    #[arg(long, default_value_t = 2.0)]
    demo_mb: f64,
    /// Directorul în care se scriu punctele de control
    #[arg(long, default_value = "germen-model")]
    out: PathBuf,
    /// Dimensiunea modelului: nano, micro, mini, small, base
    #[arg(long, default_value = "mini")]
    preset: String,
    /// Lungimea contextului (octeți). Implicit: 256, iar la --resume cea a modelului salvat
    #[arg(long)]
    seq: Option<usize>,
    /// Ferestre de text pe pas
    #[arg(long, default_value_t = 16)]
    batch: usize,
    /// Numărul total de pași de antrenare
    #[arg(long, default_value_t = 2000)]
    steps: usize,
    /// Rata de învățare maximă
    #[arg(long, default_value_t = 6e-4)]
    lr: f64,
    /// Rata finală ca fracție din cea maximă (descreștere cosinus)
    #[arg(long, default_value_t = 0.1)]
    min_lr_ratio: f64,
    /// Pași de încălzire a ratei de învățare
    #[arg(long, default_value_t = 100)]
    warmup: usize,
    /// Descompunerea ponderilor (AdamW)
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f32,
    /// Limita normei gradientului pe fiecare tensor (0 = oprit)
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f32,
    /// Validează la fiecare atâția pași (0 = oprit)
    #[arg(long, default_value_t = 100)]
    eval_every: usize,
    /// Loturi folosite la o validare
    #[arg(long, default_value_t = 8)]
    eval_batches: usize,
    /// Salvează un punct de control la fiecare atâția pași (0 = doar la final)
    #[arg(long, default_value_t = 500)]
    save_every: usize,
    /// Afișează progresul la fiecare atâția pași
    #[arg(long, default_value_t = 10)]
    log_every: usize,
    /// Generează un eșantion de text la fiecare atâția pași (0 = doar la final)
    #[arg(long, default_value_t = 0)]
    sample_every: usize,
    /// Sămânța generatoarelor aleatoare
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Fracțiunea de la sfârșitul fișierului păstrată pentru validare
    #[arg(long, default_value_t = 0.05)]
    val_fraction: f64,
    /// Continuă din punctul de control din --out
    #[arg(long)]
    resume: bool,
}

#[derive(Args)]
struct GenerateArgs {
    /// Directorul modelului antrenat
    #[arg(long, default_value = "germen-model")]
    model: PathBuf,
    /// Începutul textului (gol = pornește de la un rând nou)
    #[arg(long, default_value = "")]
    prompt: String,
    /// Câți octeți noi să genereze
    #[arg(long, default_value_t = 300)]
    max_new: usize,
    /// Temperatura: 0 = alegerea cea mai probabilă, mai mare = mai variat
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
    /// Se aleg doar primele k variante (0 = oprit)
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    /// Câți octeți recenți vede modelul la fiecare pas
    #[arg(long, default_value_t = 128)]
    window: usize,
    /// Sămânța (implicit: aleatoare)
    #[arg(long)]
    seed: Option<u64>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Some(n) = cli.threads {
        if n == 0 {
            eprintln!("eroare: --threads trebuie să fie cel puțin 1");
            return ExitCode::from(2);
        }
        if !sys::set_worker_threads(n) {
            eprintln!("avertisment: nu am putut fixa numărul de fire");
        }
    }
    let result = match cli.command {
        Command::Info => cmd_info(),
        Command::Bench(a) => cmd_bench(a),
        Command::DemoCorpus(a) => cmd_demo(a),
        Command::Train(a) => cmd_train(a),
        Command::Generate(a) => cmd_generate(a),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("eroare: {e}");
            ExitCode::FAILURE
        }
    }
}

fn parse_preset(name: &str) -> Result<Preset, String> {
    Preset::parse(name).ok_or_else(|| {
        format!("dimensiune necunoscută „{name}” (alegeți: nano, micro, mini, small, base)")
    })
}

fn cmd_info() -> Result<(), String> {
    println!("GERMEN {}", germen_core::VERSION);
    println!(
        "Mașina: {}, {} procesoare logice, {} fire de lucru, {}",
        sys::platform(),
        sys::logical_cpus(),
        sys::worker_threads(),
        sys::simd_summary()
    );
    println!();
    println!(
        "{:<7} {:>9} {:>7} {:>5} {:>6} {:>13} {:>16}",
        "model", "parametri", "straturi", "capete", "d_model", "ponderi (f32)", "ponderi+AdamW"
    );
    for p in [
        Preset::Nano,
        Preset::Micro,
        Preset::Mini,
        Preset::Small,
        Preset::Base,
    ] {
        let c = p.config(256);
        let mib = |bytes: f64| format!("{:.0} MiB", bytes / (1024.0 * 1024.0));
        println!(
            "{:<7} {:>8.2}M {:>7} {:>6} {:>7} {:>13} {:>16}",
            p.name(),
            c.num_params() as f64 / 1e6,
            c.n_layer,
            c.n_head,
            c.d_model,
            mib(c.num_params() as f64 * 4.0),
            mib(c.num_params() as f64 * 16.0),
        );
    }
    println!();
    println!(
        "Un model care nu a învățat nimic are pierderea ln(256) = 5.545 nats/octet (8 biți/octet)."
    );
    println!(
        "Memoria de activări vine în plus și crește cu --batch și --seq; vedeți `germen bench`."
    );
    Ok(())
}

fn cmd_bench(a: BenchArgs) -> Result<(), String> {
    let presets: Vec<Preset> = if a.preset.eq_ignore_ascii_case("all") {
        Preset::DEFAULT_BENCH.to_vec()
    } else {
        vec![parse_preset(&a.preset)?]
    };
    if !sys::is_optimised_build() {
        eprintln!(
            "avertisment: build de depanare — rezultatele nu sunt reprezentative; folosiți --release"
        );
    }
    let mut report = BenchReport::new();
    eprintln!(
        "Benchmark: lot {}×{} octeți, {} pași cronometrați (+{} de încălzire) pe model",
        a.batch, a.seq, a.steps, a.warmup
    );
    for p in presets {
        eprint!("  {:<6} ... ", p.name());
        let _ = std::io::stderr().flush();
        let r = bench_model(
            p.name(),
            &p.config(a.seq),
            a.batch,
            a.seq,
            a.warmup,
            a.steps,
        )?;
        eprintln!(
            "{:.0} ms/pas · {:.0} tokeni/s",
            r.ms_per_step, r.tokens_per_sec
        );
        report.results.push(r);
    }
    println!();
    println!("{}", report.render());
    if let Some(path) = a.json {
        std::fs::write(&path, report.to_json())
            .map_err(|e| format!("nu pot scrie {}: {e}", path.display()))?;
        println!("Rezultatele sunt și în {}", path.display());
    }
    Ok(())
}

fn cmd_demo(a: DemoArgs) -> Result<(), String> {
    if !(a.size_mb > 0.0 && a.size_mb <= 1024.0) {
        return Err("--size-mb trebuie să fie între 0 și 1024".into());
    }
    let text = demo_corpus((a.size_mb * 1024.0 * 1024.0) as usize, a.seed);
    std::fs::write(&a.out, &text).map_err(|e| format!("nu pot scrie {}: {e}", a.out.display()))?;
    println!(
        "Am scris {} ({:.2} MiB). Primele rânduri:\n",
        a.out.display(),
        text.len() as f64 / 1048576.0
    );
    for line in text.lines().take(3) {
        println!("  {line}");
    }
    Ok(())
}

fn cmd_train(a: TrainArgs) -> Result<(), String> {
    // Corpus.
    let corpus = match (&a.data, a.demo) {
        (Some(path), false) => Corpus::from_file(path, a.val_fraction)?,
        (None, true) => {
            let text = demo_corpus((a.demo_mb * 1024.0 * 1024.0) as usize, a.seed);
            Corpus::from_bytes(text.into_bytes(), a.val_fraction)?
        }
        _ => return Err("alegeți datele: --data FIȘIER sau --demo".into()),
    };

    // Architecture: from the checkpoint when resuming, otherwise from the preset.
    let cfg: GptConfig = if a.resume {
        checkpoint::load_config(&a.out)?
    } else {
        parse_preset(&a.preset)?.config(a.seq.unwrap_or(256))
    };
    let opts = TrainOptions {
        steps: a.steps,
        batch: a.batch,
        seq: a.seq.unwrap_or(cfg.block_size),
        lr: a.lr,
        min_lr_ratio: a.min_lr_ratio,
        warmup: a.warmup,
        weight_decay: a.weight_decay,
        grad_clip: a.grad_clip,
        beta2: 0.95,
        eval_every: a.eval_every,
        eval_batches: a.eval_batches,
        save_every: a.save_every,
        log_every: a.log_every,
        sample_every: a.sample_every,
        seed: a.seed,
    };

    // Ctrl-C: finish the current step, save, exit. A second Ctrl-C quits at once.
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        let _ = ctrlc::set_handler(move || {
            if stop.swap(true, Ordering::SeqCst) {
                std::process::exit(130);
            }
            eprintln!("\nOpresc după pasul curent și salvez (încă un Ctrl-C = ieșire imediată)...");
        });
    }

    let ln2 = std::f32::consts::LN_2;
    let report = train(
        &corpus,
        &cfg,
        &opts,
        &a.out,
        a.resume,
        &stop,
        &mut |event| match event {
            Event::Start {
                params,
                tokens_per_step,
                est_memory_bytes,
                from_step,
                total_steps,
                train_bytes,
            } => {
                println!(
                    "GERMEN {} · {:.2}M parametri · {} tokeni/pas · pașii {}→{} · {:.2} MiB de antrenare · memorie estimată ≤ {:.1} GiB",
                    germen_core::VERSION,
                    params as f64 / 1e6,
                    tokens_per_step,
                    from_step,
                    total_steps,
                    train_bytes as f64 / 1048576.0,
                    est_memory_bytes as f64 / 1073741824.0
                );
                println!(
                    "Pierdere la început, pentru un model neantrenat: ~5.545 nats/octet. Ctrl-C salvează și oprește."
                );
            }
            Event::Step {
                step,
                total,
                loss,
                lr,
                tokens_per_sec,
                elapsed,
                eta,
            } => {
                println!(
                    "pas {step:>6}/{total} · pierdere {loss:.4} · lr {lr:.2e} · {tokens_per_sec:>7.0} tok/s · scurs {} · rămas ~{}",
                    sys::human_duration(elapsed),
                    sys::human_duration(eta)
                );
            }
            Event::Eval {
                step,
                val_loss,
                best,
            } => {
                println!(
                    "  >> validare la pasul {step}: {val_loss:.4} nats/octet ({:.3} biți/octet){}",
                    val_loss / ln2,
                    if best {
                        "  ← cel mai bun până acum"
                    } else {
                        ""
                    }
                );
            }
            Event::Sample { step, text } => {
                println!("  --- eșantion la pasul {step} ---");
                for line in text.lines() {
                    println!("  | {line}");
                }
                println!("  ---");
            }
            Event::Saved { step, dir } => {
                println!(
                    "  punct de control salvat în {} (pasul {step})",
                    dir.display()
                );
            }
        },
    )?;

    println!();
    if report.interrupted {
        println!(
            "Oprit la pasul {} (poate fi reluat cu --resume).",
            report.final_step
        );
    } else {
        println!(
            "Gata: {} pași în {}.",
            report.steps_run,
            sys::human_duration(report.elapsed_secs)
        );
    }
    if let Some(v) = report.last_val_loss {
        println!(
            "Validare finală: {v:.4} nats/octet ({:.3} biți/octet).",
            v / ln2
        );
    }
    println!(
        "Modelul e în {}. Încercați: germen generate --model {}",
        a.out.display(),
        a.out.display()
    );
    Ok(())
}

fn cmd_generate(a: GenerateArgs) -> Result<(), String> {
    let device = germen_core::cpu_device();
    let (model, cfg, meta) = checkpoint::load(&a.model, &device)?;
    eprintln!(
        "Model: {:.2}M parametri, context {}, antrenat {} pași.",
        cfg.num_params() as f64 / 1e6,
        cfg.block_size,
        meta.step
    );
    let seed = a.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1)
    });
    let opts = SampleOptions {
        max_new: a.max_new,
        temperature: a.temperature,
        top_k: a.top_k,
        window: a.window,
    };
    let tok = ByteTokenizer;
    let prompt = tok.encode(&a.prompt);

    let mut out = std::io::stdout().lock();
    let _ = write!(out, "{}", a.prompt);
    let _ = out.flush();
    let mut stream = Utf8Stream::new();
    let mut rng = Rng::new(seed);
    generate(&model, &prompt, &opts, &mut rng, &device, |byte| {
        let piece = stream.push(byte);
        if !piece.is_empty() {
            let _ = write!(out, "{piece}");
            let _ = out.flush();
        }
    });
    // A character cut off by --max-new is dropped rather than shown as a replacement mark.
    let _ = writeln!(out);
    Ok(())
}
