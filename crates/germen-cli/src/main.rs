//! `germen` — command-line front end of GERMEN.
//!
//! Subcommands: `info`, `bench`, `demo-corpus`, `train`, `generate`, `verify`, `ui`.
#![deny(unsafe_code)]

use clap::{Args, Parser, Subcommand};
use germen_core::bench::{
    BenchParams, BenchReport, bench_model, bench_scaling, default_thread_counts,
};
use germen_core::bpe::Bpe;
use germen_core::checkpoint;
use germen_core::config::{GptConfig, Preset};
use germen_core::data::{Corpus, Rng, demo_corpus, sample_blocks, split_point};
use germen_core::generate::{SampleOptions, generate};
use germen_core::sys;
use germen_core::tokenizer::{Tokenizer, Utf8Stream};
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
    /// Verifică un punct de control: sumele SHA-256 din meta.json, ponderile, optimizatorul
    Verify(VerifyArgs),
    /// Pornește interfața grafică (GERMEN Studio): un server local, doar pe 127.0.0.1, și o fereastră
    Ui(UiArgs),
}

#[derive(Args)]
struct UiArgs {
    /// Folderul de lucru, cu modelele, textele și setările (implicit: ~/GERMEN)
    #[arg(long, value_name = "DIR")]
    workspace: Option<PathBuf>,
    /// Portul de pe 127.0.0.1 (implicit: 7870 sau următorul liber)
    #[arg(long, value_name = "PORT")]
    port: Option<u16>,
    /// Nu deschide browserul; doar afișează adresa
    #[arg(long)]
    no_open: bool,
    /// Cel mai mare text pe care pagina îl poate încărca, în MiB (1 … 1000000)
    #[arg(long, value_name = "MIB", default_value_t = 2048)]
    max_upload_mb: u64,
    /// Scrie pagina de demonstrație (un singur fișier HTML, fără server) și iese
    #[arg(long, value_name = "FILE", hide = true)]
    write_demo: Option<PathBuf>,
}

#[derive(Args)]
struct VerifyArgs {
    /// Directorul punctului de control
    #[arg(default_value = "germen-model")]
    dir: PathBuf,
    /// Termină (sau aruncă) o salvare întreruptă înainte de verificare; nu are nevoie de text
    #[arg(long)]
    recover: bool,
}

#[derive(Args)]
struct BenchArgs {
    /// Dimensiunea modelului: nano, micro, mini, small, base sau all (= primele patru).
    /// Implicit: all; cu --scaling, mini
    #[arg(long)]
    preset: Option<String>,
    /// Ferestre de text pe pas
    #[arg(long, default_value_t = 16)]
    batch: usize,
    /// Lungimea contextului în tokeni
    #[arg(long, default_value_t = 256)]
    seq: usize,
    /// Mărimea vocabularului (256 = octeți; un vocabular BPE mai mare face capul modelului mai scump)
    #[arg(long, default_value_t = 256)]
    vocab: usize,
    /// Pași cronometrați pentru fiecare model
    #[arg(long, default_value_t = 4)]
    steps: usize,
    /// Replici de date paralele pe pas: lotul se împarte între copii ale modelului, fiecare cu
    /// graful ei, pe fire diferite (0 = automat: câte una pe fir, cel mult ferestrele lotului; 1 =
    /// pasul obișnuit, pentru comparație). Mai multe replici decât fire pot ajuta modelele mici și
    /// strica pe cele mari: măsurați
    #[arg(long, default_value_t = 0)]
    replicas: usize,
    /// Pași de încălzire, necronometrați
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    /// Măsoară și cum crește viteza cu numărul de fire: fără valoare, 1, 2, 4, 8… până la
    /// toate procesoarele logice; sau o listă, de pildă `--scaling 1,2,4,8,16,20`
    #[arg(
        long,
        value_name = "LISTĂ",
        num_args = 0..=1,
        default_missing_value = "auto",
        value_delimiter = ','
    )]
    scaling: Option<Vec<String>>,
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
    /// Tokenizatorul: `byte` (un token = un octet; nu cere antrenare) sau `bpe` (vocabular învățat
    /// din text: de 3–4 ori mai mult text în același context). La --resume se folosește cel salvat
    #[arg(long, default_value = "byte", value_parser = ["byte", "bpe"])]
    tokenizer: String,
    /// Mărimea vocabularului BPE (257–65536); contează doar cu --tokenizer bpe
    #[arg(long, default_value_t = 2048)]
    vocab: usize,
    /// Cât din text se folosește pentru a învăța vocabularul BPE, în MiB (bucăți egal depărtate)
    #[arg(long, default_value_t = 64.0)]
    bpe_sample_mb: f64,
    /// Lungimea contextului (octeți). Implicit: 256, iar la --resume cea a modelului salvat
    #[arg(long)]
    seq: Option<usize>,
    /// Ferestre de text într-un micro-lot (cât se ține în memorie deodată)
    #[arg(long, default_value_t = 16)]
    batch: usize,
    /// Micro-loturi însumate înainte de fiecare pas: lotul efectiv = --batch × --accum, cu
    /// memoria lui --batch
    #[arg(long, default_value_t = 1)]
    accum: usize,
    /// Numărul total de pași de antrenare
    #[arg(long, default_value_t = 2000)]
    steps: usize,
    /// Replici de date paralele: ferestrele unui micro-lot se împart între copii ale modelului,
    /// fiecare cu graful ei de gradiente, pe fire diferite (0 = automat: câte una pe fir, cel mult
    /// ferestrele lotului; 1 = pasul obișnuit). Pentru folosirea deplină a procesorului alegeți un
    /// --batch cel puțin egal cu numărul de fire. Mai multe replici decât fire pot ajuta modelele
    /// mici și strica pe cele mari: măsurați cu `germen bench --replicas N`
    #[arg(long, default_value_t = 0)]
    replicas: usize,
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
    /// Limita normei globale a gradientului (0 = oprit)
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
    #[arg(long, conflicts_with = "overwrite")]
    resume: bool,
    /// Începe de la zero chiar dacă --out conține deja un punct de control (îl înlocuiește la
    /// prima salvare). Fără această opțiune, o rulare nouă într-un director cu model antrenat e
    /// refuzată, ca să nu-l pierdeți dacă uitați --resume
    #[arg(long)]
    overwrite: bool,
}

#[derive(Args)]
struct GenerateArgs {
    /// Directorul modelului antrenat
    #[arg(long, default_value = "germen-model")]
    model: PathBuf,
    /// Începutul textului (gol = pornește de la un rând nou)
    #[arg(long, default_value = "")]
    prompt: String,
    /// Câți tokeni noi să genereze
    #[arg(long, default_value_t = 300)]
    max_new: usize,
    /// Temperatura: 0 = alegerea cea mai probabilă, mai mare = mai variat
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
    /// Se aleg doar primele k variante (0 = oprit)
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    /// Contextul maxim citit de model, în tokeni; când se umple, generarea continuă din jumătatea cea mai recentă
    #[arg(long, default_value_t = 128)]
    window: usize,
    /// Sămânța (implicit: aleatoare)
    #[arg(long)]
    seed: Option<u64>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if cli.threads == Some(0) {
        eprintln!("eroare: --threads trebuie să fie cel puțin 1");
        return ExitCode::from(2);
    }
    let threads_flag_given = cli.threads.is_some();
    let requested_threads = cli.threads;
    // A thread sweep builds its own pools, so the one it starts from needs a single worker; so does
    // the Studio, whose jobs each build the pool they need.
    let threads = match &cli.command {
        Command::Bench(a) if a.scaling.is_some() => Some(1),
        Command::Ui(_) => Some(1),
        _ => cli.threads,
    };
    let command = cli.command;
    // The whole command runs as a worker of its own pool: parallel loops start faster that way
    // than from a thread outside the pool (see `sys::run_in_pool`).
    let result = sys::run_in_pool(threads, move || match command {
        Command::Info => cmd_info(),
        Command::Bench(a) => cmd_bench(a, threads_flag_given),
        Command::DemoCorpus(a) => cmd_demo(a),
        Command::Train(a) => cmd_train(a),
        Command::Generate(a) => cmd_generate(a),
        Command::Verify(a) => cmd_verify(a),
        Command::Ui(a) => cmd_ui(a, requested_threads),
    })
    .and_then(|r| r);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("eroare: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The largest value of `germen ui --max-upload-mb` (about a tebibyte: more is no limit at all).
const MAX_UPLOAD_MB: u64 = 1_000_000;

/// `germen ui`: the graphical interface. Runs until the window asks to close or Ctrl-C.
fn cmd_ui(a: UiArgs, threads: Option<usize>) -> Result<(), String> {
    if let Some(file) = &a.write_demo {
        std::fs::write(file, germen_studio::demo_page())
            .map_err(|e| format!("nu pot scrie {}: {e}", file.display()))?;
        println!("Pagina de demonstrație: {}", file.display());
        return Ok(());
    }
    let mut options = germen_studio::Options::default();
    if let Some(workspace) = a.workspace {
        options.workspace = workspace;
    }
    options.port = a.port;
    options.threads = threads;
    options.max_upload = Some(a.max_upload_mb)
        .filter(|mb| (1..=MAX_UPLOAD_MB).contains(mb))
        .map(|mb| mb * 1024 * 1024)
        .ok_or_else(|| format!("--max-upload-mb: o valoare între 1 și {MAX_UPLOAD_MB}"))?;
    let studio = germen_studio::Studio::start(options)?;
    let url = studio.url();
    println!("GERMEN Studio rulează.");
    println!("  Adresa:  {url}");
    println!("  Folder:  {}", studio.workspace().display());
    println!(
        "  Oprire:  Ctrl-C, sau „Închide” din fereastră (o antrenare în curs se salvează întâi)"
    );
    println!("  Adresa e secretă: cine o are poate porni lucrări pe acest calculator.");

    let handle = studio.shutdown_handle();
    let closing = Arc::new(AtomicBool::new(false));
    let _ = ctrlc::set_handler(move || {
        if closing.swap(true, Ordering::SeqCst) {
            std::process::exit(130);
        }
        eprintln!(
            "\nMă închid; o antrenare în curs se salvează întâi (încă un Ctrl-C = ieșire imediată)..."
        );
        handle.request();
    });
    if !a.no_open {
        match germen_studio::browser::open(&url) {
            Ok(germen_studio::browser::Opened::AppWindow) => {}
            Ok(germen_studio::browser::Opened::DefaultBrowser) => {
                println!("  Deschis în browserul implicit.");
            }
            Err(e) => println!("  {e}; deschide adresa de mai sus într-un browser."),
        }
    }
    studio.serve();
    println!("GERMEN Studio s-a închis.");
    Ok(())
}

/// Said when a checkpoint has no manifest of SHA-256 sums to be checked against.
const NO_MANIFEST_NOTE: &str = "notă: punctul de control nu are sume de control (e scris de GERMEN \
     înainte de 0.4.0), deci integritatea lui nu poate fi verificată; următoarea salvare îi adaugă un \
     manifest.";

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

/// Thread counts requested with `--scaling`: `auto` (alone) or a list of numbers.
fn parse_thread_list(list: &[String]) -> Result<Vec<usize>, String> {
    if list.len() == 1 && list[0].eq_ignore_ascii_case("auto") {
        return Ok(default_thread_counts(sys::logical_cpus()));
    }
    list.iter()
        .map(|item| {
            item.trim()
                .parse::<usize>()
                .ok()
                .filter(|&n| n >= 1)
                .ok_or_else(|| {
                    format!("--scaling: „{item}” nu e un număr de fire valid (întreg, cel puțin 1)")
                })
        })
        .collect()
}

fn cmd_bench(a: BenchArgs, threads_flag_given: bool) -> Result<(), String> {
    let scaling = a.scaling.is_some();
    let which = a
        .preset
        .as_deref()
        .unwrap_or(if scaling { "mini" } else { "all" });
    let presets: Vec<Preset> = if which.eq_ignore_ascii_case("all") {
        Preset::DEFAULT_BENCH.to_vec()
    } else {
        vec![parse_preset(which)?]
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

    if let Some(list) = &a.scaling {
        let counts = parse_thread_list(list)?;
        if threads_flag_given {
            eprintln!(
                "avertisment: --threads e ignorat cu --scaling (fiecare măsurătoare își alege singură numărul de fire)"
            );
        }
        let cpus = sys::logical_cpus();
        if let Some(too_many) = counts.iter().find(|&&n| n > cpus) {
            eprintln!(
                "avertisment: {too_many} fire depășesc cele {cpus} procesoare logice; peste acest număr viteza nu mai crește"
            );
        }
        let params = BenchParams {
            batch: a.batch,
            seq: a.seq,
            warmup: a.warmup,
            steps: a.steps,
            replicas: a.replicas,
        };
        for p in presets {
            eprintln!("  {}:", p.name());
            let series = bench_scaling(
                p.name(),
                &p.config(a.seq).with_vocab(a.vocab),
                &params,
                &counts,
                |pt| {
                    eprintln!(
                        "    {:>3} fire: {:.0} ms/pas · {:.0} tokeni/s",
                        pt.threads, pt.result.ms_per_step, pt.result.tokens_per_sec
                    );
                },
            )?;
            report.scaling.push(series);
        }
    } else {
        for p in presets {
            eprint!("  {:<6} ... ", p.name());
            let _ = std::io::stderr().flush();
            let r = bench_model(
                p.name(),
                &p.config(a.seq).with_vocab(a.vocab),
                a.batch,
                a.seq,
                a.warmup,
                a.steps,
                a.replicas,
            )?;
            eprintln!(
                "{:.0} ms/pas · {:.0} tokeni/s ({} replici)",
                r.ms_per_step, r.tokens_per_sec, r.replicas
            );
            report.results.push(r);
        }
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

/// A size given in MiB on the command line, in bytes. Zero, negative, NaN and absurd values are
/// refused: casting `1e300` to `usize` saturates, and the program would then try to build a
/// text of 16 EiB.
fn mib_to_bytes(flag: &str, mib: f64, max_mib: f64) -> Result<usize, String> {
    if !(mib.is_finite() && mib > 0.0 && mib <= max_mib) {
        return Err(format!(
            "{flag} trebuie să fie între 0 și {max_mib} (în MiB), nu {mib}"
        ));
    }
    Ok((mib * 1024.0 * 1024.0) as usize)
}

/// The share of the text kept for validation: `[0, 0.5)`.
fn check_val_fraction(fraction: f64) -> Result<(), String> {
    if !(0.0..0.5).contains(&fraction) {
        return Err(format!(
            "--val-fraction trebuie să fie în [0, 0.5), nu {fraction}"
        ));
    }
    Ok(())
}

fn cmd_demo(a: DemoArgs) -> Result<(), String> {
    let text = demo_corpus(mib_to_bytes("--size-mb", a.size_mb, 1024.0)?, a.seed);
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
    // Whatever can be checked without reading the text is checked before reading it.
    check_val_fraction(a.val_fraction)?;
    if a.val_fraction == 0.0 && a.eval_every > 0 {
        eprintln!(
            "avertisment: cu --val-fraction 0 nu există text de validare, așa că „validarea” se \
             măsoară pe chiar textul de antrenare (nu arată supraînvățarea); \
             folosiți --val-fraction 0.05 sau --eval-every 0"
        );
    }
    // Text.
    let text: Vec<u8> = match (&a.data, a.demo) {
        (Some(path), false) => {
            // A text that cannot fit is refused before it is read: reading it may well succeed,
            // and it is the tokens that then do not fit (the allocator would abort the process).
            if let Ok(info) = std::fs::metadata(path)
                && info.is_file()
            {
                let per_byte = if a.tokenizer == "bpe" && !a.resume {
                    2
                } else {
                    3
                };
                sys::check_corpus_fits(info.len(), per_byte)?;
            }
            std::fs::read(path).map_err(|e| format!("nu pot citi {}: {e}", path.display()))?
        }
        (None, true) => {
            demo_corpus(mib_to_bytes("--demo-mb", a.demo_mb, 1024.0)?, a.seed).into_bytes()
        }
        _ => return Err("alegeți datele: --data FIȘIER sau --demo".into()),
    };

    // Tokenizer: the saved one when resuming, otherwise as requested (BPE is learned from the
    // training part of the text only).
    let tokenizer = if a.resume {
        if a.out.join(checkpoint::META_FILE).exists()
            && checkpoint::load_meta(&a.out)?.files.is_empty()
        {
            eprintln!("{NO_MANIFEST_NOTE}");
        }
        let t = Tokenizer::load(&a.out)?;
        if t.name() != a.tokenizer && a.tokenizer != "byte" {
            eprintln!(
                "avertisment: la --resume se folosește tokenizatorul salvat ({}), nu --tokenizer {}",
                t.name(),
                a.tokenizer
            );
        }
        t
    } else if a.tokenizer == "bpe" {
        let budget = mib_to_bytes("--bpe-sample-mb", a.bpe_sample_mb, 4096.0)?;
        let train_part = &text[..split_point(text.len(), a.val_fraction)];
        let sample = sample_blocks(train_part, budget);
        eprintln!(
            "Învăț vocabularul BPE ({} tokeni) din {:.1} MiB de text...",
            a.vocab,
            sample.len() as f64 / 1048576.0
        );
        let started = std::time::Instant::now();
        let bpe = Bpe::train(&sample, a.vocab, &mut |done, wanted| {
            eprint!("\r  îmbinări: {done}/{wanted}   ");
            let _ = std::io::stderr().flush();
        })?;
        eprintln!(
            "\r  gata în {}: {} de tokeni.            ",
            sys::human_duration(started.elapsed().as_secs_f64()),
            bpe.vocab_size()
        );
        Tokenizer::Bpe(bpe)
    } else {
        Tokenizer::Byte
    };

    let started = std::time::Instant::now();
    let corpus = Corpus::build(&text, a.val_fraction, tokenizer)?;
    drop(text);
    eprintln!(
        "Corpus: {:.2} MiB → {} de tokeni ({:.2} octeți/token), pregătit în {}.",
        corpus.bytes as f64 / 1048576.0,
        corpus.len(),
        corpus.bytes_per_token(),
        sys::human_duration(started.elapsed().as_secs_f64())
    );

    // Architecture: from the checkpoint when resuming, otherwise from the preset (its vocabulary
    // is the tokenizer's).
    let cfg: GptConfig = if a.resume {
        checkpoint::load_config(&a.out)?
    } else {
        parse_preset(&a.preset)?
            .config(a.seq.unwrap_or(256))
            .with_vocab(corpus.tokenizer.vocab_size())
    };
    let opts = TrainOptions {
        steps: a.steps,
        batch: a.batch,
        accum: a.accum,
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
        replicas: a.replicas,
        overwrite: a.overwrite,
    };

    // Ctrl-C: finish the current step, save, exit. A second Ctrl-C quits at once.
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        let _ = ctrlc::set_handler(move || {
            if stop.swap(true, Ordering::SeqCst) {
                std::process::exit(130);
            }
            eprintln!(
                "\nOpresc și salvez ultimul pas întreg (un pas început, cu --accum mare, se abandonează); \
                 încă un Ctrl-C = ieșire imediată..."
            );
        });
    }

    let ln2 = std::f32::consts::LN_2;
    let mut bytes_per_token = 1.0f64;
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
                train_tokens,
                bytes_per_token: bpt,
                vocab_size,
                optimizer_restored,
                replicas,
            } => {
                bytes_per_token = bpt;
                println!(
                    "GERMEN {} · {:.2}M parametri · vocabular {} · {} tokeni/pas · pașii {}→{} · {} tokeni de antrenare · memorie estimată ≤ {:.1} GiB",
                    germen_core::VERSION,
                    params as f64 / 1e6,
                    vocab_size,
                    tokens_per_step,
                    from_step,
                    total_steps,
                    train_tokens,
                    est_memory_bytes as f64 / 1073741824.0
                );
                let threads = sys::worker_threads();
                if replicas == 1 {
                    println!(
                        "Calcul: {threads} fire de lucru, pasul obișnuit (o singură replică pe micro-lot)."
                    );
                } else {
                    println!(
                        "Calcul: {threads} fire de lucru, {replicas} replici pe micro-lot (cel mult {} deodată).",
                        replicas.min(threads)
                    );
                }
                if from_step > 0 {
                    println!(
                        "{}",
                        if optimizer_restored {
                            "Reluare: ponderile și starea AdamW au fost restaurate."
                        } else {
                            "Reluare: ponderile au fost restaurate, dar punctul de control nu are \
                             starea AdamW (e dinainte de 0.2); momentele pornesc de la zero."
                        }
                    );
                }
                if from_step == 0 {
                    println!(
                        "Pierdere la început, pentru un model neantrenat: ~{:.3} nats/token (= ln {vocab_size}). Ctrl-C salvează și oprește.",
                        (vocab_size as f64).ln()
                    );
                } else {
                    println!("Ctrl-C salvează și oprește.");
                }
            }
            Event::Step {
                step,
                total,
                loss,
                lr,
                grad_norm,
                tokens_per_sec,
                elapsed,
                eta,
            } => {
                let gn = if grad_norm.is_finite() {
                    format!(" · ‖g‖ {grad_norm:.2}")
                } else {
                    String::new()
                };
                println!(
                    "pas {step:>6}/{total} · pierdere {loss:.4} · lr {lr:.2e}{gn} · {tokens_per_sec:>7.0} tok/s · scurs {} · rămas ~{}",
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
                    "  >> validare la pasul {step}: {val_loss:.4} nats/token ({:.3} biți/octet){}",
                    val_loss / ln2 / bytes_per_token as f32,
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
            Event::Recovered(checkpoint::Recovery::Nothing) => {}
            Event::Recovered(found) => println!("{}", recovery_message(&found)),
        },
    )?;

    println!();
    if report.steps_run == 0 && !report.interrupted {
        println!(
            "Nimic de făcut: punctul de control e deja la pasul {} (= --steps). Pentru a continua \
             antrenarea, dați un --steps mai mare.",
            report.final_step
        );
    } else if report.interrupted {
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
            "Validare finală: {v:.4} nats/token ({:.3} biți/octet).",
            v / ln2 / bytes_per_token as f32
        );
    }
    println!(
        "Modelul e în {}. Încercați: germen generate --model {}",
        a.out.display(),
        a.out.display()
    );
    Ok(())
}

/// What `--resume` and `verify --recover` say about an interrupted save they dealt with.
fn recovery_message(found: &checkpoint::Recovery) -> String {
    match found {
        checkpoint::Recovery::Completed(files) => format!(
            "Recuperare: o salvare anterioară fusese întreruptă după ce își scrisese toate \
             fișierele; am terminat-o ({files} fișiere publicate)."
        ),
        checkpoint::Recovery::Discarded(files) => format!(
            "Recuperare: o salvare anterioară fusese întreruptă în timp ce scria; am șters \
             cele {files} fișiere neterminate, iar punctul de control vechi e intact."
        ),
        checkpoint::Recovery::Nothing => "Nicio salvare întreruptă de terminat.".to_string(),
    }
}

fn cmd_verify(a: VerifyArgs) -> Result<(), String> {
    if a.recover {
        println!("{}", recovery_message(&checkpoint::recover_locked(&a.dir)?));
    }
    let found = checkpoint::inspect(&a.dir)?;
    println!(
        "{}: punct de control întreg, la pasul {} ({:.2}M parametri).",
        a.dir.display(),
        found.meta.step,
        found.config.num_params() as f64 / 1e6
    );
    for (name, bytes) in &found.files {
        println!("  {name:<15} {bytes:>12} octeți  SHA-256 corect");
    }
    println!(
        "  {:<15} {}",
        "optimizator",
        if found.optimizer {
            "starea AdamW aparține acestor ponderi"
        } else {
            "lipsește (se pornește de la zero la --resume)"
        }
    );
    for name in &found.leftovers {
        println!(
            "  atenție: {name} e rămas dintr-o salvare întreruptă; `germen train --resume` o termină"
        );
    }
    Ok(())
}

fn cmd_generate(a: GenerateArgs) -> Result<(), String> {
    let device = germen_core::cpu_device();
    let (model, cfg, meta) = checkpoint::load(&a.model, &device)?;
    if meta.files.is_empty() {
        eprintln!("{NO_MANIFEST_NOTE}");
    }
    let tokenizer = Tokenizer::load(&a.model)?;
    if tokenizer.vocab_size() != cfg.vocab_size {
        return Err(format!(
            "tokenizatorul ({} tokeni) nu se potrivește cu modelul (vocabular {})",
            tokenizer.vocab_size(),
            cfg.vocab_size
        ));
    }
    eprintln!(
        "Model: {:.2}M parametri, tokenizator {} ({} tokeni), context {}, antrenat {} pași.",
        cfg.num_params() as f64 / 1e6,
        tokenizer.name(),
        tokenizer.vocab_size(),
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
    let mut out = std::io::stdout().lock();
    let _ = write!(out, "{}", a.prompt);
    let _ = out.flush();
    let mut stream = Utf8Stream::new();
    let mut rng = Rng::new(seed);
    generate(
        &model,
        &tokenizer,
        a.prompt.as_bytes(),
        &opts,
        &mut rng,
        &device,
        |token| {
            let piece = stream.push_token(&tokenizer, token);
            if !piece.is_empty() {
                let _ = write!(out, "{piece}");
                let _ = out.flush();
            }
        },
    );
    // A character cut off by --max-new is dropped rather than shown as a replacement mark.
    let _ = writeln!(out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_in_mib_are_checked() {
        assert_eq!(mib_to_bytes("--x", 2.0, 1024.0).unwrap(), 2 * 1024 * 1024);
        assert_eq!(mib_to_bytes("--x", 0.5, 1024.0).unwrap(), 512 * 1024);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e300, 1025.0] {
            let err = mib_to_bytes("--x", bad, 1024.0).unwrap_err();
            assert!(err.contains("--x"), "{bad}: {err}");
        }
    }

    #[test]
    fn the_validation_share_is_checked() {
        for good in [0.0, 0.05, 0.499] {
            assert!(check_val_fraction(good).is_ok(), "{good}");
        }
        for bad in [f64::NAN, -0.1, 0.5, 1.0, 7.0, f64::INFINITY] {
            assert!(check_val_fraction(bad).is_err(), "{bad}");
        }
    }
}
