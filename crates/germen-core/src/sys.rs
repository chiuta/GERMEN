//! Small, dependency-free helpers that describe the machine GERMEN runs on.

/// Number of worker threads the CPU backend will use: the size of the pool the caller runs on
/// (see [`run_in_pool`]), or of the global pool outside one.
pub fn worker_threads() -> usize {
    rayon::current_num_threads()
}

/// Stack size of the worker threads GERMEN creates.
const WORKER_STACK_BYTES: usize = 8 * 1024 * 1024;

/// Most worker threads a pool may have. Each one reserves a stack, so a typo such as
/// `--threads 1000000` would otherwise try to reserve terabytes and could freeze the machine.
pub const MAX_THREADS: usize = 1024;

/// A pool of exactly `threads` workers (at least one, at most [`MAX_THREADS`]).
pub fn build_pool(threads: usize) -> Result<rayon::ThreadPool, String> {
    if threads > MAX_THREADS {
        return Err(format!(
            "{threads} de fire de lucru e prea mult (maximul acceptat este {MAX_THREADS})"
        ));
    }
    let n = threads.max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .stack_size(WORKER_STACK_BYTES)
        .thread_name(move |i| format!("germen-{n}-{i}"))
        .build()
        .map_err(|e| format!("nu pot porni {n} fire de lucru: {e}"))
}

/// Runs `f` on a dedicated pool of `threads` workers (default: every logical processor).
///
/// Everything `f` does — tensor kernels, tokenisation, … — runs on that pool, and `f` itself is
/// one of its workers. This is measurably faster than driving the global pool from an outside
/// thread, because parallel loops then start without a hand-off between threads: on a 2-core
/// machine about +7 % for the 10.8 M-parameter model, +17 % for 2.8 M and +25 % for 0.85 M.
pub fn run_in_pool<R: Send>(
    threads: Option<usize>,
    f: impl FnOnce() -> R + Send,
) -> Result<R, String> {
    Ok(build_pool(threads.unwrap_or_else(logical_cpus))?.install(f))
}

/// Number of logical processors reported by the operating system.
pub fn logical_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Fixes the size of the GLOBAL pool. Must run BEFORE the first tensor operation.
///
/// Prefer [`run_in_pool`], which is faster. Returns `false` (and changes nothing) when the pool
/// already exists, e.g. because a tensor operation ran earlier in this process.
pub fn set_worker_threads(n: usize) -> bool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n.max(1))
        .build_global()
        .is_ok()
}

/// Vector instruction sets usable on this CPU, strongest first (e.g. `"AVX2+FMA"`).
///
/// The CPU backend selects its kernels at run time, so even a build made for a baseline x86-64
/// CPU uses these when present.
pub fn simd_summary() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let avx512 = std::arch::is_x86_feature_detected!("avx512f");
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        let fma = std::arch::is_x86_feature_detected!("fma");
        let sse42 = std::arch::is_x86_feature_detected!("sse4.2");
        let mut v = Vec::new();
        if avx512 {
            v.push("AVX-512");
        }
        if avx2 {
            v.push("AVX2");
        }
        if fma {
            v.push("FMA");
        }
        if v.is_empty() && sse42 {
            v.push("SSE4.2");
        }
        if v.is_empty() {
            v.push("SSE2");
        }
        v.join("+")
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            "NEON".to_string()
        } else {
            "scalar".to_string()
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "scalar".to_string()
    }
}

/// `"<os>/<arch>"`, e.g. `"windows/x86_64"`.
pub fn platform() -> String {
    format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Whether this binary was compiled with optimisations. A debug build is 20–100× slower and
/// makes every measurement meaningless, so the benchmark warns about it.
pub fn is_optimised_build() -> bool {
    !cfg!(debug_assertions)
}

/// Peak resident memory of this process in MiB (Linux only; `None` elsewhere).
pub fn peak_rss_mib() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
        let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib / 1024.0)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Memory the operating system reports as available for new allocations, in bytes: `MemAvailable`
/// on Linux, `ullAvailPhys` on Windows, free + inactive + speculative pages on macOS. `None`
/// elsewhere, or when the system does not answer.
pub fn available_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let info = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = info.lines().find(|l| l.starts_with("MemAvailable:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib * 1024)
    }
    #[cfg(windows)]
    {
        windows::available_physical_memory()
    }
    #[cfg(target_os = "macos")]
    {
        // Absolute path: never run whatever `vm_stat` happens to come first on PATH.
        let output = std::process::Command::new("/usr/bin/vm_stat")
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| parse_vm_stat(&String::from_utf8_lossy(&output.stdout)))
            .flatten()
    }
    #[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
    {
        None
    }
}

/// Reads macOS `vm_stat` output: the page size from its first line, then the pages that can be
/// handed to a new allocation without evicting anything in use (free, inactive, speculative).
///
/// (`vm_stat` itself already leaves the speculative pages out of "Pages free", so the three
/// counts do not overlap.) Only compiled in for macOS and tests, but parsed identically
/// everywhere so that it can be tested on any machine.
#[cfg(any(target_os = "macos", test))]
fn parse_vm_stat(text: &str) -> Option<u64> {
    let first = text.lines().next()?;
    let page_size: u64 = first
        .split("page size of")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    let pages = |label: &str| -> Option<u64> {
        let line = text.lines().find(|l| l.starts_with(label))?;
        line.split(':')
            .nth(1)?
            .trim()
            .trim_end_matches('.')
            .parse()
            .ok()
    };
    let available = pages("Pages free")?
        .checked_add(pages("Pages inactive")?)?
        .checked_add(pages("Pages speculative").unwrap_or(0))?;
    available.checked_mul(page_size)
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod windows {
    //! The one place GERMEN calls the operating system directly (no `windows-sys` dependency for
    //! a single function): `GlobalMemoryStatusEx`, which has been in kernel32 since Windows 2000.

    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    /// Physical memory that can be allocated right now (free plus reclaimable standby pages).
    pub(super) fn available_physical_memory() -> Option<u64> {
        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: `status` is a live, correctly laid-out MEMORYSTATUSEX whose `dwLength` is set,
        // which is everything the call requires; it only writes inside that structure.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        (ok != 0).then_some(status.avail_phys)
    }
}

/// How much memory a run may plan to use: 90 % of what the OS reports as available, or 256 GiB
/// where it cannot tell.
pub fn memory_limit_bytes() -> f64 {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    available_memory_bytes().map_or(256.0 * GIB, |a| 0.9 * a as f64)
}

/// Refuses a configuration whose estimated memory use cannot fit: more than 90 % of the memory
/// the OS reports as available, or — where the OS cannot tell — more than 256 GiB.
///
/// This turns a typo such as `--seq 99999` into a clear message instead of an allocation failure.
pub fn check_memory_fits(need_bytes: u64, batch: usize, seq: usize) -> Result<(), String> {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let need = need_bytes as f64 / GIB;
    let limit = memory_limit_bytes() / GIB;
    if need > limit {
        return Err(format!(
            "memoria estimată pentru lot {batch} × context {seq} este ~{need:.1} GiB, \
             iar limita acceptată aici este ~{limit:.1} GiB; reduceți --batch sau --seq"
        ));
    }
    Ok(())
}

/// Formats a duration in seconds as `"3 min 20 s"`, `"2 h 05 min"` or `"1 z 04 h"`.
pub fn human_duration(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "—".to_string();
    }
    let s = secs.round() as u64;
    if s < 60 {
        format!("{s} s")
    } else if s < 3600 {
        format!("{} min {:02} s", s / 60, s % 60)
    } else if s < 86_400 {
        format!("{} h {:02} min", s / 3600, (s % 3600) / 60)
    } else {
        format!("{} z {:02} h", s / 86_400, (s % 86_400) / 3600)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_are_readable() {
        assert_eq!(human_duration(5.0), "5 s");
        assert_eq!(human_duration(200.0), "3 min 20 s");
        assert_eq!(human_duration(7500.0), "2 h 05 min");
        assert_eq!(human_duration(90_000.0), "1 z 01 h");
        assert_eq!(human_duration(f64::NAN), "—");
    }

    #[test]
    fn memory_check_accepts_small_and_rejects_absurd_requests() {
        assert!(check_memory_fits(64 * 1024 * 1024, 1, 8).is_ok());
        let err = check_memory_fits(100_000 * 1024 * 1024 * 1024, 16, 9999).unwrap_err();
        assert!(err.contains("--batch") && err.contains("--seq"));
    }

    #[test]
    fn available_memory_is_known_on_the_platforms_germen_is_built_for() {
        // Up to 0.2.0 only Linux answered, so on Windows and macOS (where GERMEN is released)
        // `check_memory_fits` never protected anything.
        if cfg!(any(target_os = "linux", windows, target_os = "macos")) {
            let bytes = available_memory_bytes().expect("the OS must report available memory");
            assert!(bytes > 16 * 1024 * 1024, "{bytes} B is implausibly little");
            // A request for more than that must be refused with the same OS-derived limit.
            assert!(check_memory_fits(bytes.saturating_mul(2), 1, 1).is_err());
        }
    }

    #[test]
    fn vm_stat_output_is_understood() {
        let sample = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
                      Pages free:                               11263.\n\
                      Pages active:                            306417.\n\
                      Pages inactive:                          297866.\n\
                      Pages speculative:                         8001.\n\
                      Pages throttled:                              0.\n\
                      Pages wired down:                        119870.\n";
        assert_eq!(parse_vm_stat(sample), Some((11263 + 297866 + 8001) * 16384));
        assert_eq!(parse_vm_stat(""), None);
        assert_eq!(parse_vm_stat("no page size here\nPages free: 1.\n"), None);
        assert_eq!(
            parse_vm_stat("(page size of 4096 bytes)\nPages free: 1.\n"),
            None,
            "inactive pages are missing"
        );
        assert_eq!(
            parse_vm_stat("(page size of 4096 bytes)\nPages free: 5.\nPages inactive: 7.\n"),
            Some(12 * 4096),
            "speculative pages are optional"
        );
    }

    #[test]
    fn absurd_thread_counts_are_refused() {
        assert!(build_pool(MAX_THREADS + 1).is_err());
        assert!(build_pool(usize::MAX).is_err());
        assert!(run_in_pool(Some(1_000_000), || ()).is_err());
        assert!(build_pool(2).is_ok());
    }

    #[test]
    fn machine_info_is_not_empty() {
        assert!(logical_cpus() >= 1);
        assert!(worker_threads() >= 1);
    }

    #[test]
    fn run_in_pool_gives_the_closure_a_pool_of_the_requested_size() {
        assert_eq!(run_in_pool(Some(3), worker_threads).unwrap(), 3);
        assert_eq!(run_in_pool(Some(0), worker_threads).unwrap(), 1);
        assert_eq!(run_in_pool(None, worker_threads).unwrap(), logical_cpus());
        assert_eq!(run_in_pool(Some(2), || 40 + 2).unwrap(), 42);
        assert!(!simd_summary().is_empty());
        assert!(platform().contains('/'));
    }
}
