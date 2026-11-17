//! Small, dependency-free helpers that describe the machine GERMEN runs on.

/// Number of worker threads the CPU backend will use (size of the global rayon pool).
pub fn worker_threads() -> usize {
    rayon::current_num_threads()
}

/// Number of logical processors reported by the operating system.
pub fn logical_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Fixes the number of worker threads. Must run BEFORE the first tensor operation.
///
/// Returns `false` (and changes nothing) when the pool already exists, e.g. because a tensor
/// operation ran earlier in this process.
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

/// Memory the operating system reports as available for new allocations (Linux only).
pub fn available_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let info = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = info.lines().find(|l| l.starts_with("MemAvailable:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib * 1024)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Refuses a configuration whose estimated memory use cannot fit: more than 90 % of the memory
/// the OS reports as available, or — where the OS cannot tell — more than 256 GiB.
///
/// This turns a typo such as `--seq 99999` into a clear message instead of an allocation failure.
pub fn check_memory_fits(need_bytes: u64, batch: usize, seq: usize) -> Result<(), String> {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let need = need_bytes as f64 / GIB;
    let limit = available_memory_bytes().map_or(256.0, |a| 0.9 * a as f64 / GIB);
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
    fn machine_info_is_not_empty() {
        assert!(logical_cpus() >= 1);
        assert!(worker_threads() >= 1);
        assert!(!simd_summary().is_empty());
        assert!(platform().contains('/'));
    }
}
