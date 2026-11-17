//! File-system helpers shared by everything that writes into a checkpoint directory.
//!
//! * [`Staged`] — a set of files that become visible together: every file is written under a
//!   temporary name and flushed to the disk first, and only then are they all renamed into place.
//! * [`rename_over`] — a rename that survives the short "access denied" that antivirus scanners
//!   and indexers cause on Windows.
//! * [`DirLock`] — keeps a second training run out of a directory that one is already using.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Name of the lock file inside a checkpoint directory.
pub(crate) const LOCK_FILE: &str = ".germen.lock";

/// `path` with `.tmp` appended: where a file is written before it is renamed into place.
pub(crate) fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".tmp");
    path.with_file_name(name)
}

/// Forces the data of `path` onto the storage device (not merely into the operating system's
/// cache), so that a power failure after the following rename cannot leave an empty file.
fn sync_file(path: &Path) -> Result<(), String> {
    // Windows insists on write access to flush a file.
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|f| f.sync_all())
        .map_err(|e| format!("nu pot scrie pe disc {}: {e}", path.display()))
}

/// Makes the renames inside `dir` durable too (the directory entry is data of the directory).
#[cfg(unix)]
pub(crate) fn sync_dir(dir: &Path) {
    if let Ok(d) = File::open(dir) {
        let _ = d.sync_all();
    }
}

/// Windows cannot flush a directory handle; NTFS journals the renames itself.
#[cfg(not(unix))]
pub(crate) fn sync_dir(_dir: &Path) {}

/// Calls `op` until it succeeds, retrying up to `attempts` times in total while `transient` says
/// the error may go away, and waiting `first_delay`, then twice that, and so on, in between.
pub(crate) fn retry<T>(
    attempts: u32,
    first_delay: Duration,
    mut op: impl FnMut() -> io::Result<T>,
    transient: impl Fn(&io::Error) -> bool,
) -> io::Result<T> {
    let mut delay = first_delay;
    let mut attempt = 1;
    loop {
        match op() {
            Ok(value) => return Ok(value),
            Err(e) if attempt < attempts && transient(&e) => {
                std::thread::sleep(delay);
                delay = delay.saturating_mul(2);
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

/// On Windows, a file that was written a moment ago is often held open for a short while by an
/// antivirus scanner or the search indexer, and renaming over it then fails with "access denied"
/// or a sharing violation. Elsewhere such an error is real, so it is never retried.
fn is_transient_rename_error(e: &io::Error) -> bool {
    const ERROR_ACCESS_DENIED: i32 = 5;
    const ERROR_SHARING_VIOLATION: i32 = 32;
    const ERROR_LOCK_VIOLATION: i32 = 33;
    cfg!(windows)
        && (matches!(
            e.kind(),
            io::ErrorKind::PermissionDenied | io::ErrorKind::ResourceBusy
        ) || matches!(
            e.raw_os_error(),
            Some(ERROR_ACCESS_DENIED | ERROR_SHARING_VIOLATION | ERROR_LOCK_VIOLATION)
        ))
}

/// Renames `from` over `to` (replacing it), retrying for about ten seconds on Windows.
pub(crate) fn rename_over(from: &Path, to: &Path) -> Result<(), String> {
    retry(
        10,
        Duration::from_millis(20),
        || fs::rename(from, to),
        is_transient_rename_error,
    )
    .map_err(|e| format!("nu pot redenumi în {}: {e}", to.display()))
}

/// Files that are published together.
///
/// Each is first written to `<name>.tmp` and flushed to disk. [`Staged::commit`] then renames them
/// into place, in the order they were added, so the file that decides what the others mean (the
/// checkpoint's `meta.json`) goes last. A run that dies while writing (the long part: hundreds of
/// MB for a big model) leaves the previous checkpoint exactly as it was; the window in which
/// new and old files can be mixed shrinks to a handful of renames, and a run that dies inside it
/// leaves every remaining file complete under its temporary name, which is what
/// [`crate::checkpoint::recover`] finishes the set from. Dropping a `Staged` that was not
/// committed deletes its temporary files.
pub(crate) struct Staged {
    dir: PathBuf,
    /// `(temporary name, final name)` of every file added so far.
    pending: Vec<(PathBuf, PathBuf)>,
}

impl Staged {
    /// Starts a set of files for `dir`, creating it if needed.
    pub(crate) fn new(dir: &Path) -> Result<Staged, String> {
        fs::create_dir_all(dir).map_err(|e| format!("nu pot crea {}: {e}", dir.display()))?;
        Ok(Staged {
            dir: dir.to_path_buf(),
            pending: Vec::new(),
        })
    }

    /// Where the file `name` of this set is written before it is renamed into place.
    pub(crate) fn tmp_of(&self, name: &str) -> PathBuf {
        tmp_path(&self.dir.join(name))
    }

    /// Adds the file `name` with the given content.
    pub(crate) fn bytes(&mut self, name: &str, content: &[u8]) -> Result<(), String> {
        self.with(name, |tmp| {
            fs::write(tmp, content).map_err(|e| format!("nu pot scrie {}: {e}", tmp.display()))
        })
    }

    /// Adds the file `name`, produced by `write`, which receives the temporary path to write to
    /// (for writers that want a path, such as the model's `save_file`).
    pub(crate) fn with(
        &mut self,
        name: &str,
        write: impl FnOnce(&Path) -> Result<(), String>,
    ) -> Result<(), String> {
        let target = self.dir.join(name);
        let tmp = tmp_path(&target);
        // Registered first, so that a half-written file is cleaned up when `write` fails.
        self.pending.push((tmp.clone(), target));
        write(&tmp)?;
        sync_file(&tmp)
    }

    /// Renames every file into place.
    ///
    /// If the very first rename fails nothing has been published, so the staged files are deleted
    /// and the previous ones are exactly as they were. If a later one fails, the set is half
    /// published; the files not yet renamed are then *kept* under their temporary names, because
    /// they are the only way to finish the set (see [`crate::checkpoint::recover`]), and the error
    /// says so.
    pub(crate) fn commit(mut self) -> Result<(), String> {
        let pending = std::mem::take(&mut self.pending);
        for (i, (tmp, target)) in pending.iter().enumerate() {
            if let Err(e) = rename_over(tmp, target) {
                if i == 0 {
                    for (rest, _) in &pending {
                        let _ = fs::remove_file(rest);
                    }
                    return Err(e);
                }
                return Err(format!(
                    "{e}; setul a rămas publicat doar în parte ({i} din {} fișiere), iar restul \
                     au rămas ca *.tmp: la următoarea pornire cu --resume salvarea se termină \
                     singură",
                    pending.len()
                ));
            }
        }
        sync_dir(&self.dir);
        Ok(())
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        for (tmp, _) in &self.pending {
            let _ = fs::remove_file(tmp);
        }
    }
}

/// An exclusive claim on a checkpoint directory, held for as long as the value lives.
///
/// Two runs writing into one directory would interleave their files into a model that never
/// existed. The claim is an advisory file lock, which the operating system releases when the
/// process ends, however it ends. Where the file system cannot lock (some network shares), the
/// claim is skipped rather than refused.
#[derive(Debug)]
pub(crate) struct DirLock {
    _file: Option<File>,
}

impl DirLock {
    /// Claims `dir` (created if missing), or fails if another process holds it.
    pub(crate) fn acquire(dir: &Path) -> Result<DirLock, String> {
        fs::create_dir_all(dir).map_err(|e| format!("nu pot crea {}: {e}", dir.display()))?;
        let path = dir.join(LOCK_FILE);
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(|e| format!("nu pot deschide {}: {e}", path.display()))?;
        match file.try_lock() {
            Ok(()) => Ok(DirLock { _file: Some(file) }),
            Err(fs::TryLockError::WouldBlock) => Err(format!(
                "directorul {} este folosit chiar acum de o altă rulare GERMEN; \
                 așteptați să se termine sau alegeți alt --out",
                dir.display()
            )),
            Err(fs::TryLockError::Error(_)) => Ok(DirLock { _file: None }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("germen-fsutil-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn names(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    #[test]
    fn staged_files_appear_only_at_commit_and_replace_the_old_ones() {
        let dir = scratch("commit");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.txt"), "old a").unwrap();
        fs::write(dir.join("b.txt"), "old b").unwrap();

        let mut staged = Staged::new(&dir).unwrap();
        staged.bytes("a.txt", b"new a").unwrap();
        staged
            .with("b.txt", |p| {
                fs::write(p, "new b").unwrap();
                Ok(())
            })
            .unwrap();
        staged.bytes("c.txt", b"new c").unwrap();
        // Everything is written, but nothing is visible yet.
        assert_eq!(fs::read_to_string(dir.join("a.txt")).unwrap(), "old a");
        assert_eq!(fs::read_to_string(dir.join("b.txt")).unwrap(), "old b");
        assert!(!dir.join("c.txt").exists());

        staged.commit().unwrap();
        assert_eq!(fs::read_to_string(dir.join("a.txt")).unwrap(), "new a");
        assert_eq!(fs::read_to_string(dir.join("b.txt")).unwrap(), "new b");
        assert_eq!(fs::read_to_string(dir.join("c.txt")).unwrap(), "new c");
        assert_eq!(
            names(&dir),
            ["a.txt", "b.txt", "c.txt"],
            "no .tmp left over"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_abandoned_or_failed_set_leaves_the_previous_files_untouched() {
        let dir = scratch("abandon");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("weights"), "good").unwrap();

        // Dropped without a commit (what a crash between two files looks like to the survivors).
        let mut staged = Staged::new(&dir).unwrap();
        staged.bytes("weights", b"half of the new weights").unwrap();
        drop(staged);
        assert_eq!(fs::read_to_string(dir.join("weights")).unwrap(), "good");
        assert_eq!(names(&dir), ["weights"]);

        // A writer that fails part-way: the partial temporary file is removed too.
        let mut staged = Staged::new(&dir).unwrap();
        let err = staged
            .with("weights", |p| {
                fs::write(p, "partial").unwrap();
                Err("disk full".to_string())
            })
            .unwrap_err();
        assert_eq!(err, "disk full");
        drop(staged);
        assert_eq!(fs::read_to_string(dir.join("weights")).unwrap(), "good");
        assert_eq!(names(&dir), ["weights"]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn files_are_published_in_order_so_a_failure_never_reaches_the_last_one() {
        // The checkpoint's meta.json is staged last. If publishing stops half-way (here: the
        // second file cannot be replaced because a directory sits in its place), the last file
        // must still be the old one, and the files that were not published must stay, complete,
        // under their temporary names: they are what a later run finishes the set from.
        let dir = scratch("order");
        fs::create_dir_all(dir.join("blocked")).unwrap();
        fs::write(dir.join("blocked").join("inside"), "x").unwrap();
        fs::write(dir.join("first"), "old first").unwrap();
        fs::write(dir.join("meta"), "old meta").unwrap();

        let mut staged = Staged::new(&dir).unwrap();
        staged.bytes("first", b"new first").unwrap();
        staged.bytes("blocked", b"new blocked").unwrap();
        staged.bytes("meta", b"new meta").unwrap();
        let err = staged.commit().unwrap_err();
        assert!(err.contains("1 din 3"), "{err}");

        assert_eq!(fs::read_to_string(dir.join("first")).unwrap(), "new first");
        assert_eq!(fs::read_to_string(dir.join("meta")).unwrap(), "old meta");
        assert!(dir.join("blocked").is_dir());
        assert_eq!(
            fs::read_to_string(dir.join("meta.tmp")).unwrap(),
            "new meta",
            "the unpublished files stay, complete"
        );
        assert_eq!(
            names(&dir),
            ["blocked", "blocked.tmp", "first", "meta", "meta.tmp"]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failure_at_the_first_file_publishes_nothing_and_leaves_nothing() {
        // Nothing has been published yet, so there is no half-set to finish: the staged files
        // are dropped and the old ones are exactly as they were.
        let dir = scratch("first-fails");
        fs::create_dir_all(dir.join("blocked")).unwrap();
        fs::write(dir.join("blocked").join("inside"), "x").unwrap();
        fs::write(dir.join("meta"), "old meta").unwrap();

        let mut staged = Staged::new(&dir).unwrap();
        staged.bytes("blocked", b"new blocked").unwrap();
        staged.bytes("meta", b"new meta").unwrap();
        assert!(staged.commit().is_err());
        assert_eq!(fs::read_to_string(dir.join("meta")).unwrap(), "old meta");
        assert_eq!(names(&dir), ["blocked", "meta"], "no .tmp left over");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn retry_waits_out_transient_errors_only() {
        let calls = Cell::new(0);
        let flaky = |fail_times: u32, kind: io::ErrorKind| {
            calls.set(0);
            retry(
                5,
                Duration::from_millis(1),
                || {
                    calls.set(calls.get() + 1);
                    if calls.get() <= fail_times {
                        Err(io::Error::new(kind, "busy"))
                    } else {
                        Ok(calls.get())
                    }
                },
                |e| e.kind() == io::ErrorKind::PermissionDenied,
            )
        };
        assert_eq!(flaky(3, io::ErrorKind::PermissionDenied).unwrap(), 4);
        // A permanent error is returned at once, without waiting.
        assert!(flaky(3, io::ErrorKind::NotFound).is_err());
        assert_eq!(calls.get(), 1);
        // A transient one that never goes away is given up on after `attempts` tries.
        assert!(flaky(99, io::ErrorKind::PermissionDenied).is_err());
        assert_eq!(calls.get(), 5);
    }

    #[test]
    fn only_windows_retries_renames() {
        let denied = io::Error::from(io::ErrorKind::PermissionDenied);
        assert_eq!(is_transient_rename_error(&denied), cfg!(windows));
        let missing = io::Error::from(io::ErrorKind::NotFound);
        assert!(!is_transient_rename_error(&missing));
    }

    #[test]
    fn a_second_claim_on_a_directory_is_refused_until_the_first_ends() {
        let dir = scratch("lock");
        let first = DirLock::acquire(&dir).unwrap();
        let err = DirLock::acquire(&dir).unwrap_err();
        assert!(err.contains("altă rulare"), "{err}");
        drop(first);
        assert!(
            DirLock::acquire(&dir).is_ok(),
            "the lock is released on drop"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
