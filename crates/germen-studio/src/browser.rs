//! Opening the Studio in a window of its own.
//!
//! The page works in any browser; opened as an *application window* (`--app=…`, which Chrome, Edge
//! and Chromium know) it has no address bar and no tabs, and looks like a program. The address is
//! passed as one argument to the browser, never through a shell.

use std::path::PathBuf;
use std::process::{Command, Stdio};

/// How the page was opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opened {
    /// In a window of its own, without address bar.
    AppWindow,
    /// In the default browser, as a tab.
    DefaultBrowser,
}

/// The arguments that open `url` as an application window.
pub fn app_mode_args(url: &str) -> Vec<String> {
    vec![format!("--app={url}"), "--new-window".to_string()]
}

/// Browsers that can show an application window, most likely first.
fn app_mode_candidates() -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        let roots: Vec<PathBuf> = ["ProgramFiles", "ProgramFiles(x86)", "LocalAppData"]
            .iter()
            .filter_map(std::env::var_os)
            .map(PathBuf::from)
            .collect();
        for relative in [
            r"Microsoft\Edge\Application\msedge.exe",
            r"Google\Chrome\Application\chrome.exe",
            r"Chromium\Application\chrome.exe",
            r"BraveSoftware\Brave-Browser\Application\brave.exe",
        ] {
            for root in &roots {
                found.push(root.join(relative));
            }
        }
    } else if cfg!(target_os = "macos") {
        for app in [
            "Google Chrome.app/Contents/MacOS/Google Chrome",
            "Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "Chromium.app/Contents/MacOS/Chromium",
            "Brave Browser.app/Contents/MacOS/Brave Browser",
        ] {
            found.push(PathBuf::from("/Applications").join(app));
        }
    } else {
        let path = std::env::var_os("PATH").unwrap_or_default();
        for name in [
            "google-chrome",
            "google-chrome-stable",
            "microsoft-edge",
            "microsoft-edge-stable",
            "chromium",
            "chromium-browser",
            "brave-browser",
        ] {
            for dir in std::env::split_paths(&path) {
                found.push(dir.join(name));
            }
        }
    }
    found.retain(|p| p.is_file());
    found
}

fn spawn_quietly(command: &mut Command) -> bool {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .is_ok()
}

/// Opens `url`: as an application window where a Chromium-family browser is installed, otherwise in
/// the default browser. An error means neither worked; the address is still valid, so the caller
/// prints it.
pub fn open(url: &str) -> Result<Opened, String> {
    for browser in app_mode_candidates() {
        if spawn_quietly(Command::new(&browser).args(app_mode_args(url))) {
            return Ok(Opened::AppWindow);
        }
    }
    let default_browser = if cfg!(windows) {
        // `start` would hand the address to `cmd`, where `&` and `^` mean something.
        spawn_quietly(Command::new("rundll32").args(["url.dll,FileProtocolHandler", url]))
    } else if cfg!(target_os = "macos") {
        spawn_quietly(Command::new("open").arg(url))
    } else {
        spawn_quietly(Command::new("xdg-open").arg(url))
    };
    if default_browser {
        Ok(Opened::DefaultBrowser)
    } else {
        Err("nu am găsit un browser de deschis automat".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_address_is_one_argument_whatever_it_contains() {
        let url = "http://127.0.0.1:7870/#k=abc&x=1;rm -rf /";
        let args = app_mode_args(url);
        assert_eq!(args[0], format!("--app={url}"));
        assert_eq!(args.len(), 2, "no splitting on spaces or ampersands");
    }

    #[test]
    fn only_files_that_exist_are_candidates() {
        for path in app_mode_candidates() {
            assert!(path.is_file(), "{}", path.display());
        }
    }
}
