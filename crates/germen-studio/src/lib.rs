//! GERMEN Studio: the graphical interface of GERMEN.
//!
//! A small web server that listens on the loopback interface of this computer (and nowhere else),
//! and one page that it serves, which a browser shows as an application. The server is written on
//! the standard library alone: no framework, no runtime, nothing to download. The page is one HTML
//! document with its style, its script and its eight languages inside; it loads nothing from
//! anywhere.
//!
//! Who may talk to the server, and what each request may do, is described in `docs/STUDIO.md`.
//! The short form: a random token that only the opened page knows, a check of the `Host` and
//! `Origin` headers, a policy that lets the page run exactly its own script, a folder (the
//! workspace) that every name is resolved against, limits on everything that arrives.
//!
//! ```no_run
//! use germen_studio::{Options, Studio};
//!
//! let studio = Studio::start(Options::default()).expect("the server starts");
//! println!("{}", studio.url());
//! studio.serve(); // until the page asks to quit, or `shutdown_handle().request()` is called
//! ```

#![forbid(unsafe_code)]

mod api;
mod auth;
pub mod browser;
mod http;
mod jobs;
mod page;
mod work;
mod workspace;

use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// The port the Studio tries first.
pub const DEFAULT_PORT: u16 = 7870;
/// The ports tried after it, when it is taken.
const FALLBACK_PORTS: std::ops::RangeInclusive<u16> = 7871..=7889;
/// Connections served at the same moment; more are refused with 503 and retried by the page.
const MAX_CONNECTIONS: usize = 64;
/// How long a closing Studio waits for answers that are still being written.
const CLOSING_GRACE: Duration = Duration::from_secs(3);
/// Default limit for a text sent from the page, in bytes.
pub const DEFAULT_MAX_UPLOAD: u64 = 2 * 1024 * 1024 * 1024;

/// How to start.
#[derive(Clone, Debug)]
pub struct Options {
    /// The folder with the models, the texts and the settings (created if missing).
    pub workspace: PathBuf,
    /// The port. `None`: [`DEFAULT_PORT`], or the next free one. `Some(0)`: any free port. `Some(n)`:
    /// exactly `n`, or an error if it is taken.
    pub port: Option<u16>,
    /// Number of worker threads for training and generating (`None`: one per logical processor).
    pub threads: Option<usize>,
    /// The largest text the page may send, in bytes.
    pub max_upload: u64,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            workspace: default_workspace(),
            port: None,
            threads: None,
            max_upload: DEFAULT_MAX_UPLOAD,
        }
    }
}

/// The interface as one self-contained HTML page that replays a recorded training instead of
/// talking to a program: for looking at the Studio, or taking its pictures, without installing
/// anything. It starts no server and reads no file.
pub fn demo_page() -> String {
    page::demo_html()
}

/// `~/GERMEN` (the profile folder on Windows), or `./GERMEN` where there is no home.
pub fn default_workspace() -> PathBuf {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .filter(|p| p.is_absolute());
    home.unwrap_or_else(|| PathBuf::from(".")).join("GERMEN")
}

/// What the connection threads share.
pub(crate) struct Shared {
    pub(crate) port: u16,
    pub(crate) token: String,
    pub(crate) workspace: workspace::Workspace,
    pub(crate) hub: jobs::Hub,
    pub(crate) page: page::Page,
    pub(crate) threads: Option<usize>,
    pub(crate) max_upload: u64,
    shutdown: AtomicBool,
    active: AtomicUsize,
}

impl Shared {
    /// Asks the server to stop: no new connections, running jobs save and end.
    pub(crate) fn request_shutdown(&self) {
        if self.shutdown.swap(true, Ordering::SeqCst) {
            return;
        }
        // `accept` waits for a connection; give it one so that it looks at the flag.
        let _ = TcpStream::connect_timeout(
            &SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), self.port),
            Duration::from_secs(1),
        );
    }

    fn closing(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }
}

/// A handle that stops the server from another thread (from a Ctrl-C handler, say).
#[derive(Clone)]
pub struct ShutdownHandle {
    shared: Arc<Shared>,
}

impl ShutdownHandle {
    /// Asks the server to stop. A training run in progress saves first.
    pub fn request(&self) {
        self.shared.request_shutdown();
    }
}

/// A Studio that is listening.
pub struct Studio {
    shared: Arc<Shared>,
    listener: TcpListener,
}

impl Studio {
    /// Opens the workspace, binds the port and prepares the page. Nothing is served until
    /// [`Studio::serve`].
    pub fn start(options: Options) -> Result<Studio, String> {
        let workspace = workspace::Workspace::open(options.workspace.clone())?;
        let (listener, port) = bind(options.port)?;
        let token = auth::new_token();
        debug_assert_eq!(token.len(), auth::TOKEN_LEN);
        let shared = Arc::new(Shared {
            port,
            token,
            workspace,
            hub: jobs::Hub::new(),
            page: page::build(),
            threads: options.threads,
            max_upload: options.max_upload,
            shutdown: AtomicBool::new(false),
            active: AtomicUsize::new(0),
        });
        Ok(Studio { shared, listener })
    }

    /// The address to open: the server, and the token after the `#`, which a browser never sends
    /// to any server, so it cannot end up in a log or a `Referer`.
    pub fn url(&self) -> String {
        format!(
            "http://127.0.0.1:{}/#k={}",
            self.shared.port, self.shared.token
        )
    }

    /// The port the server listens on.
    pub fn port(&self) -> u16 {
        self.shared.port
    }

    /// The token that the page presents with every request.
    pub fn token(&self) -> &str {
        &self.shared.token
    }

    /// The workspace folder.
    pub fn workspace(&self) -> &std::path::Path {
        self.shared.workspace.root()
    }

    /// A handle that stops [`Studio::serve`] from another thread.
    pub fn shutdown_handle(&self) -> ShutdownHandle {
        ShutdownHandle {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Serves until a shutdown is requested (by the page, or through a [`ShutdownHandle`]); then
    /// asks the running jobs to save and end, waits for them, and returns.
    pub fn serve(self) {
        let Studio { shared, listener } = self;
        for incoming in listener.incoming() {
            if shared.closing() {
                break;
            }
            let stream = match incoming {
                Ok(stream) => stream,
                Err(_) => {
                    // A connection that was reset before it was accepted, or no file handles for
                    // the moment: look again shortly rather than spin or stop.
                    std::thread::sleep(Duration::from_millis(20));
                    continue;
                }
            };
            accept(&shared, stream);
        }
        drop(listener);
        shared.hub.close();
        shared.hub.join();
        let deadline = Instant::now() + CLOSING_GRACE;
        while shared.active.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

/// Binds the loopback address: one port when it was asked for, the first free one otherwise.
fn bind(port: Option<u16>) -> Result<(TcpListener, u16), String> {
    let try_port = |p: u16| TcpListener::bind((Ipv4Addr::LOCALHOST, p));
    let (listener, error) = match port {
        Some(p) => match try_port(p) {
            Ok(l) => (Some(l), None),
            Err(e) => (None, Some(format!("portul {p} nu poate fi folosit: {e}"))),
        },
        None => {
            let found = std::iter::once(DEFAULT_PORT)
                .chain(FALLBACK_PORTS)
                .chain(std::iter::once(0))
                .find_map(|p| try_port(p).ok());
            (
                found,
                Some("nu pot deschide niciun port pe 127.0.0.1".to_string()),
            )
        }
    };
    let Some(listener) = listener else {
        return Err(error.unwrap_or_else(|| "nu pot deschide serverul".to_string()));
    };
    let bound = listener
        .local_addr()
        .map_err(|e| format!("nu pot afla portul serverului: {e}"))?;
    Ok((listener, bound.port()))
}

/// Hands a new connection to a thread of its own, or turns it away.
fn accept(shared: &Arc<Shared>, stream: TcpStream) {
    // Only this computer can reach a loopback listener, but a listener that was ever changed to
    // another address would still not serve strangers.
    let local = stream
        .peer_addr()
        .map(|a| a.ip().is_loopback())
        .unwrap_or(false);
    if !local {
        return;
    }
    if shared.active.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
        shared.active.fetch_sub(1, Ordering::SeqCst);
        refuse(stream);
        return;
    }
    let worker = Arc::clone(shared);
    let spawned = std::thread::Builder::new()
        .name("germen-conn".into())
        .stack_size(1 << 20)
        .spawn(move || {
            serve_connection(&worker, stream);
            worker.active.fetch_sub(1, Ordering::SeqCst);
        });
    if spawned.is_err() {
        shared.active.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Tells a client that the Studio is busy, without waiting for it.
fn refuse(mut stream: TcpStream) {
    use std::io::Write;
    let _ = stream.set_write_timeout(Some(Duration::from_millis(200)));
    let _ = stream.write_all(
        b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nRetry-After: 1\r\nContent-Length: 0\r\n\r\n",
    );
}

/// One connection, one request (`Connection: close`).
fn serve_connection(shared: &Arc<Shared>, stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    let mut conn = http::Conn::new(stream);
    match http::read_head(&mut conn) {
        Ok(None) => {}
        Ok(Some(head)) => {
            let answer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                api::handle(shared, &head, &mut conn)
            }))
            .unwrap_or_else(|_| {
                http::ApiError::new(500, "internal", "eroare internă").to_response()
            });
            http::finish(&mut conn, &answer);
        }
        Err(error) => http::finish(&mut conn, &error.to_response()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_workspace_is_a_folder_called_germen_under_home() {
        let w = default_workspace();
        assert_eq!(w.file_name().and_then(|n| n.to_str()), Some("GERMEN"));
    }

    #[test]
    fn an_explicit_port_is_exact_and_a_taken_one_is_an_error() {
        let first = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let taken = first.local_addr().unwrap().port();
        let error = bind(Some(taken)).expect_err("the port is taken");
        assert!(error.contains(&taken.to_string()), "{error}");
        let (second, port) = bind(Some(0)).unwrap();
        assert_ne!(port, taken);
        assert!(second.local_addr().unwrap().ip().is_loopback());
    }

    #[test]
    fn without_a_port_a_free_one_is_found_and_it_is_loopback_only() {
        // Take the default port if it is free, so that the search has something to skip.
        let _hold = TcpListener::bind((Ipv4Addr::LOCALHOST, DEFAULT_PORT));
        let (listener, port) = bind(None).unwrap();
        assert_ne!(port, 0);
        assert_eq!(
            listener.local_addr().unwrap().ip(),
            IpAddr::V4(Ipv4Addr::LOCALHOST)
        );
    }
}
