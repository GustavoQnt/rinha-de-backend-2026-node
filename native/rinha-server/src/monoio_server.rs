// io_uring HTTP/1.1 server (Linux only). Single-thread monoio runtime per OS
// thread; SO_REUSEPORT scales accept across threads. The handler mirrors the
// sync `handle_http` logic in `main.rs` but routes I/O through monoio's
// completion-based API (buffer-ownership return pattern).

#![cfg(target_os = "linux")]

use std::env;
use std::sync::Arc;
use std::thread;
#[allow(unused_imports)]
use std::marker::PhantomData;

use monoio::buf::IoBuf;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::unix::{UnixListener, UnixStream};
use monoio::net::{ListenerOpts, TcpListener, TcpStream};
use monoio::RuntimeBuilder;

use crate::scm_recv::{self, FdQueue};
use crate::server::{Index, MAX_BODY, MAX_HEAD};
use crate::{json, response::Responses, vectorize};

const READ_BUF: usize = 128 * 1024;

pub fn serve_monoio(
    port: u16,
    sock_path: Option<String>,
    index: Arc<Index>,
    responses: Arc<Responses>,
) {
    // Leak the Arcs to obtain `'static` references. The server runs until
    // process exit, so this is the natural lifetime for these singletons and
    // avoids per-task `Arc::clone` syscalls in the hot path.
    let index_ref: &'static Index = unsafe { &*Arc::into_raw(index) };
    let responses_ref: &'static Responses = unsafe { &*Arc::into_raw(responses) };

    let workers = env::var("APP_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(num_cpus_or_default);
    eprintln!("monoio: thread-per-core workers = {workers}");

    let sock_path = sock_path.filter(|p| !p.is_empty());
    if let Some(ref path) = sock_path {
        let _ = std::fs::remove_file(path);
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
    }

    let mut handles = Vec::with_capacity(workers);
    for worker_id in 0..workers {
        let path = sock_path.clone();
        handles.push(
            thread::Builder::new()
                .name(format!("monoio-{worker_id}"))
                .spawn(move || match path {
                    Some(p) => worker_main_uds(p, worker_id, index_ref, responses_ref),
                    None => worker_main_tcp(port, worker_id, index_ref, responses_ref),
                })
                .expect("spawn monoio worker"),
        );
    }
    for h in handles {
        let _ = h.join();
    }
}

fn worker_main_tcp(
    port: u16,
    worker_id: usize,
    index: &'static Index,
    responses: &'static Responses,
) {
    let mut rt = RuntimeBuilder::<monoio::IoUringDriver>::new()
        .with_entries(1024)
        .enable_timer()
        .build()
        .expect("build monoio runtime");

    rt.block_on(async move {
        let addr = format!("0.0.0.0:{port}");
        let opts = ListenerOpts::new().reuse_addr(true).reuse_port(true);
        let listener = TcpListener::bind_with_config(&addr, &opts)
            .unwrap_or_else(|e| panic!("bind {addr}: {e}"));
        eprintln!("monoio worker {worker_id} listening on {addr} (reuseport)");

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    monoio::spawn(async move {
                        let _ = stream.set_nodelay(true);
                        handle_conn_tcp(stream, index, responses).await;
                    });
                }
                Err(err) => {
                    eprintln!("accept error on worker {worker_id}: {err}");
                    continue;
                }
            }
        }
    });
}

fn worker_main_uds(
    sock_path: String,
    worker_id: usize,
    index: &'static Index,
    responses: &'static Responses,
) {
    let mut rt = RuntimeBuilder::<monoio::IoUringDriver>::new()
        .with_entries(1024)
        .enable_timer()
        .build()
        .expect("build monoio runtime");

    // Bind synchronously via std::os::unix::net (always works), then adopt
    // into monoio. monoio's io_uring `UnixListener::bind` returns ENOTSUP on
    // some kernels (notably Docker Desktop WSL2), so we sidestep that path.
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::UnixListener as StdUnixListener;
    let std_listener = match StdUnixListener::bind(&sock_path) {
        Ok(l) => {
            let _ = std::fs::set_permissions(
                &sock_path,
                std::fs::Permissions::from_mode(0o777),
            );
            Some(l)
        }
        Err(err) => {
            eprintln!("monoio worker {worker_id} std bind {sock_path} failed: {err}");
            None
        }
    };

    rt.block_on(async move {
        let listener = std_listener.and_then(|l| match UnixListener::from_std(l) {
            Ok(m) => Some(m),
            Err(err) => {
                eprintln!("monoio worker {worker_id} from_std failed: {err}");
                None
            }
        });
        let Some(listener) = listener else {
            std::future::pending::<()>().await;
            unreachable!();
        };
        eprintln!("monoio worker {worker_id} listening on unix:{sock_path}");

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    monoio::spawn(async move {
                        handle_conn_uds(stream, index, responses).await;
                    });
                }
                Err(err) => {
                    eprintln!("accept error on worker {worker_id}: {err}");
                    continue;
                }
            }
        }
    });
}

fn num_cpus_or_default() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// ---- SCM_RIGHTS path -----------------------------------------------------
//
// Runs a single monoio runtime that consumes TCP file descriptors handed
// over by the LB on `${sock_path}.ctrl`. No own TCP/UDS listener — the LB
// owns the accept side and delegates each connection to us via SCM_RIGHTS.

pub fn serve_scm(sock_path: String, index: Arc<Index>, responses: Arc<Responses>) {
    use std::os::unix::io::FromRawFd;
    use std::sync::Mutex;

    let index_ref: &'static Index = unsafe { &*Arc::into_raw(index) };
    let responses_ref: &'static Responses = unsafe { &*Arc::into_raw(responses) };

    let ctrl_path = format!("{sock_path}.ctrl");

    // socketpair used solely to wake the runtime when fds land in the queue.
    let (notify_tx_std, notify_rx_std) =
        std::os::unix::net::UnixStream::pair().expect("socketpair");
    notify_rx_std
        .set_nonblocking(true)
        .expect("notify_rx nonblocking");
    let notify_tx = Arc::new(notify_tx_std);

    let queue: FdQueue = Arc::new(Mutex::new(Vec::new()));
    {
        let queue = queue.clone();
        let notify_tx = notify_tx.clone();
        thread::Builder::new()
            .name("scm-ctrl".into())
            .spawn(move || scm_recv::control_thread(ctrl_path, queue, notify_tx))
            .expect("spawn scm ctrl thread");
    }

    let mut rt = RuntimeBuilder::<monoio::IoUringDriver>::new()
        .with_entries(1024)
        .enable_timer()
        .build()
        .expect("build monoio runtime");

    rt.block_on(async move {
        let mut notify_rx = monoio::net::UnixStream::from_std(notify_rx_std)
            .expect("notify from_std");
        let mut buf: Vec<u8> = vec![0u8; 64];

        loop {
            let (res, b) = notify_rx.read(buf).await;
            buf = b;
            match res {
                Ok(0) | Err(_) => return,
                Ok(_) => {
                    let drained: Vec<libc::c_int> =
                        queue.lock().expect("queue poisoned").drain(..).collect();
                    for fd in drained {
                        // Bump SO_SNDBUF so write_all doesn't block on a small
                        // kernel send buffer when many connections write at once.
                        let buf: libc::c_int = 1 << 20; // 1 MiB
                        unsafe {
                            libc::setsockopt(
                                fd,
                                libc::SOL_SOCKET,
                                libc::SO_SNDBUF,
                                &buf as *const _ as *const libc::c_void,
                                std::mem::size_of_val(&buf) as libc::socklen_t,
                            );
                        }
                        let std_stream = unsafe { std::net::TcpStream::from_raw_fd(fd) };
                        if std_stream.set_nonblocking(true).is_err() {
                            continue;
                        }
                        match TcpStream::from_std(std_stream) {
                            Ok(stream) => {
                                let _ = stream.set_nodelay(true);
                                monoio::spawn(handle_conn_tcp(stream, index_ref, responses_ref));
                            }
                            Err(_) => {}
                        }
                    }
                }
            }
        }
    });
}

// --- Buffer wrappers for static and owned response payloads ---------------

struct StaticBuf(&'static [u8]);

unsafe impl IoBuf for StaticBuf {
    fn read_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }
    fn bytes_init(&self) -> usize {
        self.0.len()
    }
}

// --- Connection handler ---------------------------------------------------

async fn handle_conn_tcp(stream: TcpStream, index: &'static Index, responses: &'static Responses) {
    handle_conn(stream, index, responses).await;
}

async fn handle_conn_uds(stream: UnixStream, index: &'static Index, responses: &'static Responses) {
    handle_conn(stream, index, responses).await;
}

async fn handle_conn<S>(mut stream: S, index: &'static Index, responses: &'static Responses)
where
    S: AsyncReadRent + AsyncWriteRentExt,
{
    let mut buf: Vec<u8> = Vec::with_capacity(READ_BUF);
    let mut consumed: usize = 0;

    'conn: loop {
        // Compact any pipelined leftovers to the front of the buffer.
        if consumed > 0 {
            buf.drain(..consumed);
            consumed = 0;
        }

        // Read until we have a full header section (\r\n\r\n).
        let head_end;
        loop {
            if let Some(pos) = find_subseq(&buf, b"\r\n\r\n") {
                head_end = pos;
                break;
            }
            if buf.len() > MAX_HEAD {
                let _ = stream.write_all(StaticBuf(responses.bad(false))).await.0;
                return;
            }
            if buf.len() == buf.capacity() {
                // Should not happen with READ_BUF capacity, but be defensive.
                return;
            }
            let (res, returned) = stream.read(buf).await;
            buf = returned;
            match res {
                Ok(0) => return,
                Ok(_n) => {}
                Err(_) => return,
            }
        }

        // Parse request line + headers.
        let (route, close, content_length) = {
            let headers = &buf[..head_end];
            let request_line_end = find_subseq(headers, b"\r\n").unwrap_or(head_end);
            let request_line = &headers[..request_line_end];

            let mut parts = request_line.splitn(3, |&b| b == b' ');
            let method = parts.next().unwrap_or(&[]);
            let target = parts.next().unwrap_or(&[]);

            let route = if method == b"GET" && target == b"/ready" {
                Route::Ready
            } else if method == b"POST" && target == b"/fraud-score" {
                Route::FraudScore
            } else {
                Route::NotFound
            };

            let close = wants_close(request_line, headers);
            let cl = if matches!(route, Route::FraudScore) {
                find_content_length(headers)
            } else {
                0
            };
            (route, close, cl)
        };

        if content_length > MAX_BODY {
            let _ = stream.write_all(StaticBuf(responses.bad(false))).await.0;
            return;
        }

        let body_start = head_end + 4;
        let body_end = body_start + content_length;

        // Read until full body buffered.
        while buf.len() < body_end {
            if buf.len() == buf.capacity() {
                return;
            }
            let (res, returned) = stream.read(buf).await;
            buf = returned;
            match res {
                Ok(0) => return,
                Ok(_n) => {}
                Err(_) => return,
            }
        }

        let resp: &[u8] = match route {
            Route::Ready => responses.ready(!close),
            Route::NotFound => responses.not_found(!close),
            Route::FraudScore => {
                let body = &buf[body_start..body_end];
                match json::parse_request(body) {
                    Some(parsed) => {
                        let v = vectorize::vectorize(&parsed.to_vec_input());
                        let bucket = index.predict(&v) as usize;
                        responses.fraud(bucket, !close)
                    }
                    None => responses.bad(!close),
                }
            }
        };

        // `responses` is a `&'static Responses`, so its returned slices are
        // already `&'static [u8]` — no transmute or copy needed.
        let (res, _b) = stream.write_all(StaticBuf(resp)).await;
        if res.is_err() {
            return;
        }

        if close {
            return;
        }

        consumed = body_end;
        if consumed >= buf.len() {
            // Fully drained — keep the buffer for next request.
            buf.clear();
            consumed = 0;
        }
        continue 'conn;
    }
}

#[derive(Clone, Copy)]
enum Route {
    Ready,
    FraudScore,
    NotFound,
}

fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn find_content_length(headers: &[u8]) -> usize {
    for line in headers.split(|&b| b == b'\n') {
        let line = strip_cr(line);
        let (name, value) = match split_header(line) {
            Some(p) => p,
            None => continue,
        };
        if name.eq_ignore_ascii_case(b"content-length") {
            return parse_usize(value);
        }
    }
    0
}

fn wants_close(request_line: &[u8], headers: &[u8]) -> bool {
    let is_http10 = request_line.ends_with(b"HTTP/1.0");
    let mut explicit_close = false;
    let mut explicit_keep = false;
    for line in headers.split(|&b| b == b'\n').skip(1) {
        let line = strip_cr(line);
        let (name, value) = match split_header(line) {
            Some(p) => p,
            None => continue,
        };
        if name.eq_ignore_ascii_case(b"connection") {
            for token in value.split(|&b| b == b',') {
                let token = trim_ws(token);
                if token.eq_ignore_ascii_case(b"close") {
                    explicit_close = true;
                } else if token.eq_ignore_ascii_case(b"keep-alive") {
                    explicit_keep = true;
                }
            }
        }
    }
    if explicit_close {
        return true;
    }
    if is_http10 && !explicit_keep {
        return true;
    }
    false
}

fn strip_cr(line: &[u8]) -> &[u8] {
    if line.last() == Some(&b'\r') {
        &line[..line.len() - 1]
    } else {
        line
    }
}

fn split_header(line: &[u8]) -> Option<(&[u8], &[u8])> {
    let colon = line.iter().position(|&b| b == b':')?;
    let (name, rest) = line.split_at(colon);
    Some((trim_ws(name), trim_ws(&rest[1..])))
}

fn trim_ws(s: &[u8]) -> &[u8] {
    let start = s.iter().position(|&b| b != b' ' && b != b'\t').unwrap_or(s.len());
    let end = s
        .iter()
        .rposition(|&b| b != b' ' && b != b'\t')
        .map(|i| i + 1)
        .unwrap_or(0);
    if start <= end {
        &s[start..end]
    } else {
        &[]
    }
}

fn parse_usize(s: &[u8]) -> usize {
    let mut n = 0usize;
    for &b in s {
        if !b.is_ascii_digit() {
            break;
        }
        n = n.saturating_mul(10).saturating_add((b - b'0') as usize);
    }
    n
}
