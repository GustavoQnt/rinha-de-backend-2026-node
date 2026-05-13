#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::Arc;
#[cfg(unix)]
use std::sync::{mpsc, Mutex};
use std::thread;
use std::time::Duration;

use rinha_server::{
    ivf::Ivf, ivf_blocks::IvfBlocks, json, knn, refs::Refs, response::Responses,
    server::{Index, MAX_BODY, MAX_HEAD},
    vectorize,
};

const READ_BUF: usize = 16 * 1024;
const DEFAULT_IVF_NPROBE: usize = 64;
#[cfg(unix)]
const DEFAULT_APP_WORKERS: usize = 4;

#[derive(Clone, Copy)]
enum Route {
    Ready,
    FraudScore,
    NotFound,
}

fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn find_content_length(headers: &[u8]) -> usize {
    const NEEDLE: &[u8] = b"content-length:";
    let mut i = 0;
    while i + NEEDLE.len() <= headers.len() {
        if headers[i..i + NEEDLE.len()].eq_ignore_ascii_case(NEEDLE) {
            let mut j = i + NEEDLE.len();
            while j < headers.len() && (headers[j] == b' ' || headers[j] == b'\t') {
                j += 1;
            }
            let mut n = 0usize;
            while j < headers.len() && headers[j].is_ascii_digit() {
                n = n * 10 + (headers[j] - b'0') as usize;
                j += 1;
            }
            return n;
        }
        i += 1;
    }
    0
}

// Returns true if the request signals connection close (HTTP/1.0 default,
// or explicit `Connection: close` header).
fn wants_close(request_line: &[u8], headers: &[u8]) -> bool {
    let is_http10 = request_line.ends_with(b"HTTP/1.0");
    let mut explicit_close = false;
    let mut explicit_keep = false;

    const NEEDLE: &[u8] = b"connection:";
    let mut i = 0;
    while i + NEEDLE.len() <= headers.len() {
        if headers[i..i + NEEDLE.len()].eq_ignore_ascii_case(NEEDLE) {
            let mut j = i + NEEDLE.len();
            while j < headers.len() && (headers[j] == b' ' || headers[j] == b'\t') {
                j += 1;
            }
            let line_end = headers[j..]
                .iter()
                .position(|&b| b == b'\r' || b == b'\n')
                .map(|p| j + p)
                .unwrap_or(headers.len());
            let value = &headers[j..line_end];
            if value.eq_ignore_ascii_case(b"close") {
                explicit_close = true;
            } else if value.eq_ignore_ascii_case(b"keep-alive") {
                explicit_keep = true;
            }
            break;
        }
        i += 1;
    }

    if explicit_close {
        return true;
    }
    if is_http10 && !explicit_keep {
        return true;
    }
    false
}

fn handle_http<S: Read + Write>(mut stream: S, index: Arc<Index>, responses: Arc<Responses>) {
    let mut buf: Vec<u8> = Vec::with_capacity(READ_BUF);
    let mut chunk = [0u8; READ_BUF];

    loop {
        // Read until we have a full header section (\r\n\r\n).
        let head_end = loop {
            if let Some(pos) = find_subseq(&buf, b"\r\n\r\n") {
                break pos;
            }
            if buf.len() > MAX_HEAD {
                let _ = stream.write_all(responses.bad(false));
                return;
            }
            match stream.read(&mut chunk) {
                Ok(0) => return,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
                Err(_) => return,
            }
        };

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
            let _ = stream.write_all(responses.bad(false));
            return;
        }

        let body_start = head_end + 4;

        // Read until full body is buffered.
        while buf.len() < body_start + content_length {
            match stream.read(&mut chunk) {
                Ok(0) => return,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
                Err(_) => return,
            }
        }

        let resp: &[u8] = match route {
            Route::Ready => responses.ready(!close),
            Route::NotFound => responses.not_found(!close),
            Route::FraudScore => {
                let body = &buf[body_start..body_start + content_length];
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

        if stream.write_all(resp).is_err() {
            return;
        }

        if close {
            return;
        }

        // Drain handled bytes; preserve any pipelined leftovers for the next iteration.
        buf.drain(..body_start + content_length);
    }
}

fn handle_tcp(stream: TcpStream, index: Arc<Index>, responses: Arc<Responses>) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
    let _ = stream.set_nodelay(true);
    handle_http(stream, index, responses);
}

fn serve_tcp(port: u16, index: Arc<Index>, responses: Arc<Responses>) {
    let bind = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&bind).expect("bind tcp");
    eprintln!("rinha-server listening on {bind}");

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let index = Arc::clone(&index);
                let responses = Arc::clone(&responses);
                thread::spawn(move || handle_tcp(stream, index, responses));
            }
            Err(err) => eprintln!("accept error: {err}"),
        }
    }
}

#[cfg(unix)]
fn handle_unix(stream: UnixStream, index: Arc<Index>, responses: Arc<Responses>) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
    handle_http(stream, index, responses);
}

#[cfg(unix)]
fn serve_unix(sock_path: String, index: Arc<Index>, responses: Arc<Responses>) {
    use std::os::unix::fs::PermissionsExt;

    let _ = std::fs::remove_file(&sock_path);
    if let Some(parent) = std::path::Path::new(&sock_path).parent() {
        std::fs::create_dir_all(parent).expect("create socket dir");
    }
    let listener = UnixListener::bind(&sock_path).expect("bind unix socket");
    std::fs::set_permissions(&sock_path, std::fs::Permissions::from_mode(0o777))
        .expect("chmod unix socket");
    eprintln!("rinha-server listening on unix:{sock_path}");

    let workers = env::var("APP_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_APP_WORKERS);
    eprintln!("app workers = {workers}");

    let (tx, rx) = mpsc::sync_channel::<UnixStream>(workers * 64);
    let rx = Arc::new(Mutex::new(rx));
    for worker_id in 0..workers {
        let rx = Arc::clone(&rx);
        let index = Arc::clone(&index);
        let responses = Arc::clone(&responses);
        thread::spawn(move || loop {
            let stream = match rx.lock().expect("worker queue poisoned").recv() {
                Ok(stream) => stream,
                Err(_) => return,
            };
            handle_unix(stream, Arc::clone(&index), Arc::clone(&responses));
        });
        eprintln!("started unix worker {worker_id}");
    }

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                if tx.send(stream).is_err() {
                    return;
                }
            }
            Err(err) => eprintln!("accept error: {err}"),
        }
    }
}

fn main() {
    let port = env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080_u16);
    let refs_path =
        env::var("REFS_PATH").unwrap_or_else(|_| "resources/references.bin".to_string());

    let ivf_blocks = match env::var("IVF_BLOCKS_PATH") {
        Ok(path) if !path.is_empty() => {
            eprintln!("loading ivf-blocks (R26I v2) index from {path} ...");
            let blocks = IvfBlocks::load(&path);
            eprintln!(
                "loaded ivf-blocks K={} count={} total_blocks={}",
                blocks.k, blocks.count, blocks.total_blocks
            );
            Some(blocks)
        }
        _ => None,
    };
    let ivf = if ivf_blocks.is_some() {
        eprintln!("ivf-blocks enabled; skipping v1 IVF_PATH");
        None
    } else {
        match env::var("IVF_PATH") {
            Ok(path) if !path.is_empty() => {
                eprintln!("loading ivf index from {path} ...");
                let ivf = Ivf::load(&path);
                eprintln!("loaded ivf K={} count={}", ivf.k, ivf.count);
                Some(ivf)
            }
            _ => None,
        }
    };
    let refs = if ivf_blocks.is_some() || ivf.is_some() {
        eprintln!("IVF enabled; skipping full references.bin load");
        None
    } else {
        eprintln!("loading references from {refs_path} ...");
        let refs = Refs::load(&refs_path);
        eprintln!("loaded {} refs, scale={}", refs.count, refs.scale);
        Some(refs)
    };
    let k_for_nprobe = ivf_blocks.as_ref().map(|b| b.k).or_else(|| ivf.as_ref().map(|i| i.k));
    let nprobe = match (
        k_for_nprobe,
        env::var("IVF_NPROBE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok()),
    ) {
        (Some(k), Some(n)) => n.min(k),
        (Some(k), None) => DEFAULT_IVF_NPROBE.min(k),
        _ => 0,
    };
    if k_for_nprobe.is_some() {
        eprintln!("ivf nprobe = {nprobe}");
    }

    let two_pass = match (
        env::var("IVF_FAST_NPROBE").ok().and_then(|s| s.parse::<usize>().ok()),
        env::var("IVF_FULL_NPROBE").ok().and_then(|s| s.parse::<usize>().ok()),
    ) {
        (Some(fast), Some(full)) if k_for_nprobe.is_some() => {
            let k = k_for_nprobe.unwrap();
            let fast = fast.min(k);
            let full = full.min(k);
            eprintln!("ivf two-pass: fast_nprobe={fast} full_nprobe={full}");
            Some((fast, full))
        }
        _ => None,
    };

    let index = Arc::new(Index { refs, ivf, ivf_blocks, nprobe, two_pass });
    let responses = Arc::new(Responses::build());
    eprintln!("distance kernel = {}", knn::active_distance_kernel_name());

    // Warmup before bind: page-touch the index + dry-run predict to prime
    // iCache, dCache, and branch predictor. Eliminates the cold-start spike
    // that produced p99=13.71ms on the first prévia run of this binary.
    let warmup_rounds: usize = env::var("WARMUP_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1500);
    if let Some(blocks) = index.ivf_blocks.as_ref() {
        let t0 = std::time::Instant::now();
        knn::warmup_ivf_blocks(blocks, warmup_rounds);
        eprintln!(
            "warmup: {warmup_rounds} predict rounds + page-touch in {:?}",
            t0.elapsed()
        );
    }

    #[cfg(target_os = "linux")]
    if env::var("USE_SCM").as_deref() == Ok("1") {
        let sock = env::var("SOCK_PATH").expect("SOCK_PATH required for USE_SCM=1");
        eprintln!("server backend = monoio + SCM_RIGHTS (io_uring)");
        rinha_server::monoio_server::serve_scm(sock, index, responses);
        return;
    }

    #[cfg(target_os = "linux")]
    if env::var("USE_MONOIO").as_deref() == Ok("1") {
        let sock = env::var("SOCK_PATH").ok();
        eprintln!("server backend = monoio (io_uring)");
        rinha_server::monoio_server::serve_monoio(port, sock, index, responses);
        return;
    }

    #[cfg(unix)]
    if let Ok(sock_path) = env::var("SOCK_PATH") {
        if !sock_path.is_empty() {
            serve_unix(sock_path, index, responses);
            return;
        }
    }

    serve_tcp(port, index, responses);
}
