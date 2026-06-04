// rinha-lb: tiny TCP load balancer that hands accepted connections off to
// upstream worker processes via SCM_RIGHTS (file descriptor passing over a
// Unix Domain Socket). After the handoff the LB is no longer in the data path:
// the worker process owns the client socket directly.
//
// Linux-only by design; the body assumes libc::sendmsg. Uses a blocking accept
// loop per worker so it runs under the runner's default seccomp profile.

#[cfg(target_os = "linux")]
mod accept;
#[cfg(target_os = "linux")]
mod env;
#[cfg(target_os = "linux")]
mod scm;

#[cfg(target_os = "linux")]
fn main() {
    let cfg = env::from_env();
    eprintln!(
        "rinha-lb: port={} workers={} upstreams={:?} backlog={}",
        cfg.port, cfg.workers, cfg.upstreams, cfg.backlog
    );

    let mut handles = Vec::with_capacity(cfg.workers);
    for worker_id in 0..cfg.workers {
        let upstreams = cfg.upstreams.clone();
        let port = cfg.port;
        let backlog = cfg.backlog;
        handles.push(std::thread::spawn(move || {
            let ups: Vec<accept::Upstream> = upstreams
                .iter()
                .map(|p| accept::Upstream::new(p.to_string(), scm::connect_ctrl(p)))
                .collect();
            eprintln!(
                "rinha-lb worker {worker_id} connected to {} ctrl fds",
                ups.len()
            );

            accept::accept_loop(port, backlog, ups);
        }));
    }
    for h in handles {
        let _ = h.join();
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("rinha-lb only builds on linux (SCM_RIGHTS)");
    std::process::exit(1);
}
