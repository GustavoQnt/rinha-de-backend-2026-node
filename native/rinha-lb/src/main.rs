// rinha-lb: tiny TCP load balancer that hands accepted connections off to
// upstream worker processes via SCM_RIGHTS (file descriptor passing over a
// Unix Domain Socket). After the handoff the LB is no longer in the data
// path — the worker process owns the client socket directly.
//
// Linux-only by design; the body assumes io_uring + libc::sendmsg.

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
            let ctrl_fds: Vec<libc::c_int> =
                upstreams.iter().map(|p| scm::connect_ctrl(p)).collect();
            eprintln!("rinha-lb worker {worker_id} connected to {} ctrl fds", ctrl_fds.len());

            monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
                .with_entries(4096)
                .build()
                .expect("build io_uring runtime")
                .block_on(accept::accept_loop(port, backlog, ctrl_fds));
        }));
    }
    for h in handles {
        let _ = h.join();
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("rinha-lb only builds on linux (io_uring + SCM_RIGHTS)");
    std::process::exit(1);
}
