use crate::scm;
use socket2::{Domain, Socket, Type};
use std::net::TcpListener;
use std::os::unix::io::AsRawFd;

pub fn make_listener(port: u16, backlog: i32) -> TcpListener {
    let sock = Socket::new(Domain::IPV4, Type::STREAM, None).expect("socket");
    sock.set_reuse_address(true).expect("SO_REUSEADDR");
    sock.set_reuse_port(true).expect("SO_REUSEPORT");
    let addr: std::net::SocketAddr = format!("0.0.0.0:{port}").parse().unwrap();
    sock.bind(&addr.into()).expect("bind");
    sock.listen(backlog).expect("listen");
    sock.into()
}

// One upstream worker: its ctrl socket path (for reconnects) plus the live
// SCM_RIGHTS sender.
pub struct Upstream {
    path: String,
    sender: scm::ScmSender,
}

impl Upstream {
    pub fn new(path: String, ctrl_fd: libc::c_int) -> Self {
        Self {
            path,
            sender: scm::ScmSender::new(ctrl_fd),
        }
    }
}

// Blocking accept loop. The load balancer does almost no I/O: it accepts a TCP
// connection and hands the fd to an upstream worker with SCM_RIGHTS.
pub fn accept_loop(port: u16, backlog: i32, upstreams: Vec<Upstream>) {
    let listener = make_listener(port, backlog);
    let mut ups = upstreams;
    let mut rr: usize = 0;

    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nodelay(true).ok();
                dispatch_fd(&mut ups, &mut rr, stream.as_raw_fd());
            }
            Err(_) => continue,
        }
    }
}

// Hands `fd` to an upstream, round-robin with failover. A dead upstream (e.g.
// a restarted app container) gets one reconnect attempt per pass so it rejoins
// the rotation instead of staying orphaned until the LB itself restarts.
fn dispatch_fd(ups: &mut [Upstream], rr: &mut usize, fd: libc::c_int) -> bool {
    let n = ups.len();
    if n == 0 {
        return false;
    }
    let start = *rr % n;
    for attempt in 0..n {
        let idx = (start + attempt) % n;
        if ups[idx].sender.send_fd(fd).is_ok() {
            *rr = idx.wrapping_add(1);
            return true;
        }
        if let Some(sender) = try_reconnect(&ups[idx].path) {
            ups[idx].sender = sender;
            if ups[idx].sender.send_fd(fd).is_ok() {
                *rr = idx.wrapping_add(1);
                return true;
            }
        }
    }
    false
}

// Single non-blocking-ish reconnect attempt: UnixStream::connect fails fast
// (ENOENT/ECONNREFUSED) while the upstream is down, succeeds once it re-binds.
fn try_reconnect(path: &str) -> Option<scm::ScmSender> {
    use std::os::unix::io::IntoRawFd;
    std::os::unix::net::UnixStream::connect(path)
        .ok()
        .map(|s| scm::ScmSender::new(s.into_raw_fd()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::io::{AsRawFd, IntoRawFd};
    use std::os::unix::net::{UnixListener, UnixStream};

    fn temp_sock_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("rinha-lb-test-{tag}-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn dispatch_reconnects_to_restarted_upstream() {
        let path = temp_sock_path("reconnect");
        let path_str = path.to_str().unwrap().to_string();

        // Upstream v1 accepts the LB's initial ctrl connection, then "crashes".
        let listener = UnixListener::bind(&path).expect("bind v1");
        let ctrl = UnixStream::connect(&path)
            .expect("connect v1")
            .into_raw_fd();
        let peer1 = listener.accept().expect("accept v1").0;
        drop(peer1);
        drop(listener);
        std::fs::remove_file(&path).ok();

        // Upstream v2: the app restarted and re-bound the same ctrl path.
        let listener2 = UnixListener::bind(&path).expect("bind v2");

        let mut ups = vec![Upstream::new(path_str, ctrl)];
        let mut rr = 0usize;
        let (client, _other) = UnixStream::pair().expect("client pair");

        assert!(
            dispatch_fd(&mut ups, &mut rr, client.as_raw_fd()),
            "dispatch must fail over to the reconnected upstream"
        );
        // The restarted upstream really has the handoff connection pending.
        listener2
            .set_nonblocking(true)
            .expect("nonblocking listener");
        assert!(
            listener2.accept().is_ok(),
            "reconnect must have reached the new listener"
        );
    }

    #[test]
    fn dispatch_fails_when_all_upstreams_dead() {
        let path = temp_sock_path("dead");
        let path_str = path.to_str().unwrap().to_string();

        let listener = UnixListener::bind(&path).expect("bind");
        let ctrl = UnixStream::connect(&path).expect("connect").into_raw_fd();
        drop(listener.accept().expect("accept").0);
        drop(listener);
        std::fs::remove_file(&path).ok();

        let mut ups = vec![Upstream::new(path_str, ctrl)];
        let mut rr = 0usize;
        let (client, _other) = UnixStream::pair().expect("client pair");

        assert!(
            !dispatch_fd(&mut ups, &mut rr, client.as_raw_fd()),
            "no live upstream and no listener to reconnect to"
        );
    }
}
