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

// Blocking accept loop. The load balancer does almost no I/O: it accepts a TCP
// connection and hands the fd to an upstream worker with SCM_RIGHTS.
pub fn accept_loop(port: u16, backlog: i32, ctrl_fds: Vec<libc::c_int>) {
    let listener = make_listener(port, backlog);
    let mut senders: Vec<scm::ScmSender> = ctrl_fds.into_iter().map(scm::ScmSender::new).collect();
    let n = senders.len();
    let mut rr: usize = 0;

    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nodelay(true).ok();
                if n == 0 {
                    continue;
                }

                let start = rr % n;
                for attempt in 0..n {
                    let idx = (start + attempt) % n;
                    if senders[idx].send_fd(stream.as_raw_fd()).is_ok() {
                        rr = idx.wrapping_add(1);
                        break;
                    }
                }
            }
            Err(_) => continue,
        }
    }
}
