use crate::scm;
use monoio::net::TcpListener;
use socket2::{Domain, Socket, Type};
use std::os::unix::io::AsRawFd;

pub fn make_listener(port: u16, backlog: i32) -> std::net::TcpListener {
    let sock = Socket::new(Domain::IPV4, Type::STREAM, None).expect("socket");
    sock.set_reuse_address(true).expect("SO_REUSEADDR");
    sock.set_reuse_port(true).expect("SO_REUSEPORT");
    sock.set_nonblocking(true).expect("O_NONBLOCK");
    let addr: std::net::SocketAddr = format!("0.0.0.0:{port}").parse().unwrap();
    sock.bind(&addr.into()).expect("bind");
    sock.listen(backlog).expect("listen");
    sock.into()
}

pub async fn accept_loop(port: u16, backlog: i32, ctrl_fds: Vec<libc::c_int>) {
    let std_listener = make_listener(port, backlog);
    let listener = TcpListener::from_std(std_listener).expect("TcpListener::from_std");
    let n = ctrl_fds.len();
    let mut rr: usize = 0;

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                stream.set_nodelay(true).ok();
                let ctrl = ctrl_fds[rr % n];
                rr = rr.wrapping_add(1);
                unsafe { scm::send_fd(ctrl, stream.as_raw_fd()) };
            }
            Err(_) => continue,
        }
    }
}
