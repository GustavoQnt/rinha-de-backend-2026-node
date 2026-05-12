// Receive client TCP file descriptors handed over by the LB via SCM_RIGHTS
// on a Unix Domain Socket. Runs on a dedicated OS thread because monoio's
// io_uring driver does not expose recvmsg with ancillary data — we use
// blocking recvmsg and wake the runtime via a socketpair notification.

#![cfg(target_os = "linux")]

use std::os::unix::fs::PermissionsExt;
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex};

pub type FdQueue = Arc<Mutex<Vec<RawFd>>>;

unsafe fn recv_one_fd(sock: libc::c_int) -> Option<RawFd> {
    let cmsg_space = libc::CMSG_SPACE(std::mem::size_of::<libc::c_int>() as u32) as usize;
    let mut cmsg_buf: Vec<u8> = vec![0u8; cmsg_space];

    let mut dummy: u8 = 0;
    let mut iov = libc::iovec {
        iov_base: &mut dummy as *mut u8 as *mut libc::c_void,
        iov_len: 1,
    };

    let mut msg: libc::msghdr = std::mem::zeroed();
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
    msg.msg_controllen = cmsg_space;

    let ret = libc::recvmsg(sock, &mut msg, 0);
    if ret <= 0 {
        return None;
    }

    let cmsg = libc::CMSG_FIRSTHDR(&msg);
    if cmsg.is_null() {
        return None;
    }
    if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
        let fd = *(libc::CMSG_DATA(cmsg) as *const libc::c_int);
        return Some(fd);
    }
    None
}

fn drain_ctrl_conn(conn: UnixStream, queue: FdQueue, notify_tx: Arc<UnixStream>) {
    use std::io::Write;
    let raw = conn.as_raw_fd();
    loop {
        match unsafe { recv_one_fd(raw) } {
            None => break,
            Some(fd) => {
                queue.lock().expect("fd queue poisoned").push(fd);
                let _ = (&*notify_tx).write(&[1u8]);
            }
        }
    }
}

// Bind the .ctrl listener, accept LB connections, and dispatch each to a
// dedicated drain thread. One LB worker = one persistent ctrl conn.
pub fn control_thread(ctrl_path: String, queue: FdQueue, notify_tx: Arc<UnixStream>) {
    let _ = std::fs::remove_file(&ctrl_path);
    if let Some(parent) = std::path::Path::new(&ctrl_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let listener = UnixListener::bind(&ctrl_path)
        .unwrap_or_else(|e| panic!("bind ctrl socket {ctrl_path}: {e}"));
    let _ = std::fs::set_permissions(&ctrl_path, std::fs::Permissions::from_mode(0o666));
    eprintln!("scm_recv: ctrl socket bound at {ctrl_path}");

    for conn in listener.incoming() {
        match conn {
            Ok(conn) => {
                let queue = queue.clone();
                let notify_tx = notify_tx.clone();
                std::thread::spawn(move || drain_ctrl_conn(conn, queue, notify_tx));
            }
            Err(err) => eprintln!("scm_recv: accept error: {err}"),
        }
    }
}
