use std::os::unix::net::UnixStream;
use std::os::unix::io::IntoRawFd;
use std::time::Duration;

// Block until the upstream ctrl socket is reachable. The app may take a moment
// to bind its .ctrl listener after startup, so retry with a small sleep instead
// of crashing the LB on first connect failure.
pub fn connect_ctrl(path: &str) -> libc::c_int {
    loop {
        match UnixStream::connect(path) {
            Ok(s) => return s.into_raw_fd(),
            Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

// Hand `fd` to the peer on `ctrl` via SCM_RIGHTS. One byte of payload is
// required by the kernel even when only ancillary data is meaningful.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn send_fd(ctrl: libc::c_int, fd: libc::c_int) {
    let cmsg_space = libc::CMSG_SPACE(std::mem::size_of::<libc::c_int>() as u32) as usize;
    let mut cmsg_buf: Vec<u8> = vec![0u8; cmsg_space];

    let mut dummy: u8 = 0;
    let mut iov = libc::iovec {
        iov_base: &mut dummy as *mut u8 as *mut libc::c_void,0
        iov_len: 1,
    };

    let mut msg: libc::msghdr = std::mem::zeroed();
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
    msg.msg_controllen = cmsg_space;

    let cmsg = libc::CMSG_FIRSTHDR(&msg);
    (*cmsg).cmsg_level = libc::SOL_SOCKET;
    (*cmsg).cmsg_type = libc::SCM_RIGHTS;
    (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<libc::c_int>() as u32) as _;
    *(libc::CMSG_DATA(cmsg) as *mut libc::c_int) = fd;

    libc::sendmsg(ctrl, &msg, 0);
}
