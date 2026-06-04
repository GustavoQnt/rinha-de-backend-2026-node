use std::io;
use std::os::unix::io::IntoRawFd;
use std::os::unix::net::UnixStream;
use std::time::Duration;

const CMSG_BUF_LEN: usize = 64;

#[repr(C)]
union CmsgBuffer {
    hdr: libc::cmsghdr,
    bytes: [u8; CMSG_BUF_LEN],
}

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

pub struct ScmSender {
    ctrl: libc::c_int,
    cmsg_buf: CmsgBuffer,
}

// Owns the ctrl fd: closing on drop lets a reconnect replace a dead sender
// without leaking the old descriptor.
impl Drop for ScmSender {
    fn drop(&mut self) {
        if self.ctrl >= 0 {
            unsafe {
                libc::close(self.ctrl);
            }
        }
    }
}

impl ScmSender {
    pub fn new(ctrl: libc::c_int) -> Self {
        Self {
            ctrl,
            cmsg_buf: CmsgBuffer {
                bytes: [0u8; CMSG_BUF_LEN],
            },
        }
    }

    // Hand `fd` to the peer via SCM_RIGHTS. One byte of payload is required by
    // the kernel even when only ancillary data is meaningful.
    pub fn send_fd(&mut self, fd: libc::c_int) -> io::Result<()> {
        let cmsg_space =
            unsafe { libc::CMSG_SPACE(std::mem::size_of::<libc::c_int>() as u32) as usize };
        debug_assert!(cmsg_space <= CMSG_BUF_LEN);
        let control = unsafe { &mut self.cmsg_buf.bytes[..cmsg_space] };
        control.fill(0);

        let mut dummy: u8 = 0;
        let mut iov = libc::iovec {
            iov_base: &mut dummy as *mut u8 as *mut libc::c_void,
            iov_len: 1,
        };

        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
        msg.msg_controllen = cmsg_space;

        unsafe {
            let cmsg = libc::CMSG_FIRSTHDR(&msg);
            (*cmsg).cmsg_level = libc::SOL_SOCKET;
            (*cmsg).cmsg_type = libc::SCM_RIGHTS;
            (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<libc::c_int>() as u32) as _;
            *(libc::CMSG_DATA(cmsg) as *mut libc::c_int) = fd;

            let sent = libc::sendmsg(self.ctrl, &msg, libc::MSG_NOSIGNAL);
            if sent == 1 {
                Ok(())
            } else if sent < 0 {
                Err(io::Error::last_os_error())
            } else {
                Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "short SCM_RIGHTS send",
                ))
            }
        }
    }
}

#[cfg(test)]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn send_fd(ctrl: libc::c_int, fd: libc::c_int) -> io::Result<()> {
    ScmSender::new(ctrl).send_fd(fd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::io::AsRawFd;

    #[test]
    fn send_fd_reports_bad_control_fd() {
        let (client, _peer) = UnixStream::pair().expect("socketpair");

        let err = unsafe { send_fd(-1, client.as_raw_fd()) }
            .expect_err("invalid control fd must be reported");

        assert_eq!(err.raw_os_error(), Some(libc::EBADF));
    }
}
