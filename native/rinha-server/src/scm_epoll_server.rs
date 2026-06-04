#![cfg(target_os = "linux")]

use crate::response::Responses;
use crate::server::{Index, MAX_BODY, MAX_HEAD};
use crate::{json, vectorize};
use std::io;
use std::os::fd::{IntoRawFd, RawFd};
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;

const READ_BUF: usize = MAX_HEAD + MAX_BODY + 4;
const MAX_EVENTS: usize = 256;
const MAX_FD: usize = 65_536;
const LISTENER_TOKEN: u64 = u64::MAX;
const DEFAULT_CONN_POOL: usize = 64;
const DEFAULT_RECV_FD_BUDGET: usize = 64;

pub fn serve_scm_epoll(sock_path: String, index: Arc<Index>, responses: Arc<Responses>) {
    let index: &'static Index = unsafe { &*Arc::into_raw(index) };
    let responses: &'static Responses = unsafe { &*Arc::into_raw(responses) };
    let ctrl_path = format!("{sock_path}.ctrl");

    let listener_fd = bind_ctrl_listener(&ctrl_path)
        .unwrap_or_else(|err| panic!("bind ctrl socket {ctrl_path}: {err}"));
    let epfd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
    if epfd < 0 {
        panic!("epoll_create1: {}", io::Error::last_os_error());
    }

    epoll_add(
        epfd,
        listener_fd,
        LISTENER_TOKEN,
        libc::EPOLLIN | libc::EPOLLRDHUP,
    )
    .unwrap_or_else(|err| panic!("epoll add ctrl listener: {err}"));
    configure_epoll_busy_poll(epfd);

    eprintln!(
        "scm_epoll: ctrl socket bound at {ctrl_path} (EPOLL_IDLE_US={}, 0 = block)",
        env_u32("EPOLL_IDLE_US", 0)
    );

    let mut server = EpollServer::new(epfd, listener_fd, index, responses);
    server.run();
}

struct EpollServer {
    epfd: RawFd,
    listener_fd: RawFd,
    index: &'static Index,
    responses: &'static Responses,
    conns: Vec<Option<Box<Conn>>>,
    conn_pool: Vec<Box<Conn>>,
    conn_pool_cap: usize,
    is_control: Vec<bool>,
    recv_fd_budget: usize,
}

struct Conn {
    in_buf: Box<[u8; READ_BUF]>,
    in_len: usize,
    write_resp: Option<&'static [u8]>,
    write_pos: usize,
    close_after_write: bool,
    peer_closed: bool,
}

impl Conn {
    fn new() -> Box<Self> {
        Box::new(Self {
            in_buf: Box::new([0; READ_BUF]),
            in_len: 0,
            write_resp: None,
            write_pos: 0,
            close_after_write: false,
            peer_closed: false,
        })
    }

    #[inline]
    fn reset(&mut self) {
        self.in_len = 0;
        self.write_resp = None;
        self.write_pos = 0;
        self.close_after_write = false;
        self.peer_closed = false;
    }

    #[inline]
    fn consume(&mut self, n: usize) {
        if n >= self.in_len {
            self.in_len = 0;
            return;
        }
        self.in_buf.copy_within(n..self.in_len, 0);
        self.in_len -= n;
    }

    #[inline]
    fn has_pending_write(&self) -> bool {
        self.write_resp.is_some()
    }
}

impl EpollServer {
    fn new(
        epfd: RawFd,
        listener_fd: RawFd,
        index: &'static Index,
        responses: &'static Responses,
    ) -> Self {
        let conn_pool_cap = env_usize("SCM_CONN_POOL_CAP", DEFAULT_CONN_POOL);
        let mut conns = Vec::with_capacity(MAX_FD);
        conns.resize_with(MAX_FD, || None);
        let mut is_control = Vec::with_capacity(MAX_FD);
        is_control.resize(MAX_FD, false);
        let mut conn_pool = Vec::with_capacity(conn_pool_cap);
        for _ in 0..conn_pool_cap {
            conn_pool.push(Conn::new());
        }
        Self {
            epfd,
            listener_fd,
            index,
            responses,
            conns,
            conn_pool,
            conn_pool_cap,
            is_control,
            recv_fd_budget: env_usize("SCM_RECV_FD_BUDGET", DEFAULT_RECV_FD_BUDGET),
        }
    }

    fn run(&mut self) {
        let mut events = [libc::epoll_event { events: 0, u64: 0 }; MAX_EVENTS];
        let mut idle = IdleWait::from_env();
        loop {
            let n = wait_events(self.epfd, &mut events, &mut idle);
            if n < 0 {
                let err = io::Error::last_os_error();
                if err.raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                eprintln!("scm_epoll: epoll_wait error: {err}");
                continue;
            }

            for ev in events.iter().take(n as usize) {
                let token = ev.u64;
                if token == LISTENER_TOKEN {
                    self.accept_control_conns();
                    continue;
                }

                let fd = token as RawFd;
                if fd < 0 || fd as usize >= MAX_FD {
                    continue;
                }
                if self.is_control[fd as usize] {
                    let closed = self.handle_control(fd, ev.events);
                    if closed {
                        self.close_control(fd);
                    }
                } else {
                    self.handle_client(fd, ev.events);
                }
            }
        }
    }

    fn accept_control_conns(&mut self) {
        loop {
            let fd = unsafe {
                libc::accept4(
                    self.listener_fd,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
                )
            };
            if fd >= 0 {
                if fd as usize >= MAX_FD {
                    unsafe { libc::close(fd) };
                    continue;
                }
                self.is_control[fd as usize] = true;
                if epoll_add(self.epfd, fd, fd as u64, libc::EPOLLIN | libc::EPOLLRDHUP).is_err() {
                    self.close_control(fd);
                }
                continue;
            }
            if would_block_or_interrupted() {
                return;
            }
            return;
        }
    }

    fn handle_control(&mut self, fd: RawFd, events: u32) -> bool {
        if events & ((libc::EPOLLHUP | libc::EPOLLERR | libc::EPOLLRDHUP) as u32) != 0 {
            return true;
        }
        if events & (libc::EPOLLIN as u32) == 0 {
            return false;
        }

        for _ in 0..self.recv_fd_budget {
            match recv_fd_nb(fd) {
                RecvFdResult::Fd(client_fd) => self.register_client(client_fd),
                RecvFdResult::WouldBlock => return false,
                RecvFdResult::Closed => return true,
            }
        }
        false
    }

    fn register_client(&mut self, fd: RawFd) {
        if fd < 0 || fd as usize >= MAX_FD {
            unsafe { libc::close(fd) };
            return;
        }
        if set_nonblocking(fd).is_err() {
            unsafe { libc::close(fd) };
            return;
        }
        set_tcp_nodelay(fd);
        set_tcp_quickack(fd);
        set_socket_busy_poll(fd);

        let mut conn = self.alloc_conn();
        conn.reset();
        let idx = fd as usize;
        if self.conns[idx].is_some() {
            self.close_client(fd);
        }
        self.conns[idx] = Some(conn);

        if epoll_add(self.epfd, fd, fd as u64, libc::EPOLLIN | libc::EPOLLRDHUP).is_err() {
            self.close_client(fd);
            return;
        }

        self.handle_client(fd, libc::EPOLLIN as u32);
    }

    fn handle_client(&mut self, fd: RawFd, events: u32) {
        let mut close = false;
        if events & (libc::EPOLLERR as u32) != 0 {
            close = true;
        }
        if events & ((libc::EPOLLHUP | libc::EPOLLRDHUP) as u32) != 0 {
            if let Some(conn) = self.conn_mut(fd) {
                conn.peer_closed = true;
            }
        }

        if !close && events & (libc::EPOLLIN as u32) != 0 {
            close = !self.read_available(fd);
        }
        if !close {
            close = self.process_requests(fd);
        }
        if close {
            self.close_client(fd);
        }
    }

    fn read_available(&mut self, fd: RawFd) -> bool {
        let Some(conn) = self.conn_mut(fd) else {
            return false;
        };

        loop {
            if conn.in_len == READ_BUF {
                return false;
            }
            let dst = unsafe { conn.in_buf.as_mut_ptr().add(conn.in_len) };
            let space = READ_BUF - conn.in_len;
            let n = unsafe { libc::recv(fd, dst.cast(), space, 0) };
            if n > 0 {
                conn.in_len += n as usize;
                continue;
            }
            if n == 0 {
                conn.peer_closed = true;
                return true;
            }
            let err = io::Error::last_os_error();
            let code = err.raw_os_error().unwrap_or(0);
            if code == libc::EAGAIN || code == libc::EWOULDBLOCK {
                return true;
            }
            if code == libc::EINTR {
                continue;
            }
            return false;
        }
    }

    fn process_requests(&mut self, fd: RawFd) -> bool {
        loop {
            if self.conn(fd).is_none() {
                return true;
            }
            let peer_closed = self.conn(fd).map(|c| c.peer_closed).unwrap_or(false);

            if self
                .conn(fd)
                .map(|c| c.has_pending_write())
                .unwrap_or(false)
            {
                match self.flush_pending(fd) {
                    FlushResult::Done { close_after } => {
                        if close_after {
                            return true;
                        }
                        update_interest(self.epfd, fd, false);
                    }
                    FlushResult::Pending => {
                        update_interest(self.epfd, fd, true);
                        return false;
                    }
                    FlushResult::Closed => return true,
                }
            }

            let parsed = {
                let Some(conn) = self.conn(fd) else {
                    return true;
                };
                parse_request(&conn.in_buf[..conn.in_len])
            };

            let (resp, consumed, close_after) = match parsed {
                ParseStatus::Need => return peer_closed,
                ParseStatus::Bad => {
                    let consumed = self.conn(fd).map(|c| c.in_len).unwrap_or(0);
                    (self.responses.bad(false), consumed, true)
                }
                ParseStatus::Got {
                    route,
                    body_start,
                    body_end,
                    close,
                } => {
                    let resp = {
                        let Some(conn) = self.conn(fd) else {
                            return true;
                        };
                        let body = &conn.in_buf[body_start..body_end];
                        self.handle_route(route, body, !close)
                    };
                    (resp, body_end, close || peer_closed)
                }
            };

            if let Some(conn) = self.conn_mut(fd) {
                conn.consume(consumed);
            } else {
                return true;
            }

            match write_response(fd, resp, 0) {
                WriteResult::Done => {
                    if close_after {
                        return true;
                    }
                    continue;
                }
                WriteResult::Pending(pos) => {
                    if let Some(conn) = self.conn_mut(fd) {
                        conn.write_resp = Some(resp);
                        conn.write_pos = pos;
                        conn.close_after_write = close_after;
                    }
                    update_interest(self.epfd, fd, true);
                    return false;
                }
                WriteResult::Closed => return true,
            }
        }
    }

    fn handle_route(&self, route: Route, body: &[u8], keep_alive: bool) -> &'static [u8] {
        match route {
            Route::Ready => self.responses.ready(keep_alive),
            Route::NotFound => self.responses.not_found(keep_alive),
            Route::FraudScore => match classify(self.index, body) {
                Some(bucket) => self.responses.fraud(bucket, keep_alive),
                // Unclassifiable body: an HTTP error weighs 5 in the scoring
                // while a wrong guess weighs 1 (FP) or 3 (FN); at the dataset's
                // ~44% fraud rate denying is the cheapest blind guess
                // (0.56 expected weight vs 1.33 for approving).
                None => self.responses.fraud(FALLBACK_DENY_BUCKET, keep_alive),
            },
        }
    }

    fn flush_pending(&mut self, fd: RawFd) -> FlushResult {
        let Some(conn) = self.conn_mut(fd) else {
            return FlushResult::Closed;
        };
        let Some(resp) = conn.write_resp else {
            return FlushResult::Done {
                close_after: conn.close_after_write,
            };
        };
        match write_response(fd, resp, conn.write_pos) {
            WriteResult::Done => {
                let close_after = conn.close_after_write;
                conn.write_resp = None;
                conn.write_pos = 0;
                conn.close_after_write = false;
                FlushResult::Done { close_after }
            }
            WriteResult::Pending(pos) => {
                conn.write_pos = pos;
                FlushResult::Pending
            }
            WriteResult::Closed => FlushResult::Closed,
        }
    }

    fn close_control(&mut self, fd: RawFd) {
        if fd >= 0 && (fd as usize) < self.is_control.len() {
            self.is_control[fd as usize] = false;
        }
        epoll_del(self.epfd, fd);
        unsafe { libc::close(fd) };
    }

    fn close_client(&mut self, fd: RawFd) {
        epoll_del(self.epfd, fd);
        if fd >= 0 && (fd as usize) < self.conns.len() {
            if let Some(mut conn) = self.conns[fd as usize].take() {
                conn.reset();
                if self.conn_pool.len() < self.conn_pool_cap {
                    self.conn_pool.push(conn);
                }
            }
        }
        unsafe { libc::close(fd) };
    }

    fn alloc_conn(&mut self) -> Box<Conn> {
        self.conn_pool.pop().unwrap_or_else(Conn::new)
    }

    fn conn(&self, fd: RawFd) -> Option<&Conn> {
        self.conns.get(fd as usize)?.as_deref()
    }

    fn conn_mut(&mut self, fd: RawFd) -> Option<&mut Conn> {
        self.conns.get_mut(fd as usize)?.as_deref_mut()
    }
}

/// approved:false, fraud_score 0.6 — the first "deny" bucket.
const FALLBACK_DENY_BUCKET: usize = 3;

/// Total classification: parse + vectorize + predict, never panics.
/// Returns None for anything unclassifiable (parse failure, non-finite vector,
/// or an unexpected panic caught before it can kill the single-threaded event
/// loop — one poisoned request must cost one fallback response, not the
/// process).
fn classify(index: &Index, body: &[u8]) -> Option<usize> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let parsed = json::parse_request(body)?;
        let v = vectorize::vectorize(&parsed.to_vec_input());
        if v.iter().any(|x| x.is_nan()) {
            return None;
        }
        Some(index.predict(&v) as usize)
    }));
    match result {
        Ok(bucket) => bucket,
        Err(_) => {
            eprintln!("scm_epoll: classify panicked; served fallback deny");
            None
        }
    }
}

#[derive(Clone, Copy)]
enum Route {
    Ready,
    FraudScore,
    NotFound,
}

enum ParseStatus {
    Need,
    Bad,
    Got {
        route: Route,
        body_start: usize,
        body_end: usize,
        close: bool,
    },
}

fn parse_request(buf: &[u8]) -> ParseStatus {
    let Some(head_end) = find_subseq(buf, b"\r\n\r\n") else {
        if buf.len() > MAX_HEAD {
            return ParseStatus::Bad;
        }
        return ParseStatus::Need;
    };
    if head_end > MAX_HEAD {
        return ParseStatus::Bad;
    }

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
    let body_start = head_end + 4;
    let content_length = find_content_length(headers);
    if content_length > MAX_BODY {
        return ParseStatus::Bad;
    }
    let body_end = body_start + content_length;
    if buf.len() < body_end {
        return ParseStatus::Need;
    }

    if !matches!(route, Route::FraudScore) {
        return ParseStatus::Got {
            route,
            body_start,
            body_end,
            close,
        };
    }

    ParseStatus::Got {
        route,
        body_start,
        body_end,
        close,
    }
}

enum WriteResult {
    Done,
    Pending(usize),
    Closed,
}

enum FlushResult {
    Done { close_after: bool },
    Pending,
    Closed,
}

fn write_response(fd: RawFd, resp: &'static [u8], mut pos: usize) -> WriteResult {
    while pos < resp.len() {
        let n = unsafe {
            libc::send(
                fd,
                resp.as_ptr().add(pos).cast(),
                resp.len() - pos,
                libc::MSG_NOSIGNAL,
            )
        };
        if n > 0 {
            pos += n as usize;
            continue;
        }
        if n == 0 {
            return WriteResult::Closed;
        }
        let err = io::Error::last_os_error();
        let code = err.raw_os_error().unwrap_or(0);
        if code == libc::EAGAIN || code == libc::EWOULDBLOCK {
            return WriteResult::Pending(pos);
        }
        if code == libc::EINTR {
            continue;
        }
        return WriteResult::Closed;
    }
    WriteResult::Done
}

enum RecvFdResult {
    Fd(RawFd),
    WouldBlock,
    Closed,
}

fn recv_fd_nb(fd: RawFd) -> RecvFdResult {
    let mut data = [0u8; 1];
    let mut iov = libc::iovec {
        iov_base: data.as_mut_ptr().cast(),
        iov_len: data.len(),
    };
    let mut control = [0u8; 64];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr().cast();
    msg.msg_controllen = control.len();

    let n = unsafe { libc::recvmsg(fd, &mut msg, libc::MSG_DONTWAIT | libc::MSG_CMSG_CLOEXEC) };
    if n < 0 {
        let err = io::Error::last_os_error();
        let code = err.raw_os_error().unwrap_or(0);
        if code == libc::EAGAIN || code == libc::EWOULDBLOCK {
            return RecvFdResult::WouldBlock;
        }
        if code == libc::EINTR {
            return RecvFdResult::WouldBlock;
        }
        return RecvFdResult::Closed;
    }
    if n == 0 {
        return RecvFdResult::Closed;
    }

    unsafe {
        let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
        while !cmsg.is_null() {
            if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                let data = libc::CMSG_DATA(cmsg).cast::<RawFd>();
                return RecvFdResult::Fd(*data);
            }
            cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
        }
    }
    RecvFdResult::Closed
}

fn bind_ctrl_listener(path: &str) -> io::Result<RawFd> {
    let _ = std::fs::remove_file(path);
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let listener = std::os::unix::net::UnixListener::bind(path)?;
    listener.set_nonblocking(true)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o666))?;
    Ok(listener.into_raw_fd())
}

fn epoll_add(epfd: RawFd, fd: RawFd, token: u64, events: i32) -> io::Result<()> {
    let mut ev = libc::epoll_event {
        events: events as u32,
        u64: token,
    };
    let rc = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd, &mut ev) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn update_interest(epfd: RawFd, fd: RawFd, want_write: bool) {
    let mut events = libc::EPOLLRDHUP;
    if want_write {
        events |= libc::EPOLLOUT;
    } else {
        events |= libc::EPOLLIN;
    }
    let mut ev = libc::epoll_event {
        events: events as u32,
        u64: fd as u64,
    };
    unsafe {
        libc::epoll_ctl(epfd, libc::EPOLL_CTL_MOD, fd, &mut ev);
    }
}

fn epoll_del(epfd: RawFd, fd: RawFd) {
    let mut ev = libc::epoll_event { events: 0, u64: 0 };
    unsafe {
        libc::epoll_ctl(epfd, libc::EPOLL_CTL_DEL, fd, &mut ev);
    }
}

/// Idle-wait tuning for the event loop.
///
/// With `EPOLL_IDLE_US > 0` the loop never blocks indefinitely: it waits with a
/// microsecond timeout (`epoll_pwait2`), so the worker keeps waking even when no
/// request is in flight. On the mostly-idle stretches of the load ramp this keeps
/// the pinned core out of deep C-states and the wake-up path warm, at a duty
/// cycle far below the container CPU quota. `EPOLL_IDLE_US=0` (default) keeps
/// the previous behavior of blocking until an event arrives.
struct IdleWait {
    timeout: Option<libc::timespec>,
    fallback_ms: libc::c_int,
    pwait2_usable: bool,
}

impl IdleWait {
    fn from_env() -> Self {
        Self::for_micros(env_u32("EPOLL_IDLE_US", 0) as u64)
    }

    fn for_micros(idle_us: u64) -> Self {
        let timeout = (idle_us > 0).then(|| libc::timespec {
            tv_sec: (idle_us / 1_000_000) as libc::time_t,
            tv_nsec: ((idle_us % 1_000_000) * 1_000) as libc::c_long,
        });
        Self {
            timeout,
            fallback_ms: ((idle_us + 999) / 1_000).max(1) as libc::c_int,
            pwait2_usable: true,
        }
    }
}

/// Waits for epoll events honoring the idle tuning. Returns the raw
/// `epoll_wait`/`epoll_pwait2` result; callers keep handling `n < 0` (EINTR).
fn wait_events(epfd: RawFd, events: &mut [libc::epoll_event], idle: &mut IdleWait) -> libc::c_int {
    let max_events = events.len() as libc::c_int;
    let Some(timeout) = idle.timeout else {
        return unsafe { libc::epoll_wait(epfd, events.as_mut_ptr(), max_events, -1) };
    };
    if idle.pwait2_usable {
        let n = unsafe {
            libc::epoll_pwait2(
                epfd,
                events.as_mut_ptr(),
                max_events,
                &timeout,
                std::ptr::null(),
            )
        };
        if n >= 0 {
            return n;
        }
        let code = io::Error::last_os_error().raw_os_error().unwrap_or(0);
        if code != libc::ENOSYS && code != libc::EPERM {
            return n;
        }
        // Kernel too old (ENOSYS) or syscall filtered by seccomp (EPERM):
        // fall back to a millisecond timeout for good.
        idle.pwait2_usable = false;
        eprintln!(
            "scm_epoll: epoll_pwait2 unavailable (errno {code}); using epoll_wait({} ms)",
            idle.fallback_ms
        );
    }
    unsafe { libc::epoll_wait(epfd, events.as_mut_ptr(), max_events, idle.fallback_ms) }
}

fn set_nonblocking(fd: RawFd) -> io::Result<()> {
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL);
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        if libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

fn set_tcp_nodelay(fd: RawFd) {
    set_int_sockopt(fd, libc::IPPROTO_TCP, libc::TCP_NODELAY, 1);
}

fn set_tcp_quickack(fd: RawFd) {
    set_int_sockopt(fd, libc::IPPROTO_TCP, libc::TCP_QUICKACK, 1);
}

fn set_socket_busy_poll(fd: RawFd) {
    let us = env_i32("SO_BUSY_POLL_US", 0);
    if us > 0 {
        set_int_sockopt(fd, libc::SOL_SOCKET, libc::SO_BUSY_POLL, us);
    }
}

fn set_int_sockopt(fd: RawFd, level: i32, opt: i32, value: i32) {
    unsafe {
        libc::setsockopt(
            fd,
            level,
            opt,
            &value as *const i32 as *const libc::c_void,
            std::mem::size_of::<i32>() as libc::socklen_t,
        );
    }
}

#[repr(C)]
struct EpollParams {
    busy_poll_usecs: u32,
    busy_poll_budget: u16,
    prefer_busy_poll: u8,
    _pad: u8,
}

const fn iow(ty: u32, nr: u32, size: u32) -> libc::c_ulong {
    ((1u32 << 30) | (size << 16) | (ty << 8) | nr) as libc::c_ulong
}

const EPIOCSPARAMS: libc::c_ulong = iow(0x8A, 0x01, std::mem::size_of::<EpollParams>() as u32);

fn configure_epoll_busy_poll(epfd: RawFd) {
    let usecs = env_u32("EPOLL_BUSY_POLL_US", 0);
    let prefer = env_u32("EPOLL_PREFER_BUSY_POLL", 0) as u8;
    if usecs == 0 && prefer == 0 {
        return;
    }
    let params = EpollParams {
        busy_poll_usecs: usecs,
        busy_poll_budget: env_u32("EPOLL_BUSY_POLL_BUDGET", 8) as u16,
        prefer_busy_poll: prefer,
        _pad: 0,
    };
    unsafe {
        libc::ioctl(epfd, EPIOCSPARAMS as _, &params as *const EpollParams);
    }
}

fn would_block_or_interrupted() -> bool {
    let err = io::Error::last_os_error();
    let code = err.raw_os_error().unwrap_or(0);
    code == libc::EAGAIN || code == libc::EWOULDBLOCK || code == libc::EINTR
}

fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn find_content_length(headers: &[u8]) -> usize {
    let Some(value) = header_value(headers, b"content-length") else {
        return 0;
    };
    let mut n = 0usize;
    for &b in value {
        if !b.is_ascii_digit() {
            break;
        }
        n = n.saturating_mul(10).saturating_add((b - b'0') as usize);
    }
    n
}

fn wants_close(request_line: &[u8], headers: &[u8]) -> bool {
    if request_line.ends_with(b"HTTP/1.0") {
        return !header_has_token(headers, b"connection:", b"keep-alive");
    }
    header_has_token(headers, b"connection:", b"close")
}

fn header_has_token(headers: &[u8], name: &[u8], token: &[u8]) -> bool {
    let name = name.strip_suffix(b":").unwrap_or(name);
    header_value(headers, name)
        .map(|value| {
            value
                .split(|&b| b == b',')
                .any(|part| trim_ws(part).eq_ignore_ascii_case(token))
        })
        .unwrap_or(false)
}

fn header_value<'a>(headers: &'a [u8], name: &[u8]) -> Option<&'a [u8]> {
    for line in headers.split(|&b| b == b'\n').skip(1) {
        let line = strip_cr(line);
        let Some(colon) = line.iter().position(|&b| b == b':') else {
            continue;
        };
        let (line_name, rest) = line.split_at(colon);
        if trim_ws(line_name).eq_ignore_ascii_case(name) {
            return Some(trim_ws(&rest[1..]));
        }
    }
    None
}

fn strip_cr(line: &[u8]) -> &[u8] {
    if line.last() == Some(&b'\r') {
        &line[..line.len() - 1]
    } else {
        line
    }
}

fn trim_ws(s: &[u8]) -> &[u8] {
    let start = s
        .iter()
        .position(|&b| b != b' ' && b != b'\t')
        .unwrap_or(s.len());
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

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_i32(name: &str, default: i32) -> i32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn new_epoll() -> RawFd {
        let epfd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
        assert!(epfd >= 0, "epoll_create1: {}", io::Error::last_os_error());
        epfd
    }

    fn close_fd(fd: RawFd) {
        unsafe {
            libc::close(fd);
        }
    }

    fn socketpair_with_ready_byte(epfd: RawFd) -> [RawFd; 2] {
        let mut fds = [0 as RawFd; 2];
        let rc = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
        assert_eq!(rc, 0, "socketpair: {}", io::Error::last_os_error());
        epoll_add(epfd, fds[0], fds[0] as u64, libc::EPOLLIN).expect("epoll add socketpair");
        let byte = [1u8];
        let sent = unsafe { libc::send(fds[1], byte.as_ptr().cast(), 1, 0) };
        assert_eq!(sent, 1, "send: {}", io::Error::last_os_error());
        fds
    }

    #[test]
    fn idle_timeout_returns_zero_events_instead_of_blocking() {
        let epfd = new_epoll();
        let mut events = [libc::epoll_event { events: 0, u64: 0 }; 8];
        let mut idle = IdleWait::for_micros(20_000);
        let start = Instant::now();
        let n = wait_events(epfd, &mut events, &mut idle);
        let elapsed = start.elapsed();
        close_fd(epfd);
        assert_eq!(n, 0, "expected timeout with no events");
        assert!(
            elapsed >= Duration::from_millis(10),
            "returned too early: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(500),
            "took too long: {elapsed:?}"
        );
    }

    #[test]
    fn ready_event_returns_immediately_with_idle_timeout() {
        let epfd = new_epoll();
        let fds = socketpair_with_ready_byte(epfd);
        let mut events = [libc::epoll_event { events: 0, u64: 0 }; 8];
        let mut idle = IdleWait::for_micros(1_000_000);
        let start = Instant::now();
        let n = wait_events(epfd, &mut events, &mut idle);
        let elapsed = start.elapsed();
        close_fd(fds[0]);
        close_fd(fds[1]);
        close_fd(epfd);
        assert_eq!(n, 1, "expected the ready event");
        assert!(
            elapsed < Duration::from_millis(100),
            "ready event should not wait for the idle timeout: {elapsed:?}"
        );
    }

    #[test]
    fn ms_fallback_times_out_when_pwait2_unusable() {
        let epfd = new_epoll();
        let mut events = [libc::epoll_event { events: 0, u64: 0 }; 8];
        let mut idle = IdleWait::for_micros(20_000);
        idle.pwait2_usable = false;
        let start = Instant::now();
        let n = wait_events(epfd, &mut events, &mut idle);
        let elapsed = start.elapsed();
        close_fd(epfd);
        assert_eq!(n, 0, "expected timeout with no events");
        assert!(
            elapsed >= Duration::from_millis(10),
            "returned too early: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(500),
            "took too long: {elapsed:?}"
        );
    }

    #[test]
    fn zero_idle_blocks_until_event() {
        let epfd = new_epoll();
        let fds = socketpair_with_ready_byte(epfd);
        let mut events = [libc::epoll_event { events: 0, u64: 0 }; 8];
        let mut idle = IdleWait::for_micros(0);
        let n = wait_events(epfd, &mut events, &mut idle);
        close_fd(fds[0]);
        close_fd(fds[1]);
        close_fd(epfd);
        assert_eq!(n, 1, "expected the ready event from the blocking branch");
    }
}
