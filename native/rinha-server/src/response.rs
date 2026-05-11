// HTTP response builder with programmatically computed Content-Length.
// All response bytes are assembled once at startup and stored in `Responses`.

pub const FRAUD_JSON: [&[u8]; 6] = [
    b"{\"approved\":true,\"fraud_score\":0}",
    b"{\"approved\":true,\"fraud_score\":0.2}",
    b"{\"approved\":true,\"fraud_score\":0.4}",
    b"{\"approved\":false,\"fraud_score\":0.6}",
    b"{\"approved\":false,\"fraud_score\":0.8}",
    b"{\"approved\":false,\"fraud_score\":1}",
];

const READY_BODY: &[u8] = b"ready";

pub fn build_response(
    status: &str,
    content_type: Option<&str>,
    body: &[u8],
    keep_alive: bool,
) -> Vec<u8> {
    let conn: &[u8] = if keep_alive { b"keep-alive" } else { b"close" };
    let mut len_buf = itoa_buf();
    let len_str = u64_to_ascii(&mut len_buf, body.len() as u64);

    let mut out = Vec::with_capacity(96 + body.len());
    out.extend_from_slice(b"HTTP/1.1 ");
    out.extend_from_slice(status.as_bytes());
    out.extend_from_slice(b"\r\n");
    if let Some(ct) = content_type {
        out.extend_from_slice(b"Content-Type: ");
        out.extend_from_slice(ct.as_bytes());
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"Content-Length: ");
    out.extend_from_slice(len_str);
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(b"Connection: ");
    out.extend_from_slice(conn);
    out.extend_from_slice(b"\r\n\r\n");
    out.extend_from_slice(body);
    out
}

fn itoa_buf() -> [u8; 20] {
    [0u8; 20]
}

fn u64_to_ascii(buf: &mut [u8; 20], mut n: u64) -> &[u8] {
    if n == 0 {
        buf[0] = b'0';
        return &buf[..1];
    }
    let mut i = buf.len();
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    &buf[i..]
}

pub struct Responses {
    fraud_keep: [Vec<u8>; 6],
    fraud_close: [Vec<u8>; 6],
    ready_keep: Vec<u8>,
    ready_close: Vec<u8>,
    not_found_keep: Vec<u8>,
    not_found_close: Vec<u8>,
    bad_keep: Vec<u8>,
    bad_close: Vec<u8>,
}

impl Responses {
    pub fn build() -> Self {
        let fraud_keep = std::array::from_fn(|i| {
            build_response("200 OK", Some("application/json"), FRAUD_JSON[i], true)
        });
        let fraud_close = std::array::from_fn(|i| {
            build_response("200 OK", Some("application/json"), FRAUD_JSON[i], false)
        });
        Self {
            fraud_keep,
            fraud_close,
            ready_keep: build_response("200 OK", Some("text/plain"), READY_BODY, true),
            ready_close: build_response("200 OK", Some("text/plain"), READY_BODY, false),
            not_found_keep: build_response("404 Not Found", None, b"", true),
            not_found_close: build_response("404 Not Found", None, b"", false),
            bad_keep: build_response("400 Bad Request", None, b"", true),
            bad_close: build_response("400 Bad Request", None, b"", false),
        }
    }

    #[inline]
    pub fn fraud(&self, bucket: usize, keep_alive: bool) -> &[u8] {
        if keep_alive {
            &self.fraud_keep[bucket]
        } else {
            &self.fraud_close[bucket]
        }
    }
    #[inline]
    pub fn ready(&self, keep_alive: bool) -> &[u8] {
        if keep_alive {
            &self.ready_keep
        } else {
            &self.ready_close
        }
    }
    #[inline]
    pub fn not_found(&self, keep_alive: bool) -> &[u8] {
        if keep_alive {
            &self.not_found_keep
        } else {
            &self.not_found_close
        }
    }
    #[inline]
    pub fn bad(&self, keep_alive: bool) -> &[u8] {
        if keep_alive {
            &self.bad_keep
        } else {
            &self.bad_close
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_content_length(resp: &[u8]) -> Option<usize> {
        let header_end = resp.windows(4).position(|w| w == b"\r\n\r\n")?;
        let needle = b"Content-Length: ";
        let mut i = 0;
        while i + needle.len() <= header_end {
            if resp[i..i + needle.len()].eq_ignore_ascii_case(needle) {
                let mut n = 0usize;
                let mut j = i + needle.len();
                while j < header_end && resp[j].is_ascii_digit() {
                    n = n * 10 + (resp[j] - b'0') as usize;
                    j += 1;
                }
                return Some(n);
            }
            i += 1;
        }
        None
    }

    fn body_of<'a>(resp: &'a [u8]) -> &'a [u8] {
        let p = resp.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        &resp[p + 4..]
    }

    #[test]
    fn fraud_bodies_have_matching_content_length() {
        let r = Responses::build();
        for bucket in 0..6 {
            for &keep in &[true, false] {
                let resp = r.fraud(bucket, keep);
                let cl = parse_content_length(resp).expect("Content-Length present");
                let body = body_of(resp);
                assert_eq!(cl, body.len(), "bucket={bucket} keep={keep}");
                assert_eq!(body, FRAUD_JSON[bucket]);
                assert!(!body.ends_with(b"\n"));
            }
        }
    }

    #[test]
    fn ready_and_error_responses_have_matching_content_length() {
        let r = Responses::build();
        for resp in [
            r.ready(true),
            r.ready(false),
            r.not_found(true),
            r.not_found(false),
            r.bad(true),
            r.bad(false),
        ] {
            let cl = parse_content_length(resp).expect("Content-Length present");
            assert_eq!(cl, body_of(resp).len());
        }
    }

    #[test]
    fn fraud_bodies_are_exact_spec_strings() {
        assert_eq!(FRAUD_JSON[0], b"{\"approved\":true,\"fraud_score\":0}");
        assert_eq!(FRAUD_JSON[1], b"{\"approved\":true,\"fraud_score\":0.2}");
        assert_eq!(FRAUD_JSON[2], b"{\"approved\":true,\"fraud_score\":0.4}");
        assert_eq!(FRAUD_JSON[3], b"{\"approved\":false,\"fraud_score\":0.6}");
        assert_eq!(FRAUD_JSON[4], b"{\"approved\":false,\"fraud_score\":0.8}");
        assert_eq!(FRAUD_JSON[5], b"{\"approved\":false,\"fraud_score\":1}");
    }

    #[test]
    fn keep_alive_header_reflects_flag() {
        let r = Responses::build();
        assert!(find_subseq(r.fraud(0, true), b"Connection: keep-alive\r\n").is_some());
        assert!(find_subseq(r.fraud(0, false), b"Connection: close\r\n").is_some());
    }

    fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }
}
