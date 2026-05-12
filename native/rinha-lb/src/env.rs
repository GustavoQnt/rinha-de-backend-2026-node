use std::sync::Arc;

pub struct LbEnv {
    pub port: u16,
    pub upstreams: Arc<Vec<Arc<str>>>,
    pub workers: usize,
    pub backlog: i32,
}

pub fn from_env() -> LbEnv {
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9999);

    let raw = std::env::var("UPSTREAMS")
        .expect("UPSTREAMS env var required (comma-separated .ctrl UDS paths)");
    let upstreams: Vec<Arc<str>> = raw
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(Arc::from)
        .collect();
    assert!(!upstreams.is_empty(), "UPSTREAMS must contain at least one path");

    let workers = std::env::var("WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(1);

    let backlog: i32 = std::env::var("BACKLOG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(65535);

    LbEnv {
        port,
        upstreams: Arc::new(upstreams),
        workers,
        backlog,
    }
}
