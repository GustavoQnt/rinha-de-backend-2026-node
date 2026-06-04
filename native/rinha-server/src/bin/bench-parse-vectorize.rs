// Microbench: per-request CPU cost of parse + vectorize, in isolation.
// No index load needed — answers "how much app-CPU is spent before predict".
// Usage: bench-parse-vectorize test/test-data.requests.ndjson

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use rinha_server::{json, vectorize};

fn pct(sorted: &[u64], q: f64) -> u64 {
    sorted[((sorted.len() - 1) as f64 * q) as usize]
}

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "test/test-data.requests.ndjson".to_string());
    let file = File::open(&path).expect("open requests");
    let lines: Vec<Vec<u8>> = BufReader::new(file)
        .lines()
        .map(|l| l.expect("read line").into_bytes())
        .filter(|l| !l.is_empty())
        .collect();
    eprintln!("loaded {} requests from {path}", lines.len());

    // Warm the branch predictor / caches.
    let mut sink = 0u64;
    for _ in 0..3 {
        for body in &lines {
            if let Some(p) = json::parse_request(body) {
                let v = vectorize::vectorize(&p.to_vec_input());
                sink = sink.wrapping_add(v[0].to_bits());
            }
        }
    }

    let mut ns: Vec<u64> = Vec::with_capacity(lines.len());
    for body in &lines {
        let t = Instant::now();
        if let Some(p) = json::parse_request(body) {
            let v = vectorize::vectorize(&p.to_vec_input());
            sink = sink.wrapping_add(v[0].to_bits());
        }
        ns.push(t.elapsed().as_nanos() as u64);
    }
    unsafe { std::ptr::read_volatile(&sink) };

    ns.sort_unstable();
    let mean = ns.iter().sum::<u64>() as f64 / ns.len() as f64;
    let f = |n: u64| format!("{:.2}µs", n as f64 / 1000.0);
    println!(
        "PARSE+VECTORIZE n={} mean={} p50={} p90={} p99={} p999={} max={}",
        ns.len(),
        format!("{:.2}µs", mean / 1000.0),
        f(pct(&ns, 0.50)),
        f(pct(&ns, 0.90)),
        f(pct(&ns, 0.99)),
        f(pct(&ns, 0.999)),
        f(*ns.last().unwrap()),
    );
}
