// Phase 4b harness: sweep IVF nprobe over the preview corpus and report
// (top-5 mismatches, bucket mismatches, edge-case bucket mismatches, latency).
//
// Strategy: load references + IVF + requests once, pre-vectorize all queries,
// run full_scan_top5 once to build the ground-truth baseline (top-5 indexes
// and derived bucket per case), then loop over each --nprobes value reusing
// the baseline. This makes high-nprobe runs the bottleneck of the sweep, and
// the low-nprobe end (where the latency answer lives) cheap.
//
// Inputs:
//   resources/references.bin
//   resources/references.ivf.bin
//   test/test-data.requests.ndjson
//
// Output: CSV on stdout, one row per nprobe. Progress on stderr.
//
// Edge case = case whose full_scan-derived bucket is 2 (fraud_score=0.4) or
// 3 (fraud_score=0.6) — the boundary the multi-grid sweep already flagged.

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use rinha_server::{
    ivf::Ivf,
    json,
    knn::{full_scan_top5, ivf_top5, quantize_query, TOP_K},
    refs::{Refs, DIM},
    vectorize,
};

const DEFAULT_NPROBES: &[usize] = &[768, 512, 384, 256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8];

fn parse_args() -> (String, String, String, Vec<usize>) {
    let mut refs_path = "resources/references.bin".to_string();
    let mut ivf_path = "resources/references.ivf.bin".to_string();
    let mut requests_path = "test/test-data.requests.ndjson".to_string();
    let mut nprobes: Option<Vec<usize>> = None;
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--refs" => {
                refs_path = args[i + 1].clone();
                i += 2;
            }
            "--ivf" => {
                ivf_path = args[i + 1].clone();
                i += 2;
            }
            "--requests" => {
                requests_path = args[i + 1].clone();
                i += 2;
            }
            "--nprobes" => {
                let parsed: Vec<usize> = args[i + 1]
                    .split(',')
                    .map(|s| s.trim().parse().expect("--nprobes int"))
                    .collect();
                nprobes = Some(parsed);
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    let nprobes = nprobes.unwrap_or_else(|| DEFAULT_NPROBES.to_vec());
    (refs_path, ivf_path, requests_path, nprobes)
}

#[inline(always)]
fn bucket_from_top5(refs: &Refs, indexes: &[u32; TOP_K]) -> u8 {
    let mut b = 0u8;
    for i in 0..TOP_K {
        if refs.label(indexes[i] as usize) == 1 {
            b += 1;
        }
    }
    b
}

fn main() {
    let (refs_path, ivf_path, requests_path, nprobes) = parse_args();

    eprintln!("loading {} ...", refs_path);
    let refs = Refs::load(&refs_path);
    eprintln!("refs: count={} scale={}", refs.count, refs.scale);

    eprintln!("loading {} ...", ivf_path);
    let ivf = Ivf::load(&ivf_path);
    eprintln!("ivf: K={} count={}", ivf.k, ivf.count);
    assert_eq!(refs.count, ivf.count, "refs/ivf count mismatch");
    assert_eq!(refs.scale, ivf.scale, "refs/ivf scale mismatch");

    for &n in &nprobes {
        assert!(n <= ivf.k, "nprobe {n} exceeds K={}", ivf.k);
    }

    eprintln!("loading queries from {} ...", requests_path);
    let file = File::open(&requests_path).expect("open requests ndjson");
    let reader = BufReader::new(file);

    let mut queries: Vec<[i16; DIM]> = Vec::new();
    for line in reader.lines() {
        let line = line.expect("read line");
        if line.trim().is_empty() {
            continue;
        }
        let parsed = json::parse_request(line.as_bytes()).expect("parse request");
        let v = vectorize::vectorize(&parsed.to_vec_input());
        queries.push(quantize_query(&v, refs.scale));
    }
    let total = queries.len();
    eprintln!("queries: {}", total);

    eprintln!("computing full_scan baseline (one-time) ...");
    let t0 = Instant::now();
    let mut full_top: Vec<[u32; TOP_K]> = Vec::with_capacity(total);
    let mut full_bucket: Vec<u8> = Vec::with_capacity(total);
    for (i, q) in queries.iter().enumerate() {
        let top = full_scan_top5(&refs, q);
        let b = bucket_from_top5(&refs, &top);
        full_top.push(top);
        full_bucket.push(b);
        if (i + 1) % 1000 == 0 {
            let secs = t0.elapsed().as_secs_f64();
            eprintln!(
                "full_scan {} / {} ({:.1}/s, eta {:.0}s)",
                i + 1,
                total,
                (i + 1) as f64 / secs,
                (total - i - 1) as f64 * secs / (i + 1) as f64,
            );
        }
    }
    let baseline_secs = t0.elapsed().as_secs_f64();
    eprintln!("full_scan baseline done in {:.1}s", baseline_secs);

    let edge_mask: Vec<bool> = full_bucket.iter().map(|&b| b == 2 || b == 3).collect();
    let edge_count = edge_mask.iter().filter(|&&x| x).count();
    eprintln!("edge cases (bucket in {{2,3}}): {} / {}", edge_count, total);

    println!("nprobe,total,top5_mismatches,bucket_mismatches,edge_cases,edge_bucket_mismatches,ms_mean,ms_p50,ms_p95,ms_p99,ms_max,sweep_secs");

    for &nprobe in &nprobes {
        eprintln!("\n--- nprobe={} ---", nprobe);
        let t0 = Instant::now();
        let mut latencies_ns: Vec<u64> = Vec::with_capacity(total);
        let mut top5_mm: usize = 0;
        let mut bucket_mm: usize = 0;
        let mut edge_bucket_mm: usize = 0;

        for (i, q) in queries.iter().enumerate() {
            let t1 = Instant::now();
            let top = ivf_top5(&ivf, q, nprobe);
            let dt = t1.elapsed().as_nanos() as u64;
            latencies_ns.push(dt);

            if top != full_top[i] {
                top5_mm += 1;
            }
            let b = bucket_from_top5(&refs, &top);
            if b != full_bucket[i] {
                bucket_mm += 1;
                if edge_mask[i] {
                    edge_bucket_mm += 1;
                }
            }

            if (i + 1) % 5000 == 0 {
                let secs = t0.elapsed().as_secs_f64();
                eprintln!(
                    "  nprobe={} {} / {} ({:.0}/s)",
                    nprobe,
                    i + 1,
                    total,
                    (i + 1) as f64 / secs
                );
            }
        }
        let sweep_secs = t0.elapsed().as_secs_f64();

        latencies_ns.sort_unstable();
        let mean_ns = latencies_ns.iter().sum::<u64>() as f64 / latencies_ns.len() as f64;
        let p50_ns = latencies_ns[latencies_ns.len() / 2] as f64;
        let p95_ns = latencies_ns[(latencies_ns.len() * 95) / 100] as f64;
        let p99_ns = latencies_ns[(latencies_ns.len() * 99) / 100] as f64;
        let max_ns = *latencies_ns.last().unwrap() as f64;

        eprintln!(
            "  done: top5_mm={} bucket_mm={} edge_bucket_mm={} mean={:.3}ms p99={:.3}ms in {:.1}s",
            top5_mm,
            bucket_mm,
            edge_bucket_mm,
            mean_ns / 1.0e6,
            p99_ns / 1.0e6,
            sweep_secs,
        );

        println!(
            "{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1}",
            nprobe,
            total,
            top5_mm,
            bucket_mm,
            edge_count,
            edge_bucket_mm,
            mean_ns / 1.0e6,
            p50_ns / 1.0e6,
            p95_ns / 1.0e6,
            p99_ns / 1.0e6,
            max_ns / 1.0e6,
            sweep_secs,
        );
    }
}
