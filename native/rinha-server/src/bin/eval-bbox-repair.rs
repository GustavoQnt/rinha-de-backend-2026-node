// Detection + latency eval for bounded bbox-repair IVF search.
//
// Reports per (seed, cap) pair: total queries, bucket mismatches vs ground
// truth, fraud/legit verdict crossings (approved_mismatches), edge-case
// breakdown, and latency percentiles. CSV on stdout.
//
// Usage:
//   eval-bbox-repair \
//     --refs resources/references.bin \
//     --blocks resources/references.ivf-blocks-K4096.bin \
//     --requests test/test-data.requests.ndjson \
//     --expected-buckets test/test-data.expected-buckets.ndjson \
//     --configs "1:16,3:32,5:64"

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use rinha_server::{
    ivf_blocks::IvfBlocks,
    json,
    knn::{full_scan_top5, predict_bucket_ivf_blocks_bbox_repair, quantize_query},
    refs::{Refs, DIM},
    vectorize,
};

struct Args {
    refs_path: String,
    blocks_path: String,
    requests_path: String,
    expected_buckets_path: Option<String>,
    configs: Vec<(usize, usize)>,
}

fn parse_args() -> Args {
    let mut refs_path = "resources/references.bin".to_string();
    let mut blocks_path = "resources/references.ivf-blocks-K4096.bin".to_string();
    let mut requests_path = "test/test-data.requests.ndjson".to_string();
    let mut expected_buckets_path: Option<String> = None;
    let mut configs: Vec<(usize, usize)> = vec![(1, 16), (3, 32), (5, 64), (5, 96)];
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--refs" => {
                refs_path = args[i + 1].clone();
                i += 2;
            }
            "--blocks" => {
                blocks_path = args[i + 1].clone();
                i += 2;
            }
            "--requests" => {
                requests_path = args[i + 1].clone();
                i += 2;
            }
            "--expected-buckets" => {
                expected_buckets_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--configs" => {
                configs = args[i + 1]
                    .split(',')
                    .map(|pair| {
                        let mut parts = pair.split(':');
                        let s: usize = parts.next().unwrap().parse().expect("seed");
                        let c: usize = parts.next().unwrap().parse().expect("cap");
                        (s, c)
                    })
                    .collect();
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    Args {
        refs_path,
        blocks_path,
        requests_path,
        expected_buckets_path,
        configs,
    }
}

fn bucket_from_label_set(labels: &[u8]) -> u8 {
    labels.iter().filter(|&&l| l == 1).count() as u8
}

fn main() {
    let args = parse_args();
    eprintln!("loading {} ...", args.refs_path);
    let refs = Refs::load(&args.refs_path);
    eprintln!("refs: count={} scale={}", refs.count, refs.scale);

    eprintln!("loading {} ...", args.blocks_path);
    let blocks = IvfBlocks::load(&args.blocks_path);
    eprintln!("blocks: K={} count={}", blocks.k, blocks.count);

    eprintln!("loading queries from {} ...", args.requests_path);
    let file = File::open(&args.requests_path).expect("open requests");
    // Two parallel arrays: raw f64 vectors for bbox-repair (which quantizes
    // internally), and quantized i16 vectors for the full-scan ground truth.
    let mut queries: Vec<[f64; DIM]> = Vec::with_capacity(64_000);
    let mut queries_q: Vec<[i16; DIM]> = Vec::with_capacity(64_000);
    for line in BufReader::new(file).lines() {
        let line = line.expect("read line");
        if line.trim().is_empty() {
            continue;
        }
        let parsed = json::parse_request(line.as_bytes()).expect("parse request");
        let v = vectorize::vectorize(&parsed.to_vec_input());
        queries_q.push(quantize_query(&v, refs.scale));
        queries.push(v);
    }
    eprintln!("queries: {}", queries.len());

    let expected_buckets: Option<Vec<u8>> = args.expected_buckets_path.as_ref().map(|path| {
        eprintln!("loading expected buckets from {} ...", path);
        let file = File::open(path).expect("open expected-buckets");
        BufReader::new(file)
            .lines()
            .map(|l| l.unwrap().trim().parse::<u8>().expect("bucket u8"))
            .collect()
    });
    assert_eq!(
        expected_buckets
            .as_ref()
            .map(|b| b.len())
            .unwrap_or(queries.len()),
        queries.len(),
    );

    let ground_truth: Vec<u8> = if let Some(eb) = expected_buckets.as_ref() {
        eb.clone()
    } else {
        eprintln!("computing ground-truth full-scan buckets ...");
        queries_q
            .iter()
            .map(|q| {
                let top = full_scan_top5(&refs, q);
                let labels: Vec<u8> = top.iter().map(|&idx| refs.label(idx as usize)).collect();
                bucket_from_label_set(&labels)
            })
            .collect()
    };

    let edge_mask: Vec<bool> = ground_truth.iter().map(|&b| b == 2 || b == 3).collect();
    let edge_total: usize = edge_mask.iter().filter(|&&e| e).count();

    println!(
        "seed,cap,total,bucket_mm,approved_mm,edge_total,edge_mm,ms_mean,ms_p50,ms_p95,ms_p99,ms_max,secs",
    );

    for (seed, cap) in args.configs {
        eprintln!("--- seed={seed} cap={cap} ---");
        let t_start = Instant::now();
        let mut latencies_ns: Vec<u64> = Vec::with_capacity(queries.len());
        let mut bucket_mm = 0usize;
        let mut approved_mm = 0usize;
        let mut edge_mm = 0usize;

        for (i, q) in queries.iter().enumerate() {
            let t1 = Instant::now();
            let b = predict_bucket_ivf_blocks_bbox_repair(&blocks, q, seed, cap);
            latencies_ns.push(t1.elapsed().as_nanos() as u64);

            let gt = ground_truth[i];
            if b != gt {
                bucket_mm += 1;
                if (b < 3) != (gt < 3) {
                    approved_mm += 1;
                }
                if edge_mask[i] {
                    edge_mm += 1;
                }
            }
        }

        let sweep_secs = t_start.elapsed().as_secs_f64();
        latencies_ns.sort_unstable();
        let total = latencies_ns.len();
        let pct = |p: f64| latencies_ns[((total as f64) * p) as usize] as f64;
        let mean = latencies_ns.iter().sum::<u64>() as f64 / total as f64;
        let p50 = pct(0.50);
        let p95 = pct(0.95);
        let p99 = pct(0.99);
        let max = *latencies_ns.last().unwrap() as f64;

        eprintln!(
            "  done: bucket_mm={} approved_mm={} edge_mm={}/{} mean={:.3}ms p99={:.3}ms in {:.1}s",
            bucket_mm,
            approved_mm,
            edge_mm,
            edge_total,
            mean / 1e6,
            p99 / 1e6,
            sweep_secs,
        );
        println!(
            "{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1}",
            seed,
            cap,
            total,
            bucket_mm,
            approved_mm,
            edge_total,
            edge_mm,
            mean / 1e6,
            p50 / 1e6,
            p95 / 1e6,
            p99 / 1e6,
            max / 1e6,
            sweep_secs,
        );
    }
}
