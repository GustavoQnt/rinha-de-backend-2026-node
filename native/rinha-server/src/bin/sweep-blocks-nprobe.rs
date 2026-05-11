// Phase B sweep: nprobe sweep over R26I v2 (block-SoA) IVF, comparing top-5
// and bucket against full-scan ground truth from references.bin. AVX2 forced.
//
// Reports per nprobe: top-5 mismatches, bucket mismatches, edge-bucket
// mismatches, latency percentiles. CSV on stdout.
//
// Usage:
//   sweep-blocks-nprobe --refs resources/references.bin \
//                       --blocks resources/references.ivf-blocks-K4096.bin \
//                       --requests test/test-data.requests.ndjson \
//                       --expected-buckets test/test-data.expected-buckets.ndjson \
//                       --nprobes 256,192,128,96,64,48,32

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use rinha_server::{
    ivf_blocks::IvfBlocks,
    json,
    knn::{full_scan_top5, ivf_blocks_top5, quantize_query, TOP_K},
    refs::{Refs, DIM},
    vectorize,
};

const DEFAULT_NPROBES: &[usize] = &[512, 384, 256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8];
const DEFAULT_MAX_MISMATCHES: usize = 20;

struct Args {
    refs_path: String,
    blocks_path: String,
    requests_path: String,
    expected_buckets_path: Option<String>,
    nprobes: Vec<usize>,
    analyze_mismatches: bool,
    max_mismatches: usize,
}

fn parse_args() -> Args {
    let mut refs_path = "resources/references.bin".to_string();
    let mut blocks_path = "resources/references.ivf-blocks-K4096.bin".to_string();
    let mut requests_path = "test/test-data.requests.ndjson".to_string();
    let mut expected_buckets_path: Option<String> = None;
    let mut nprobes: Option<Vec<usize>> = None;
    let mut analyze_mismatches = false;
    let mut max_mismatches = DEFAULT_MAX_MISMATCHES;
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--refs" => { refs_path = args[i + 1].clone(); i += 2; }
            "--blocks" => { blocks_path = args[i + 1].clone(); i += 2; }
            "--requests" => { requests_path = args[i + 1].clone(); i += 2; }
            "--expected-buckets" => {
                expected_buckets_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--nprobes" => {
                nprobes = Some(
                    args[i + 1].split(',').map(|s| s.trim().parse().expect("--nprobes int")).collect()
                );
                i += 2;
            }
            "--analyze-mismatches" | "--mismatches" => {
                analyze_mismatches = true;
                i += 1;
            }
            "--max-mismatches" => {
                max_mismatches = args[i + 1].parse().expect("--max-mismatches int");
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    let nprobes = nprobes.unwrap_or_else(|| DEFAULT_NPROBES.to_vec());
    Args {
        refs_path,
        blocks_path,
        requests_path,
        expected_buckets_path,
        nprobes,
        analyze_mismatches,
        max_mismatches,
    }
}

fn extract_request_id(line: &str) -> Option<String> {
    let bytes = line.as_bytes();
    let key_pos = bytes.windows(4).position(|w| w == b"\"id\"")?;
    let mut i = key_pos + 4;
    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\r' | b'\n') {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b':' {
        return None;
    }
    i += 1;
    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\r' | b'\n') {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'"' {
        return None;
    }
    i += 1;
    let start = i;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Some(String::from_utf8_lossy(&bytes[start..i]).into_owned()),
            b'\\' => i += 2,
            _ => i += 1,
        }
    }
    None
}

fn load_expected_buckets(path: &str) -> Vec<u8> {
    let file = File::open(path).unwrap_or_else(|e| panic!("open expected buckets {path}: {e}"));
    let reader = BufReader::new(file);
    let mut buckets = Vec::new();
    for line in reader.lines() {
        let line = line.expect("read expected bucket line");
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let b: u8 = trimmed.parse().expect("expected bucket int");
        assert!(b <= 5, "expected bucket must be 0..5, got {b}");
        buckets.push(b);
    }
    buckets
}

#[inline(always)]
fn bucket_from_top5(refs: &Refs, indexes: &[u32; TOP_K]) -> u8 {
    let mut b = 0u8;
    for i in 0..TOP_K {
        if refs.label(indexes[i] as usize) == 1 { b += 1; }
    }
    b
}

fn main() {
    let args = parse_args();

    eprintln!("loading {} ...", args.refs_path);
    let refs = Refs::load(&args.refs_path);
    eprintln!("refs: count={} scale={}", refs.count, refs.scale);

    eprintln!("loading {} ...", args.blocks_path);
    let blocks = IvfBlocks::load(&args.blocks_path);
    eprintln!(
        "blocks: K={} count={} total_blocks={}",
        blocks.k, blocks.count, blocks.total_blocks
    );
    assert_eq!(refs.count, blocks.count);
    assert_eq!(refs.scale, blocks.scale);

    // Cluster size distribution.
    let sizes: Vec<usize> = (0..blocks.k)
        .map(|c| {
            let mut count = 0usize;
            for b in blocks.cluster_block_range(c) {
                for &o in blocks.block_origin_slots(b) {
                    if o != u32::MAX { count += 1; }
                }
            }
            count
        })
        .collect();
    let mut sorted = sizes.clone();
    sorted.sort_unstable();
    let p = |q: f64| -> usize { sorted[((sorted.len() as f64 - 1.0) * q).round() as usize] };
    eprintln!(
        "cluster size: min={} p25={} p50={} p75={} p90={} p95={} p99={} max={} mean={:.1}",
        sorted[0], p(0.25), p(0.50), p(0.75), p(0.90), p(0.95), p(0.99), *sorted.last().unwrap(),
        sorted.iter().sum::<usize>() as f64 / sorted.len() as f64,
    );

    for &n in &args.nprobes {
        assert!(n <= blocks.k, "nprobe {n} exceeds K={}", blocks.k);
    }

    eprintln!("loading queries from {} ...", args.requests_path);
    let file = File::open(&args.requests_path).expect("open requests");
    let reader = BufReader::new(file);

    let mut queries: Vec<[i16; DIM]> = Vec::new();
    let mut request_ids: Vec<String> = Vec::new();
    for line in reader.lines() {
        let line = line.expect("read line");
        if line.trim().is_empty() { continue; }
        request_ids.push(extract_request_id(&line).unwrap_or_else(|| format!("case-{}", queries.len())));
        let parsed = json::parse_request(line.as_bytes()).expect("parse");
        let v = vectorize::vectorize(&parsed.to_vec_input());
        queries.push(quantize_query(&v, refs.scale));
    }
    let total = queries.len();
    eprintln!("queries: {}", total);

    let mut full_top: Option<Vec<[u32; TOP_K]>> = None;
    let full_bucket: Vec<u8> = if let Some(path) = &args.expected_buckets_path {
        eprintln!("loading expected buckets from {} ...", path);
        let buckets = load_expected_buckets(path);
        assert_eq!(
            buckets.len(),
            total,
            "expected bucket count must match query count"
        );
        eprintln!("using fixture buckets; top5_mismatches will be NA");
        buckets
    } else {
        eprintln!("computing full_scan baseline ...");
        let t0 = Instant::now();
        let mut tops: Vec<[u32; TOP_K]> = Vec::with_capacity(total);
        let mut buckets: Vec<u8> = Vec::with_capacity(total);
        for (i, q) in queries.iter().enumerate() {
            let top = full_scan_top5(&refs, q);
            buckets.push(bucket_from_top5(&refs, &top));
            tops.push(top);
            if (i + 1) % 2000 == 0 {
                let s = t0.elapsed().as_secs_f64();
                eprintln!("baseline {}/{} eta={:.0}s", i + 1, total, (total - i - 1) as f64 * s / (i + 1) as f64);
            }
        }
        eprintln!("baseline done in {:.1}s", t0.elapsed().as_secs_f64());
        full_top = Some(tops);
        buckets
    };
    let full_top = full_top.as_ref();

    let edge_mask: Vec<bool> = full_bucket.iter().map(|&b| b == 2 || b == 3).collect();
    let edge_count = edge_mask.iter().filter(|&&x| x).count();
    eprintln!("edge cases (bucket in {{2,3}}): {} / {}", edge_count, total);

    println!("nprobe,total,top5_mismatches,bucket_mismatches,edge_cases,edge_bucket_mismatches,ms_mean,ms_p50,ms_p95,ms_p99,ms_max,sweep_secs");

    for &nprobe in &args.nprobes {
        eprintln!("\n--- nprobe={} ---", nprobe);
        let t0 = Instant::now();
        let mut latencies_ns: Vec<u64> = Vec::with_capacity(total);
        let mut top5_mm = 0usize;
        let mut bucket_mm = 0usize;
        let mut edge_bucket_mm = 0usize;
        let mut bucket_matrix = [[0usize; 6]; 6];
        let mut top5_samples: Vec<String> = Vec::new();
        let mut bucket_samples: Vec<String> = Vec::new();

        for (i, q) in queries.iter().enumerate() {
            let t1 = Instant::now();
            let top = ivf_blocks_top5(&blocks, q, nprobe);
            latencies_ns.push(t1.elapsed().as_nanos() as u64);

            if full_top.is_some_and(|tops| top != tops[i]) {
                top5_mm += 1;
                if args.analyze_mismatches && top5_samples.len() < args.max_mismatches {
                    let fast_bucket = bucket_from_top5(&refs, &top);
                    let expected_top = full_top.expect("full_top");
                    top5_samples.push(format!(
                        "    top5 case_idx={} id={} expected_bucket={} fast_bucket={} expected_top5={:?} fast_top5={:?}",
                        i, request_ids[i], full_bucket[i], fast_bucket, expected_top[i], top,
                    ));
                }
            }
            let b = bucket_from_top5(&refs, &top);
            if b != full_bucket[i] {
                bucket_mm += 1;
                bucket_matrix[full_bucket[i] as usize][b as usize] += 1;
                if edge_mask[i] { edge_bucket_mm += 1; }
                if args.analyze_mismatches && bucket_samples.len() < args.max_mismatches {
                    if let Some(expected_top) = full_top {
                        bucket_samples.push(format!(
                            "    bucket case_idx={} id={} expected_bucket={} fast_bucket={} expected_top5={:?} fast_top5={:?}",
                            i, request_ids[i], full_bucket[i], b, expected_top[i], top,
                        ));
                    } else {
                        bucket_samples.push(format!(
                            "    bucket case_idx={} id={} expected_bucket={} fast_bucket={} fast_top5={:?}",
                            i, request_ids[i], full_bucket[i], b, top,
                        ));
                    }
                }
            }
        }
        let sweep_secs = t0.elapsed().as_secs_f64();

        latencies_ns.sort_unstable();
        let mean = latencies_ns.iter().sum::<u64>() as f64 / latencies_ns.len() as f64;
        let p50 = latencies_ns[latencies_ns.len() / 2] as f64;
        let p95 = latencies_ns[(latencies_ns.len() * 95) / 100] as f64;
        let p99 = latencies_ns[(latencies_ns.len() * 99) / 100] as f64;
        let max = *latencies_ns.last().unwrap() as f64;

        eprintln!(
            "  done: top5_mm={} bucket_mm={} edge_bucket_mm={} mean={:.3}ms p99={:.3}ms in {:.1}s",
            top5_mm, bucket_mm, edge_bucket_mm, mean / 1e6, p99 / 1e6, sweep_secs
        );
        if args.analyze_mismatches {
            if bucket_mm == 0 {
                eprintln!("  bucket mismatch matrix: none");
            } else {
                eprintln!("  bucket mismatch matrix expected->fast:");
                for expected in 0..6 {
                    let row_sum: usize = bucket_matrix[expected].iter().sum();
                    if row_sum == 0 {
                        continue;
                    }
                    eprintln!(
                        "    expected {}: to0={} to1={} to2={} to3={} to4={} to5={}",
                        expected,
                        bucket_matrix[expected][0],
                        bucket_matrix[expected][1],
                        bucket_matrix[expected][2],
                        bucket_matrix[expected][3],
                        bucket_matrix[expected][4],
                        bucket_matrix[expected][5],
                    );
                }
            }
            if !bucket_samples.is_empty() {
                eprintln!("  first bucket mismatches:");
                for sample in &bucket_samples {
                    eprintln!("{sample}");
                }
            }
            if !top5_samples.is_empty() {
                eprintln!("  first top5 mismatches:");
                for sample in &top5_samples {
                    eprintln!("{sample}");
                }
            }
        }
        println!(
            "{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1}",
            nprobe,
            total,
            if full_top.is_some() { top5_mm.to_string() } else { "NA".to_string() },
            bucket_mm,
            edge_count,
            edge_bucket_mm,
            mean / 1e6, p50 / 1e6, p95 / 1e6, p99 / 1e6, max / 1e6, sweep_secs,
        );
    }
}
