// Phase 5 gate: compare forced scalar IVF top-5 against forced AVX2 IVF top-5
// over the preview corpus. This validates the SIMD distance kernel independently
// from bucket-only parity.

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use rinha_server::{
    ivf::Ivf,
    json,
    knn::{ivf_top5_avx2, ivf_top5_scalar, quantize_query},
    vectorize,
};

fn parse_args() -> (String, String, usize) {
    let mut ivf_path = "resources/references.ivf.bin".to_string();
    let mut requests_path = "test/test-data.requests.ndjson".to_string();
    let mut nprobe = 64usize;

    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ivf" => {
                ivf_path = args[i + 1].clone();
                i += 2;
            }
            "--requests" => {
                requests_path = args[i + 1].clone();
                i += 2;
            }
            "--nprobe" => {
                nprobe = args[i + 1].parse().expect("--nprobe int");
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    (ivf_path, requests_path, nprobe)
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn main() {
    eprintln!("AVX2 validation skipped: target architecture is not x86/x86_64");
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn main() {
    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("AVX2 validation skipped: CPU does not report avx2");
        return;
    }

    let (ivf_path, requests_path, nprobe) = parse_args();

    eprintln!("loading {} ...", ivf_path);
    let ivf = Ivf::load(&ivf_path);
    eprintln!("ivf: K={} count={} scale={}", ivf.k, ivf.count, ivf.scale);
    assert!(nprobe <= ivf.k, "nprobe {nprobe} exceeds K={}", ivf.k);

    eprintln!("loading queries from {} ...", requests_path);
    let file = File::open(&requests_path).expect("open requests ndjson");
    let reader = BufReader::new(file);

    let mut queries = Vec::new();
    for line in reader.lines() {
        let line = line.expect("read line");
        if line.trim().is_empty() {
            continue;
        }
        let parsed = json::parse_request(line.as_bytes()).expect("parse request");
        let v = vectorize::vectorize(&parsed.to_vec_input());
        queries.push(quantize_query(&v, ivf.scale));
    }

    eprintln!("loaded {} queries", queries.len());

    let scalar_start = Instant::now();
    let mut scalar_results = Vec::with_capacity(queries.len());
    let mut checksum = 0u64;
    for q in &queries {
        let top = ivf_top5_scalar(&ivf, q, nprobe);
        checksum ^= top
            .iter()
            .fold(0u64, |acc, &x| acc.wrapping_mul(16_777_619) ^ x as u64);
        scalar_results.push(top);
    }
    let scalar_ms = scalar_start.elapsed().as_secs_f64() * 1000.0;

    let avx2_start = Instant::now();
    let mut mismatches = Vec::new();
    for (case_idx, q) in queries.iter().enumerate() {
        let avx2 = ivf_top5_avx2(&ivf, q, nprobe).expect("avx2 available");
        checksum ^= avx2
            .iter()
            .fold(0u64, |acc, &x| acc.wrapping_mul(16_777_619) ^ x as u64);
        if scalar_results[case_idx] != avx2 {
            mismatches.push((case_idx, scalar_results[case_idx], avx2));
        }
        if (case_idx + 1) % 5000 == 0 {
            eprintln!(
                "processed {} cases, mismatches={}",
                case_idx + 1,
                mismatches.len()
            );
        }
    }
    let avx2_ms = avx2_start.elapsed().as_secs_f64() * 1000.0;

    if mismatches.is_empty() {
        println!(
            "AVX2_IVF_PARITY_OK total={} nprobe={} scalar_ms={:.1} avx2_ms={:.1} speedup={:.3} checksum={}",
            queries.len(),
            nprobe,
            scalar_ms,
            avx2_ms,
            scalar_ms / avx2_ms,
            checksum,
        );
    } else {
        eprintln!(
            "AVX2 IVF mismatches: {} / {}",
            mismatches.len(),
            queries.len()
        );
        for (case_idx, scalar, avx2) in mismatches.iter().take(20) {
            eprintln!("MISMATCH case_idx={case_idx} scalar={scalar:?} avx2={avx2:?}");
        }
        std::process::exit(1);
    }
}
