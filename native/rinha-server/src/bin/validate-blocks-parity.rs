// Phase A gate: compare R26I v1 (row-major IVF) top-5 against R26I v2 (block-SoA)
// top-5 over the preview corpus, using forced AVX2 on both. Mismatches must be 0.
//
// Usage:
//   validate-blocks-parity --ivf resources/references.ivf.bin \
//                          --blocks resources/references.ivf-blocks.bin \
//                          --requests test/test-data.requests.ndjson \
//                          --nprobe 32

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use rinha_server::{
    ivf::Ivf,
    ivf_blocks::IvfBlocks,
    json,
    knn::{ivf_blocks_top5_scalar, ivf_top5_scalar, quantize_query},
    vectorize,
};

fn parse_args() -> (String, String, String, usize) {
    let mut ivf_path = "resources/references.ivf.bin".to_string();
    let mut blocks_path = "resources/references.ivf-blocks.bin".to_string();
    let mut requests_path = "test/test-data.requests.ndjson".to_string();
    let mut nprobe = 32usize;

    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ivf" => { ivf_path = args[i + 1].clone(); i += 2; }
            "--blocks" => { blocks_path = args[i + 1].clone(); i += 2; }
            "--requests" => { requests_path = args[i + 1].clone(); i += 2; }
            "--nprobe" => { nprobe = args[i + 1].parse().expect("--nprobe int"); i += 2; }
            other => panic!("unknown arg: {other}"),
        }
    }
    (ivf_path, blocks_path, requests_path, nprobe)
}

fn main() {
    let (ivf_path, blocks_path, requests_path, nprobe) = parse_args();

    eprintln!("loading v1 {} ...", ivf_path);
    let ivf = Ivf::load(&ivf_path);
    eprintln!("v1: K={} count={} scale={}", ivf.k, ivf.count, ivf.scale);

    eprintln!("loading v2 {} ...", blocks_path);
    let blocks = IvfBlocks::load(&blocks_path);
    eprintln!(
        "v2: K={} count={} scale={} total_blocks={}",
        blocks.k, blocks.count, blocks.scale, blocks.total_blocks
    );

    assert_eq!(ivf.k, blocks.k, "K mismatch between v1 and v2");
    assert_eq!(ivf.count, blocks.count, "count mismatch between v1 and v2");
    assert_eq!(ivf.scale, blocks.scale, "scale mismatch between v1 and v2");
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

    eprintln!("loaded {} queries; nprobe={nprobe}", queries.len());

    let v1_start = Instant::now();
    let mut v1_results = Vec::with_capacity(queries.len());
    for q in &queries {
        v1_results.push(ivf_top5_scalar(&ivf, q, nprobe));
    }
    let v1_ms = v1_start.elapsed().as_secs_f64() * 1000.0;

    let v2_start = Instant::now();
    let mut mismatches = Vec::new();
    for (case_idx, q) in queries.iter().enumerate() {
        let v2 = ivf_blocks_top5_scalar(&blocks, q, nprobe);
        if v1_results[case_idx] != v2 {
            mismatches.push((case_idx, v1_results[case_idx], v2));
        }
        if (case_idx + 1) % 5000 == 0 {
            eprintln!("processed {} cases, mismatches={}", case_idx + 1, mismatches.len());
        }
    }
    let v2_ms = v2_start.elapsed().as_secs_f64() * 1000.0;

    if mismatches.is_empty() {
        println!(
            "BLOCKS_PARITY_OK total={} nprobe={} v1_ms={:.1} v2_ms={:.1} ratio={:.3}",
            queries.len(),
            nprobe,
            v1_ms,
            v2_ms,
            v1_ms / v2_ms,
        );
    } else {
        eprintln!("BLOCKS mismatches: {} / {}", mismatches.len(), queries.len());
        for (case_idx, v1, v2) in mismatches.iter().take(20) {
            eprintln!("MISMATCH case_idx={case_idx} v1={v1:?} v2={v2:?}");
        }
        std::process::exit(1);
    }
}
