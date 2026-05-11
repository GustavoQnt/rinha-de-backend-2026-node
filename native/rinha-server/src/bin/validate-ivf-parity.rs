// Phase 4a gate: full-scan vs IVF-at-nprobe=K bit-identical top-5 over the entire
// preview corpus. In-process, no HTTP, no thread spawn.
//
// Inputs:
//   resources/references.bin
//   resources/references.ivf.bin   (built by `node scripts/build-ivf-index.js`)
//   test/test-data.requests.ndjson (built by `node scripts/extract-test-requests.js`)
//
// Optional: --nprobe N (defaults to K).

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use rinha_server::{
    ivf::Ivf,
    json,
    knn::{full_scan_top5, ivf_top5, quantize_query},
    refs::Refs,
    vectorize,
};

fn parse_args() -> (String, String, String, Option<usize>) {
    let mut refs_path = "resources/references.bin".to_string();
    let mut ivf_path = "resources/references.ivf.bin".to_string();
    let mut requests_path = "test/test-data.requests.ndjson".to_string();
    let mut nprobe: Option<usize> = None;
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
            "--nprobe" => {
                nprobe = Some(args[i + 1].parse().expect("--nprobe int"));
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    (refs_path, ivf_path, requests_path, nprobe)
}

fn extract_id(line: &[u8]) -> &[u8] {
    let needle = b"\"id\":";
    if let Some(p) = line.windows(needle.len()).position(|w| w == needle) {
        let mut j = p + needle.len();
        while j < line.len() && line[j] == b' ' {
            j += 1;
        }
        if j < line.len() && line[j] == b'"' {
            j += 1;
            let start = j;
            while j < line.len() && line[j] != b'"' {
                j += 1;
            }
            return &line[start..j];
        }
    }
    b""
}

fn main() {
    let (refs_path, ivf_path, requests_path, nprobe_arg) = parse_args();

    eprintln!("loading {} ...", refs_path);
    let refs = Refs::load(&refs_path);
    eprintln!("refs: count={} scale={}", refs.count, refs.scale);

    eprintln!("loading {} ...", ivf_path);
    let ivf = Ivf::load(&ivf_path);
    eprintln!("ivf: K={} count={}", ivf.k, ivf.count);
    assert_eq!(refs.count, ivf.count, "refs/ivf count mismatch");
    assert_eq!(refs.scale, ivf.scale, "refs/ivf scale mismatch");

    let nprobe = nprobe_arg.unwrap_or(ivf.k);
    eprintln!("validating top-5 parity at nprobe={} (K={})", nprobe, ivf.k);

    let file = File::open(&requests_path).expect("open requests ndjson");
    let reader = BufReader::new(file);

    let mut total = 0usize;
    let mut mismatches: Vec<(String, [u32; 5], [u32; 5])> = Vec::new();
    let t0 = Instant::now();

    for line in reader.lines() {
        let line = line.expect("read line");
        if line.trim().is_empty() {
            continue;
        }

        let parsed = match json::parse_request(line.as_bytes()) {
            Some(p) => p,
            None => panic!("parse failed at case {total}: {line}"),
        };
        let id = String::from_utf8_lossy(extract_id(line.as_bytes())).into_owned();
        let v = vectorize::vectorize(&parsed.to_vec_input());
        let q = quantize_query(&v, refs.scale);

        let full = full_scan_top5(&refs, &q);
        let ivf_t = ivf_top5(&ivf, &q, nprobe);

        if full != ivf_t {
            mismatches.push((id, full, ivf_t));
        }

        total += 1;
        if total % 1000 == 0 {
            let secs = t0.elapsed().as_secs_f64();
            eprintln!(
                "processed {} cases in {:.1}s ({:.1}/s)",
                total,
                secs,
                total as f64 / secs
            );
        }
    }

    let secs = t0.elapsed().as_secs_f64();
    println!(
        "validated {} cases in {:.1}s ({:.1}/s) — {} mismatches",
        total,
        secs,
        total as f64 / secs,
        mismatches.len()
    );

    if !mismatches.is_empty() {
        for (id, full, ivf) in mismatches.iter().take(10) {
            println!("MISMATCH id={id} full={full:?} ivf={ivf:?}");
        }
        std::process::exit(1);
    }

    println!("PARITY OK");
}
