use std::env;
use std::time::Instant;

use rinha_server::{ivf::Ivf, refs::DIM};

fn parse_args() -> (String, usize, usize, u64) {
    let mut ivf_path = "resources/references.ivf.bin".to_string();
    let mut samples = 10_000usize;
    let mut nprobe = 32usize;
    let mut seed = 0x5EED_2026u64;

    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ivf" => {
                ivf_path = args[i + 1].clone();
                i += 2;
            }
            "--samples" => {
                samples = args[i + 1].parse().expect("--samples int");
                i += 2;
            }
            "--nprobe" => {
                nprobe = args[i + 1].parse().expect("--nprobe int");
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().expect("--seed int");
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    (ivf_path, samples, nprobe, seed)
}

#[inline(always)]
fn squared_distance(a: &[i16], b: &[i16]) -> i64 {
    let mut d = 0i64;
    for j in 0..DIM {
        let diff = a[j] as i64 - b[j] as i64;
        d += diff * diff;
    }
    d
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn sample_positions(count: usize, samples: usize, seed: u64) -> Vec<usize> {
    let samples = samples.min(count);
    let start = seed as usize % count;
    let mut step = ((seed >> 17) as usize | 1) % count;
    if step == 0 {
        step = 1;
    }
    while gcd(step, count) != 1 {
        step += 2;
        if step >= count {
            step = 1;
        }
    }

    (0..samples)
        .map(|i| (start + i.wrapping_mul(step)) % count)
        .collect()
}

fn top_n_centroids(ivf: &Ivf, q: &[i16], nprobe: usize) -> Vec<usize> {
    let mut heap: Vec<(i64, usize)> = Vec::with_capacity(nprobe + 1);
    for c in 0..ivf.k {
        let d = squared_distance(q, ivf.centroid(c));
        if heap.len() < nprobe {
            heap.push((d, c));
            if heap.len() == nprobe {
                heap.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            }
        } else {
            let last = heap[nprobe - 1];
            if d < last.0 || (d == last.0 && c < last.1) {
                heap[nprobe - 1] = (d, c);
                let mut i = nprobe - 1;
                while i > 0 {
                    let prev = heap[i - 1];
                    let cur = heap[i];
                    if cur.0 < prev.0 || (cur.0 == prev.0 && cur.1 < prev.1) {
                        heap.swap(i, i - 1);
                        i -= 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }
    heap.into_iter().map(|(_, c)| c).collect()
}

fn nearest_neighbor(ivf: &Ivf, q: &[i16], self_origin: u32, nprobe: usize) -> (i64, u8) {
    let probes = top_n_centroids(ivf, q, nprobe);
    let mut best_d = i64::MAX;
    let mut best_label = 0u8;

    for &c in &probes {
        for p in ivf.cluster_range(c) {
            if ivf.origin[p] == self_origin {
                continue;
            }
            let d = squared_distance(q, ivf.vector_perm(p));
            if d < best_d || (d == best_d && ivf.origin[p] < self_origin) {
                best_d = d;
                best_label = ivf.labels[p];
            }
        }
    }

    (best_d, best_label)
}

fn percentile(sorted: &[i64], pct: f64) -> i64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * pct).round() as usize;
    sorted[idx]
}

fn main() {
    let (ivf_path, samples, nprobe, seed) = parse_args();
    eprintln!("loading {ivf_path} ...");
    let ivf = Ivf::load(&ivf_path);
    assert!(nprobe > 0 && nprobe <= ivf.k, "invalid nprobe={nprobe}");
    eprintln!(
        "ivf: K={} count={} scale={} samples={} nprobe={} seed={}",
        ivf.k, ivf.count, ivf.scale, samples, nprobe, seed
    );

    let positions = sample_positions(ivf.count, samples, seed);
    let start = Instant::now();
    let mut distances = Vec::with_capacity(positions.len());
    let mut fraud_samples = 0usize;
    let mut legit_samples = 0usize;
    let mut same_label_nn = 0usize;
    let mut cross_label_nn = 0usize;

    for (i, &p) in positions.iter().enumerate() {
        let q = ivf.vector_perm(p);
        let label = ivf.labels[p];
        if label == 1 {
            fraud_samples += 1;
        } else {
            legit_samples += 1;
        }

        let (d, nn_label) = nearest_neighbor(&ivf, q, ivf.origin[p], nprobe);
        distances.push(d);
        if nn_label == label {
            same_label_nn += 1;
        } else {
            cross_label_nn += 1;
        }

        if (i + 1) % 1000 == 0 {
            let secs = start.elapsed().as_secs_f64();
            eprintln!(
                "processed {} / {} ({:.0}/s)",
                i + 1,
                positions.len(),
                (i + 1) as f64 / secs
            );
        }
    }

    distances.sort_unstable();
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "NN_DENSITY samples={} nprobe={} secs={:.2}",
        positions.len(),
        nprobe,
        elapsed
    );
    println!("labels fraud={} legit={}", fraud_samples, legit_samples);
    println!(
        "nearest_label same={} cross={} cross_pct={:.2}",
        same_label_nn,
        cross_label_nn,
        100.0 * cross_label_nn as f64 / positions.len() as f64
    );
    println!(
        "squared_distance min={} p01={} p05={} p10={} p25={} p50={} p75={} p90={} p95={} p99={} max={}",
        distances[0],
        percentile(&distances, 0.01),
        percentile(&distances, 0.05),
        percentile(&distances, 0.10),
        percentile(&distances, 0.25),
        percentile(&distances, 0.50),
        percentile(&distances, 0.75),
        percentile(&distances, 0.90),
        percentile(&distances, 0.95),
        percentile(&distances, 0.99),
        distances[distances.len() - 1],
    );
}
