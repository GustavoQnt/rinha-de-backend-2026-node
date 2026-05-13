// Simulate bbox-prune with bbox bounds quantized to i8 per-dim. For each dim,
// compute global min/max across the full dataset → per-dim step = (gmax-gmin)/255.
// Quantizing rounds outward (min -= step/2, max += step/2), inflating each bbox.
// Measures how much that inflation degrades the prune rate vs i16 bbox.

use std::env;

use rinha_server::{ivf_blocks::{IvfBlocks, BLOCK_SIZE}, refs::DIM};

const TOP_K: usize = 5;

fn squared(a: &[i16], b: &[i16]) -> i64 {
    let mut s = 0i64;
    for i in 0..DIM {
        let d = a[i] as i64 - b[i] as i64;
        s += d * d;
    }
    s
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((sorted.len() - 1) as f64 * pct).round() as usize;
    sorted[idx]
}

fn nearest_centroid(ivf: &IvfBlocks, q: &[i16; DIM]) -> usize {
    let mut best = i64::MAX;
    let mut bc = 0usize;
    for c in 0..ivf.k {
        let d = squared(q, ivf.centroid(c));
        if d < best { best = d; bc = c; }
    }
    bc
}

fn scan_cluster(ivf: &IvfBlocks, c: usize, q: &[i16; DIM], topk: &mut [i64; TOP_K]) {
    let mut tmp = [0i16; DIM];
    for block in ivf.cluster_block_range(c) {
        let payload = ivf.block_payload(block);
        let origins = ivf.block_origin_slots(block);
        for slot in 0..BLOCK_SIZE {
            if origins[slot] == u32::MAX { continue; }
            for d in 0..DIM { tmp[d] = payload[d * BLOCK_SIZE + slot]; }
            let dist = squared(q, &tmp);
            if dist < topk[TOP_K - 1] {
                let mut p = TOP_K - 1;
                while p > 0 && dist < topk[p - 1] { topk[p] = topk[p - 1]; p -= 1; }
                topk[p] = dist;
            }
        }
    }
}

fn sample_positions(count: usize, samples: usize, seed: u64) -> Vec<usize> {
    let samples = samples.min(count);
    let mut step = ((seed >> 17) as usize | 1) % count;
    if step == 0 { step = 1; }
    while gcd(step, count) != 1 { step += 2; if step >= count { step = 1; } }
    let start = (seed as usize) % count;
    (0..samples).map(|i| (start + i.wrapping_mul(step)) % count).collect()
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 { let r = a % b; a = b; b = r; }
    a
}

fn reconstruct(ivf: &IvfBlocks, origin: u32) -> Option<[i16; DIM]> {
    for block in 0..ivf.total_blocks {
        let origins = ivf.block_origin_slots(block);
        for slot in 0..BLOCK_SIZE {
            if origins[slot] == origin {
                let payload = ivf.block_payload(block);
                let mut v = [0i16; DIM];
                for d in 0..DIM { v[d] = payload[d * BLOCK_SIZE + slot]; }
                return Some(v);
            }
        }
    }
    None
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| "resources/references.ivf-blocks-K4096.bin".to_string());
    let samples: usize = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1500);

    eprintln!("loading {path} ...");
    let ivf = IvfBlocks::load(&path);
    eprintln!("ivf: K={} count={}", ivf.k, ivf.count);

    eprintln!("computing tight i16 bboxes and global per-dim range ...");
    let mut bmins = vec![i16::MAX; ivf.k * DIM];
    let mut bmaxs = vec![i16::MIN; ivf.k * DIM];
    let mut gmin = [i16::MAX; DIM];
    let mut gmax = [i16::MIN; DIM];
    for c in 0..ivf.k {
        for block in ivf.cluster_block_range(c) {
            let payload = ivf.block_payload(block);
            let origins = ivf.block_origin_slots(block);
            for slot in 0..BLOCK_SIZE {
                if origins[slot] == u32::MAX { continue; }
                for d in 0..DIM {
                    let v = payload[d * BLOCK_SIZE + slot];
                    let idx = c * DIM + d;
                    if v < bmins[idx] { bmins[idx] = v; }
                    if v > bmaxs[idx] { bmaxs[idx] = v; }
                    if v < gmin[d] { gmin[d] = v; }
                    if v > gmax[d] { gmax[d] = v; }
                }
            }
        }
    }

    // Per-dim quant step (i16 units that one i8 bin represents).
    let mut step = [1f64; DIM];
    for d in 0..DIM {
        step[d] = ((gmax[d] as f64 - gmin[d] as f64) / 255.0).max(1.0);
        eprintln!("  dim {}: range=[{}..{}] step={:.2}", d, gmin[d], gmax[d], step[d]);
    }

    // Inflate bbox bounds outward by step/2 (= what i8 quantization adds at worst).
    let mut bmins_i8 = bmins.clone();
    let mut bmaxs_i8 = bmaxs.clone();
    for c in 0..ivf.k {
        for d in 0..DIM {
            let half = (step[d] / 2.0).ceil() as i32;
            let idx = c * DIM + d;
            bmins_i8[idx] = (bmins[idx] as i32 - half).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            bmaxs_i8[idx] = (bmaxs[idx] as i32 + half).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }

    eprintln!("sampling {samples} queries ...");
    let positions = sample_positions(ivf.count, samples, 0x5EED_2026);

    let mut prune_rates_i16: Vec<f64> = Vec::with_capacity(samples);
    let mut prune_rates_i8: Vec<f64> = Vec::with_capacity(samples);

    for (i, &origin) in positions.iter().enumerate() {
        let q = match reconstruct(&ivf, origin as u32) {
            Some(v) => v,
            None => continue,
        };
        let seed = nearest_centroid(&ivf, &q);
        let mut topk = [i64::MAX; TOP_K];
        scan_cluster(&ivf, seed, &q, &mut topk);
        let worst = topk[TOP_K - 1];

        let mut pruned_i16 = 0usize;
        let mut pruned_i8 = 0usize;
        for c in 0..ivf.k {
            if c == seed { continue; }
            let mut lb16 = 0i64;
            let mut lb8 = 0i64;
            for d in 0..DIM {
                let t = q[d] as i64;
                let lo16 = bmins[c * DIM + d] as i64;
                let hi16 = bmaxs[c * DIM + d] as i64;
                let g16 = if t < lo16 { lo16 - t } else if t > hi16 { t - hi16 } else { 0 };
                lb16 += g16 * g16;
                let lo8 = bmins_i8[c * DIM + d] as i64;
                let hi8 = bmaxs_i8[c * DIM + d] as i64;
                let g8 = if t < lo8 { lo8 - t } else if t > hi8 { t - hi8 } else { 0 };
                lb8 += g8 * g8;
            }
            if lb16 > worst { pruned_i16 += 1; }
            if lb8  > worst { pruned_i8  += 1; }
        }
        prune_rates_i16.push(pruned_i16 as f64 / (ivf.k - 1) as f64);
        prune_rates_i8 .push(pruned_i8  as f64 / (ivf.k - 1) as f64);

        if (i + 1) % 200 == 0 { eprintln!("  processed {}/{}", i + 1, samples); }
    }

    prune_rates_i16.sort_by(|a, b| a.partial_cmp(b).unwrap());
    prune_rates_i8.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p16 = |q: f64| percentile(&prune_rates_i16, q);
    let p8  = |q: f64| percentile(&prune_rates_i8, q);
    println!(
        "I16 PRUNE n={} min={:.3} p10={:.3} p50={:.3} p90={:.3} mean={:.3}",
        prune_rates_i16.len(), prune_rates_i16[0],
        p16(0.10), p16(0.50), p16(0.90),
        prune_rates_i16.iter().sum::<f64>() / prune_rates_i16.len() as f64,
    );
    println!(
        "I8  PRUNE n={} min={:.3} p10={:.3} p50={:.3} p90={:.3} mean={:.3}",
        prune_rates_i8.len(), prune_rates_i8[0],
        p8(0.10), p8(0.50), p8(0.90),
        prune_rates_i8.iter().sum::<f64>() / prune_rates_i8.len() as f64,
    );

    let visited_i16 = (1.0 - prune_rates_i16.iter().sum::<f64>() / prune_rates_i16.len() as f64) * (ivf.k - 1) as f64 + 1.0;
    let visited_i8  = (1.0 - prune_rates_i8 .iter().sum::<f64>() / prune_rates_i8.len() as f64) * (ivf.k - 1) as f64 + 1.0;
    println!("EXPECTED_CLUSTERS_VISITED i16={:.1}  i8={:.1}  (out of {})", visited_i16, visited_i8, ivf.k);
}
