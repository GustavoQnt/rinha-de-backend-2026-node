// Simulate bbox-prune-by-dim on real reference vectors as queries. For each
// query: seed = nearest centroid, top-5 from seed → worst. Then for every
// other cluster compute the per-dim bbox lower bound and count how many
// clusters would be pruned (lb > worst). Reports prune rate distribution.
//
// If p50 prune rate ≥ ~95%, bbox-i8 is worth implementing. Below ~70%, drop it.

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

fn compute_bboxes(ivf: &IvfBlocks) -> (Vec<i16>, Vec<i16>) {
    let mut mins = vec![i16::MAX; ivf.k * DIM];
    let mut maxs = vec![i16::MIN; ivf.k * DIM];
    for c in 0..ivf.k {
        for block in ivf.cluster_block_range(c) {
            let payload = ivf.block_payload(block);
            let origins = ivf.block_origin_slots(block);
            for slot in 0..BLOCK_SIZE {
                if origins[slot] == u32::MAX { continue; }
                for d in 0..DIM {
                    let v = payload[d * BLOCK_SIZE + slot];
                    let idx = c * DIM + d;
                    if v < mins[idx] { mins[idx] = v; }
                    if v > maxs[idx] { maxs[idx] = v; }
                }
            }
        }
    }
    // Clusters with no vectors: keep degenerate (max < min) so lb is huge → always pruned.
    (mins, maxs)
}

fn bbox_lower_bound(bmin: &[i16], bmax: &[i16], q: &[i16; DIM]) -> i64 {
    let mut s = 0i64;
    for d in 0..DIM {
        let t = q[d] as i64;
        let lo = bmin[d] as i64;
        let hi = bmax[d] as i64;
        let g = if t < lo { lo - t } else if t > hi { t - hi } else { 0 };
        s += g * g;
    }
    s
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
    // brute scan blocks to find origin (only used for sampling; slow but OK).
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
    let samples: usize = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(2000);

    eprintln!("loading {path} ...");
    let ivf = IvfBlocks::load(&path);
    eprintln!("ivf: K={} count={}", ivf.k, ivf.count);

    eprintln!("computing bboxes ...");
    let (bmins, bmaxs) = compute_bboxes(&ivf);

    eprintln!("sampling {samples} queries ...");
    let positions = sample_positions(ivf.count, samples, 0x5EED_2026);

    let mut prune_rates: Vec<f64> = Vec::with_capacity(samples);
    let mut seed_cluster_sizes: Vec<usize> = Vec::with_capacity(samples);

    for (i, &origin) in positions.iter().enumerate() {
        let q = match reconstruct(&ivf, origin as u32) {
            Some(v) => v,
            None => continue,
        };
        let seed = nearest_centroid(&ivf, &q);
        let mut topk = [i64::MAX; TOP_K];
        scan_cluster(&ivf, seed, &q, &mut topk);
        seed_cluster_sizes.push(ivf.cluster_block_range(seed).len() * BLOCK_SIZE);
        let worst = topk[TOP_K - 1];

        let mut pruned = 0usize;
        for c in 0..ivf.k {
            if c == seed { continue; }
            let bm = &bmins[c * DIM..(c + 1) * DIM];
            let bM = &bmaxs[c * DIM..(c + 1) * DIM];
            let lb = bbox_lower_bound(bm, bM, &q);
            if lb > worst { pruned += 1; }
        }
        prune_rates.push(pruned as f64 / (ivf.k - 1) as f64);

        if (i + 1) % 200 == 0 {
            eprintln!("  processed {}/{}", i + 1, samples);
        }
    }

    prune_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| percentile(&prune_rates, q);
    println!(
        "PRUNE_RATE n={} min={:.3} p10={:.3} p25={:.3} p50={:.3} p75={:.3} p90={:.3} p99={:.3} max={:.3}",
        prune_rates.len(), prune_rates[0],
        p(0.10), p(0.25), p(0.50), p(0.75), p(0.90), p(0.99),
        prune_rates[prune_rates.len() - 1],
    );

    let mean_prune = prune_rates.iter().sum::<f64>() / prune_rates.len() as f64;
    let clusters_visited_mean = (1.0 - mean_prune) * (ivf.k - 1) as f64 + 1.0;
    println!(
        "EXPECTED_CLUSTERS_VISITED_PER_QUERY mean={:.1} (out of {})  mean_prune_rate={:.3}",
        clusters_visited_mean, ivf.k, mean_prune,
    );
}
