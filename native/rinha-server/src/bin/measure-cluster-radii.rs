// Measure per-cluster radius and nearest-centroid spacing for the IVF index.
// Output decides whether radius-prune is viable: if p50(radius) >> p50(spacing),
// the triangle-inequality lower bound rarely fires and the prune dies.

use std::env;

use rinha_server::{ivf_blocks::{IvfBlocks, BLOCK_SIZE}, refs::DIM};

fn squared(a: &[i16], b: &[i16]) -> i64 {
    let mut s = 0i64;
    for i in 0..DIM {
        let d = a[i] as i64 - b[i] as i64;
        s += d * d;
    }
    s
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * pct).round() as usize;
    sorted[idx]
}

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "resources/references.ivf-blocks-K4096.bin".to_string());
    eprintln!("loading {path} ...");
    let ivf = IvfBlocks::load(&path);
    eprintln!("ivf: K={} count={} scale={} total_blocks={}", ivf.k, ivf.count, ivf.scale, ivf.total_blocks);

    let mut radii_sq: Vec<i64> = Vec::with_capacity(ivf.k);
    let mut cluster_sizes: Vec<usize> = Vec::with_capacity(ivf.k);

    let mut tmp = [0i16; DIM];
    for c in 0..ivf.k {
        let centroid = ivf.centroid(c);
        let mut max_d = 0i64;
        let mut count = 0usize;
        for block in ivf.cluster_block_range(c) {
            let payload = ivf.block_payload(block);
            let origins = ivf.block_origin_slots(block);
            for slot in 0..BLOCK_SIZE {
                if origins[slot] == u32::MAX {
                    continue;
                }
                count += 1;
                for d in 0..DIM {
                    tmp[d] = payload[d * BLOCK_SIZE + slot];
                }
                let dist = squared(&tmp, centroid);
                if dist > max_d {
                    max_d = dist;
                }
            }
        }
        radii_sq.push(max_d);
        cluster_sizes.push(count);
    }

    // Nearest-centroid spacing per cluster.
    let mut nn_centroid_sq: Vec<i64> = Vec::with_capacity(ivf.k);
    for c in 0..ivf.k {
        let cc = ivf.centroid(c);
        let mut best = i64::MAX;
        for o in 0..ivf.k {
            if o == c {
                continue;
            }
            let d = squared(cc, ivf.centroid(o));
            if d < best {
                best = d;
            }
        }
        nn_centroid_sq.push(best);
    }

    let mut radii: Vec<f64> = radii_sq.iter().map(|&v| (v as f64).sqrt()).collect();
    let mut spacing: Vec<f64> = nn_centroid_sq.iter().map(|&v| (v as f64).sqrt()).collect();
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
    spacing.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut sizes_sorted = cluster_sizes.clone();
    sizes_sorted.sort_unstable();
    let p50_size = sizes_sorted[sizes_sorted.len() / 2];
    let p99_size = sizes_sorted[sizes_sorted.len() * 99 / 100];
    let max_size = sizes_sorted[sizes_sorted.len() - 1];
    let empty = sizes_sorted.iter().filter(|&&s| s == 0).count();
    println!("cluster_size p50={} p99={} max={} empty={}", p50_size, p99_size, max_size, empty);

    let p = |s: &[f64], q: f64| percentile(s, q);
    println!(
        "RADIUS    min={:.1} p10={:.1} p25={:.1} p50={:.1} p75={:.1} p90={:.1} p99={:.1} max={:.1}",
        radii[0], p(&radii, 0.10), p(&radii, 0.25), p(&radii, 0.50),
        p(&radii, 0.75), p(&radii, 0.90), p(&radii, 0.99), radii[radii.len() - 1],
    );
    println!(
        "SPACING   min={:.1} p10={:.1} p25={:.1} p50={:.1} p75={:.1} p90={:.1} p99={:.1} max={:.1}",
        spacing[0], p(&spacing, 0.10), p(&spacing, 0.25), p(&spacing, 0.50),
        p(&spacing, 0.75), p(&spacing, 0.90), p(&spacing, 0.99), spacing[spacing.len() - 1],
    );

    // Ratio per cluster: radius / nearest-centroid distance. < 0.5 means
    // clusters are well-separated and the bbox prune fires aggressively;
    // > 1.0 means clusters overlap and prune dies on arrival.
    let mut ratios: Vec<f64> = (0..ivf.k)
        .map(|c| {
            let r = (radii_sq[c] as f64).sqrt();
            let s = (nn_centroid_sq[c] as f64).sqrt();
            if s > 0.0 { r / s } else { f64::INFINITY }
        })
        .collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "R/S_RATIO p10={:.3} p25={:.3} p50={:.3} p75={:.3} p90={:.3} p99={:.3} max={:.3}",
        p(&ratios, 0.10), p(&ratios, 0.25), p(&ratios, 0.50),
        p(&ratios, 0.75), p(&ratios, 0.90), p(&ratios, 0.99),
        ratios[ratios.len() - 1],
    );

    let over_05 = ratios.iter().filter(|&&r| r > 0.5).count();
    let over_10 = ratios.iter().filter(|&&r| r > 1.0).count();
    println!(
        "RATIO_HISTOGRAM clusters_with_r>0.5*spacing={} ({:.1}%) r>1.0*spacing={} ({:.1}%)",
        over_05, 100.0 * over_05 as f64 / ivf.k as f64,
        over_10, 100.0 * over_10 as f64 / ivf.k as f64,
    );
}
