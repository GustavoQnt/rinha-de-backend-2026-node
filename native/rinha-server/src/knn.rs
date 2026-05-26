use crate::ivf::Ivf;
use crate::ivf_blocks::{IvfBlocks, BLOCK_SIZE};
use crate::refs::{Refs, DIM, DIM_PADDED};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::sync::OnceLock;

pub const TOP_K: usize = 5;

#[inline(always)]
pub fn quantize_query(query: &[f64; 14], scale: u32) -> [i16; DIM] {
    let scale = scale as f64;
    let mut q = [0i16; DIM];
    for i in 0..DIM {
        q[i] = (query[i] * scale).round() as i16;
    }
    q
}

#[inline(always)]
fn squared_distance_scalar(a: &[i16], b: &[i16]) -> i64 {
    let mut d: i64 = 0;
    for j in 0..DIM {
        let diff = a[j] as i64 - b[j] as i64;
        d += diff * diff;
    }
    d
}

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128i, __m256i, _mm256_add_epi64, _mm256_castsi256_si128, _mm256_cvtepi16_epi32,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_mul_epi32, _mm256_set1_epi32,
    _mm256_setzero_si256, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi32,
    _mm_add_epi16, _mm_loadu_si128, _mm_madd_epi16, _mm_max_epi16, _mm_setzero_si128,
    _mm_storeu_si128, _mm_subs_epi16,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi64, _mm256_castsi256_si128, _mm256_cvtepi16_epi32,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_mul_epi32, _mm256_set1_epi32,
    _mm256_setzero_si256, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi32,
    _mm_add_epi16, _mm_loadu_si128, _mm_madd_epi16, _mm_max_epi16, _mm_setzero_si128,
    _mm_storeu_si128, _mm_subs_epi16,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn squared_distance_avx2_checked(a: &[i16], b: &[i16]) -> i64 {
    debug_assert!(std::is_x86_feature_detected!("avx2"));
    unsafe { squared_distance_avx2(a, b) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn squared_distance_avx2(a: &[i16], b: &[i16]) -> i64 {
    debug_assert!(a.len() >= DIM);
    debug_assert!(b.len() >= DIM);

    let av = _mm_loadu_si128(a.as_ptr() as *const __m128i);
    let bv = _mm_loadu_si128(b.as_ptr() as *const __m128i);
    let sums = squared_distance_8x_i16(av, bv);

    let mut lanes = [0i64; 4];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, sums);
    let mut d = lanes[0] + lanes[1] + lanes[2] + lanes[3];

    for j in 8..DIM {
        let diff = a[j] as i64 - b[j] as i64;
        d += diff * diff;
    }
    d
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn squared_distance_avx2_padded_checked(a: &[i16], b: &[i16]) -> i64 {
    debug_assert!(std::is_x86_feature_detected!("avx2"));
    unsafe { squared_distance_avx2_padded(a, b) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn squared_distance_avx2_padded(a: &[i16], b: &[i16]) -> i64 {
    debug_assert!(a.len() >= DIM_PADDED);
    debug_assert!(b.len() >= DIM_PADDED);

    let av = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
    let bv = _mm256_loadu_si256(b.as_ptr() as *const __m256i);

    let a_lo = _mm256_castsi256_si128(av);
    let b_lo = _mm256_castsi256_si128(bv);
    let a_hi = _mm256_extracti128_si256::<1>(av);
    let b_hi = _mm256_extracti128_si256::<1>(bv);

    let lo = squared_distance_8x_i16(a_lo, b_lo);
    let hi = squared_distance_8x_i16(a_hi, b_hi);
    let sums = _mm256_add_epi64(lo, hi);

    let mut lanes = [0i64; 4];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, sums);
    lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn squared_distance_8x_i16(a: __m128i, b: __m128i) -> __m256i {
    let ai = _mm256_cvtepi16_epi32(a);
    let bi = _mm256_cvtepi16_epi32(b);
    let diff = _mm256_sub_epi32(ai, bi);

    let even_sq = _mm256_mul_epi32(diff, diff);
    let odd = _mm256_srli_epi64(diff, 32);
    let odd_sq = _mm256_mul_epi32(odd, odd);
    _mm256_add_epi64(even_sq, odd_sq)
}

#[inline(always)]
fn squared_distance(a: &[i16], b: &[i16]) -> i64 {
    squared_distance_scalar(a, b)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn avx2_enabled_for_runtime() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RINHA_USE_AVX2").as_deref() == Ok("1")
            && std::is_x86_feature_detected!("avx2")
    })
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn avx2_padded_enabled_for_runtime() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("RINHA_USE_AVX2_PADDED16").as_deref() == Ok("1"))
}

#[inline(always)]
fn try_insert_top(distances: &mut [i64; TOP_K], indexes: &mut [u32; TOP_K], d: i64, orig_idx: u32) {
    // Lex order on (distance, original_index): smaller distance wins; on tie,
    // smaller original index wins. Bit-identical between full-scan and IVF.
    let last = TOP_K - 1;
    if d > distances[last] {
        return;
    }
    if d == distances[last] && orig_idx >= indexes[last] {
        return;
    }
    let mut pos = last;
    while pos > 0 {
        let prev_d = distances[pos - 1];
        let prev_i = indexes[pos - 1];
        let should_move = d < prev_d || (d == prev_d && orig_idx < prev_i);
        if !should_move {
            break;
        }
        distances[pos] = prev_d;
        indexes[pos] = prev_i;
        pos -= 1;
    }
    distances[pos] = d;
    indexes[pos] = orig_idx;
}

#[inline(always)]
fn bucket_from_top5(refs: &Refs, indexes: &[u32; TOP_K]) -> u8 {
    let mut bucket = 0u8;
    for i in 0..TOP_K {
        if refs.label(indexes[i] as usize) == 1 {
            bucket += 1;
        }
    }
    bucket
}

#[inline(always)]
fn bucket_from_ivf_top5(ivf: &Ivf, indexes: &[u32; TOP_K]) -> u8 {
    let mut bucket = 0u8;
    for i in 0..TOP_K {
        if ivf.label_by_origin(indexes[i] as usize) == 1 {
            bucket += 1;
        }
    }
    bucket
}

/// Full-scan top-K neighbors. Returns original-index identifiers (which equals the
/// position in `refs` since refs are not permuted). Tie-break: lower original index wins.
pub fn full_scan_top5(refs: &Refs, q: &[i16; DIM]) -> [u32; TOP_K] {
    let mut distances = [i64::MAX; TOP_K];
    let mut indexes = [0u32; TOP_K];

    for i in 0..refs.count {
        let rv = refs.vector(i);
        let d = squared_distance(q, rv);
        try_insert_top(&mut distances, &mut indexes, d, i as u32);
    }
    indexes
}

pub fn predict_bucket(refs: &Refs, query: &[f64; 14]) -> u8 {
    let q = quantize_query(query, refs.scale);
    let top = full_scan_top5(refs, &q);
    bucket_from_top5(refs, &top)
}

/// IVF top-K neighbors. Selects top `nprobe` centroids by L2 distance, then scans only
/// those clusters. The reported indexes are *original* (unpermuted) so the tie-break is
/// identical to full-scan and bucket lookup against `refs.labels` is direct.
pub fn ivf_top5(ivf: &Ivf, q: &[i16; DIM], nprobe: usize) -> [u32; TOP_K] {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if avx2_enabled_for_runtime() {
            if avx2_padded_enabled_for_runtime() && ivf.has_padded_vectors() {
                return ivf_top5_avx2_padded(ivf, q, nprobe);
            }
            return ivf_top5_with_distance(ivf, q, nprobe, squared_distance_avx2_checked);
        }
    }
    ivf_top5_with_distance(ivf, q, nprobe, squared_distance_scalar)
}

fn ivf_top5_with_distance<F>(ivf: &Ivf, q: &[i16; DIM], nprobe: usize, distance: F) -> [u32; TOP_K]
where
    F: Fn(&[i16], &[i16]) -> i64 + Copy,
{
    debug_assert!(nprobe <= ivf.k);

    // Top-nprobe centroids by squared distance, lex tie-break on centroid index.
    // For nprobe == K we just need every centroid; skip selection and iterate 0..K.
    let mut distances = [i64::MAX; TOP_K];
    let mut indexes = [0u32; TOP_K];

    if nprobe >= ivf.k {
        for c in 0..ivf.k {
            scan_cluster(ivf, c, q, &mut distances, &mut indexes, distance);
        }
    } else {
        let probes = top_n_centroids(ivf, q, nprobe, distance);
        for &c in &probes[..nprobe] {
            scan_cluster(ivf, c, q, &mut distances, &mut indexes, distance);
        }
    }
    indexes
}

pub fn ivf_top5_scalar(ivf: &Ivf, q: &[i16; DIM], nprobe: usize) -> [u32; TOP_K] {
    ivf_top5_with_distance(ivf, q, nprobe, squared_distance_scalar)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn ivf_top5_avx2(ivf: &Ivf, q: &[i16; DIM], nprobe: usize) -> Option<[u32; TOP_K]> {
    if std::is_x86_feature_detected!("avx2") {
        if ivf.has_padded_vectors() {
            Some(ivf_top5_avx2_padded(ivf, q, nprobe))
        } else {
            Some(ivf_top5_with_distance(
                ivf,
                q,
                nprobe,
                squared_distance_avx2_checked,
            ))
        }
    } else {
        None
    }
}

pub fn active_distance_kernel_name() -> &'static str {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if avx2_enabled_for_runtime() {
            if avx2_padded_enabled_for_runtime() {
                return "avx2-padded16";
            }
            return "avx2";
        }
    }
    "scalar"
}

#[inline(always)]
fn scan_cluster<F>(
    ivf: &Ivf,
    c: usize,
    q: &[i16; DIM],
    distances: &mut [i64; TOP_K],
    indexes: &mut [u32; TOP_K],
    distance: F,
) where
    F: Fn(&[i16], &[i16]) -> i64 + Copy,
{
    let range = ivf.cluster_range(c);
    for p in range {
        let rv = ivf.vector_perm(p);
        let d = distance(q, rv);
        let orig_idx = ivf.origin[p];
        try_insert_top(distances, indexes, d, orig_idx);
    }
}

/// Returns the `nprobe` cluster indexes nearest to `q`, in ascending-distance order
/// with lex tie-break on cluster index. Allocates a small Vec so we can hand back any
/// nprobe up to K; nprobe is bounded by K (≤2048 in practice).
fn top_n_centroids<F>(ivf: &Ivf, q: &[i16; DIM], nprobe: usize, distance: F) -> Vec<usize>
where
    F: Fn(&[i16], &[i16]) -> i64 + Copy,
{
    let mut heap: Vec<(i64, usize)> = Vec::with_capacity(nprobe + 1);
    for c in 0..ivf.k {
        let cv = ivf.centroid(c);
        let d = distance(q, cv);
        if heap.len() < nprobe {
            heap.push((d, c));
            if heap.len() == nprobe {
                heap.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            }
        } else {
            let last = heap[nprobe - 1];
            if d < last.0 || (d == last.0 && c < last.1) {
                // insert keeping sort
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn ivf_top5_avx2_padded(ivf: &Ivf, q: &[i16; DIM], nprobe: usize) -> [u32; TOP_K] {
    debug_assert!(nprobe <= ivf.k);
    let q = pad_query(q);
    let mut distances = [i64::MAX; TOP_K];
    let mut indexes = [0u32; TOP_K];

    if nprobe >= ivf.k {
        for c in 0..ivf.k {
            scan_cluster_padded(ivf, c, &q, &mut distances, &mut indexes);
        }
    } else {
        let probes = top_n_centroids_padded(ivf, &q, nprobe);
        for &c in &probes[..nprobe] {
            scan_cluster_padded(ivf, c, &q, &mut distances, &mut indexes);
        }
    }
    indexes
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn pad_query(q: &[i16; DIM]) -> [i16; DIM_PADDED] {
    let mut padded = [0i16; DIM_PADDED];
    padded[..DIM].copy_from_slice(q);
    padded
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn scan_cluster_padded(
    ivf: &Ivf,
    c: usize,
    q: &[i16; DIM_PADDED],
    distances: &mut [i64; TOP_K],
    indexes: &mut [u32; TOP_K],
) {
    let range = ivf.cluster_range(c);
    for p in range {
        let rv = ivf.vector_perm_padded(p);
        let d = squared_distance_avx2_padded_checked(q, rv);
        let orig_idx = ivf.origin[p];
        try_insert_top(distances, indexes, d, orig_idx);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn top_n_centroids_padded(ivf: &Ivf, q: &[i16; DIM_PADDED], nprobe: usize) -> Vec<usize> {
    let mut heap: Vec<(i64, usize)> = Vec::with_capacity(nprobe + 1);
    for c in 0..ivf.k {
        let cv = ivf.centroid_padded(c);
        let d = squared_distance_avx2_padded_checked(q, cv);
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

pub fn predict_bucket_ivf(ivf: &Ivf, query: &[f64; 14], nprobe: usize) -> u8 {
    let q = quantize_query(query, ivf.scale);
    let top = ivf_top5(ivf, &q, nprobe);
    bucket_from_ivf_top5(ivf, &top)
}

// =====================================================================
// Block-SoA IVF (R26I v2) scan path. See `ivf_blocks.rs` for the data layout.
//
// Per block (8 slots × 14 dims, SoA-by-block), we compute 8 squared distances
// in parallel, accumulating in i64 to avoid overflow (14 × max_diff² ≈ 6e10).
//
// Padded slots carry vector [i16::MAX; 14] so their distance to any clamped
// query is enormous; we additionally guard with origin == u32::MAX to avoid
// inserting them into the top-5 even on a fresh (i64::MAX-filled) state.
// =====================================================================

#[inline(always)]
fn block_distances_scalar(query: &[i16; DIM], block: &[i16]) -> [i64; BLOCK_SIZE] {
    debug_assert!(block.len() >= DIM * BLOCK_SIZE);
    let mut d = [0i64; BLOCK_SIZE];
    for dim in 0..DIM {
        let q = query[dim] as i64;
        let base = dim * BLOCK_SIZE;
        for slot in 0..BLOCK_SIZE {
            let diff = block[base + slot] as i64 - q;
            d[slot] += diff * diff;
        }
    }
    d
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn block_distances_avx2_checked(query: &[i16; DIM], block: &[i16]) -> [i64; BLOCK_SIZE] {
    debug_assert!(std::is_x86_feature_detected!("avx2"));
    unsafe { block_distances_avx2(query, block) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn block_distances_avx2(query: &[i16; DIM], block: &[i16]) -> [i64; BLOCK_SIZE] {
    debug_assert!(block.len() >= DIM * BLOCK_SIZE);

    // Two i64x4 accumulators; together they cover 8 slots after interleave.
    let mut acc_even = _mm256_setzero_si256();
    let mut acc_odd = _mm256_setzero_si256();

    for dim in 0..DIM {
        // Load 8 i16 lanes for this dim across 8 slots, widen to 8 i32 lanes.
        let lane_ptr = block.as_ptr().add(dim * BLOCK_SIZE) as *const __m128i;
        let slot_i16x8 = _mm_loadu_si128(lane_ptr);
        let slot_i32x8 = _mm256_cvtepi16_epi32(slot_i16x8);

        let q_i32x8 = _mm256_set1_epi32(query[dim] as i32);
        let diff = _mm256_sub_epi32(slot_i32x8, q_i32x8);

        // 8 i32 squares -> 8 i64, split across two registers via mul_epi32 +
        // shifted-mul_epi32 trick (same as squared_distance_8x_i16).
        let even_sq = _mm256_mul_epi32(diff, diff);
        let odd = _mm256_srli_epi64(diff, 32);
        let odd_sq = _mm256_mul_epi32(odd, odd);

        acc_even = _mm256_add_epi64(acc_even, even_sq);
        acc_odd = _mm256_add_epi64(acc_odd, odd_sq);
    }

    let mut even = [0i64; 4];
    let mut odd = [0i64; 4];
    _mm256_storeu_si256(even.as_mut_ptr() as *mut __m256i, acc_even);
    _mm256_storeu_si256(odd.as_mut_ptr() as *mut __m256i, acc_odd);

    // Interleave: even -> slots 0,2,4,6; odd -> slots 1,3,5,7.
    [
        even[0], odd[0], even[1], odd[1], even[2], odd[2], even[3], odd[3],
    ]
}

#[inline(always)]
fn scan_block_soa<F>(
    ivf: &IvfBlocks,
    block_idx: usize,
    q: &[i16; DIM],
    distances: &mut [i64; TOP_K],
    indexes: &mut [u32; TOP_K],
    block_distances: F,
) where
    F: Fn(&[i16; DIM], &[i16]) -> [i64; BLOCK_SIZE] + Copy,
{
    let block = ivf.block_payload(block_idx);
    let origins = ivf.block_origin_slots(block_idx);
    let d = block_distances(q, block);
    for slot in 0..BLOCK_SIZE {
        let origin = origins[slot];
        if origin == u32::MAX {
            continue; // padded slot
        }
        try_insert_top(distances, indexes, d[slot], origin);
    }
}

fn top_n_centroids_blocks<F>(
    ivf: &IvfBlocks,
    q: &[i16; DIM],
    nprobe: usize,
    distance: F,
) -> Vec<usize>
where
    F: Fn(&[i16], &[i16]) -> i64 + Copy,
{
    let mut heap: Vec<(i64, usize)> = Vec::with_capacity(nprobe + 1);
    for c in 0..ivf.k {
        let cv = ivf.centroid(c);
        let d = distance(q, cv);
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

fn ivf_blocks_top5_with<FB, FC>(
    ivf: &IvfBlocks,
    q: &[i16; DIM],
    nprobe: usize,
    block_distances: FB,
    centroid_distance: FC,
) -> [u32; TOP_K]
where
    FB: Fn(&[i16; DIM], &[i16]) -> [i64; BLOCK_SIZE] + Copy,
    FC: Fn(&[i16], &[i16]) -> i64 + Copy,
{
    debug_assert!(nprobe <= ivf.k);
    let mut distances = [i64::MAX; TOP_K];
    let mut indexes = [0u32; TOP_K];

    if nprobe >= ivf.k {
        for c in 0..ivf.k {
            for block_idx in ivf.cluster_block_range(c) {
                scan_block_soa(
                    ivf,
                    block_idx,
                    q,
                    &mut distances,
                    &mut indexes,
                    block_distances,
                );
            }
        }
    } else {
        let probes = top_n_centroids_blocks(ivf, q, nprobe, centroid_distance);
        for &c in &probes[..nprobe] {
            for block_idx in ivf.cluster_block_range(c) {
                scan_block_soa(
                    ivf,
                    block_idx,
                    q,
                    &mut distances,
                    &mut indexes,
                    block_distances,
                );
            }
        }
    }
    indexes
}

pub fn ivf_blocks_top5_scalar(ivf: &IvfBlocks, q: &[i16; DIM], nprobe: usize) -> [u32; TOP_K] {
    ivf_blocks_top5_with(
        ivf,
        q,
        nprobe,
        block_distances_scalar,
        squared_distance_scalar,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn ivf_blocks_top5_avx2(
    ivf: &IvfBlocks,
    q: &[i16; DIM],
    nprobe: usize,
) -> Option<[u32; TOP_K]> {
    if std::is_x86_feature_detected!("avx2") {
        Some(ivf_blocks_top5_with(
            ivf,
            q,
            nprobe,
            block_distances_avx2_checked,
            squared_distance_avx2_checked,
        ))
    } else {
        None
    }
}

pub fn ivf_blocks_top5(ivf: &IvfBlocks, q: &[i16; DIM], nprobe: usize) -> [u32; TOP_K] {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if avx2_enabled_for_runtime() {
            return ivf_blocks_top5_with(
                ivf,
                q,
                nprobe,
                block_distances_avx2_checked,
                squared_distance_avx2_checked,
            );
        }
    }
    ivf_blocks_top5_with(
        ivf,
        q,
        nprobe,
        block_distances_scalar,
        squared_distance_scalar,
    )
}

#[inline(always)]
fn bucket_from_blocks_top5(ivf: &IvfBlocks, indexes: &[u32; TOP_K]) -> u8 {
    let mut bucket = 0u8;
    for i in 0..TOP_K {
        if ivf.label_by_origin(indexes[i] as usize) == 1 {
            bucket += 1;
        }
    }
    bucket
}

pub fn predict_bucket_ivf_blocks(ivf: &IvfBlocks, query: &[f64; 14], nprobe: usize) -> u8 {
    let q = quantize_query(query, ivf.scale);
    let top = ivf_blocks_top5(ivf, &q, nprobe);
    bucket_from_blocks_top5(ivf, &top)
}

// Warmup: forces all index pages resident in RAM, primes the AVX2 codegen
// path / iCache / branch predictor, and exercises both the fast-pass and
// escalation paths. Runs synchronously on the main thread before any listener
// is bound. The early prévia run produced p99=13.71ms while a re-run of the
// same binary scored p99=1.38ms — symptom of cold pages + cold predictor on
// the first burst of real traffic. The rinha banca takes ~1s to ramp up,
// which is not enough to absorb that cost.
pub fn warmup_ivf_blocks(ivf: &IvfBlocks, rounds: usize) {
    warmup_ivf_blocks_with(ivf, rounds, None);
}

pub fn warmup_ivf_blocks_bbox(
    ivf: &IvfBlocks,
    rounds: usize,
    seed_clusters: usize,
    visit_cap: usize,
) {
    warmup_ivf_blocks_with(ivf, rounds, Some((seed_clusters, visit_cap)));
}

fn warmup_ivf_blocks_with(ivf: &IvfBlocks, rounds: usize, bbox: Option<(usize, usize)>) {
    // Phase 1: page-touch every backing array at 4 KiB stride to force the
    // kernel to populate the VMA before traffic arrives. read_volatile prevents
    // the loop from being optimized away.
    let stride = 4096;
    touch_i16(&ivf.centroids, stride);
    touch_i16(&ivf.block_vectors, stride);
    touch_u32(&ivf.block_offsets, stride);
    touch_u32(&ivf.block_origin, stride);
    touch_u8(&ivf.block_labels, stride);
    touch_u8(&ivf.labels_by_origin, stride);
    touch_i16(&ivf.bbox_min, stride);
    touch_i16(&ivf.bbox_max, stride);

    // Phase 2: exercise the full predict path with a deterministic LCG. Hits
    // both the fast-pass and full-pass branches because we sweep the input
    // space wide enough to land all 6 bucket outcomes.
    let mut s: u64 = 0xC0FFEE_DEAD_BEEFu64;
    let mut acc: u64 = 0;
    for _ in 0..rounds {
        let mut q = [0f64; 14];
        for slot in &mut q {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // map upper 32 bits to roughly [-3, 3], the scale of a quantized feature
            let v = ((s >> 32) as i32 as f64) / (i32::MAX as f64) * 3.0;
            *slot = v;
        }
        let bucket = if let Some((seed, cap)) = bbox {
            predict_bucket_ivf_blocks_bbox_repair(ivf, &q, seed, cap)
        } else {
            let qi = quantize_query(&q, ivf.scale);
            let top = ivf_blocks_top5(ivf, &qi, IVF_FULL_NPROBE_WARMUP);
            bucket_from_blocks_top5(ivf, &top)
        };
        // read_volatile sink so codegen can't elide the loop.
        unsafe {
            std::ptr::read_volatile(&bucket);
        }
        acc = acc.wrapping_add(bucket as u64);
    }
    // Sink for the accumulator too, belt-and-suspenders.
    unsafe {
        std::ptr::read_volatile(&acc);
    }
}

// Match the production full nprobe so the warmup exercises the same cluster
// scan depth used by borderline requests.
const IVF_FULL_NPROBE_WARMUP: usize = 96;

#[inline(never)]
fn touch_i16(buf: &[i16], stride_bytes: usize) {
    let step = (stride_bytes / std::mem::size_of::<i16>()).max(1);
    let mut sink: i16 = 0;
    let mut i = 0;
    while i < buf.len() {
        sink ^= unsafe { std::ptr::read_volatile(&buf[i]) };
        i += step;
    }
    unsafe {
        std::ptr::read_volatile(&sink);
    }
}

#[inline(never)]
fn touch_u32(buf: &[u32], stride_bytes: usize) {
    let step = (stride_bytes / std::mem::size_of::<u32>()).max(1);
    let mut sink: u32 = 0;
    let mut i = 0;
    while i < buf.len() {
        sink ^= unsafe { std::ptr::read_volatile(&buf[i]) };
        i += step;
    }
    unsafe {
        std::ptr::read_volatile(&sink);
    }
}

#[inline(never)]
fn touch_u8(buf: &[u8], stride_bytes: usize) {
    let step = stride_bytes.max(1);
    let mut sink: u8 = 0;
    let mut i = 0;
    while i < buf.len() {
        sink ^= unsafe { std::ptr::read_volatile(&buf[i]) };
        i += step;
    }
    unsafe {
        std::ptr::read_volatile(&sink);
    }
}

pub fn predict_bucket_ivf_blocks_two_pass(
    ivf: &IvfBlocks,
    query: &[f64; 14],
    fast_nprobe: usize,
    full_nprobe: usize,
) -> u8 {
    let q = quantize_query(query, ivf.scale);
    let top = ivf_blocks_top5(ivf, &q, fast_nprobe);
    let bucket = bucket_from_blocks_top5(ivf, &top);
    // Asymmetric short-circuit: only trust "clear legit" (bucket==0) from the
    // fast pass. Empirically, at low fast_nprobe a fast verdict of bucket==5
    // ("clear fraud") produces FPs at ~7x the rate of FNs from bucket==0 —
    // fraud clusters are locally denser than the global population, so a tiny
    // scan can coincidentally land on 5 fraud neighbors when the true K-NN
    // would include legit refs from unprobed clusters. Always escalate when
    // the fast pass sees any fraud signal.
    if bucket == 0 {
        return 0;
    }
    let top_full = ivf_blocks_top5(ivf, &q, full_nprobe);
    bucket_from_blocks_top5(ivf, &top_full)
}

// ---- Bounded bbox-repair IVF search ------------------------------------
//
// Hybrid between top-N nprobe and exact full scan. Seeds top-5 by scanning
// the closest few centroids, then walks the remaining clusters in ascending
// bbox-LB order. A cluster is skipped (provably safe by triangle inequality)
// when its L2² lower-bound exceeds the current worst-of-top-5. A hard
// `visit_cap` on the *total* number of clusters scanned bounds the tail —
// pathological queries that would otherwise walk hundreds of clusters give
// up at the cap, trading a small theoretical recall miss for predictable p99.

// L2² lower-bound from a quantized i16 query to a cluster's axis-aligned
// bounding box. For each dim the squared per-axis gap (0 when the query
// coordinate lies inside the box on that axis) is summed. Early-exits the
// moment the running sum exceeds `stop_after`.
#[inline]
fn bbox_lower_bound_l2(bbox_min: &[i16], bbox_max: &[i16], q: &[i16; DIM], stop_after: i64) -> i64 {
    let mut sum: i64 = 0;
    for d in 0..DIM {
        let target = q[d] as i64;
        let lo = bbox_min[d] as i64;
        let hi = bbox_max[d] as i64;
        let gap = if target < lo {
            lo - target
        } else if target > hi {
            target - hi
        } else {
            0
        };
        sum += gap * gap;
        if sum > stop_after {
            return sum;
        }
    }
    sum
}

// AVX2 variant of `bbox_lower_bound_l2`: loads the first 8 dims of bbox/query
// into an __m128i and computes the squared per-axis gaps in parallel. The
// remaining 6 dims (DIM=14) go through a scalar tail.
//
// Math per lane (i16):
//   gap = max(lo - q, 0) + max(q - hi, 0)        // exactly one of these is non-zero
// Then `pmaddwd` computes a*c + b*d on adjacent i16 lanes and stores them as
// i32s — perfect for summing squared distances. Worst-case lane value is
// (2 * 32767)² which exceeds i16 range, so we use saturating subs and stay
// within the legal i16 input domain of `pmaddwd`; the i32 accumulator absorbs
// the products without overflow for our scale (≤ 5.6e9 across 14 dims).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn bbox_lower_bound_l2_avx2(
    bbox_min: &[i16],
    bbox_max: &[i16],
    q: &[i16; DIM],
    stop_after: i64,
) -> i64 {
    let qv = _mm_loadu_si128(q.as_ptr() as *const __m128i);
    let lov = _mm_loadu_si128(bbox_min.as_ptr() as *const __m128i);
    let hiv = _mm_loadu_si128(bbox_max.as_ptr() as *const __m128i);

    let zero = _mm_setzero_si128();
    // Saturating subtractions clamp at i16 bounds; max-with-0 gives the
    // positive gap only. At most one of (lov-qv) and (qv-hiv) is non-zero
    // per lane (mutually exclusive — query is either left, inside, or right
    // of the interval).
    let gap_lo = _mm_max_epi16(_mm_subs_epi16(lov, qv), zero);
    let gap_hi = _mm_max_epi16(_mm_subs_epi16(qv, hiv), zero);
    let gap = _mm_add_epi16(gap_lo, gap_hi);

    // pmaddwd: pairs of adjacent i16 lanes -> i32 sums of squares.
    let sq = _mm_madd_epi16(gap, gap);
    let mut lanes = [0i32; 4];
    _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, sq);
    let mut sum: i64 =
        lanes[0] as i64 + lanes[1] as i64 + lanes[2] as i64 + lanes[3] as i64;

    if sum > stop_after {
        return sum;
    }

    // Tail dims 8..DIM scalar; only 6 iterations.
    for d in 8..DIM {
        let target = q[d] as i64;
        let lo = bbox_min[d] as i64;
        let hi = bbox_max[d] as i64;
        let gap = if target < lo {
            lo - target
        } else if target > hi {
            target - hi
        } else {
            0
        };
        sum += gap * gap;
        if sum > stop_after {
            return sum;
        }
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn bbox_lower_bound_l2_avx2_checked(
    bbox_min: &[i16],
    bbox_max: &[i16],
    q: &[i16; DIM],
    stop_after: i64,
) -> i64 {
    debug_assert!(std::is_x86_feature_detected!("avx2"));
    unsafe { bbox_lower_bound_l2_avx2(bbox_min, bbox_max, q, stop_after) }
}

// Closest N centroid indexes by squared L2, written into a fixed-size buffer.
// Insertion sort over an [i64; N] beats a heap for small N (≤ 5).
#[inline]
fn top_n_centroids_fixed_blocks<FC, const N: usize>(
    ivf: &IvfBlocks,
    q: &[i16; DIM],
    distance: FC,
    out: &mut [u32; N],
) where
    FC: Fn(&[i16], &[i16]) -> i64 + Copy,
{
    let mut dists = [i64::MAX; N];
    *out = [0u32; N];
    for c in 0..ivf.k {
        let d = distance(q, ivf.centroid(c));
        if d >= dists[N - 1] {
            continue;
        }
        let mut pos = N - 1;
        while pos > 0 && d < dists[pos - 1] {
            dists[pos] = dists[pos - 1];
            out[pos] = out[pos - 1];
            pos -= 1;
        }
        dists[pos] = d;
        out[pos] = c as u32;
    }
}

// Upper limit on candidates kept between phase 2's filter and sort. After
// the seed phase prunes `worst` aggressively, the typical query keeps fewer
// than 30 candidates above the bbox-LB threshold. 256 is a comfortable
// safety margin without blowing L1d on the per-query stack frame.
const BBOX_CAND_CAP: usize = 256;

fn predict_bucket_ivf_blocks_bbox_repair_with<FB, FC, FL>(
    ivf: &IvfBlocks,
    query: &[f64; 14],
    seed_clusters: usize,
    visit_cap: usize,
    block_distances: FB,
    centroid_distance: FC,
    bbox_lb: FL,
) -> u8
where
    FB: Fn(&[i16; DIM], &[i16]) -> [i64; BLOCK_SIZE] + Copy,
    FC: Fn(&[i16], &[i16]) -> i64 + Copy,
    FL: Fn(&[i16], &[i16], &[i16; DIM], i64) -> i64 + Copy,
{
    let q = quantize_query(query, ivf.scale);
    let mut distances = [i64::MAX; TOP_K];
    let mut indexes = [0u32; TOP_K];

    // Phase 1: scan the `seed_clusters` closest centroids to fill an initial
    // top-5. A small N (3-5) gives a tight `worst` very quickly so phase 2
    // can prune aggressively.
    const MAX_SEED: usize = 5;
    let seeds_n = seed_clusters.max(1).min(MAX_SEED);
    let mut seeds = [0u32; MAX_SEED];
    top_n_centroids_fixed_blocks::<_, MAX_SEED>(ivf, &q, centroid_distance, &mut seeds);
    let seed_slice = &seeds[..seeds_n];
    for &c in seed_slice {
        for block_idx in ivf.cluster_block_range(c as usize) {
            scan_block_soa(
                ivf,
                block_idx,
                &q,
                &mut distances,
                &mut indexes,
                block_distances,
            );
        }
    }

    // Phase 2: gather every non-seed cluster whose bbox-LB is below the
    // current top-5 worst, then visit them in ascending-LB order. Each
    // visit only shrinks `worst`, so the post-seed value is a safe upper
    // bound to filter against — clusters dropped here would never beat
    // the top-5 even after further refinement.
    //
    // Stack-resident scratch: parallel arrays of (lb, cluster_id) sized
    // for ~256 candidates. After the worst-of-top-5 filter, real queries
    // sit at a few dozen entries, so insertion sort beats a generic
    // sort_unstable: O(N²) with N≈20 is ~150 cycles vs ~30k for a 4096
    // sort_unstable that the prior version paid every single query.
    let mut cand_lb = [0i64; BBOX_CAND_CAP];
    let mut cand_c = [0u32; BBOX_CAND_CAP];
    let mut cand_n = 0usize;

    let worst_after_seed = distances[TOP_K - 1];
    for c in 0..ivf.k {
        if seed_slice.iter().any(|&s| s as usize == c) {
            continue;
        }
        // `stop_after` = current worst tells the LB to bail early if the
        // partial sum already exceeds the threshold — useful because
        // ~99% of clusters end up rejected here.
        let lb = bbox_lb(
            ivf.bbox_min_for(c),
            ivf.bbox_max_for(c),
            &q,
            worst_after_seed,
        );
        if lb >= worst_after_seed {
            continue;
        }
        if cand_n < BBOX_CAND_CAP {
            cand_lb[cand_n] = lb;
            cand_c[cand_n] = c as u32;
            cand_n += 1;
        }
    }

    // Insertion sort over the (typically small) filtered set. Stops as
    // soon as the new key is already in order, so best-case is O(N) for
    // an already-sorted input — common when seeds dominate proximity.
    for i in 1..cand_n {
        let key_lb = cand_lb[i];
        let key_c = cand_c[i];
        let mut j = i;
        while j > 0 && cand_lb[j - 1] > key_lb {
            cand_lb[j] = cand_lb[j - 1];
            cand_c[j] = cand_c[j - 1];
            j -= 1;
        }
        cand_lb[j] = key_lb;
        cand_c[j] = key_c;
    }

    let mut visited = seeds_n;
    for i in 0..cand_n {
        if visited >= visit_cap {
            break;
        }
        let worst = distances[TOP_K - 1];
        if cand_lb[i] >= worst {
            break;
        }
        let c = cand_c[i] as usize;
        for block_idx in ivf.cluster_block_range(c) {
            scan_block_soa(
                ivf,
                block_idx,
                &q,
                &mut distances,
                &mut indexes,
                block_distances,
            );
        }
        visited += 1;
    }

    bucket_from_blocks_top5(ivf, &indexes)
}

pub fn predict_bucket_ivf_blocks_bbox_repair(
    ivf: &IvfBlocks,
    query: &[f64; 14],
    seed_clusters: usize,
    visit_cap: usize,
) -> u8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if avx2_enabled_for_runtime() {
            return predict_bucket_ivf_blocks_bbox_repair_with(
                ivf,
                query,
                seed_clusters,
                visit_cap,
                block_distances_avx2_checked,
                squared_distance_avx2_checked,
                bbox_lower_bound_l2_avx2_checked,
            );
        }
    }
    predict_bucket_ivf_blocks_bbox_repair_with(
        ivf,
        query,
        seed_clusters,
        visit_cap,
        block_distances_scalar,
        squared_distance_scalar,
        bbox_lower_bound_l2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::refs::Refs;

    fn synth_refs(n: usize) -> Refs {
        let dim = DIM;
        let mut vectors = Vec::with_capacity(n * dim);
        let mut labels = Vec::with_capacity(n);
        // Deterministic but varied: each ref is i + j across dims.
        for i in 0..n {
            for j in 0..dim {
                vectors.push(((i + j) as i16).wrapping_mul(7));
            }
            labels.push((i % 2) as u8);
        }
        Refs {
            count: n,
            scale: 10000,
            vectors,
            labels,
        }
    }

    fn build_ivf(refs: &Refs, k: usize) -> Ivf {
        // Cheap-centroids: first K refs (matches the Node builder).
        let mut centroids = Vec::with_capacity(k * DIM);
        for i in 0..k {
            centroids.extend_from_slice(refs.vector(i));
        }

        // Single-pass nearest assignment.
        let mut assign = vec![0u32; refs.count];
        let mut sizes = vec![0u32; k];
        for i in 0..refs.count {
            let v = refs.vector(i);
            let mut best_c = 0usize;
            let mut best_d = i64::MAX;
            for c in 0..k {
                let cv = &centroids[c * DIM..(c + 1) * DIM];
                let d = squared_distance(v, cv);
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            assign[i] = best_c as u32;
            sizes[best_c] += 1;
        }

        let mut offsets = vec![0u32; k + 1];
        let mut acc = 0u32;
        for c in 0..k {
            offsets[c] = acc;
            acc += sizes[c];
        }
        offsets[k] = acc;

        let mut cursor = vec![0u32; k];
        let mut perm_vectors = vec![0i16; refs.count * DIM];
        let mut perm_labels = vec![0u8; refs.count];
        let mut perm_origin = vec![0u32; refs.count];
        for i in 0..refs.count {
            let c = assign[i] as usize;
            let slot = (offsets[c] + cursor[c]) as usize;
            cursor[c] += 1;
            perm_vectors[slot * DIM..(slot + 1) * DIM].copy_from_slice(refs.vector(i));
            perm_labels[slot] = refs.label(i);
            perm_origin[slot] = i as u32;
        }

        Ivf {
            k,
            count: refs.count,
            scale: refs.scale,
            centroids_padded: Some(crate::ivf::pad_vectors(&centroids, k)),
            centroids,
            offsets,
            vectors_padded: Some(crate::ivf::pad_vectors(&perm_vectors, refs.count)),
            vectors: perm_vectors,
            labels: perm_labels,
            labels_by_origin: refs.labels.clone(),
            origin: perm_origin,
        }
    }

    fn quantized(v: &[i16]) -> [i16; DIM] {
        let mut q = [0i16; DIM];
        q.copy_from_slice(v);
        q
    }

    #[test]
    fn ivf_at_nprobe_k_matches_full_scan_top5() {
        let refs = synth_refs(800);
        let ivf = build_ivf(&refs, 32);

        // Ten arbitrary queries — use existing refs as queries with a perturbation so we
        // exercise both exact-hit and near-hit cases.
        for sample in (0..10).map(|i| i * 73 % refs.count) {
            let mut q = quantized(refs.vector(sample));
            q[0] = q[0].wrapping_add(3);
            q[7] = q[7].wrapping_sub(2);

            let full = full_scan_top5(&refs, &q);
            let ivf_top = ivf_top5(&ivf, &q, ivf.k);
            assert_eq!(full, ivf_top, "sample={sample}");
        }
    }

    #[test]
    fn ivf_at_nprobe_k_matches_full_scan_bucket() {
        let refs = synth_refs(500);
        let ivf = build_ivf(&refs, 16);
        for sample in (0..20).map(|i| i * 41 % refs.count) {
            let q = quantized(refs.vector(sample));
            let full_top = full_scan_top5(&refs, &q);
            let ivf_t = ivf_top5(&ivf, &q, ivf.k);
            assert_eq!(full_top, ivf_t);
            assert_eq!(
                bucket_from_top5(&refs, &full_top),
                bucket_from_top5(&refs, &ivf_t)
            );
        }
    }

    #[test]
    fn tie_break_prefers_lower_original_index() {
        // Construct a 2-ref dataset with identical vectors so the distance is tied.
        // Lower-index ref must win.
        let refs = Refs {
            count: 2,
            scale: 10000,
            vectors: vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14,
            ],
            labels: vec![0, 1],
        };
        let q: [i16; DIM] = [0; DIM];
        let top = full_scan_top5(&refs, &q);
        // With only 2 refs, slots 2..5 stay at default 0 — they are dummies.
        assert_eq!(top[0], 0, "lower original index must win the tie");
        assert_eq!(top[1], 1);
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_distance_matches_scalar_for_adversarial_patterns() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping AVX2 distance test: CPU does not report avx2");
            return;
        }

        let cases: [([i16; DIM], [i16; DIM]); 5] = [
            ([0; DIM], [0; DIM]),
            ([32767; DIM], [-32768; DIM]),
            ([-10000; DIM], [10000; DIM]),
            (
                [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14],
                [-14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -4, 3, -2, 1],
            ),
            (
                [
                    1234, -5678, 9012, -12345, 23456, -30000, 42, -42, 777, -888, 999, -1111, 2222,
                    -3333,
                ],
                [
                    -4321, 8765, -2109, 12345, -23456, 30000, -42, 42, -777, 888, -999, 1111,
                    -2222, 3333,
                ],
            ),
        ];

        for (a, b) in cases {
            let mut padded_a = [0i16; DIM_PADDED];
            let mut padded_b = [0i16; DIM_PADDED];
            padded_a[..DIM].copy_from_slice(&a);
            padded_b[..DIM].copy_from_slice(&b);
            assert_eq!(
                squared_distance_scalar(&a, &b),
                squared_distance_avx2_checked(&a, &b),
            );
            assert_eq!(
                squared_distance_scalar(&a, &b),
                squared_distance_avx2_padded_checked(&padded_a, &padded_b),
            );
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_padded_distance_matches_scalar_when_tail_is_zero() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping AVX2 padded distance test: CPU does not report avx2");
            return;
        }

        let scalar_a = [
            1234, -5678, 9012, -12345, 23456, -30000, 42, -42, 777, -888, 999, -1111, 2222, -3333,
        ];
        let scalar_b = [
            -4321, 8765, -2109, 12345, -23456, 30000, -42, 42, -777, 888, -999, 1111, -2222, 3333,
        ];
        let mut padded_a = [0i16; DIM_PADDED];
        let mut padded_b = [0i16; DIM_PADDED];
        padded_a[..DIM].copy_from_slice(&scalar_a);
        padded_b[..DIM].copy_from_slice(&scalar_b);

        assert_eq!(
            squared_distance_scalar(&scalar_a, &scalar_b),
            squared_distance_avx2_padded_checked(&padded_a, &padded_b),
        );
    }

    fn build_ivf_blocks_from_ivf(ivf: &Ivf) -> crate::ivf_blocks::IvfBlocks {
        // Convert v1 IVF (row-major within each cluster) to v2 block-SoA.
        let k = ivf.k;
        let count = ivf.count;
        let mut block_offsets = vec![0u32; k + 1];
        let mut acc = 0u32;
        for c in 0..k {
            let size = ivf.offsets[c + 1] - ivf.offsets[c];
            let blocks = ((size as usize + BLOCK_SIZE - 1) / BLOCK_SIZE) as u32;
            block_offsets[c] = acc;
            acc += blocks;
        }
        block_offsets[k] = acc;
        let total_blocks = acc as usize;

        let mut block_vectors = vec![i16::MAX; total_blocks * DIM * BLOCK_SIZE];
        let mut block_origin = vec![u32::MAX; total_blocks * BLOCK_SIZE];
        let mut block_labels = vec![0xFFu8; total_blocks * BLOCK_SIZE];

        for c in 0..k {
            let ref_start = ivf.offsets[c] as usize;
            let ref_end = ivf.offsets[c + 1] as usize;
            let size = ref_end - ref_start;
            let base_block = block_offsets[c] as usize;
            for i in 0..size {
                let block = base_block + i / BLOCK_SIZE;
                let slot = i % BLOCK_SIZE;
                let perm_idx = ref_start + i;
                let v = ivf.vector_perm(perm_idx);
                let block_base = block * DIM * BLOCK_SIZE;
                for d in 0..DIM {
                    block_vectors[block_base + d * BLOCK_SIZE + slot] = v[d];
                }
                block_origin[block * BLOCK_SIZE + slot] = ivf.origin[perm_idx];
                block_labels[block * BLOCK_SIZE + slot] = ivf.labels[perm_idx];
            }
        }

        let (bbox_min, bbox_max) =
            crate::ivf_blocks::compute_bboxes(k, &block_offsets, &block_vectors, &block_origin);

        crate::ivf_blocks::IvfBlocks {
            k,
            count,
            scale: ivf.scale,
            total_blocks,
            centroids: ivf.centroids.clone(),
            block_offsets,
            block_vectors,
            block_origin,
            block_labels,
            labels_by_origin: ivf.labels_by_origin.clone(),
            bbox_min,
            bbox_max,
        }
    }

    #[test]
    fn ivf_blocks_scalar_matches_ivf_v1_top5() {
        let refs = synth_refs(800);
        let ivf = build_ivf(&refs, 32);
        let blocks = build_ivf_blocks_from_ivf(&ivf);

        for sample in (0..10).map(|i| i * 73 % refs.count) {
            let mut q = quantized(refs.vector(sample));
            q[0] = q[0].wrapping_add(3);
            q[7] = q[7].wrapping_sub(2);

            // Same nprobe (full scan) and same query → identical top-5.
            let v1 = ivf_top5(&ivf, &q, ivf.k);
            let v2 = ivf_blocks_top5_scalar(&blocks, &q, blocks.k);
            assert_eq!(v1, v2, "scalar sample={sample}");
        }
    }

    #[test]
    fn ivf_blocks_scalar_matches_full_scan_at_nprobe_k() {
        let refs = synth_refs(500);
        let ivf = build_ivf(&refs, 16);
        let blocks = build_ivf_blocks_from_ivf(&ivf);
        for sample in (0..20).map(|i| i * 41 % refs.count) {
            let q = quantized(refs.vector(sample));
            let full = full_scan_top5(&refs, &q);
            let v2 = ivf_blocks_top5_scalar(&blocks, &q, blocks.k);
            assert_eq!(full, v2, "sample={sample}");
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn ivf_blocks_avx2_matches_scalar_top5() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping AVX2 block-SoA test: CPU does not report avx2");
            return;
        }
        let refs = synth_refs(800);
        let ivf = build_ivf(&refs, 32);
        let blocks = build_ivf_blocks_from_ivf(&ivf);

        for sample in (0..10).map(|i| i * 73 % refs.count) {
            let mut q = quantized(refs.vector(sample));
            q[0] = q[0].wrapping_add(3);
            q[7] = q[7].wrapping_sub(2);

            let scalar = ivf_blocks_top5_scalar(&blocks, &q, blocks.k);
            let avx2 = ivf_blocks_top5_avx2(&blocks, &q, blocks.k).expect("avx2");
            assert_eq!(scalar, avx2, "sample={sample}");
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn block_distances_avx2_matches_scalar_for_adversarial_patterns() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping AVX2 block_distances test: CPU does not report avx2");
            return;
        }

        // 1 block of 8 slots × 14 dims, SoA-by-block. Place adversarial values across
        // all (slot, dim) combinations to stress sub/widen/square paths.
        let mut block = vec![0i16; DIM * BLOCK_SIZE];
        let presets: [[i16; DIM]; BLOCK_SIZE] = [
            [0; DIM],
            [32767; DIM],
            [-32768; DIM],
            [10000; DIM],
            [-10000; DIM],
            [
                1234, -5678, 9012, -12345, 23456, -30000, 42, -42, 777, -888, 999, -1111, 2222,
                -3333,
            ],
            [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14],
            [i16::MAX; DIM], // sentinel padded slot
        ];
        for slot in 0..BLOCK_SIZE {
            for d in 0..DIM {
                block[d * BLOCK_SIZE + slot] = presets[slot][d];
            }
        }

        let queries: [[i16; DIM]; 4] = [
            [0; DIM],
            [12345; DIM],
            [-12345; DIM],
            [-7, 11, -13, 17, -19, 23, -29, 31, -37, 41, -43, 47, -53, 59],
        ];
        for q in &queries {
            let scalar = block_distances_scalar(q, &block);
            let avx2 = block_distances_avx2_checked(q, &block);
            assert_eq!(scalar, avx2, "query={q:?}");
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_ivf_padded_path_matches_scalar_top5() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping AVX2 padded IVF test: CPU does not report avx2");
            return;
        }

        let refs = synth_refs(800);
        let ivf = build_ivf(&refs, 32);
        for sample in (0..10).map(|i| i * 73 % refs.count) {
            let mut q = quantized(refs.vector(sample));
            q[0] = q[0].wrapping_add(3);
            q[7] = q[7].wrapping_sub(2);

            let scalar = ivf_top5_scalar(&ivf, &q, ivf.k);
            let avx2 = ivf_top5_avx2(&ivf, &q, ivf.k).expect("avx2 available");
            assert_eq!(scalar, avx2, "sample={sample}");
        }
    }
}
