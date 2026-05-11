use napi::bindgen_prelude::*;
use napi_derive::napi;

const DIM: usize = 14;
const K: usize = 5;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn distance_14_impl(query: *const i16, reference: *const i16) -> i64 {
    use std::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm_madd_epi16, _mm_storeu_si128, _mm_sub_epi16,
    };

    let q = _mm_loadu_si128(query as *const __m128i);
    let r = _mm_loadu_si128(reference as *const __m128i);
    let diff = _mm_sub_epi16(q, r);
    let pairs = _mm_madd_epi16(diff, diff);
    let mut lanes = [0_i32; 4];
    _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, pairs);

    let mut distance = lanes[0] as i64 + lanes[1] as i64 + lanes[2] as i64 + lanes[3] as i64;
    for dim in 8..DIM {
        let diff = *query.add(dim) as i64 - *reference.add(dim) as i64;
        distance += diff * diff;
    }

    distance
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn distance_14_impl(query: *const i16, reference: *const i16) -> i64 {
    let mut distance = 0_i64;
    for dim in 0..DIM {
        let diff = *query.add(dim) as i64 - *reference.add(dim) as i64;
        distance += diff * diff;
    }
    distance
}

fn insert_top_k(distances: &mut [i64; K], indexes: &mut [usize; K], distance: i64, index: usize) {
    if distance < distances[K - 1] {
        let mut pos = K - 1;
        while pos > 0 && distance < distances[pos - 1] {
            distances[pos] = distances[pos - 1];
            indexes[pos] = indexes[pos - 1];
            pos -= 1;
        }
        distances[pos] = distance;
        indexes[pos] = index;
    }
}

fn bucket_from_indexes(labels: &[u8], indexes: [usize; K]) -> u32 {
    let mut bucket = 0_u32;
    for idx in indexes {
        if labels[idx] == 1 {
            bucket += 1;
        }
    }
    bucket
}

#[napi(js_name = "distance14")]
pub fn distance_14(query: Int16Array, refs: Int16Array, ref_index: u32) -> Result<i64> {
    let query = query.as_ref();
    let refs = refs.as_ref();
    let base = ref_index as usize * DIM;

    if query.len() != DIM {
        return Err(Error::from_reason(format!(
            "query must have {DIM} dimensions, got {}",
            query.len()
        )));
    }
    if base + DIM > refs.len() {
        return Err(Error::from_reason("reference index out of range"));
    }

    Ok(unsafe { distance_14_impl(query.as_ptr(), refs.as_ptr().add(base)) })
}

#[napi(js_name = "knnBucket")]
pub fn knn_bucket(
    query: Int16Array,
    refs: Int16Array,
    labels: Uint8Array,
    count: u32,
) -> Result<u32> {
    let query = query.as_ref();
    let refs = refs.as_ref();
    let labels = labels.as_ref();
    let count = count as usize;

    if query.len() != DIM {
        return Err(Error::from_reason(format!(
            "query must have {DIM} dimensions, got {}",
            query.len()
        )));
    }
    if refs.len() < count * DIM {
        return Err(Error::from_reason(
            "reference vector buffer shorter than count * dim",
        ));
    }
    if labels.len() < count {
        return Err(Error::from_reason("label buffer shorter than count"));
    }
    if count < K {
        return Err(Error::from_reason("at least 5 references are required"));
    }

    let mut distances = [i64::MAX; K];
    let mut indexes = [usize::MAX; K];

    for i in 0..count {
        let base = i * DIM;
        let distance = unsafe { distance_14_impl(query.as_ptr(), refs.as_ptr().add(base)) };
        insert_top_k(&mut distances, &mut indexes, distance, i);
    }

    Ok(bucket_from_indexes(labels, indexes))
}

#[napi(js_name = "knnBucketCandidates")]
pub fn knn_bucket_candidates(
    query: Int16Array,
    refs: Int16Array,
    labels: Uint8Array,
    candidates: Uint32Array,
) -> Result<u32> {
    let query = query.as_ref();
    let refs = refs.as_ref();
    let labels = labels.as_ref();
    let candidates = candidates.as_ref();

    if query.len() != DIM {
        return Err(Error::from_reason(format!(
            "query must have {DIM} dimensions, got {}",
            query.len()
        )));
    }
    if candidates.len() < K {
        return Err(Error::from_reason("at least 5 candidates are required"));
    }

    let mut distances = [i64::MAX; K];
    let mut indexes = [usize::MAX; K];

    for candidate in candidates {
        let i = *candidate as usize;
        let base = i * DIM;
        if base + DIM > refs.len() {
            return Err(Error::from_reason("candidate index out of reference range"));
        }
        if i >= labels.len() {
            return Err(Error::from_reason("candidate index out of label range"));
        }

        let distance = unsafe { distance_14_impl(query.as_ptr(), refs.as_ptr().add(base)) };
        insert_top_k(&mut distances, &mut indexes, distance, i);
    }

    Ok(bucket_from_indexes(labels, indexes))
}

const FEATURE_FIELDS: usize = 4;
const FLAG_FIXED: u32 = 1;
const FLAG_SENTINEL: u32 = 2;

#[inline]
fn quantized_to_bin(value: i16, scale: i32, bins: i32, flags: u32) -> i32 {
    let value = value as i32;
    if (flags & FLAG_SENTINEL) != 0 && value < 0 {
        return 0;
    }
    let clamped = if value < 0 {
        0
    } else if value > scale {
        scale
    } else {
        value
    };
    let mut bin = ((clamped as i64) * (bins as i64) / (scale as i64)) as i32;
    if bin >= bins {
        bin = bins - 1;
    }
    if (flags & FLAG_SENTINEL) != 0 {
        bin + 1
    } else {
        bin
    }
}

#[inline]
fn bins_to_key(bins: &[i32], features: &[u32]) -> u32 {
    let mut key: u64 = 0;
    let mut multiplier: u64 = 1;
    let count = bins.len();
    for i in 0..count {
        let radix = features[i * FEATURE_FIELDS + 3] as u64;
        key += (bins[i] as u64) * multiplier;
        multiplier *= radix;
    }
    key as u32
}

#[inline]
fn find_bucket(bucket_table: &[u32], bucket_count: usize, key: u32) -> Option<usize> {
    let mut lo: i64 = 0;
    let mut hi: i64 = bucket_count as i64 - 1;
    while lo <= hi {
        let mid = ((lo + hi) >> 1) as usize;
        let cur = bucket_table[mid * 3];
        if cur == key {
            return Some(mid);
        }
        if cur < key {
            lo = mid as i64 + 1;
        } else {
            hi = mid as i64 - 1;
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
unsafe fn scan_bucket(
    bucket_index: usize,
    bucket_table: &[u32],
    postings: &[u32],
    query_ptr: *const i16,
    refs_ptr: *const i16,
    refs_len: usize,
    labels_len: usize,
    seen: &mut [u8],
    visited: &mut Vec<usize>,
    distances: &mut [i64; K],
    indexes: &mut [usize; K],
    unique_count: &mut usize,
) -> Result<()> {
    let start = bucket_table[bucket_index * 3 + 1] as usize;
    let length = bucket_table[bucket_index * 3 + 2] as usize;
    for slot in 0..length {
        let i = postings[start + slot] as usize;
        if i >= labels_len {
            return Err(Error::from_reason("posting index out of label range"));
        }
        let base = i * DIM;
        if base + DIM > refs_len {
            return Err(Error::from_reason("posting index out of reference range"));
        }
        if seen[i] != 0 {
            continue;
        }
        seen[i] = 1;
        visited.push(i);
        *unique_count += 1;
        let distance = distance_14_impl(query_ptr, refs_ptr.add(base));
        insert_top_k(distances, indexes, distance, i);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn walk_probes(
    pos: usize,
    remaining: i32,
    bins_query: &[i32],
    bins_current: &mut [i32],
    features: &[u32],
    bucket_table: &[u32],
    bucket_count: usize,
    postings: &[u32],
    query_ptr: *const i16,
    refs_ptr: *const i16,
    refs_len: usize,
    labels_len: usize,
    seen: &mut [u8],
    visited: &mut Vec<usize>,
    distances: &mut [i64; K],
    indexes: &mut [usize; K],
    unique_count: &mut usize,
    target_unique: usize,
    radius_only: i32,
) -> Result<()> {
    if pos == bins_query.len() {
        // Skip if total cost != radius_only (we want exact-radius shells per pass).
        let mut cost = 0_i32;
        for i in 0..bins_query.len() {
            cost += (bins_current[i] - bins_query[i]).abs();
        }
        if cost != radius_only {
            return Ok(());
        }
        let key = bins_to_key(bins_current, features);
        if let Some(bucket_index) = find_bucket(bucket_table, bucket_count, key) {
            scan_bucket(
                bucket_index,
                bucket_table,
                postings,
                query_ptr,
                refs_ptr,
                refs_len,
                labels_len,
                seen,
                visited,
                distances,
                indexes,
                unique_count,
            )?;
        }
        return Ok(());
    }

    let feature_base = pos * FEATURE_FIELDS;
    let bins_count = features[feature_base + 1] as i32;
    let flags = features[feature_base + 2];
    let radix = features[feature_base + 3] as i32;
    let original = bins_query[pos];
    let fixed = (flags & FLAG_FIXED) != 0
        || ((flags & FLAG_SENTINEL) != 0 && original == 0);
    let _ = bins_count;

    if fixed {
        bins_current[pos] = original;
        walk_probes(
            pos + 1,
            remaining,
            bins_query,
            bins_current,
            features,
            bucket_table,
            bucket_count,
            postings,
            query_ptr,
            refs_ptr,
            refs_len,
            labels_len,
            seen,
            visited,
            distances,
            indexes,
            unique_count,
            target_unique,
            radius_only,
        )?;
        return Ok(());
    }

    let min = (original - remaining).max(0);
    let max = (original + remaining).min(radix - 1);
    let mut value = min;
    while value <= max {
        let cost = (value - original).abs();
        if cost <= remaining {
            bins_current[pos] = value;
            walk_probes(
                pos + 1,
                remaining - cost,
                bins_query,
                bins_current,
                features,
                bucket_table,
                bucket_count,
                postings,
                query_ptr,
                refs_ptr,
                refs_len,
                labels_len,
                seen,
                visited,
                distances,
                indexes,
                unique_count,
                target_unique,
                radius_only,
            )?;
        }
        value += 1;
    }
    Ok(())
}

#[napi(js_name = "multiGridKnnBucket")]
#[allow(clippy::too_many_arguments)]
pub fn multi_grid_knn_bucket(
    query: Int16Array,
    refs: Int16Array,
    labels: Uint8Array,
    bucket_tables: Vec<Uint32Array>,
    postings_list: Vec<Uint32Array>,
    features_list: Vec<Uint32Array>,
    scale: i32,
    per_grid_min_candidates: u32,
    max_radius: u32,
    seen_scratch: Uint8Array,
) -> Result<u32> {
    let query = query.as_ref();
    let refs = refs.as_ref();
    let labels = labels.as_ref();
    let mut seen_scratch = seen_scratch;
    let seen = seen_scratch.as_mut();

    if query.len() != DIM {
        return Err(Error::from_reason(format!(
            "query must have {DIM} dimensions, got {}",
            query.len()
        )));
    }
    let grid_count = bucket_tables.len();
    if postings_list.len() != grid_count || features_list.len() != grid_count {
        return Err(Error::from_reason("grid descriptor list length mismatch"));
    }
    if seen.len() < labels.len() {
        return Err(Error::from_reason(
            "seen scratch buffer shorter than labels length",
        ));
    }

    for cell in seen.iter_mut() {
        *cell = 0;
    }

    let mut distances = [i64::MAX; K];
    let mut indexes = [usize::MAX; K];
    let mut visited: Vec<usize> = Vec::new();
    let mut unique_count = 0_usize;

    let target_unique = per_grid_min_candidates as usize;

    for g in 0..grid_count {
        let bucket_table = bucket_tables[g].as_ref();
        let postings = postings_list[g].as_ref();
        let features = features_list[g].as_ref();
        if features.len() % FEATURE_FIELDS != 0 {
            return Err(Error::from_reason(
                "features array length must be a multiple of 4",
            ));
        }
        let feature_count = features.len() / FEATURE_FIELDS;
        if bucket_table.len() % 3 != 0 {
            return Err(Error::from_reason(
                "bucket table length must be a multiple of 3",
            ));
        }
        let bucket_count = bucket_table.len() / 3;

        let unique_before = unique_count;
        let mut bins_query = vec![0_i32; feature_count];
        for i in 0..feature_count {
            let dim_idx = features[i * FEATURE_FIELDS] as usize;
            if dim_idx >= DIM {
                return Err(Error::from_reason("feature dim out of range"));
            }
            let bins = features[i * FEATURE_FIELDS + 1] as i32;
            let flags = features[i * FEATURE_FIELDS + 2];
            bins_query[i] = quantized_to_bin(query[dim_idx], scale, bins, flags);
        }
        let mut bins_current = bins_query.clone();

        let max_r = max_radius as i32;
        for radius in 0..=max_r {
            unsafe {
                walk_probes(
                    0,
                    radius,
                    &bins_query,
                    &mut bins_current,
                    features,
                    bucket_table,
                    bucket_count,
                    postings,
                    query.as_ptr(),
                    refs.as_ptr(),
                    refs.len(),
                    labels.len(),
                    seen,
                    &mut visited,
                    &mut distances,
                    &mut indexes,
                    &mut unique_count,
                    target_unique,
                    radius,
                )?;
            }
            if unique_count - unique_before >= target_unique {
                break;
            }
        }
    }

    for mark in &visited {
        seen[*mark] = 0;
    }

    if unique_count < K {
        return Err(Error::from_reason(
            "at least 5 unique candidates are required",
        ));
    }

    Ok(bucket_from_indexes(labels, indexes))
}

#[napi(js_name = "knnBucketMultiCandidates")]
pub fn knn_bucket_multi_candidates(
    query: Int16Array,
    refs: Int16Array,
    labels: Uint8Array,
    candidate_groups: Vec<Uint32Array>,
    seen_scratch: Uint8Array,
) -> Result<u32> {
    let query = query.as_ref();
    let refs = refs.as_ref();
    let labels = labels.as_ref();
    let mut seen = seen_scratch;
    let seen_buf = seen.as_mut();

    if query.len() != DIM {
        return Err(Error::from_reason(format!(
            "query must have {DIM} dimensions, got {}",
            query.len()
        )));
    }
    if seen_buf.len() < labels.len() {
        return Err(Error::from_reason(
            "seen scratch buffer shorter than labels length",
        ));
    }

    for cell in seen_buf.iter_mut() {
        *cell = 0;
    }

    let mut distances = [i64::MAX; K];
    let mut indexes = [usize::MAX; K];
    let mut visited_marks: Vec<usize> = Vec::new();
    let mut total_unique = 0_usize;

    for group in &candidate_groups {
        for candidate in group.as_ref() {
            let i = *candidate as usize;
            if i >= labels.len() {
                for mark in &visited_marks {
                    seen_buf[*mark] = 0;
                }
                return Err(Error::from_reason("candidate index out of label range"));
            }
            let base = i * DIM;
            if base + DIM > refs.len() {
                for mark in &visited_marks {
                    seen_buf[*mark] = 0;
                }
                return Err(Error::from_reason("candidate index out of reference range"));
            }
            if seen_buf[i] != 0 {
                continue;
            }
            seen_buf[i] = 1;
            visited_marks.push(i);
            total_unique += 1;

            let distance = unsafe { distance_14_impl(query.as_ptr(), refs.as_ptr().add(base)) };
            insert_top_k(&mut distances, &mut indexes, distance, i);
        }
    }

    for mark in &visited_marks {
        seen_buf[*mark] = 0;
    }

    if total_unique < K {
        return Err(Error::from_reason(
            "at least 5 unique candidates are required",
        ));
    }

    Ok(bucket_from_indexes(labels, indexes))
}
