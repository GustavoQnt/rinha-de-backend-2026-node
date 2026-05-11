// IVF (R26I v1) index loader. See `scripts/build-ivf-index.js` for the file format.

use std::fs::File;
use std::io::Read;

use crate::refs::{DIM, DIM_PADDED};

const MAGIC: &[u8; 4] = b"R26I";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 24;

pub struct Ivf {
    pub k: usize,
    pub count: usize,
    pub scale: u32,
    pub centroids: Vec<i16>,
    pub centroids_padded: Option<Vec<i16>>,
    pub offsets: Vec<u32>,
    pub vectors: Vec<i16>,
    pub vectors_padded: Option<Vec<i16>>,
    pub labels: Vec<u8>,
    pub labels_by_origin: Vec<u8>,
    pub origin: Vec<u32>,
}

unsafe impl Send for Ivf {}
unsafe impl Sync for Ivf {}

impl Ivf {
    pub fn load(path: &str) -> Self {
        let mut file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let len = file.metadata().map(|m| m.len() as usize).unwrap_or(0);
        let mut bytes = Vec::with_capacity(len);
        file.read_to_end(&mut bytes).expect("read ivf file");

        assert!(bytes.len() >= HEADER_BYTES, "ivf file too small");
        assert_eq!(&bytes[0..4], MAGIC, "invalid ivf magic");

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, VERSION, "unsupported ivf version: {version}");

        let k = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let scale = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let count = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
        assert_eq!(dim, DIM, "unexpected ivf dim: {dim}");

        let centroids_bytes = k * DIM * 2;
        let offsets_bytes = (k + 1) * 4;
        let vectors_bytes = count * DIM * 2;
        let labels_bytes = count;
        let origin_bytes = count * 4;

        let p_centroids = HEADER_BYTES;
        let p_offsets = p_centroids + centroids_bytes;
        let p_vectors = p_offsets + offsets_bytes;
        let p_labels = p_vectors + vectors_bytes;
        let p_origin = p_labels + labels_bytes;
        let expected_end = p_origin + origin_bytes;
        assert_eq!(bytes.len(), expected_end, "unexpected ivf file size");

        let centroids: Vec<i16> = bytes[p_centroids..p_offsets]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let offsets: Vec<u32> = bytes[p_offsets..p_vectors]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let vectors: Vec<i16> = bytes[p_vectors..p_labels]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let use_padded = std::env::var("RINHA_USE_AVX2_PADDED16").as_deref() == Ok("1");
        let centroids_padded = use_padded.then(|| pad_vectors(&centroids, k));
        let vectors_padded = use_padded.then(|| pad_vectors(&vectors, count));
        let labels = bytes[p_labels..p_origin].to_vec();
        let origin: Vec<u32> = bytes[p_origin..expected_end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut labels_by_origin = vec![0u8; count];
        for p in 0..count {
            labels_by_origin[origin[p] as usize] = labels[p];
        }

        // Sanity: offsets must be monotone and cover exactly count.
        assert_eq!(offsets.len(), k + 1);
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[k] as usize, count);

        Ivf {
            k,
            count,
            scale,
            centroids,
            centroids_padded,
            offsets,
            vectors,
            vectors_padded,
            labels,
            labels_by_origin,
            origin,
        }
    }

    #[inline(always)]
    pub fn centroid(&self, c: usize) -> &[i16] {
        &self.centroids[c * DIM..(c + 1) * DIM]
    }

    #[inline(always)]
    pub fn centroid_padded(&self, c: usize) -> &[i16] {
        let centroids = self
            .centroids_padded
            .as_ref()
            .expect("padded centroids loaded");
        &centroids[c * DIM_PADDED..(c + 1) * DIM_PADDED]
    }

    #[inline(always)]
    pub fn cluster_range(&self, c: usize) -> std::ops::Range<usize> {
        self.offsets[c] as usize..self.offsets[c + 1] as usize
    }

    #[inline(always)]
    pub fn vector_perm(&self, p: usize) -> &[i16] {
        &self.vectors[p * DIM..(p + 1) * DIM]
    }

    #[inline(always)]
    pub fn vector_perm_padded(&self, p: usize) -> &[i16] {
        let vectors = self.vectors_padded.as_ref().expect("padded vectors loaded");
        &vectors[p * DIM_PADDED..(p + 1) * DIM_PADDED]
    }

    #[inline(always)]
    pub fn has_padded_vectors(&self) -> bool {
        self.centroids_padded.is_some() && self.vectors_padded.is_some()
    }

    #[inline(always)]
    pub fn label_by_origin(&self, i: usize) -> u8 {
        self.labels_by_origin[i]
    }
}

pub fn pad_vectors(raw: &[i16], count: usize) -> Vec<i16> {
    assert_eq!(raw.len(), count * DIM);
    let mut padded = vec![0i16; count * DIM_PADDED];
    for i in 0..count {
        let raw_start = i * DIM;
        let padded_start = i * DIM_PADDED;
        padded[padded_start..padded_start + DIM].copy_from_slice(&raw[raw_start..raw_start + DIM]);
    }
    padded
}
