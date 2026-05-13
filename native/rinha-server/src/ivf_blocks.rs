// IVF block-SoA (R26I v2) loader. See `scripts/build-ivf-blocks-index.js` for the
// file format and `docs/plans/2026-05-07-block-soa-ivf-plan.md` for rationale.

use std::fs::File;
use std::io::Read;

use crate::refs::DIM;

pub const BLOCK_SIZE: usize = 8;
pub const BLOCK_LANES: usize = DIM * BLOCK_SIZE; // 112 i16 per block

const MAGIC: &[u8; 4] = b"R26I";
const VERSION: u32 = 2;
const HEADER_BYTES: usize = 32;

pub struct IvfBlocks {
    pub k: usize,
    pub count: usize,
    pub scale: u32,
    pub total_blocks: usize,
    pub centroids: Vec<i16>,             // K × DIM
    pub block_offsets: Vec<u32>,         // (K+1), cumulative blocks
    pub block_vectors: Vec<i16>,         // total_blocks × 112, SoA-by-block
    pub block_origin: Vec<u32>,          // total_blocks × 8, u32::MAX for padding
    pub block_labels: Vec<u8>,           // total_blocks × 8, 0xFF for padding
    pub labels_by_origin: Vec<u8>,       // count
}

unsafe impl Send for IvfBlocks {}
unsafe impl Sync for IvfBlocks {}

impl IvfBlocks {
    pub fn load(path: &str) -> Self {
        let mut file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let len = file.metadata().map(|m| m.len() as usize).unwrap_or(0);
        let mut bytes = Vec::with_capacity(len);
        file.read_to_end(&mut bytes).expect("read ivf-blocks file");

        assert!(bytes.len() >= HEADER_BYTES, "ivf-blocks file too small");
        assert_eq!(&bytes[0..4], MAGIC, "invalid ivf-blocks magic");

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, VERSION, "unsupported ivf-blocks version: {version}");

        let k = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let scale = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let count = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
        let block_size = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
        let total_blocks = u32::from_le_bytes(bytes[28..32].try_into().unwrap()) as usize;
        assert_eq!(dim, DIM, "unexpected ivf-blocks dim: {dim}");
        assert_eq!(block_size, BLOCK_SIZE, "unexpected block_size: {block_size}");

        let centroids_bytes = k * DIM * 2;
        let block_offsets_bytes = (k + 1) * 4;
        let block_vectors_bytes = total_blocks * BLOCK_LANES * 2;
        let block_origin_bytes = total_blocks * BLOCK_SIZE * 4;
        let block_labels_bytes = total_blocks * BLOCK_SIZE;
        let labels_by_origin_bytes = count;

        let p_centroids = HEADER_BYTES;
        let p_block_offsets = p_centroids + centroids_bytes;
        let p_block_vectors = p_block_offsets + block_offsets_bytes;
        let p_block_origin = p_block_vectors + block_vectors_bytes;
        let p_block_labels = p_block_origin + block_origin_bytes;
        let p_labels_by_origin = p_block_labels + block_labels_bytes;
        let expected_end = p_labels_by_origin + labels_by_origin_bytes;
        assert_eq!(bytes.len(), expected_end, "unexpected ivf-blocks file size");

        let centroids: Vec<i16> = bytes[p_centroids..p_block_offsets]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let block_offsets: Vec<u32> = bytes[p_block_offsets..p_block_vectors]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let block_vectors: Vec<i16> = bytes[p_block_vectors..p_block_origin]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let block_origin: Vec<u32> = bytes[p_block_origin..p_block_labels]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let block_labels = bytes[p_block_labels..p_labels_by_origin].to_vec();
        let labels_by_origin = bytes[p_labels_by_origin..expected_end].to_vec();

        assert_eq!(block_offsets.len(), k + 1);
        assert_eq!(block_offsets[0], 0);
        assert_eq!(block_offsets[k] as usize, total_blocks);

        IvfBlocks {
            k,
            count,
            scale,
            total_blocks,
            centroids,
            block_offsets,
            block_vectors,
            block_origin,
            block_labels,
            labels_by_origin,
        }
    }

    #[inline(always)]
    pub fn centroid(&self, c: usize) -> &[i16] {
        &self.centroids[c * DIM..(c + 1) * DIM]
    }

    /// Cluster's block range as [start_block, end_block).
    #[inline(always)]
    pub fn cluster_block_range(&self, c: usize) -> std::ops::Range<usize> {
        self.block_offsets[c] as usize..self.block_offsets[c + 1] as usize
    }

    /// 112 i16 of one block, SoA-by-block layout: dim d, slot s -> offset d*8 + s.
    #[inline(always)]
    pub fn block_payload(&self, block: usize) -> &[i16] {
        &self.block_vectors[block * BLOCK_LANES..(block + 1) * BLOCK_LANES]
    }

    #[inline(always)]
    pub fn block_origin_slots(&self, block: usize) -> &[u32] {
        &self.block_origin[block * BLOCK_SIZE..(block + 1) * BLOCK_SIZE]
    }

    #[inline(always)]
    pub fn label_by_origin(&self, i: usize) -> u8 {
        self.labels_by_origin[i]
    }
}
