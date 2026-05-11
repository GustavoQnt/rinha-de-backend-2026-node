use std::fs::File;
use std::io::Read;

const MAGIC: &[u8; 4] = b"R26B";
const VERSION: u32 = 1;
pub const DIM: usize = 14;
pub const DIM_PADDED: usize = 16;
const HEADER_BYTES: usize = 24;

pub struct Refs {
    pub count: usize,
    pub scale: u32,
    pub vectors: Vec<i16>,
    pub labels: Vec<u8>,
}

unsafe impl Send for Refs {}
unsafe impl Sync for Refs {}

impl Refs {
    pub fn load(path: &str) -> Self {
        let mut file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let len = file.metadata().map(|m| m.len() as usize).unwrap_or(0);
        let mut bytes = Vec::with_capacity(len);
        file.read_to_end(&mut bytes).expect("read references.bin");

        assert!(bytes.len() >= HEADER_BYTES, "file too small");
        assert_eq!(&bytes[0..4], MAGIC, "invalid magic");

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, VERSION, "unsupported version: {version}");

        let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        assert_eq!(dim, DIM, "unexpected dim: {dim}");

        let scale = u32::from_le_bytes(bytes[16..20].try_into().unwrap());

        let vectors_end = HEADER_BYTES + count * DIM * 2;
        let labels_end = vectors_end + count;
        assert_eq!(bytes.len(), labels_end, "unexpected file size");

        let raw = &bytes[HEADER_BYTES..vectors_end];
        let vectors: Vec<i16> = raw
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let labels = bytes[vectors_end..labels_end].to_vec();

        Refs {
            count,
            scale,
            vectors,
            labels,
        }
    }

    #[inline(always)]
    pub fn vector(&self, i: usize) -> &[i16] {
        &self.vectors[i * DIM..(i + 1) * DIM]
    }

    #[inline(always)]
    pub fn label(&self, i: usize) -> u8 {
        self.labels[i]
    }
}
