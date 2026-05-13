use crate::ivf::Ivf;
use crate::ivf_blocks::IvfBlocks;
use crate::knn;
use crate::refs::Refs;

pub const MAX_HEAD: usize = 16 * 1024;
pub const MAX_BODY: usize = 64 * 1024;

pub struct Index {
    pub refs: Option<Refs>,
    pub ivf: Option<Ivf>,
    pub ivf_blocks: Option<IvfBlocks>,
    pub nprobe: usize,
}

impl Index {
    #[inline]
    pub fn predict(&self, v: &[f64; 14]) -> u8 {
        if let Some(blocks) = &self.ivf_blocks {
            // Bbox repair is the default when blocks are loaded: provably
            // exact recall, with most clusters pruned via L2² lower-bound.
            return knn::predict_bucket_ivf_blocks_bbox_repair(blocks, v);
        }
        match &self.ivf {
            Some(ivf) => knn::predict_bucket_ivf(ivf, v, self.nprobe),
            None => knn::predict_bucket(self.refs.as_ref().expect("refs loaded"), v),
        }
    }
}
