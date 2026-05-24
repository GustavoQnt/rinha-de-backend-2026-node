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
    pub two_pass: Option<(usize, usize)>,
    pub bbox_repair: Option<(usize, usize)>, // (seed_clusters, visit_cap)
}

impl Index {
    pub fn predict(&self, v: &[f64; 14]) -> u8 {
        if let Some(blocks) = &self.ivf_blocks {
            if let Some((seed, cap)) = self.bbox_repair {
                return knn::predict_bucket_ivf_blocks_bbox_repair(blocks, v, seed, cap);
            }
            if let Some((fast, full)) = self.two_pass {
                return knn::predict_bucket_ivf_blocks_two_pass(blocks, v, fast, full);
            }
            return knn::predict_bucket_ivf_blocks(blocks, v, self.nprobe);
        }
        match &self.ivf {
            Some(ivf) => knn::predict_bucket_ivf(ivf, v, self.nprobe),
            None => knn::predict_bucket(self.refs.as_ref().expect("refs loaded"), v),
        }
    }
}
