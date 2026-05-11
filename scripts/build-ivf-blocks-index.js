/**
 * Block-SoA IVF builder (R26I v2). See docs/plans/2026-05-07-block-soa-ivf-plan.md.
 *
 * Reuses k-means / assignment from `scripts/build-ivf-index.js` so centroids and
 * cluster membership match v1 byte-for-byte for the same inputs/seed.
 *
 * File format (R26I v2, all little-endian):
 *
 *   header (32 bytes):
 *     magic        "R26I"        4 bytes
 *     version      u32           = 2
 *     K            u32           number of centroids
 *     dim          u32           = 14
 *     scale        u32           = 10000
 *     count        u32           number of references
 *     block_size   u32           = 8
 *     total_blocks u32
 *
 *   centroids:        K * dim * 2 bytes        (int16 LE)
 *   block_offsets:    (K+1) * 4 bytes          (uint32 LE) — cumulative BLOCK index per cluster
 *   block_vectors:    total_blocks * 112 * 2   (int16 LE, SoA-by-block: 14 dims × 8 slots)
 *   block_origin:     total_blocks * 8 * 4     (uint32 LE) — u32::MAX for padding
 *   block_labels:     total_blocks * 8 * 1     (u8)        — 0xFF for padding
 *   labels_by_origin: count * 1 byte           (u8)
 *
 * Block payload layout (112 int16 = 14 dims × 8 slots) inside one block:
 *   [v0_d0  v1_d0  v2_d0  v3_d0  v4_d0  v5_d0  v6_d0  v7_d0]   ← 8 lanes, dim 0
 *   [v0_d1  v1_d1  ...                                      ]   ← 8 lanes, dim 1
 *   ...
 *   [v0_d13 v1_d13 ...                                      ]   ← 8 lanes, dim 13
 *
 * Padded slots get vector i16::MAX in every dim — distance to any clamped query is
 * orders of magnitude above any real ref, so they cannot enter top-5.
 *
 * Usage:
 *   node scripts/build-ivf-blocks-index.js [--K 1024] [--centroids first|sample|kmeans]
 *                                          [--iterations 25] [--seed 12626]
 *                                          [--input resources/references.bin]
 *                                          [--output resources/references.ivf-blocks.bin]
 */

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  readReferences,
  pickCentroids,
  sampleCentroids,
  kMeansCentroids,
  assignClusters,
  buildPermutation,
} from './build-ivf-index.js';

const MAGIC_IVF = 'R26I';
const HEADER_BYTES = 32;
const VERSION = 2;
const DIM = 14;
const BLOCK_SIZE = 8;
const PAD_LANES_I16 = 0x7FFF;        // i16::MAX
const PAD_ORIGIN_U32 = 0xFFFFFFFF;   // u32::MAX
const PAD_LABEL_U8 = 0xFF;

function parseArgs(argv) {
  const opts = {
    K: 1024,
    input: 'resources/references.bin',
    output: 'resources/references.ivf-blocks.bin',
    metaPath: 'resources/references.ivf-blocks.meta.json',
    centroidStrategy: 'kmeans',
    iterations: 25,
    seed: 12626,
    progressEvery: 250_000,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '--K') opts.K = Number(argv[++i]);
    else if (a === '--centroids') opts.centroidStrategy = argv[++i];
    else if (a === '--iterations') opts.iterations = Number(argv[++i]);
    else if (a === '--seed') opts.seed = Number(argv[++i]);
    else if (a === '--input') opts.input = argv[++i];
    else if (a === '--output') opts.output = argv[++i];
    else if (a === '--meta') opts.metaPath = argv[++i];
    else if (a === '--no-meta') opts.metaPath = null;
    else if (a === '--early-stop') opts.earlyStopRatio = Number(argv[++i]);
    else throw new Error(`unknown argument: ${a}`);
  }
  if (!Number.isInteger(opts.K) || opts.K <= 0 || opts.K > 65535) {
    throw new Error(`invalid --K: ${opts.K}`);
  }
  if (!['first', 'sample', 'kmeans', 'kmeans++'].includes(opts.centroidStrategy)) {
    throw new Error(`invalid --centroids: ${opts.centroidStrategy}`);
  }
  return opts;
}

function chooseCentroids(refs, opts) {
  const K = opts.K;
  if (opts.centroidStrategy === 'first') {
    console.error(`picking ${K} centroids (first K refs)`);
    return pickCentroids(refs, K);
  }
  if (opts.centroidStrategy === 'sample') {
    console.error(`picking ${K} centroids (seeded random sample, seed=${opts.seed})`);
    return sampleCentroids(refs, K, opts.seed);
  }
  if (opts.centroidStrategy === 'kmeans++') {
    console.error(`picking ${K} centroids (k-means with kmeans++ init, iterations=${opts.iterations}, seed=${opts.seed}, earlyStop=${opts.earlyStopRatio ?? 'off'})`);
    return kMeansCentroids(refs, K, {
      iterations: opts.iterations,
      seed: opts.seed,
      progressEvery: opts.progressEvery,
      init: 'kmeans++',
      earlyStopRatio: opts.earlyStopRatio,
    });
  }
  console.error(`picking ${K} centroids (k-means, iterations=${opts.iterations}, seed=${opts.seed})`);
  return kMeansCentroids(refs, K, {
    iterations: opts.iterations,
    seed: opts.seed,
    progressEvery: opts.progressEvery,
  });
}

/**
 * Convert a permuted (cluster-contiguous) ref array into block-SoA layout.
 * Returns:
 *   - blockOffsets: Uint32Array(K+1) cumulative BLOCK index per cluster
 *   - blockVectors: Int16Array(totalBlocks * 112) SoA-by-block
 *   - blockOrigin:  Uint32Array(totalBlocks * 8)
 *   - blockLabels:  Uint8Array(totalBlocks * 8)
 *   - totalBlocks:  number
 */
function buildBlockSoA(refs, perm, K) {
  // Original (v1) offsets are per-ref. Re-derive cluster sizes from them.
  const blockOffsets = new Uint32Array(K + 1);
  let acc = 0;
  for (let c = 0; c < K; c += 1) {
    const size = perm.offsets[c + 1] - perm.offsets[c];
    const blocks = Math.ceil(size / BLOCK_SIZE);
    blockOffsets[c] = acc;
    acc += blocks;
  }
  blockOffsets[K] = acc;
  const totalBlocks = acc;

  const blockVectors = new Int16Array(totalBlocks * 112);
  const blockOrigin = new Uint32Array(totalBlocks * BLOCK_SIZE);
  const blockLabels = new Uint8Array(totalBlocks * BLOCK_SIZE);

  // Initialize padding sentinels everywhere; valid slots will overwrite.
  blockVectors.fill(PAD_LANES_I16);
  blockOrigin.fill(PAD_ORIGIN_U32);
  blockLabels.fill(PAD_LABEL_U8);

  for (let c = 0; c < K; c += 1) {
    const refStart = perm.offsets[c];
    const refEnd = perm.offsets[c + 1];
    const size = refEnd - refStart;
    const baseBlock = blockOffsets[c];

    for (let i = 0; i < size; i += 1) {
      const block = baseBlock + Math.floor(i / BLOCK_SIZE);
      const slot = i % BLOCK_SIZE;
      const refIdx = refStart + i;
      const blockBase = block * 112;
      const refBase = refIdx * DIM;
      // SoA-by-block: dim d, slot s -> blockBase + d * 8 + s
      for (let d = 0; d < DIM; d += 1) {
        blockVectors[blockBase + d * BLOCK_SIZE + slot] = perm.permVectors[refBase + d];
      }
      blockOrigin[block * BLOCK_SIZE + slot] = perm.permOrigin[refIdx];
      blockLabels[block * BLOCK_SIZE + slot] = perm.permLabels[refIdx];
    }
  }

  return { blockOffsets, blockVectors, blockOrigin, blockLabels, totalBlocks };
}

function writeFile(outputPath, refs, centroids, K, soa) {
  const tmp = `${outputPath}.tmp`;
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  const header = Buffer.alloc(HEADER_BYTES);
  header.write(MAGIC_IVF, 0, 'ascii');
  header.writeUInt32LE(VERSION, 4);
  header.writeUInt32LE(K, 8);
  header.writeUInt32LE(DIM, 12);
  header.writeUInt32LE(refs.scale, 16);
  header.writeUInt32LE(refs.count, 20);
  header.writeUInt32LE(BLOCK_SIZE, 24);
  header.writeUInt32LE(soa.totalBlocks, 28);

  // labels_by_origin: derive from refs.labels (origin == raw position in refs.bin).
  const labelsByOrigin = Buffer.from(refs.labels.buffer, refs.labels.byteOffset, refs.count);

  const fd = fs.openSync(tmp, 'w');
  try {
    fs.writeSync(fd, header);
    fs.writeSync(fd, Buffer.from(centroids.buffer, centroids.byteOffset, centroids.byteLength));
    fs.writeSync(fd, Buffer.from(soa.blockOffsets.buffer, soa.blockOffsets.byteOffset, soa.blockOffsets.byteLength));
    fs.writeSync(fd, Buffer.from(soa.blockVectors.buffer, soa.blockVectors.byteOffset, soa.blockVectors.byteLength));
    fs.writeSync(fd, Buffer.from(soa.blockOrigin.buffer, soa.blockOrigin.byteOffset, soa.blockOrigin.byteLength));
    fs.writeSync(fd, soa.blockLabels);
    fs.writeSync(fd, labelsByOrigin);
  } finally {
    fs.closeSync(fd);
  }

  fs.renameSync(tmp, outputPath);
  return fs.statSync(outputPath).size;
}

function writeMeta(metaPath, meta) {
  if (!metaPath) return;
  fs.mkdirSync(path.dirname(metaPath), { recursive: true });
  fs.writeFileSync(`${metaPath}.tmp`, `${JSON.stringify(meta, null, 2)}\n`);
  fs.renameSync(`${metaPath}.tmp`, metaPath);
}

export function buildIvfBlocksIndex(rawOpts = {}) {
  const opts = {
    K: 1024,
    input: 'resources/references.bin',
    output: 'resources/references.ivf-blocks.bin',
    metaPath: 'resources/references.ivf-blocks.meta.json',
    centroidStrategy: 'kmeans',
    iterations: 25,
    seed: 12626,
    progressEvery: 250_000,
    ...rawOpts,
  };

  const t0 = Date.now();
  console.error(`reading ${opts.input} ...`);
  const refs = readReferences(opts.input);
  console.error(`refs: count=${refs.count} dim=${refs.dim} scale=${refs.scale}`);

  const K = opts.K;
  const centroids = chooseCentroids(refs, opts);

  console.error(`assigning ${refs.count} refs to nearest of ${K} centroids ...`);
  const { assignments, sizes } = assignClusters(refs, centroids, K, { progressEvery: opts.progressEvery });

  let minSize = Infinity, maxSize = 0;
  for (let c = 0; c < K; c += 1) {
    if (sizes[c] < minSize) minSize = sizes[c];
    if (sizes[c] > maxSize) maxSize = sizes[c];
  }
  console.error(`cluster sizes: min=${minSize} max=${maxSize} mean=${refs.count / K}`);

  console.error(`permuting refs by cluster ...`);
  const perm = buildPermutation(refs, assignments, sizes, K);

  console.error(`building block-SoA layout (block_size=${BLOCK_SIZE}) ...`);
  const soa = buildBlockSoA(refs, perm, K);
  console.error(`total_blocks=${soa.totalBlocks}, padded_slots=${soa.totalBlocks * BLOCK_SIZE - refs.count}`);

  console.error(`writing ${opts.output} ...`);
  const bytes = writeFile(opts.output, refs, centroids, K, soa);
  const seconds = ((Date.now() - t0) / 1000).toFixed(1);
  console.error(`done in ${seconds}s, ${bytes} bytes`);

  const meta = {
    format: MAGIC_IVF,
    version: VERSION,
    K,
    dim: DIM,
    scale: refs.scale,
    count: refs.count,
    block_size: BLOCK_SIZE,
    total_blocks: soa.totalBlocks,
    padded_slots: soa.totalBlocks * BLOCK_SIZE - refs.count,
    cluster_size_min: minSize,
    cluster_size_max: maxSize,
    cluster_size_mean: refs.count / K,
    centroid_strategy: opts.centroidStrategy,
    kmeans_iterations: opts.centroidStrategy === 'kmeans' ? opts.iterations : 0,
    seed: opts.centroidStrategy === 'first' ? null : opts.seed,
    bytes,
    output: opts.output,
    input: opts.input,
    generated_at: new Date().toISOString(),
  };
  writeMeta(opts.metaPath, meta);
  return meta;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const meta = buildIvfBlocksIndex(opts);
  console.log(JSON.stringify(meta, null, 2));
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => { console.error(err); process.exitCode = 1; });
}
