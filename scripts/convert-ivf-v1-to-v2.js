/**
 * Convert R26I v1 (row-major IVF) → R26I v2 (block-SoA-of-8) without rebuilding.
 *
 * Reads `resources/references.ivf.bin` and emits `resources/references.ivf-blocks.bin`
 * with identical centroids/assignments — only the per-cluster vector layout is
 * transformed from row-major to SoA-by-block (8 lanes × 14 dims per block).
 *
 * This skips the expensive k-means rerun: parity to the v1 production index is
 * guaranteed by construction.
 *
 * Usage:
 *   node scripts/convert-ivf-v1-to-v2.js
 *     [--input resources/references.ivf.bin]
 *     [--refs resources/references.bin]
 *     [--output resources/references.ivf-blocks.bin]
 */

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const MAGIC = 'R26I';
const VERSION_V1 = 1;
const VERSION_V2 = 2;
const V1_HEADER_BYTES = 24;
const V2_HEADER_BYTES = 32;
const DIM = 14;
const BLOCK_SIZE = 8;
const PAD_LANES_I16 = 0x7FFF;        // i16::MAX
const PAD_ORIGIN_U32 = 0xFFFFFFFF;   // u32::MAX
const PAD_LABEL_U8 = 0xFF;

function parseArgs(argv) {
  const opts = {
    input: 'resources/references.ivf.bin',
    refs: 'resources/references.bin',
    output: 'resources/references.ivf-blocks.bin',
    metaPath: 'resources/references.ivf-blocks.meta.json',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '--input') opts.input = argv[++i];
    else if (a === '--refs') opts.refs = argv[++i];
    else if (a === '--output') opts.output = argv[++i];
    else if (a === '--meta') opts.metaPath = argv[++i];
    else if (a === '--no-meta') opts.metaPath = null;
    else throw new Error(`unknown argument: ${a}`);
  }
  return opts;
}

function readV1(path) {
  const buf = fs.readFileSync(path);
  if (buf.length < V1_HEADER_BYTES) throw new Error('v1 file too small');
  if (buf.toString('ascii', 0, 4) !== MAGIC) throw new Error('bad magic');
  const version = buf.readUInt32LE(4);
  if (version !== VERSION_V1) throw new Error(`expected v1, got v${version}`);
  const K = buf.readUInt32LE(8);
  const dim = buf.readUInt32LE(12);
  const scale = buf.readUInt32LE(16);
  const count = buf.readUInt32LE(20);
  if (dim !== DIM) throw new Error(`unexpected dim ${dim}`);

  const pCentroids = V1_HEADER_BYTES;
  const pOffsets = pCentroids + K * DIM * 2;
  const pVectors = pOffsets + (K + 1) * 4;
  const pLabels = pVectors + count * DIM * 2;
  const pOrigin = pLabels + count;
  const pEnd = pOrigin + count * 4;
  if (buf.length !== pEnd) throw new Error(`unexpected v1 file size ${buf.length} vs ${pEnd}`);

  // Zero-copy typed array views over the file buffer.
  const centroids = new Int16Array(buf.buffer, buf.byteOffset + pCentroids, K * DIM);
  const offsets = new Uint32Array(buf.buffer, buf.byteOffset + pOffsets, K + 1);
  const vectors = new Int16Array(buf.buffer, buf.byteOffset + pVectors, count * DIM);
  const labels = Buffer.from(buf.buffer, buf.byteOffset + pLabels, count);
  const origin = new Uint32Array(buf.buffer, buf.byteOffset + pOrigin, count);

  return { K, dim, scale, count, centroids, offsets, vectors, labels, origin };
}

function readRefsLabels(path) {
  // R26B v1: 24-byte header + count*DIM*2 vectors + count labels.
  const buf = fs.readFileSync(path);
  if (buf.toString('ascii', 0, 4) !== 'R26B') throw new Error('bad refs magic');
  const count = buf.readUInt32LE(8);
  const labelsStart = 24 + count * DIM * 2;
  if (buf.length < labelsStart + count) throw new Error('refs file too small');
  return Buffer.from(buf.buffer, buf.byteOffset + labelsStart, count);
}

function buildBlockSoA(v1) {
  const { K, count, offsets, vectors, labels: permLabels, origin: permOrigin } = v1;
  // Compute total blocks and per-cluster block offsets.
  const blockOffsets = new Uint32Array(K + 1);
  let acc = 0;
  for (let c = 0; c < K; c += 1) {
    const size = offsets[c + 1] - offsets[c];
    blockOffsets[c] = acc;
    acc += Math.ceil(size / BLOCK_SIZE);
  }
  blockOffsets[K] = acc;
  const totalBlocks = acc;

  const blockVectors = new Int16Array(totalBlocks * 112);
  const blockOrigin = new Uint32Array(totalBlocks * BLOCK_SIZE);
  const blockLabels = new Uint8Array(totalBlocks * BLOCK_SIZE);
  blockVectors.fill(PAD_LANES_I16);
  blockOrigin.fill(PAD_ORIGIN_U32);
  blockLabels.fill(PAD_LABEL_U8);

  for (let c = 0; c < K; c += 1) {
    const refStart = offsets[c];
    const refEnd = offsets[c + 1];
    const size = refEnd - refStart;
    const baseBlock = blockOffsets[c];
    for (let i = 0; i < size; i += 1) {
      const block = baseBlock + Math.floor(i / BLOCK_SIZE);
      const slot = i % BLOCK_SIZE;
      const refIdx = refStart + i;
      const blockBase = block * 112;
      const refBase = refIdx * DIM;
      for (let d = 0; d < DIM; d += 1) {
        blockVectors[blockBase + d * BLOCK_SIZE + slot] = vectors[refBase + d];
      }
      blockOrigin[block * BLOCK_SIZE + slot] = permOrigin[refIdx];
      blockLabels[block * BLOCK_SIZE + slot] = permLabels[refIdx];
    }
  }

  return { blockOffsets, blockVectors, blockOrigin, blockLabels, totalBlocks };
}

function writeV2(outputPath, v1, soa, labelsByOrigin) {
  const tmp = `${outputPath}.tmp`;
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  const header = Buffer.alloc(V2_HEADER_BYTES);
  header.write(MAGIC, 0, 'ascii');
  header.writeUInt32LE(VERSION_V2, 4);
  header.writeUInt32LE(v1.K, 8);
  header.writeUInt32LE(DIM, 12);
  header.writeUInt32LE(v1.scale, 16);
  header.writeUInt32LE(v1.count, 20);
  header.writeUInt32LE(BLOCK_SIZE, 24);
  header.writeUInt32LE(soa.totalBlocks, 28);

  const fd = fs.openSync(tmp, 'w');
  try {
    fs.writeSync(fd, header);
    fs.writeSync(fd, Buffer.from(v1.centroids.buffer, v1.centroids.byteOffset, v1.centroids.byteLength));
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

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const t0 = Date.now();
  console.error(`reading v1 ${opts.input} ...`);
  const v1 = readV1(opts.input);
  console.error(`v1: K=${v1.K} count=${v1.count} scale=${v1.scale}`);

  console.error(`reading refs labels from ${opts.refs} ...`);
  const labelsByOrigin = readRefsLabels(opts.refs);
  if (labelsByOrigin.length !== v1.count) {
    throw new Error(`refs count mismatch: ${labelsByOrigin.length} vs ${v1.count}`);
  }

  console.error(`building block-SoA layout ...`);
  const soa = buildBlockSoA(v1);
  console.error(`total_blocks=${soa.totalBlocks} padded_slots=${soa.totalBlocks * BLOCK_SIZE - v1.count}`);

  console.error(`writing ${opts.output} ...`);
  const bytes = writeV2(opts.output, v1, soa, labelsByOrigin);
  const seconds = ((Date.now() - t0) / 1000).toFixed(1);
  console.error(`done in ${seconds}s, ${bytes} bytes`);

  const meta = {
    format: MAGIC,
    version: VERSION_V2,
    K: v1.K,
    dim: DIM,
    scale: v1.scale,
    count: v1.count,
    block_size: BLOCK_SIZE,
    total_blocks: soa.totalBlocks,
    padded_slots: soa.totalBlocks * BLOCK_SIZE - v1.count,
    derived_from: opts.input,
    bytes,
    output: opts.output,
    generated_at: new Date().toISOString(),
  };
  if (opts.metaPath) {
    fs.writeFileSync(`${opts.metaPath}.tmp`, `${JSON.stringify(meta, null, 2)}\n`);
    fs.renameSync(`${opts.metaPath}.tmp`, opts.metaPath);
  }
  console.log(JSON.stringify(meta, null, 2));
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => { console.error(err); process.exitCode = 1; });
}
