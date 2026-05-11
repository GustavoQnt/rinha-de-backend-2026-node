/**
 * Describe an R26I IVF file (v1 or v2): cluster-size distribution + headline stats.
 *
 * Usage:
 *   node scripts/describe-ivf.js [--input resources/references.ivf.bin]
 *   node scripts/describe-ivf.js [--input resources/references.ivf-blocks.bin]
 */

import fs from 'node:fs';
import { pathToFileURL } from 'node:url';

const MAGIC = 'R26I';

function parseArgs(argv) {
  const opts = { input: 'resources/references.ivf.bin' };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '--input') opts.input = argv[++i];
    else throw new Error(`unknown argument: ${a}`);
  }
  return opts;
}

function describeV1(buf) {
  const K = buf.readUInt32LE(8);
  const dim = buf.readUInt32LE(12);
  const scale = buf.readUInt32LE(16);
  const count = buf.readUInt32LE(20);
  const pCentroids = 24;
  const pOffsets = pCentroids + K * dim * 2;
  const offsets = new Uint32Array(buf.buffer, buf.byteOffset + pOffsets, K + 1);
  const sizes = new Array(K);
  for (let c = 0; c < K; c += 1) sizes[c] = offsets[c + 1] - offsets[c];
  return { version: 1, K, dim, scale, count, sizes };
}

function describeV2(buf) {
  const K = buf.readUInt32LE(8);
  const dim = buf.readUInt32LE(12);
  const scale = buf.readUInt32LE(16);
  const count = buf.readUInt32LE(20);
  const blockSize = buf.readUInt32LE(24);
  const totalBlocks = buf.readUInt32LE(28);

  const pCentroids = 32;
  const pBlockOffsets = pCentroids + K * dim * 2;
  const pBlockVectors = pBlockOffsets + (K + 1) * 4;
  const pBlockOrigin = pBlockVectors + totalBlocks * 112 * 2;

  const blockOffsets = new Uint32Array(buf.buffer, buf.byteOffset + pBlockOffsets, K + 1);
  const blockOrigin = new Uint32Array(buf.buffer, buf.byteOffset + pBlockOrigin, totalBlocks * blockSize);

  // True per-cluster size = count of non-padded slots in [start_block, end_block).
  const sizes = new Array(K);
  for (let c = 0; c < K; c += 1) {
    const start = blockOffsets[c];
    const end = blockOffsets[c + 1];
    let n = 0;
    for (let b = start; b < end; b += 1) {
      const base = b * blockSize;
      for (let s = 0; s < blockSize; s += 1) {
        if (blockOrigin[base + s] !== 0xFFFFFFFF) n += 1;
      }
    }
    sizes[c] = n;
  }
  return { version: 2, K, dim, scale, count, totalBlocks, blockSize, sizes };
}

function summarize(sizes) {
  const sorted = [...sizes].sort((a, b) => a - b);
  const N = sorted.length;
  const p = (q) => sorted[Math.min(N - 1, Math.max(0, Math.round((N - 1) * q)))];
  const mean = sorted.reduce((acc, v) => acc + v, 0) / N;
  const variance = sorted.reduce((acc, v) => acc + (v - mean) ** 2, 0) / N;
  // Histogram in log-2 buckets capped at a reasonable upper edge.
  const buckets = [0, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600];
  const histogram = {};
  for (let i = 0; i < buckets.length; i += 1) {
    const lo = buckets[i];
    const hi = i + 1 < buckets.length ? buckets[i + 1] : Infinity;
    const label = hi === Infinity ? `${lo}+` : `${lo}-${hi - 1}`;
    histogram[label] = sizes.filter((v) => v >= lo && v < hi).length;
  }
  return {
    K: N,
    min: sorted[0],
    p1: p(0.01),
    p5: p(0.05),
    p25: p(0.25),
    p50: p(0.50),
    p75: p(0.75),
    p90: p(0.90),
    p95: p(0.95),
    p99: p(0.99),
    max: sorted[N - 1],
    mean: +mean.toFixed(2),
    stddev: +Math.sqrt(variance).toFixed(2),
    cv: +(Math.sqrt(variance) / mean).toFixed(4),
    histogram,
  };
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  const buf = fs.readFileSync(opts.input);
  if (buf.toString('ascii', 0, 4) !== MAGIC) throw new Error('bad magic');
  const version = buf.readUInt32LE(4);
  const desc = version === 1 ? describeV1(buf) : version === 2 ? describeV2(buf) : null;
  if (!desc) throw new Error(`unsupported version ${version}`);
  const stats = summarize(desc.sizes);
  const out = {
    file: opts.input,
    bytes: buf.length,
    version: desc.version,
    K: desc.K,
    dim: desc.dim,
    scale: desc.scale,
    count: desc.count,
    block_size: desc.blockSize,
    total_blocks: desc.totalBlocks,
    cluster_size: stats,
  };
  console.log(JSON.stringify(out, null, 2));
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  try { main(); } catch (e) { console.error(e); process.exitCode = 1; }
}
