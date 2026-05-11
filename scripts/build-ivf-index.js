/**
 * Phase 4a/4b IVF index builder.
 *
 * Reads `resources/references.bin` (R26B v1) and emits `resources/references.ivf.bin`
 * (R26I v1). Phase 4a uses cheap centroids; Phase 4b can use seeded k-means.
 * After centroids are selected, refs are assigned to their nearest centroid and then
 * permuted so each cluster is contiguous.
 *
 * File format (R26I v1, all little-endian):
 *
 *   header (24 bytes):
 *     magic     "R26I"        4 bytes
 *     version   u32           = 1
 *     K         u32           number of centroids
 *     dim       u32           = 14
 *     scale     u32           = 10000
 *     count     u32           number of references
 *
 *   centroids:        K * dim * 2 bytes      (int16 LE)
 *   offsets:          (K+1) * 4 bytes        (uint32 LE) — permuted-array offsets per cluster
 *   permuted vectors: count * dim * 2 bytes  (int16 LE)
 *   permuted labels:  count * 1 byte         (0=legit, 1=fraud)
 *   permuted origin:  count * 4 bytes        (uint32 LE) — index into original references.bin
 *
 * Centroid choices:
 *   first   - first K refs, deterministic Phase 4a baseline
 *   sample  - seeded random sample, cheap Phase 4a alternative
 *   kmeans  - seeded Lloyd iterations, Phase 4b
 *
 * Usage:
 *   node scripts/build-ivf-index.js [--K 1024] [--centroids first|sample|kmeans]
 *                                   [--iterations 25] [--seed 12626]
 *                                   [--input resources/references.bin]
 *                                   [--output resources/references.ivf.bin]
 */

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const MAGIC_REFS = 'R26B';
const MAGIC_IVF = 'R26I';
const REFS_HEADER_BYTES = 24;
const IVF_HEADER_BYTES = 24;
const DIM = 14;
const VERSION = 1;

function parseArgs(argv) {
  const opts = {
    K: 1024,
    input: 'resources/references.bin',
    output: 'resources/references.ivf.bin',
    metaPath: 'resources/references.ivf.meta.json',
    centroidStrategy: 'first',
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
    else throw new Error(`unknown argument: ${a}`);
  }
  if (!Number.isInteger(opts.K) || opts.K <= 0 || opts.K > 65535) {
    throw new Error(`invalid --K: ${opts.K}`);
  }
  if (!['first', 'sample', 'kmeans', 'kmeans++'].includes(opts.centroidStrategy)) {
    throw new Error(`invalid --centroids: ${opts.centroidStrategy}`);
  }
  if (!Number.isInteger(opts.iterations) || opts.iterations < 0) {
    throw new Error(`invalid --iterations: ${opts.iterations}`);
  }
  if (!Number.isInteger(opts.seed)) {
    throw new Error(`invalid --seed: ${opts.seed}`);
  }
  return opts;
}

function readReferences(filePath) {
  const buf = fs.readFileSync(filePath);
  if (buf.length < REFS_HEADER_BYTES) throw new Error('references.bin too small');
  const magic = buf.toString('ascii', 0, 4);
  if (magic !== MAGIC_REFS) throw new Error(`bad magic: ${magic}`);
  const version = buf.readUInt32LE(4);
  if (version !== VERSION) throw new Error(`unsupported version: ${version}`);
  const count = buf.readUInt32LE(8);
  const dim = buf.readUInt32LE(12);
  const scale = buf.readUInt32LE(16);
  if (dim !== DIM) throw new Error(`unexpected dim: ${dim}`);
  const vectorsBytes = count * DIM * 2;
  const labelsStart = REFS_HEADER_BYTES + vectorsBytes;
  if (buf.length !== labelsStart + count) throw new Error('unexpected references.bin size');
  // Vectors as Int16Array (need a fresh underlying buffer with proper alignment).
  const vectors = new Int16Array(count * DIM);
  for (let i = 0; i < vectors.length; i += 1) {
    vectors[i] = buf.readInt16LE(REFS_HEADER_BYTES + i * 2);
  }
  const labels = Buffer.from(buf.buffer, buf.byteOffset + labelsStart, count);
  return { count, scale, dim, vectors, labels };
}

function pickCentroids(refs, K) {
  if (K > refs.count) throw new Error(`K (${K}) > count (${refs.count})`);
  // First K vectors. Deterministic. Plan permits this for Phase 4a.
  const centroids = new Int16Array(K * DIM);
  centroids.set(refs.vectors.subarray(0, K * DIM));
  return centroids;
}

function seededRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleIndexes(count, K, seed) {
  if (K > count) throw new Error(`K (${K}) > count (${count})`);
  const rng = seededRng(seed);
  const indexes = new Uint32Array(K);
  const seen = new Set();
  let i = 0;
  while (i < K) {
    const candidate = Math.floor(rng() * count);
    if (seen.has(candidate)) continue;
    seen.add(candidate);
    indexes[i] = candidate;
    i += 1;
  }
  return indexes;
}

/**
 * k-means++ initialization: D²-weighted seeding over a 50k random sample of refs.
 * Same shape as jairoblatt's `kmeans_plus_plus_init` (rinha-2026-rust/src/build_index.rs).
 *
 * Returns K centroids as Int16Array(K * DIM). Deterministic for a given seed.
 */
function kMeansPlusPlusCentroids(refs, K, seed, options = {}) {
  if (K > refs.count) throw new Error(`K (${K}) > count (${refs.count})`);
  const sampleSize = Math.min(refs.count, options.sampleSize ?? 50_000);
  const rng = seededRng(seed);

  // Random sample (with replacement, like the reference impl).
  const sample = new Uint32Array(sampleSize);
  for (let i = 0; i < sampleSize; i += 1) {
    sample[i] = Math.floor(rng() * refs.count);
  }

  const centroids = new Int16Array(K * DIM);
  const minDists = new Float64Array(sampleSize);
  for (let i = 0; i < sampleSize; i += 1) minDists[i] = Number.POSITIVE_INFINITY;

  // First centroid: random pick from the sample.
  const first = sample[Math.floor(rng() * sampleSize)];
  centroids.set(refs.vectors.subarray(first * DIM, (first + 1) * DIM), 0);

  const last = new Int32Array(DIM);
  for (let c = 1; c < K; c += 1) {
    const cOff = (c - 1) * DIM;
    for (let j = 0; j < DIM; j += 1) last[j] = centroids[cOff + j];

    // Update D²(sample[i]) = min(D²(sample[i]), dist²(sample[i], last centroid)).
    let total = 0;
    for (let i = 0; i < sampleSize; i += 1) {
      const off = sample[i] * DIM;
      let d = 0;
      for (let j = 0; j < DIM; j += 1) {
        const diff = refs.vectors[off + j] - last[j];
        d += diff * diff;
      }
      if (d < minDists[i]) minDists[i] = d;
      total += minDists[i];
    }

    // Sample next centroid with prob ∝ D².
    const r = rng() * total;
    let cum = 0;
    let chosen = sampleSize - 1;
    for (let i = 0; i < sampleSize; i += 1) {
      cum += minDists[i];
      if (cum >= r) {
        chosen = i;
        break;
      }
    }
    const idx = sample[chosen];
    centroids.set(refs.vectors.subarray(idx * DIM, (idx + 1) * DIM), c * DIM);
  }
  return centroids;
}

function sampleCentroids(refs, K, seed) {
  const indexes = sampleIndexes(refs.count, K, seed);
  const centroids = new Int16Array(K * DIM);
  for (let c = 0; c < K; c += 1) {
    const idx = indexes[c];
    centroids.set(refs.vectors.subarray(idx * DIM, (idx + 1) * DIM), c * DIM);
  }
  return centroids;
}

function nearestCentroid(query, centroids, K) {
  let bestIdx = 0;
  let bestDist = Number.POSITIVE_INFINITY;
  for (let c = 0; c < K; c += 1) {
    const off = c * DIM;
    let d = 0;
    for (let j = 0; j < DIM; j += 1) {
      const diff = query[j] - centroids[off + j];
      d += diff * diff;
      if (d >= bestDist) break;
    }
    if (d < bestDist) {
      bestDist = d;
      bestIdx = c;
    }
  }
  return bestIdx;
}

function assignClusters(refs, centroids, K, options = {}) {
  const progressEvery = options.progressEvery ?? 250_000;
  const t0 = Date.now();
  const assignments = new Uint16Array(refs.count); // K <= 65535
  const sizes = new Uint32Array(K);
  const q = new Int32Array(DIM);
  for (let i = 0; i < refs.count; i += 1) {
    const off = i * DIM;
    for (let j = 0; j < DIM; j += 1) q[j] = refs.vectors[off + j];
    const c = nearestCentroid(q, centroids, K);
    assignments[i] = c;
    sizes[c] += 1;
    if (progressEvery > 0 && (i + 1) % progressEvery === 0) {
      const seconds = ((Date.now() - t0) / 1000).toFixed(1);
      console.error(`assigned ${i + 1}/${refs.count} in ${seconds}s`);
    }
  }
  return { assignments, sizes };
}

function kMeansCentroids(refs, K, options = {}) {
  const iterations = options.iterations ?? 25;
  const seed = options.seed ?? 12626;
  const progressEvery = options.progressEvery ?? 250_000;
  const init = options.init ?? 'sample'; // 'sample' or 'kmeans++'
  // Early-stop ratio: if `changed / count < earlyStopRatio` after an iteration,
  // we converge and return. Set to null/0 to disable (default disabled to keep
  // existing builds bit-stable).
  const earlyStopRatio = options.earlyStopRatio ?? null;
  let centroids;
  if (init === 'kmeans++') {
    console.error(`kmeans++ init (sample=${Math.min(refs.count, options.sampleSize ?? 50_000)}) ...`);
    centroids = kMeansPlusPlusCentroids(refs, K, seed, { sampleSize: options.sampleSize });
  } else {
    centroids = sampleCentroids(refs, K, seed);
  }
  if (iterations === 0) return centroids;

  const q = new Int32Array(DIM);
  const sums = new Float64Array(K * DIM);
  const sizes = new Uint32Array(K);
  const prevAssign = earlyStopRatio ? new Uint16Array(refs.count).fill(65535) : null;
  const rngIndexes = sampleIndexes(refs.count, Math.min(refs.count, K + iterations), seed ^ 0xA5A5A5A5);
  let reseedCursor = 0;

  for (let iter = 0; iter < iterations; iter += 1) {
    sums.fill(0);
    sizes.fill(0);
    let changed = 0;
    const t0 = Date.now();

    for (let i = 0; i < refs.count; i += 1) {
      const off = i * DIM;
      for (let j = 0; j < DIM; j += 1) q[j] = refs.vectors[off + j];
      const c = nearestCentroid(q, centroids, K);
      if (prevAssign && prevAssign[i] !== c) {
        prevAssign[i] = c;
        changed += 1;
      }
      const sumOff = c * DIM;
      for (let j = 0; j < DIM; j += 1) sums[sumOff + j] += q[j];
      sizes[c] += 1;

      if (progressEvery > 0 && (i + 1) % progressEvery === 0) {
        const seconds = ((Date.now() - t0) / 1000).toFixed(1);
        console.error(`kmeans iter ${iter + 1}/${iterations}: assigned ${i + 1}/${refs.count} in ${seconds}s`);
      }
    }

    const next = new Int16Array(K * DIM);
    for (let c = 0; c < K; c += 1) {
      const centroidOff = c * DIM;
      if (sizes[c] === 0) {
        const idx = rngIndexes[reseedCursor % rngIndexes.length];
        reseedCursor += 1;
        next.set(refs.vectors.subarray(idx * DIM, (idx + 1) * DIM), centroidOff);
        continue;
      }

      for (let j = 0; j < DIM; j += 1) {
        next[centroidOff + j] = Math.round(sums[centroidOff + j] / sizes[c]);
      }
    }
    centroids = next;

    if (earlyStopRatio && iter > 0) {
      const ratio = changed / refs.count;
      console.error(`kmeans iter ${iter + 1}: changed=${changed} (${(ratio * 100).toFixed(3)}%)`);
      if (ratio < earlyStopRatio) {
        console.error(`early-stop: changed ratio ${ratio.toFixed(5)} < ${earlyStopRatio}`);
        break;
      }
    }
  }

  return centroids;
}

function buildPermutation(refs, assignments, sizes, K) {
  const offsets = new Uint32Array(K + 1);
  let acc = 0;
  for (let c = 0; c < K; c += 1) {
    offsets[c] = acc;
    acc += sizes[c];
  }
  offsets[K] = acc;
  if (acc !== refs.count) throw new Error('size accumulation mismatch');

  const cursor = new Uint32Array(K);
  const permVectors = new Int16Array(refs.count * DIM);
  const permLabels = Buffer.allocUnsafe(refs.count);
  const permOrigin = new Uint32Array(refs.count);

  for (let i = 0; i < refs.count; i += 1) {
    const c = assignments[i];
    const slot = offsets[c] + cursor[c];
    cursor[c] += 1;
    permVectors.set(refs.vectors.subarray(i * DIM, (i + 1) * DIM), slot * DIM);
    permLabels[slot] = refs.labels[i];
    permOrigin[slot] = i;
  }

  return { offsets, permVectors, permLabels, permOrigin };
}

function writeIvfFile(outputPath, refs, centroids, K, perm) {
  const tmp = `${outputPath}.tmp`;
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  const header = Buffer.alloc(IVF_HEADER_BYTES);
  header.write(MAGIC_IVF, 0, 'ascii');
  header.writeUInt32LE(VERSION, 4);
  header.writeUInt32LE(K, 8);
  header.writeUInt32LE(DIM, 12);
  header.writeUInt32LE(refs.scale, 16);
  header.writeUInt32LE(refs.count, 20);

  const fd = fs.openSync(tmp, 'w');
  try {
    fs.writeSync(fd, header);
    fs.writeSync(fd, Buffer.from(centroids.buffer, centroids.byteOffset, centroids.byteLength));
    fs.writeSync(fd, Buffer.from(perm.offsets.buffer, perm.offsets.byteOffset, perm.offsets.byteLength));
    fs.writeSync(fd, Buffer.from(perm.permVectors.buffer, perm.permVectors.byteOffset, perm.permVectors.byteLength));
    fs.writeSync(fd, perm.permLabels);
    fs.writeSync(fd, Buffer.from(perm.permOrigin.buffer, perm.permOrigin.byteOffset, perm.permOrigin.byteLength));
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
    console.error(`picking ${K} centroids (k-means with kmeans++ init, iterations=${opts.iterations}, seed=${opts.seed})`);
    return kMeansCentroids(refs, K, {
      iterations: opts.iterations,
      seed: opts.seed,
      progressEvery: opts.progressEvery,
      init: 'kmeans++',
    });
  }
  console.error(`picking ${K} centroids (k-means, iterations=${opts.iterations}, seed=${opts.seed})`);
  return kMeansCentroids(refs, K, {
    iterations: opts.iterations,
    seed: opts.seed,
    progressEvery: opts.progressEvery,
  });
}

function buildIvfIndex(rawOpts = {}) {
  const opts = {
    K: 1024,
    input: 'resources/references.bin',
    output: 'resources/references.ivf.bin',
    metaPath: 'resources/references.ivf.meta.json',
    centroidStrategy: 'first',
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

  // Sanity: cluster size stats (helps catch degenerate centroids early).
  let minSize = Infinity, maxSize = 0;
  for (let c = 0; c < K; c += 1) {
    if (sizes[c] < minSize) minSize = sizes[c];
    if (sizes[c] > maxSize) maxSize = sizes[c];
  }
  console.error(`cluster sizes: min=${minSize} max=${maxSize} mean=${refs.count / K}`);

  console.error(`permuting refs by cluster ...`);
  const perm = buildPermutation(refs, assignments, sizes, K);

  console.error(`writing ${opts.output} ...`);
  const bytes = writeIvfFile(opts.output, refs, centroids, K, perm);
  const seconds = ((Date.now() - t0) / 1000).toFixed(1);
  console.error(`done in ${seconds}s, ${bytes} bytes`);

  const meta = {
    format: MAGIC_IVF,
    version: VERSION,
    K,
    dim: DIM,
    scale: refs.scale,
    count: refs.count,
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
  const meta = buildIvfIndex(opts);
  console.log(JSON.stringify(meta, null, 2));
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}

export {
  readReferences,
  pickCentroids,
  sampleCentroids,
  kMeansPlusPlusCentroids,
  kMeansCentroids,
  assignClusters,
  buildPermutation,
  buildIvfIndex,
};
