import fs from 'node:fs';
import path from 'node:path';

const MAGIC = 'R26G';
const VERSION = 1;
const HEADER_BYTES = 24;
const BUCKET_ENTRY_UINT32S = 3;
const BUCKET_ENTRY_BYTES = BUCKET_ENTRY_UINT32S * 4;

export const GRID_FEATURES = [
  { dim: 5, bins: 4, sentinel: true },
  { dim: 6, bins: 4, sentinel: true },
  { dim: 7, bins: 8 },
  { dim: 8, bins: 8 },
  { dim: 9, bins: 2, fixed: true },
  { dim: 10, bins: 2, fixed: true },
  { dim: 11, bins: 2, fixed: true },
  { dim: 12, bins: 4 },
  { dim: 13, bins: 4 },
];

function featureRadix(feature) {
  return feature.sentinel ? feature.bins + 1 : feature.bins;
}

function quantizedToBin(value, scale, feature) {
  if (feature.sentinel && value < 0) return 0;

  const clamped = value < 0 ? 0 : value > scale ? scale : value;
  const bin = Math.min(feature.bins - 1, Math.floor((clamped / scale) * feature.bins));
  return feature.sentinel ? bin + 1 : bin;
}

function binsToKey(bins, features) {
  let key = 0;
  let multiplier = 1;

  for (let i = 0; i < features.length; i += 1) {
    key += bins[i] * multiplier;
    multiplier *= featureRadix(features[i]);
  }

  return key >>> 0;
}

export function vectorToGridBins(vector, scale, features = GRID_FEATURES) {
  const bins = new Array(features.length);
  for (let i = 0; i < features.length; i += 1) {
    const feature = features[i];
    bins[i] = quantizedToBin(vector[feature.dim], scale, feature);
  }
  return bins;
}

export function vectorToGridKey(vector, scale, features = GRID_FEATURES) {
  return binsToKey(vectorToGridBins(vector, scale, features), features);
}

export function buildGridIndex(refs, features = GRID_FEATURES) {
  const buckets = new Map();

  for (let i = 0; i < refs.count; i += 1) {
    const base = i * refs.dim;
    const key = vectorToGridKey(refs.vectors.subarray(base, base + refs.dim), refs.scale, features);
    const bucket = buckets.get(key);
    if (bucket === undefined) buckets.set(key, [i]);
    else bucket.push(i);
  }

  const keys = Array.from(buckets.keys()).sort((a, b) => a - b);
  const bucketTable = new Uint32Array(keys.length * BUCKET_ENTRY_UINT32S);
  const postings = new Uint32Array(refs.count);
  let offset = 0;

  for (let i = 0; i < keys.length; i += 1) {
    const key = keys[i];
    const indexes = buckets.get(key);
    bucketTable[i * 3] = key;
    bucketTable[i * 3 + 1] = offset;
    bucketTable[i * 3 + 2] = indexes.length;
    postings.set(indexes, offset);
    offset += indexes.length;
  }

  return {
    magic: MAGIC,
    version: VERSION,
    count: refs.count,
    scale: refs.scale,
    features,
    bucketCount: keys.length,
    postingCount: refs.count,
    bucketTable,
    postings,
  };
}

function findBucket(index, key) {
  let lo = 0;
  let hi = index.bucketCount - 1;

  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const current = index.bucketTable[mid * 3];
    if (current === key) return mid;
    if (current < key) lo = mid + 1;
    else hi = mid - 1;
  }

  return -1;
}

function collectKey(index, key, out) {
  const bucketIndex = findBucket(index, key);
  if (bucketIndex < 0) return;

  const start = index.bucketTable[bucketIndex * 3 + 1];
  const length = index.bucketTable[bucketIndex * 3 + 2];
  for (let i = 0; i < length; i += 1) out.push(index.postings[start + i]);
}

function generateProbeKeys(bins, features, radius, visit) {
  const current = bins.slice();

  function walk(pos, remaining) {
    if (pos === features.length) {
      visit(binsToKey(current, features));
      return;
    }

    const feature = features[pos];
    const original = bins[pos];
    const radix = featureRadix(feature);
    const fixed = feature.fixed || (feature.sentinel && original === 0);

    if (fixed) {
      current[pos] = original;
      walk(pos + 1, remaining);
      return;
    }

    const min = Math.max(0, original - remaining);
    const max = Math.min(radix - 1, original + remaining);
    for (let value = min; value <= max; value += 1) {
      const cost = Math.abs(value - original);
      if (cost <= remaining) {
        current[pos] = value;
        walk(pos + 1, remaining - cost);
      }
    }
  }

  walk(0, radius);
}

export function candidateIndexesForVector(index, vector, options = {}) {
  const minCandidates = options.minCandidates ?? 10_000;
  const maxRadius = options.maxRadius ?? 1;
  const features = index.features ?? GRID_FEATURES;
  const bins = vectorToGridBins(vector, index.scale, features);
  const indexes = [];
  const seenKeys = new Set();
  let usedRadius = 0;

  for (let radius = 0; radius <= maxRadius; radius += 1) {
    usedRadius = radius;
    generateProbeKeys(bins, features, radius, (key) => {
      if (seenKeys.has(key)) return;
      seenKeys.add(key);
      collectKey(index, key, indexes);
    });

    if (indexes.length >= minCandidates) break;
  }

  return {
    indexes,
    radius: usedRadius,
    keysVisited: seenKeys.size,
  };
}

export function saveGridIndex(index, outputPath) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  const bytes = HEADER_BYTES + index.bucketTable.length * 4 + index.postings.length * 4;
  const buffer = Buffer.allocUnsafe(bytes);

  buffer.write(MAGIC, 0, 'ascii');
  buffer.writeUInt32LE(VERSION, 4);
  buffer.writeUInt32LE(index.count, 8);
  buffer.writeUInt32LE(index.bucketCount, 12);
  buffer.writeUInt32LE(index.postingCount, 16);
  buffer.writeUInt32LE(index.scale, 20);

  Buffer.from(index.bucketTable.buffer, index.bucketTable.byteOffset, index.bucketTable.byteLength)
    .copy(buffer, HEADER_BYTES);
  Buffer.from(index.postings.buffer, index.postings.byteOffset, index.postings.byteLength)
    .copy(buffer, HEADER_BYTES + index.bucketTable.byteLength);

  fs.writeFileSync(`${outputPath}.tmp`, buffer);
  fs.renameSync(`${outputPath}.tmp`, outputPath);
}

export function loadGridIndex(filePath) {
  const bytes = fs.readFileSync(filePath);
  const magic = bytes.toString('ascii', 0, 4);
  const version = bytes.readUInt32LE(4);
  const count = bytes.readUInt32LE(8);
  const bucketCount = bytes.readUInt32LE(12);
  const postingCount = bytes.readUInt32LE(16);
  const scale = bytes.readUInt32LE(20);

  if (magic !== MAGIC) throw new Error(`invalid grid index magic: ${magic}`);
  if (version !== VERSION) throw new Error(`unsupported grid index version: ${version}`);

  const bucketBytes = bucketCount * BUCKET_ENTRY_BYTES;
  const expectedBytes = HEADER_BYTES + bucketBytes + postingCount * 4;
  if (bytes.length !== expectedBytes) {
    throw new Error(`invalid grid index size: got ${bytes.length}, expected ${expectedBytes}`);
  }

  return {
    bytes,
    magic,
    version,
    count,
    scale,
    features: GRID_FEATURES,
    bucketCount,
    postingCount,
    bucketTable: new Uint32Array(bytes.buffer, bytes.byteOffset + HEADER_BYTES, bucketCount * BUCKET_ENTRY_UINT32S),
    postings: new Uint32Array(bytes.buffer, bytes.byteOffset + HEADER_BYTES + bucketBytes, postingCount),
  };
}
