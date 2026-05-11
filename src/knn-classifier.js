import fs from 'node:fs';

const MAGIC = 'R26B';
const VERSION = 1;
const DIM = 14;
const HEADER_BYTES = 24;
const DEFAULT_REFERENCES_BIN = 'resources/references.bin';

function quantizeValue(value, scale) {
  const q = Math.round(value * scale);
  if (q < -32768 || q > 32767) {
    throw new RangeError(`quantized value out of int16 range: ${q}`);
  }
  return q;
}

function readHeader(bytes) {
  return {
    magic: bytes.toString('ascii', 0, 4),
    version: bytes.readUInt32LE(4),
    count: bytes.readUInt32LE(8),
    dim: bytes.readUInt32LE(12),
    scale: bytes.readUInt32LE(16),
    reserved: bytes.readUInt32LE(20),
  };
}

function quantizeQuery(query, scale) {
  const out = new Int16Array(DIM);
  for (let i = 0; i < DIM; i += 1) out[i] = quantizeValue(query[i], scale);
  return out;
}

export function loadBinaryReferences(filePath = DEFAULT_REFERENCES_BIN) {
  const bytes = fs.readFileSync(filePath);
  const header = readHeader(bytes);

  if (header.magic !== MAGIC) throw new Error(`invalid references magic: ${header.magic}`);
  if (header.version !== VERSION) throw new Error(`unsupported references version: ${header.version}`);
  if (header.dim !== DIM) throw new Error(`unsupported references dim: ${header.dim}`);

  const expectedBytes = HEADER_BYTES + header.count * header.dim * 2 + header.count;
  if (bytes.length !== expectedBytes) {
    throw new Error(`invalid references size: got ${bytes.length}, expected ${expectedBytes}`);
  }

  const vectorOffset = bytes.byteOffset + HEADER_BYTES;
  const labelsOffset = HEADER_BYTES + header.count * header.dim * 2;

  return {
    bytes,
    count: header.count,
    dim: header.dim,
    scale: header.scale,
    vectors: new Int16Array(bytes.buffer, vectorOffset, header.count * header.dim),
    labels: new Uint8Array(bytes.buffer, bytes.byteOffset + labelsOffset, header.count),
  };
}

export function predictBinaryKnnBucket(refs, query) {
  const q = quantizeQuery(query, refs.scale);
  const distances = [Infinity, Infinity, Infinity, Infinity, Infinity];
  const indexes = [-1, -1, -1, -1, -1];

  for (let i = 0; i < refs.count; i += 1) {
    const base = i * DIM;
    let d = 0;
    for (let j = 0; j < DIM; j += 1) {
      const diff = q[j] - refs.vectors[base + j];
      d += diff * diff;
    }

    if (d < distances[4]) {
      let pos = 4;
      while (pos > 0 && d < distances[pos - 1]) {
        distances[pos] = distances[pos - 1];
        indexes[pos] = indexes[pos - 1];
        pos -= 1;
      }
      distances[pos] = d;
      indexes[pos] = i;
    }
  }

  let bucket = 0;
  for (let i = 0; i < 5; i += 1) {
    if (refs.labels[indexes[i]] === 1) bucket += 1;
  }

  return {
    bucket,
    fraudScore: bucket / 5,
    approved: bucket < 3,
    distances,
    indexes,
  };
}

export function createBinaryKnnClassifier(filePath = DEFAULT_REFERENCES_BIN, vectorize) {
  const refs = loadBinaryReferences(filePath);

  return {
    refs,
    classifyFraudBucket(payload) {
      if (vectorize === undefined) {
        throw new Error('binary k-NN classifier requires a vectorize function');
      }
      return predictBinaryKnnBucket(refs, vectorize(payload)).bucket;
    },
  };
}
