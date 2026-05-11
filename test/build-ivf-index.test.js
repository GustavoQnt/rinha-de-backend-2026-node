import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  buildIvfIndex,
  kMeansCentroids,
  kMeansPlusPlusCentroids,
  readReferences,
} from '../scripts/build-ivf-index.js';

const DIM = 14;

function writeSyntheticReferences(filePath, vectors, labels, scale = 10000) {
  const count = vectors.length;
  const headerBytes = 24;
  const buf = Buffer.alloc(headerBytes + count * DIM * 2 + count);
  buf.write('R26B', 0, 'ascii');
  buf.writeUInt32LE(1, 4);
  buf.writeUInt32LE(count, 8);
  buf.writeUInt32LE(DIM, 12);
  buf.writeUInt32LE(scale, 16);

  let pos = headerBytes;
  for (const vector of vectors) {
    assert.equal(vector.length, DIM);
    for (const value of vector) {
      buf.writeInt16LE(value, pos);
      pos += 2;
    }
  }
  for (let i = 0; i < labels.length; i += 1) {
    buf[pos + i] = labels[i];
  }
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, buf);
}

function vector(firstDim) {
  const v = new Array(DIM).fill(0);
  v[0] = firstDim;
  return v;
}

test('kMeansCentroids is deterministic for a fixed seed', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ivf-kmeans-'));
  const input = path.join(tmp, 'references.bin');
  writeSyntheticReferences(input, [
    vector(-1000),
    vector(-900),
    vector(-800),
    vector(900),
    vector(1000),
    vector(1100),
  ], [0, 0, 0, 1, 1, 1]);

  const refs = readReferences(input);
  const a = kMeansCentroids(refs, 2, { iterations: 6, seed: 42 });
  const b = kMeansCentroids(refs, 2, { iterations: 6, seed: 42 });

  assert.deepEqual(Array.from(a), Array.from(b));
  assert.deepEqual(
    [a[0], a[DIM]].sort((x, y) => x - y),
    [-900, 1000],
  );
});

test('kMeansPlusPlusCentroids is deterministic and separates well-separated clusters', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ivf-kmeanspp-'));
  const input = path.join(tmp, 'references.bin');
  // Two well-separated clusters at -1000 and +1000.
  writeSyntheticReferences(input, [
    vector(-1010), vector(-1000), vector(-990), vector(-980), vector(-970),
    vector(970), vector(980), vector(990), vector(1000), vector(1010),
  ], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

  const refs = readReferences(input);
  const a = kMeansPlusPlusCentroids(refs, 2, 42, { sampleSize: 8 });
  const b = kMeansPlusPlusCentroids(refs, 2, 42, { sampleSize: 8 });
  assert.deepEqual(Array.from(a), Array.from(b), 'deterministic for fixed seed');

  // First-dim values must straddle 0 (one cluster around -1000, one around +1000).
  const c0 = a[0];
  const c1 = a[DIM];
  assert.ok((c0 < 0 && c1 > 0) || (c0 > 0 && c1 < 0),
    `kmeans++ should pick one centroid per region, got [${c0}, ${c1}]`);
});

test('buildIvfIndex writes k-means metadata and a valid R26I header', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ivf-build-'));
  const input = path.join(tmp, 'references.bin');
  const output = path.join(tmp, 'references.ivf.bin');
  const metaPath = path.join(tmp, 'references.ivf.meta.json');
  writeSyntheticReferences(input, [
    vector(-1000),
    vector(-900),
    vector(-800),
    vector(900),
    vector(1000),
    vector(1100),
  ], [0, 0, 0, 1, 1, 1]);

  const meta = buildIvfIndex({
    K: 2,
    input,
    output,
    metaPath,
    centroidStrategy: 'kmeans',
    iterations: 6,
    seed: 42,
    progressEvery: 0,
  });

  const bytes = fs.readFileSync(output);
  assert.equal(bytes.toString('ascii', 0, 4), 'R26I');
  assert.equal(bytes.readUInt32LE(4), 1);
  assert.equal(bytes.readUInt32LE(8), 2);
  assert.equal(bytes.readUInt32LE(12), DIM);
  assert.equal(bytes.readUInt32LE(20), 6);

  assert.equal(meta.centroid_strategy, 'kmeans');
  assert.equal(meta.kmeans_iterations, 6);
  assert.equal(meta.seed, 42);
  assert.deepEqual(JSON.parse(fs.readFileSync(metaPath, 'utf8')), meta);
});
