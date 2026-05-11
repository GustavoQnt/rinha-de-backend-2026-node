import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { buildReferencesBinFromRecords } from '../scripts/build-references-bin.js';
import { loadBinaryReferences, predictBinaryKnnBucket } from '../src/knn-classifier.js';

const tmpDir = path.join(process.cwd(), 'temporary-results', 'knn-classifier-test');

function vector(firstDim) {
  const v = new Array(14).fill(0);
  v[0] = firstDim;
  return v;
}

test('predictBinaryKnnBucket uses strict top-5 ties in original reference order', async () => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
  fs.mkdirSync(tmpDir, { recursive: true });

  const outputPath = path.join(tmpDir, 'references.bin');
  await buildReferencesBinFromRecords([
    { vector: vector(0), label: 'fraud' },
    { vector: vector(0), label: 'fraud' },
    { vector: vector(0), label: 'legit' },
    { vector: vector(0), label: 'fraud' },
    { vector: vector(0), label: 'legit' },
    { vector: vector(0), label: 'legit' },
  ], { outputPath, metaPath: null });

  const refs = loadBinaryReferences(outputPath);
  const result = predictBinaryKnnBucket(refs, vector(0));

  assert.equal(result.bucket, 3);
  assert.equal(result.fraudScore, 0.6);
  assert.deepEqual(result.indexes, [0, 1, 2, 3, 4]);
});
