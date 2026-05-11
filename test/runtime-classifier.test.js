import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { buildReferencesBinFromRecords } from '../scripts/build-references-bin.js';
import { createRuntimeClassifier } from '../src/runtime-classifier.js';

const tmpDir = path.join(process.cwd(), 'temporary-results', 'runtime-classifier-test');

function vector(firstDim) {
  const v = new Array(14).fill(0);
  v[0] = firstDim;
  return v;
}

test('createRuntimeClassifier selects binary k-NN from explicit options', async () => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
  fs.mkdirSync(tmpDir, { recursive: true });

  const referencesBinPath = path.join(tmpDir, 'references.bin');
  await buildReferencesBinFromRecords([
    { vector: vector(0), label: 'fraud' },
    { vector: vector(0), label: 'fraud' },
    { vector: vector(0), label: 'legit' },
    { vector: vector(0), label: 'fraud' },
    { vector: vector(0), label: 'legit' },
  ], { outputPath: referencesBinPath, metaPath: null });

  const runtime = createRuntimeClassifier({
    classifierName: 'knn-bin',
    referencesBinPath,
  });

  assert.equal(runtime.name, 'knn-bin');
  assert.equal(runtime.predictVectorBucket(vector(0)), 3);
});
