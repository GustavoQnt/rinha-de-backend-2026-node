import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import {
  buildGridIndex,
  candidateIndexesForVector,
  loadGridIndex,
  saveGridIndex,
  vectorToGridKey,
} from '../src/grid-index.js';

const tmpDir = path.join(process.cwd(), 'temporary-results', 'grid-index-test');

function qVector(values = {}) {
  const v = new Int16Array(14);
  for (const [dim, value] of Object.entries(values)) v[Number(dim)] = value;
  return v;
}

test('vectorToGridKey separates null-history sentinels from normalized values', () => {
  const withSentinel = qVector({ 5: -10000, 6: -10000, 7: 2500 });
  const withZero = qVector({ 5: 0, 6: 0, 7: 2500 });

  assert.notEqual(vectorToGridKey(withSentinel, 10000), vectorToGridKey(withZero, 10000));
  assert.equal(vectorToGridKey(withSentinel, 10000), vectorToGridKey(withSentinel, 10000));
});

test('candidateIndexesForVector returns indexed references in original order', () => {
  const vectors = new Int16Array(4 * 14);
  vectors.set(qVector({ 7: 1000 }), 0);
  vectors.set(qVector({ 7: 1000 }), 14);
  vectors.set(qVector({ 7: 9000 }), 28);
  vectors.set(qVector({ 7: 1000 }), 42);

  const refs = { count: 4, dim: 14, scale: 10000, vectors };
  const index = buildGridIndex(refs);
  const candidates = candidateIndexesForVector(index, qVector({ 7: 1000 }), {
    minCandidates: 1,
    maxRadius: 0,
  });

  assert.deepEqual(candidates.indexes, [0, 1, 3]);
  assert.equal(candidates.radius, 0);
});

test('grid index can be built with a custom feature set', () => {
  const vectors = new Int16Array(3 * 14);
  vectors.set(qVector({ 0: 1000 }), 0);
  vectors.set(qVector({ 0: 9000 }), 14);
  vectors.set(qVector({ 0: 1000 }), 28);

  const refs = { count: 3, dim: 14, scale: 10000, vectors };
  const features = [{ dim: 0, bins: 2 }];
  const index = buildGridIndex(refs, features);
  const candidates = candidateIndexesForVector(index, qVector({ 0: 1000 }), {
    minCandidates: 1,
    maxRadius: 0,
  });

  assert.deepEqual(candidates.indexes, [0, 2]);
});

test('multi-grid candidates are deduped and sorted by reference order', async () => {
  const { buildMultiGridIndex, candidateIndexesForMultiGrid } = await import('../src/multi-grid-index.js');
  const vectors = new Int16Array(4 * 14);
  vectors.set(qVector({ 0: 1000, 7: 9000 }), 0);
  vectors.set(qVector({ 0: 9000, 7: 1000 }), 14);
  vectors.set(qVector({ 0: 1000, 7: 1000 }), 28);
  vectors.set(qVector({ 0: 9000, 7: 9000 }), 42);

  const refs = { count: 4, dim: 14, scale: 10000, vectors };
  const index = buildMultiGridIndex(refs, [
    [{ dim: 0, bins: 2 }],
    [{ dim: 7, bins: 2 }],
  ]);
  const candidates = candidateIndexesForMultiGrid(index, qVector({ 0: 1000, 7: 1000 }), {
    minCandidates: 1,
    maxRadius: 0,
  });

  assert.deepEqual(candidates.indexes, [0, 1, 2]);
  assert.equal(candidates.gridResults.length, 2);
  assert.equal(candidates.totalBeforeDedupe, 4);
});

test('grid index binary save/load preserves candidate lookup', () => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
  fs.mkdirSync(tmpDir, { recursive: true });

  const vectors = new Int16Array(3 * 14);
  vectors.set(qVector({ 8: 1000, 12: 5000 }), 0);
  vectors.set(qVector({ 8: 1000, 12: 5000 }), 14);
  vectors.set(qVector({ 8: 9000, 12: 5000 }), 28);

  const refs = { count: 3, dim: 14, scale: 10000, vectors };
  const outputPath = path.join(tmpDir, 'references.grid.bin');
  const original = buildGridIndex(refs);

  saveGridIndex(original, outputPath);
  const loaded = loadGridIndex(outputPath);
  const candidates = candidateIndexesForVector(loaded, qVector({ 8: 1000, 12: 5000 }), {
    minCandidates: 1,
    maxRadius: 0,
  });

  assert.equal(loaded.count, 3);
  assert.equal(loaded.bucketCount, original.bucketCount);
  assert.deepEqual(candidates.indexes, [0, 1]);
});
