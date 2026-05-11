import assert from 'node:assert/strict';
import test from 'node:test';
import { loadNativeKnn } from '../src/native-knn.js';

function vector(firstDim) {
  const v = new Int16Array(14);
  v[0] = firstDim;
  return v;
}

test('native knnBucket preserves strict top-5 ties in original order', () => {
  const native = loadNativeKnn();
  const refs = new Int16Array(6 * 14);
  refs.set(vector(0), 0);
  refs.set(vector(0), 14);
  refs.set(vector(0), 28);
  refs.set(vector(0), 42);
  refs.set(vector(0), 56);
  refs.set(vector(0), 70);

  const labels = new Uint8Array([1, 1, 0, 1, 0, 0]);

  assert.equal(native.distance14(vector(0), refs, 0), 0);
  assert.equal(native.knnBucket(vector(0), refs, labels, 6), 3);
});

test('native knnBucketCandidates searches only the provided candidates', () => {
  const native = loadNativeKnn();
  const refs = new Int16Array(8 * 14);
  refs.set(vector(0), 0);
  refs.set(vector(0), 14);
  refs.set(vector(0), 28);
  refs.set(vector(0), 42);
  refs.set(vector(0), 56);
  refs.set(vector(0), 70);
  refs.set(vector(9000), 84);
  refs.set(vector(10000), 98);

  const labels = new Uint8Array([1, 1, 0, 1, 0, 0, 1, 1]);
  const candidates = new Uint32Array([1, 2, 3, 4, 5]);

  assert.equal(native.knnBucketCandidates(vector(0), refs, labels, candidates), 2);
});

test('native knnBucketMultiCandidates dedupes overlapping groups', () => {
  const native = loadNativeKnn();
  const refs = new Int16Array(8 * 14);
  refs.set(vector(0), 0);
  refs.set(vector(0), 14);
  refs.set(vector(0), 28);
  refs.set(vector(0), 42);
  refs.set(vector(0), 56);
  refs.set(vector(0), 70);
  refs.set(vector(9000), 84);
  refs.set(vector(10000), 98);

  const labels = new Uint8Array([1, 1, 0, 1, 0, 0, 1, 1]);
  const groupA = new Uint32Array([1, 2, 3]);
  const groupB = new Uint32Array([3, 4, 5, 1]);
  const seen = new Uint8Array(labels.length);

  assert.equal(
    native.knnBucketMultiCandidates(vector(0), refs, labels, [groupA, groupB], seen),
    2,
  );
  for (const cell of seen) assert.equal(cell, 0);
});
