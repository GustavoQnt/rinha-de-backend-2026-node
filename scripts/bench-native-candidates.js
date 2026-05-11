import fs from 'node:fs';
import { vectorize } from '../src/vectorizer.js';
import { loadNativeKnn } from '../src/native-knn.js';
import { loadBinaryReferences } from '../src/knn-classifier.js';
import { candidateIndexesForVector, loadGridIndex } from '../src/grid-index.js';
import { quantizeValue } from './build-references-bin.js';

const argv = process.argv.slice(2);
const BIN_PATH = argv.includes('--bin') ? argv[argv.indexOf('--bin') + 1] : 'resources/references.bin';
const INDEX_PATH = argv.includes('--index') ? argv[argv.indexOf('--index') + 1] : 'resources/references.grid.bin';
const TEST_LIMIT = argv.includes('--tests') ? +argv[argv.indexOf('--tests') + 1] : 100;
const MIN_CANDIDATES = argv.includes('--min-candidates') ? +argv[argv.indexOf('--min-candidates') + 1] : 50_000;
const MAX_RADIUS = argv.includes('--max-radius') ? +argv[argv.indexOf('--max-radius') + 1] : 3;
const EDGE_ONLY = argv.includes('--edge-only');

function quantizeQuery(query, scale) {
  const out = new Int16Array(query.length);
  for (let i = 0; i < query.length; i += 1) out[i] = quantizeValue(query[i], scale);
  return out;
}

function jsBucketFromCandidates(refs, query, candidates) {
  const distances = [Infinity, Infinity, Infinity, Infinity, Infinity];
  const indexes = [-1, -1, -1, -1, -1];

  for (let c = 0; c < candidates.length; c += 1) {
    const i = candidates[c];
    const base = i * refs.dim;
    let distance = 0;
    for (let dim = 0; dim < refs.dim; dim += 1) {
      const diff = query[dim] - refs.vectors[base + dim];
      distance += diff * diff;
    }

    if (distance < distances[4]) {
      let pos = 4;
      while (pos > 0 && distance < distances[pos - 1]) {
        distances[pos] = distances[pos - 1];
        indexes[pos] = indexes[pos - 1];
        pos -= 1;
      }
      distances[pos] = distance;
      indexes[pos] = i;
    }
  }

  let bucket = 0;
  for (let i = 0; i < 5; i += 1) {
    if (refs.labels[indexes[i]] === 1) bucket += 1;
  }
  return bucket;
}

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

function bucketToScore(bucket) {
  return Math.round((bucket / 5) * 10000) / 10000;
}

console.log('Loading native addon...');
const native = loadNativeKnn();

console.log(`Loading references from ${BIN_PATH}...`);
let t0 = Date.now();
const refs = loadBinaryReferences(BIN_PATH);
console.log(`  loaded ${refs.count} refs in ${Date.now() - t0}ms`);

console.log(`Loading grid index from ${INDEX_PATH}...`);
t0 = Date.now();
const index = loadGridIndex(INDEX_PATH);
console.log(`  loaded ${index.bucketCount} buckets in ${Date.now() - t0}ms`);

const data = JSON.parse(fs.readFileSync('test/test-data.json', 'utf8'));
const entries = EDGE_ONLY
  ? data.entries.filter((entry) => entry.expected_fraud_score === 0.4 || entry.expected_fraud_score === 0.6)
  : data.entries;
const nTests = Math.min(entries.length, TEST_LIMIT);
const queries = new Array(nTests);
const candidates = new Array(nTests);
const candidateCounts = [];

t0 = Date.now();
for (let i = 0; i < nTests; i += 1) {
  const query = quantizeQuery(vectorize(entries[i].request), refs.scale);
  const candidateResult = candidateIndexesForVector(index, query, {
    minCandidates: MIN_CANDIDATES,
    maxRadius: MAX_RADIUS,
  });
  queries[i] = query;
  candidates[i] = Uint32Array.from(candidateResult.indexes);
  candidateCounts.push(candidates[i].length);
}
const candidateMs = Date.now() - t0;

let jsScoreMatch = 0;
t0 = Date.now();
for (let i = 0; i < nTests; i += 1) {
  const bucket = jsBucketFromCandidates(refs, queries[i], candidates[i]);
  if (Math.abs(bucketToScore(bucket) - entries[i].expected_fraud_score) < 1e-9) jsScoreMatch += 1;
}
const jsMs = Date.now() - t0;

let nativeScoreMatch = 0;
t0 = Date.now();
for (let i = 0; i < nTests; i += 1) {
  const bucket = native.knnBucketCandidates(queries[i], refs.vectors, refs.labels, candidates[i]);
  if (Math.abs(bucketToScore(bucket) - entries[i].expected_fraud_score) < 1e-9) nativeScoreMatch += 1;
}
const nativeMs = Date.now() - t0;

candidateCounts.sort((a, b) => a - b);
const totalCandidates = candidateCounts.reduce((sum, n) => sum + n, 0);

console.log(`\nResults (${nTests} ${EDGE_ONLY ? 'edge' : 'general'} tests):`);
console.log(`  score_match_js=${jsScoreMatch}/${nTests}`);
console.log(`  score_match_native=${nativeScoreMatch}/${nTests}`);
console.log('  candidates:');
console.log(`    avg=${(totalCandidates / Math.max(1, nTests)).toFixed(1)}`);
console.log(`    p50=${percentile(candidateCounts, 0.50)}`);
console.log(`    p95=${percentile(candidateCounts, 0.95)}`);
console.log(`    max=${candidateCounts[candidateCounts.length - 1] ?? 0}`);
console.log('  timings:');
console.log(`    candidate_generation_ms=${candidateMs}`);
console.log(`    js_scan_ms=${jsMs}`);
console.log(`    native_scan_ms=${nativeMs}`);
console.log(`    js_scan_qps=${(nTests / (jsMs / 1000)).toFixed(1)}`);
console.log(`    native_scan_qps=${(nTests / (nativeMs / 1000)).toFixed(1)}`);
console.log(`    native_avg_ms_per_query=${(nativeMs / Math.max(1, nTests)).toFixed(3)}`);

if (nativeScoreMatch !== jsScoreMatch) {
  process.exitCode = 1;
}
