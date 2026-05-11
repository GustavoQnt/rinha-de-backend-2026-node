import fs from 'node:fs';
import { vectorize } from '../src/vectorizer.js';
import { loadNativeKnn } from '../src/native-knn.js';
import { loadBinaryReferences } from '../src/knn-classifier.js';
import {
  buildMultiGridIndex,
  candidateGroupsForMultiGrid,
  candidateIndexesForMultiGrid,
  packMultiGridForNative,
  MULTI_GRID_FEATURE_SETS,
} from '../src/multi-grid-index.js';
import { quantizeValue } from './build-references-bin.js';

const argv = process.argv.slice(2);
const BIN_PATH = argv.includes('--bin') ? argv[argv.indexOf('--bin') + 1] : 'resources/references.bin';
const TEST_LIMIT = argv.includes('--tests') ? +argv[argv.indexOf('--tests') + 1] : 200;
const PER_GRID = argv.includes('--per-grid-min') ? +argv[argv.indexOf('--per-grid-min') + 1] : 10_000;
const MAX_RADIUS = argv.includes('--max-radius') ? +argv[argv.indexOf('--max-radius') + 1] : 1;
const GRID_COUNT = argv.includes('--grids') ? +argv[argv.indexOf('--grids') + 1] : MULTI_GRID_FEATURE_SETS.length;
const EDGE_ONLY = argv.includes('--edge-only');

function quantizeQuery(query, scale) {
  const out = new Int16Array(query.length);
  for (let i = 0; i < query.length; i += 1) out[i] = quantizeValue(query[i], scale);
  return out;
}

function bucketToScore(bucket) {
  return Math.round((bucket / 5) * 10000) / 10000;
}

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

console.log('Loading native addon...');
const native = loadNativeKnn();

console.log(`Loading references from ${BIN_PATH}...`);
let t0 = Date.now();
const refs = loadBinaryReferences(BIN_PATH);
console.log(`  loaded ${refs.count} refs in ${Date.now() - t0}ms`);

const featureSets = MULTI_GRID_FEATURE_SETS.slice(0, GRID_COUNT);
console.log(`Building ${featureSets.length} grid indexes...`);
t0 = Date.now();
const multiIndex = buildMultiGridIndex(refs, featureSets);
console.log(`  built in ${Date.now() - t0}ms`);

const data = JSON.parse(fs.readFileSync('test/test-data.json', 'utf8'));
const entries = EDGE_ONLY
  ? data.entries.filter((entry) => entry.expected_fraud_score === 0.4 || entry.expected_fraud_score === 0.6)
  : data.entries;
const nTests = Math.min(entries.length, TEST_LIMIT);
const queries = new Array(nTests);

for (let i = 0; i < nTests; i += 1) {
  queries[i] = quantizeQuery(vectorize(entries[i].request), refs.scale);
}

// JS path: dedupe + native knnBucketCandidates
let jsCandMs = 0;
let jsScanMs = 0;
let jsScoreMatch = 0;
const jsCandCounts = [];
for (let i = 0; i < nTests; i += 1) {
  t0 = Date.now();
  const result = candidateIndexesForMultiGrid(multiIndex, queries[i], {
    perGridMinCandidates: PER_GRID,
    maxRadius: MAX_RADIUS,
    sortCandidates: false,
  });
  jsCandMs += Date.now() - t0;
  jsCandCounts.push(result.indexes.length);
  const candidates = Uint32Array.from(result.indexes);

  t0 = Date.now();
  const bucket = native.knnBucketCandidates(queries[i], refs.vectors, refs.labels, candidates);
  jsScanMs += Date.now() - t0;
  if (Math.abs(bucketToScore(bucket) - entries[i].expected_fraud_score) < 1e-9) jsScoreMatch += 1;
}

// Native path: per-grid groups + knnBucketMultiCandidates (dedupe + scan in native)
const seenScratch = new Uint8Array(refs.count);
let nvCandMs = 0;
let nvScanMs = 0;
let nvScoreMatch = 0;
for (let i = 0; i < nTests; i += 1) {
  t0 = Date.now();
  const { groups } = candidateGroupsForMultiGrid(multiIndex, queries[i], {
    perGridMinCandidates: PER_GRID,
    maxRadius: MAX_RADIUS,
  });
  nvCandMs += Date.now() - t0;

  t0 = Date.now();
  const bucket = native.knnBucketMultiCandidates(queries[i], refs.vectors, refs.labels, groups, seenScratch);
  nvScanMs += Date.now() - t0;
  if (Math.abs(bucketToScore(bucket) - entries[i].expected_fraud_score) < 1e-9) nvScoreMatch += 1;
}

// Native path C: full native multi-grid (probes + dedupe + scan in Rust)
const packed = packMultiGridForNative(multiIndex);
let fullMs = 0;
let fullScoreMatch = 0;
for (let i = 0; i < nTests; i += 1) {
  t0 = Date.now();
  const bucket = native.multiGridKnnBucket(
    queries[i],
    refs.vectors,
    refs.labels,
    packed.bucketTables,
    packed.postingsList,
    packed.featuresList,
    refs.scale,
    PER_GRID,
    MAX_RADIUS,
    seenScratch,
  );
  fullMs += Date.now() - t0;
  if (Math.abs(bucketToScore(bucket) - entries[i].expected_fraud_score) < 1e-9) fullScoreMatch += 1;
}

jsCandCounts.sort((a, b) => a - b);
const totalCands = jsCandCounts.reduce((s, n) => s + n, 0);

console.log(`\nResults (${nTests} ${EDGE_ONLY ? 'edge' : 'general'} tests):`);
console.log(`  score_match_js_dedupe=${jsScoreMatch}/${nTests}`);
console.log(`  score_match_native_dedupe=${nvScoreMatch}/${nTests}`);
console.log(`  score_match_native_full=${fullScoreMatch}/${nTests}`);
console.log('  candidates_after_dedupe:');
console.log(`    avg=${(totalCands / Math.max(1, nTests)).toFixed(1)}`);
console.log(`    p50=${percentile(jsCandCounts, 0.5)}`);
console.log(`    p95=${percentile(jsCandCounts, 0.95)}`);
console.log(`    max=${jsCandCounts[jsCandCounts.length - 1] ?? 0}`);
console.log('  timings:');
console.log(`    js_dedupe_cand_ms=${jsCandMs}  scan_ms=${jsScanMs}  total=${jsCandMs + jsScanMs}`);
console.log(`    native_dedupe_cand_ms=${nvCandMs}  scan_ms=${nvScanMs}  total=${nvCandMs + nvScanMs}`);
console.log(`    native_full_total_ms=${fullMs}  qps=${(nTests / (fullMs / 1000)).toFixed(1)}  avg_ms=${(fullMs / Math.max(1, nTests)).toFixed(3)}`);
console.log(`    js_total_qps=${(nTests / ((jsCandMs + jsScanMs) / 1000)).toFixed(1)}`);
console.log(`    native_total_qps=${(nTests / ((nvCandMs + nvScanMs) / 1000)).toFixed(1)}`);
console.log(`    native_avg_ms_per_query=${((nvCandMs + nvScanMs) / Math.max(1, nTests)).toFixed(3)}`);

if (jsScoreMatch !== nvScoreMatch) {
  console.error('\nMismatch between JS-dedupe and native-dedupe paths.');
  process.exitCode = 1;
}
