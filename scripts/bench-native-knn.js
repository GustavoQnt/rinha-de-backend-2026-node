import fs from 'node:fs';
import { loadNativeKnn } from '../src/native-knn.js';
import { loadBinaryReferences } from '../src/knn-classifier.js';
import { vectorize } from '../src/vectorizer.js';
import { quantizeValue } from './build-references-bin.js';

const argv = process.argv.slice(2);
const BIN_PATH = argv.includes('--bin') ? argv[argv.indexOf('--bin') + 1] : 'resources/references.bin';
const TEST_LIMIT = argv.includes('--tests') ? +argv[argv.indexOf('--tests') + 1] : 10;

function quantizeQuery(query, scale) {
  const out = new Int16Array(query.length);
  for (let i = 0; i < query.length; i += 1) out[i] = quantizeValue(query[i], scale);
  return out;
}

console.log('Loading native addon...');
const native = loadNativeKnn();

console.log(`Loading references from ${BIN_PATH}...`);
let t0 = Date.now();
const refs = loadBinaryReferences(BIN_PATH);
console.log(`  loaded ${refs.count} refs in ${Date.now() - t0}ms`);

const data = JSON.parse(fs.readFileSync('test/test-data.json', 'utf8'));
const nTests = Math.min(data.entries.length, TEST_LIMIT);
let approvedMatch = 0;
let scoreMatch = 0;
const mismatches = [];

t0 = Date.now();
for (let i = 0; i < nTests; i += 1) {
  const entry = data.entries[i];
  const query = quantizeQuery(vectorize(entry.request), refs.scale);
  const bucket = native.knnBucket(query, refs.vectors, refs.labels, refs.count);
  const score = Math.round((bucket / 5) * 10000) / 10000;
  const approved = score < 0.6;

  if (approved === entry.expected_approved) approvedMatch += 1;
  if (Math.abs(score - entry.expected_fraud_score) < 1e-9) scoreMatch += 1;
  else if (mismatches.length < 20) {
    mismatches.push({
      idx: i,
      id: entry.request.id,
      expected_approved: entry.expected_approved,
      actual_approved: approved,
      expected_score: entry.expected_fraud_score,
      actual_score: score,
    });
  }
}

const elapsedMs = Date.now() - t0;
console.log(`\nResults (${nTests} tests x ${refs.count} native refs):`);
console.log(`  approved_match=${approvedMatch}/${nTests}`);
console.log(`  score_match=${scoreMatch}/${nTests}`);
console.log(`  elapsed=${elapsedMs}ms`);
console.log(`  rate=${(nTests / (elapsedMs / 1000)).toFixed(1)} tests/s`);
console.log(`  avg_ms_per_query=${(elapsedMs / Math.max(1, nTests)).toFixed(2)}`);

if (mismatches.length > 0) {
  console.log('\nFirst mismatches:');
  for (const mismatch of mismatches) {
    console.log(`  [${mismatch.idx}] ${mismatch.id}: expected approved=${mismatch.expected_approved} score=${mismatch.expected_score}, got approved=${mismatch.actual_approved} score=${mismatch.actual_score}`);
  }
}

if (approvedMatch !== nTests || scoreMatch !== nTests) {
  process.exitCode = 1;
}
