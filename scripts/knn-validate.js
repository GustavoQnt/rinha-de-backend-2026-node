import fs from 'node:fs';
import zlib from 'node:zlib';
import { vectorize } from '../src/vectorizer.js';

const argv = process.argv.slice(2);
const REF_LIMIT = argv.includes('--refs') ? +argv[argv.indexOf('--refs') + 1] : Infinity;
const TEST_LIMIT = argv.includes('--tests') ? +argv[argv.indexOf('--tests') + 1] : Infinity;
const VERBOSE = argv.includes('--verbose');

console.log(`Loading references (limit ${REF_LIMIT})...`);
let t0 = Date.now();
const refsRaw = JSON.parse(zlib.gunzipSync(fs.readFileSync('resources/references.json.gz')).toString('utf8'));
console.log(`  loaded ${refsRaw.length} refs in ${Date.now() - t0}ms`);

const N_REFS = Math.min(refsRaw.length, REF_LIMIT);

t0 = Date.now();
// Pack vectors into a Float64Array for tight loop access. 14 floats per ref.
const refVecs = new Float64Array(N_REFS * 14);
const refLabels = new Uint8Array(N_REFS); // 1 = fraud, 0 = legit
for (let i = 0; i < N_REFS; i += 1) {
  const r = refsRaw[i];
  for (let j = 0; j < 14; j += 1) refVecs[i * 14 + j] = r.vector[j];
  refLabels[i] = r.label === 'fraud' ? 1 : 0;
}
console.log(`  packed in ${Date.now() - t0}ms`);

console.log('Loading test entries...');
const data = JSON.parse(fs.readFileSync('test/test-data.json', 'utf8'));
const N_TESTS = Math.min(data.entries.length, TEST_LIMIT);
console.log(`  ${N_TESTS} entries`);

function knnPredict(query) {
  // top-5 with strict <
  const dists = [Infinity, Infinity, Infinity, Infinity, Infinity];
  const idxs = [-1, -1, -1, -1, -1];

  for (let i = 0; i < N_REFS; i += 1) {
    const base = i * 14;
    let d = 0;
    for (let j = 0; j < 14; j += 1) {
      const diff = query[j] - refVecs[base + j];
      d += diff * diff;
    }
    // Insert with strict <: only displace existing entries
    if (d < dists[4]) {
      let pos = 4;
      while (pos > 0 && d < dists[pos - 1]) {
        dists[pos] = dists[pos - 1];
        idxs[pos] = idxs[pos - 1];
        pos -= 1;
      }
      dists[pos] = d;
      idxs[pos] = i;
    }
  }

  let fraudCount = 0;
  for (let k = 0; k < 5; k += 1) {
    if (refLabels[idxs[k]] === 1) fraudCount += 1;
  }
  const score = fraudCount / 5;
  // round4 like the generator
  const rounded = Math.round(score * 10000) / 10000;
  return { score: rounded, approved: rounded < 0.6, dists, idxs };
}

let approvedMatch = 0;
let approvedMismatch = 0;
let scoreMatch = 0;
let scoreMismatch = 0;
const mismatchSamples = [];

const tStart = Date.now();
for (let i = 0; i < N_TESTS; i += 1) {
  const entry = data.entries[i];
  const query = vectorize(entry.request);
  const result = knnPredict(query);

  if (result.approved === entry.expected_approved) approvedMatch += 1; else {
    approvedMismatch += 1;
    if (mismatchSamples.length < 20) {
      mismatchSamples.push({
        idx: i,
        id: entry.request.id,
        expected_approved: entry.expected_approved,
        actual_approved: result.approved,
        expected_score: entry.expected_fraud_score,
        actual_score: result.score,
        query,
        dists: result.dists,
      });
    }
  }
  if (Math.abs(result.score - entry.expected_fraud_score) < 1e-9) scoreMatch += 1;
  else scoreMismatch += 1;

  if (VERBOSE && i < 5) {
    console.log(`  [${i}] id=${entry.request.id} expected_approved=${entry.expected_approved} actual_approved=${result.approved} expected_score=${entry.expected_fraud_score} actual_score=${result.score}`);
  }
}
const ms = Date.now() - tStart;
const ratePerSec = (N_TESTS / (ms / 1000));

console.log(`\nResults (${N_TESTS} tests × ${N_REFS} refs):`);
console.log(`  approved_match=${approvedMatch}  approved_mismatch=${approvedMismatch}`);
console.log(`  score_match=${scoreMatch}  score_mismatch=${scoreMismatch}`);
console.log(`  elapsed=${ms}ms  rate=${ratePerSec.toFixed(1)} tests/s`);

if (mismatchSamples.length > 0) {
  console.log('\nFirst mismatches:');
  for (const m of mismatchSamples) {
    console.log(`  [${m.idx}] ${m.id}: expected approved=${m.expected_approved} score=${m.expected_score}, got approved=${m.actual_approved} score=${m.actual_score}`);
    console.log(`    query: [${m.query.join(', ')}]`);
    console.log(`    top-5 dists: [${m.dists.map(d => d.toFixed(6)).join(', ')}]`);
  }
}
