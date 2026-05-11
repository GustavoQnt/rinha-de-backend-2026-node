import fs from 'node:fs';
import { vectorize } from '../src/vectorizer.js';
import {
  DIM,
  HEADER_BYTES,
  quantizeValue,
  readReferencesBinHeader,
} from './build-references-bin.js';

const argv = process.argv.slice(2);
const BIN_PATH = argv.includes('--bin') ? argv[argv.indexOf('--bin') + 1] : 'resources/references.bin';
const TEST_LIMIT = argv.includes('--tests') ? +argv[argv.indexOf('--tests') + 1] : Infinity;
const VERBOSE = argv.includes('--verbose');

console.log(`Loading quantized references from ${BIN_PATH}...`);
let t0 = Date.now();
const header = readReferencesBinHeader(BIN_PATH);
if (header.magic !== 'R26B') throw new Error(`invalid magic: ${header.magic}`);
if (header.version !== 1) throw new Error(`unsupported version: ${header.version}`);
if (header.dim !== DIM) throw new Error(`unsupported dim: ${header.dim}`);

const expectedBytes = HEADER_BYTES + header.count * header.dim * 2 + header.count;
if (header.bytes !== expectedBytes) {
  throw new Error(`invalid file size: got ${header.bytes}, expected ${expectedBytes}`);
}

const bin = fs.readFileSync(BIN_PATH);
const refVecs = new Int16Array(bin.buffer, bin.byteOffset + HEADER_BYTES, header.count * DIM);
const labelsOffset = HEADER_BYTES + header.count * DIM * 2;
const refLabels = new Uint8Array(bin.buffer, bin.byteOffset + labelsOffset, header.count);
console.log(`  loaded ${header.count} refs in ${Date.now() - t0}ms`);

console.log('Loading test entries...');
const data = JSON.parse(fs.readFileSync('test/test-data.json', 'utf8'));
const N_TESTS = Math.min(data.entries.length, TEST_LIMIT);
console.log(`  ${N_TESTS} entries`);

function quantizeQuery(query) {
  const out = new Int16Array(DIM);
  for (let i = 0; i < DIM; i += 1) out[i] = quantizeValue(query[i], header.scale);
  return out;
}

function knnPredict(query) {
  const q = quantizeQuery(query);
  const dists = [Infinity, Infinity, Infinity, Infinity, Infinity];
  const idxs = [-1, -1, -1, -1, -1];

  for (let i = 0; i < header.count; i += 1) {
    const base = i * DIM;
    let d = 0;
    for (let j = 0; j < DIM; j += 1) {
      const diff = q[j] - refVecs[base + j];
      d += diff * diff;
    }

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
  const score = Math.round((fraudCount / 5) * 10000) / 10000;
  return { score, approved: score < 0.6, dists, idxs };
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

  if (result.approved === entry.expected_approved) approvedMatch += 1;
  else {
    approvedMismatch += 1;
    if (mismatchSamples.length < 20) {
      mismatchSamples.push({
        idx: i,
        id: entry.request.id,
        expected_approved: entry.expected_approved,
        actual_approved: result.approved,
        expected_score: entry.expected_fraud_score,
        actual_score: result.score,
        dists: result.dists,
        idxs: result.idxs,
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
const ratePerSec = N_TESTS / (ms / 1000);

console.log(`\nResults (${N_TESTS} tests x ${header.count} quantized refs):`);
console.log(`  approved_match=${approvedMatch}  approved_mismatch=${approvedMismatch}`);
console.log(`  score_match=${scoreMatch}  score_mismatch=${scoreMismatch}`);
console.log(`  elapsed=${ms}ms  rate=${ratePerSec.toFixed(1)} tests/s`);

if (mismatchSamples.length > 0) {
  console.log('\nFirst mismatches:');
  for (const m of mismatchSamples) {
    console.log(`  [${m.idx}] ${m.id}: expected approved=${m.expected_approved} score=${m.expected_score}, got approved=${m.actual_approved} score=${m.actual_score}`);
    console.log(`    top-5 idxs: [${m.idxs.join(', ')}]`);
    console.log(`    top-5 dists: [${m.dists.join(', ')}]`);
  }
}
