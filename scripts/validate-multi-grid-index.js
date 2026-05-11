import fs from 'node:fs';
import { vectorize } from '../src/vectorizer.js';
import { loadBinaryReferences } from '../src/knn-classifier.js';
import {
  buildMultiGridIndex,
  candidateIndexesForMultiGrid,
  MULTI_GRID_FEATURE_SETS,
} from '../src/multi-grid-index.js';
import { quantizeValue } from './build-references-bin.js';

const argv = process.argv.slice(2);
if (argv.includes('--help')) {
  console.log(`Usage: node scripts/validate-multi-grid-index.js [options]

Options:
  --bin <path>                    Source R26B references.bin (default: resources/references.bin)
  --tests <n>                     Number of preview entries to validate (default: 100)
  --skip <n>                      Skip n selected entries before validation (default: 0)
  --edge-only                     Validate only expected fraud scores 0.4 and 0.6
  --grids <n>                     Use the first n configured grids (default: all)
  --per-grid-min-candidates <n>   Stop each grid probe once it reaches n candidates (default: 10000)
  --max-radius <n>                Maximum Manhattan probe radius per grid (default: 1)
  --unsorted-candidates           Keep union discovery order instead of sorting by reference index
`);
  process.exit(0);
}

const BIN_PATH = argv.includes('--bin') ? argv[argv.indexOf('--bin') + 1] : 'resources/references.bin';
const TEST_LIMIT = argv.includes('--tests') ? +argv[argv.indexOf('--tests') + 1] : 100;
const SKIP = argv.includes('--skip') ? +argv[argv.indexOf('--skip') + 1] : 0;
const EDGE_ONLY = argv.includes('--edge-only');
const GRID_COUNT = argv.includes('--grids') ? +argv[argv.indexOf('--grids') + 1] : MULTI_GRID_FEATURE_SETS.length;
const PER_GRID_MIN_CANDIDATES = argv.includes('--per-grid-min-candidates')
  ? +argv[argv.indexOf('--per-grid-min-candidates') + 1]
  : 10_000;
const MAX_RADIUS = argv.includes('--max-radius') ? +argv[argv.indexOf('--max-radius') + 1] : 1;
const SORT_CANDIDATES = !argv.includes('--unsorted-candidates');

function quantizeQuery(query, scale) {
  const out = new Int16Array(query.length);
  for (let i = 0; i < query.length; i += 1) out[i] = quantizeValue(query[i], scale);
  return out;
}

function predictFromCandidates(refs, query, candidates) {
  const distances = [Infinity, Infinity, Infinity, Infinity, Infinity];
  const indexes = [-1, -1, -1, -1, -1];

  for (let c = 0; c < candidates.length; c += 1) {
    const i = candidates[c];
    const base = i * refs.dim;
    let d = 0;
    for (let j = 0; j < refs.dim; j += 1) {
      const diff = query[j] - refs.vectors[base + j];
      d += diff * diff;
    }

    if (d < distances[4]) {
      let pos = 4;
      while (pos > 0 && d < distances[pos - 1]) {
        distances[pos] = distances[pos - 1];
        indexes[pos] = indexes[pos - 1];
        pos -= 1;
      }
      distances[pos] = d;
      indexes[pos] = i;
    }
  }

  let fraudCount = 0;
  for (let i = 0; i < 5; i += 1) {
    if (indexes[i] >= 0 && refs.labels[indexes[i]] === 1) fraudCount += 1;
  }

  const score = Math.round((fraudCount / 5) * 10000) / 10000;
  return { score, approved: score < 0.6, distances, indexes };
}

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

const featureSets = MULTI_GRID_FEATURE_SETS.slice(0, GRID_COUNT);

console.log(`Loading references from ${BIN_PATH}...`);
let t0 = Date.now();
const refs = loadBinaryReferences(BIN_PATH);
console.log(`  loaded ${refs.count} refs in ${Date.now() - t0}ms`);

console.log(`Building ${featureSets.length} grid indexes...`);
t0 = Date.now();
const index = buildMultiGridIndex(refs, featureSets);
const buildMs = Date.now() - t0;
console.log(`  built in ${buildMs}ms`);
for (let i = 0; i < index.indexes.length; i += 1) {
  console.log(`  grid_${i}: buckets=${index.indexes[i].bucketCount}`);
}

const data = JSON.parse(fs.readFileSync('test/test-data.json', 'utf8'));
const entries = EDGE_ONLY
  ? data.entries.filter((entry) => entry.expected_fraud_score === 0.4 || entry.expected_fraud_score === 0.6)
  : data.entries;
const selectedEntries = entries.slice(SKIP);
const nTests = Math.min(selectedEntries.length, TEST_LIMIT);
const candidateCounts = [];
const totalBeforeDedupeCounts = [];
const mismatches = [];
let approvedMatch = 0;
let scoreMatch = 0;
let emptyCandidateQueries = 0;
let candidateMs = 0;
let scanMs = 0;

for (let i = 0; i < nTests; i += 1) {
  const entry = selectedEntries[i];
  const query = quantizeQuery(vectorize(entry.request), refs.scale);

  t0 = Date.now();
  const candidateResult = candidateIndexesForMultiGrid(index, query, {
    perGridMinCandidates: PER_GRID_MIN_CANDIDATES,
    maxRadius: MAX_RADIUS,
    sortCandidates: SORT_CANDIDATES,
  });
  candidateMs += Date.now() - t0;

  candidateCounts.push(candidateResult.indexes.length);
  totalBeforeDedupeCounts.push(candidateResult.totalBeforeDedupe);

  if (candidateResult.indexes.length === 0) {
    emptyCandidateQueries += 1;
    continue;
  }

  t0 = Date.now();
  const result = predictFromCandidates(refs, query, candidateResult.indexes);
  scanMs += Date.now() - t0;

  if (result.approved === entry.expected_approved) approvedMatch += 1;
  if (Math.abs(result.score - entry.expected_fraud_score) < 1e-9) scoreMatch += 1;
  else if (mismatches.length < 20) {
    mismatches.push({
      idx: i,
      selected_idx: SKIP + i,
      id: entry.request.id,
      expected_approved: entry.expected_approved,
      actual_approved: result.approved,
      expected_score: entry.expected_fraud_score,
      actual_score: result.score,
      candidates: candidateResult.indexes.length,
      total_before_dedupe: candidateResult.totalBeforeDedupe,
      grid_results: candidateResult.gridResults,
      indexes: result.indexes,
      distances: result.distances,
    });
  }
}

candidateCounts.sort((a, b) => a - b);
totalBeforeDedupeCounts.sort((a, b) => a - b);
const totalCandidates = candidateCounts.reduce((sum, n) => sum + n, 0);
const totalBeforeDedupe = totalBeforeDedupeCounts.reduce((sum, n) => sum + n, 0);

console.log(`\nResults (${nTests} ${EDGE_ONLY ? 'edge' : 'general'} tests):`);
console.log(`  approved_match=${approvedMatch}/${nTests}`);
console.log(`  score_match=${scoreMatch}/${nTests}`);
console.log(`  empty_candidate_queries=${emptyCandidateQueries}`);
console.log(`  build_ms=${buildMs}`);
console.log('  candidates_after_dedupe:');
console.log(`    avg=${(totalCandidates / Math.max(1, nTests)).toFixed(1)}`);
console.log(`    p50=${percentile(candidateCounts, 0.50)}`);
console.log(`    p95=${percentile(candidateCounts, 0.95)}`);
console.log(`    max=${candidateCounts[candidateCounts.length - 1] ?? 0}`);
console.log('  candidates_before_dedupe:');
console.log(`    avg=${(totalBeforeDedupe / Math.max(1, nTests)).toFixed(1)}`);
console.log(`    p95=${percentile(totalBeforeDedupeCounts, 0.95)}`);
console.log('  timings:');
console.log(`    candidate_generation_ms=${candidateMs}`);
console.log(`    scan_ms=${scanMs}`);
console.log(`    end_to_end_qps=${(nTests / ((candidateMs + scanMs) / 1000)).toFixed(1)}`);

if (mismatches.length > 0) {
  console.log('\nFirst score mismatches:');
  for (const mismatch of mismatches) {
    console.log(`  [${mismatch.selected_idx}] ${mismatch.id}: expected approved=${mismatch.expected_approved} score=${mismatch.expected_score}, got approved=${mismatch.actual_approved} score=${mismatch.actual_score}`);
    console.log(`    candidates=${mismatch.candidates} before_dedupe=${mismatch.total_before_dedupe}`);
    console.log(`    top-5 idxs=[${mismatch.indexes.join(', ')}]`);
    console.log(`    top-5 dists=[${mismatch.distances.join(', ')}]`);
    console.log(`    grids=${JSON.stringify(mismatch.grid_results)}`);
  }
}

if (approvedMatch !== nTests || scoreMatch !== nTests) {
  process.exitCode = 1;
}
