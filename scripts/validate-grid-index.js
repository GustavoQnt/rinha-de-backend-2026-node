import fs from 'node:fs';
import { vectorize } from '../src/vectorizer.js';
import { candidateIndexesForVector, loadGridIndex } from '../src/grid-index.js';
import { loadBinaryReferences } from '../src/knn-classifier.js';
import { quantizeValue } from './build-references-bin.js';

const argv = process.argv.slice(2);
if (argv.includes('--help')) {
  console.log(`Usage: node scripts/validate-grid-index.js [options]

Options:
  --bin <path>              Source R26B references.bin (default: resources/references.bin)
  --index <path>            Source R26G grid index (default: resources/references.grid.bin)
  --tests <n>               Number of preview entries to validate (default: 100)
  --min-candidates <n>      Stop probing once at least this many candidates are found (default: 10000)
  --max-radius <n>          Maximum Manhattan probe radius over grid bins (default: 1)
  --force-max-radius        Always probe all keys through max radius
  --fallback-full-scan-under <n>
                            Use full reference scan when candidate count is below n
  --edge-only               Validate only expected fraud scores 0.4 and 0.6
`);
  process.exit(0);
}
const BIN_PATH = argv.includes('--bin') ? argv[argv.indexOf('--bin') + 1] : 'resources/references.bin';
const INDEX_PATH = argv.includes('--index') ? argv[argv.indexOf('--index') + 1] : 'resources/references.grid.bin';
const TEST_LIMIT = argv.includes('--tests') ? +argv[argv.indexOf('--tests') + 1] : 100;
const MIN_CANDIDATES = argv.includes('--min-candidates') ? +argv[argv.indexOf('--min-candidates') + 1] : 10_000;
const MAX_RADIUS = argv.includes('--max-radius') ? +argv[argv.indexOf('--max-radius') + 1] : 1;
const FORCE_MAX_RADIUS = argv.includes('--force-max-radius');
const FALLBACK_FULL_SCAN_UNDER = argv.includes('--fallback-full-scan-under')
  ? +argv[argv.indexOf('--fallback-full-scan-under') + 1]
  : 0;
const EDGE_ONLY = argv.includes('--edge-only');

function quantizeQuery(query, scale) {
  const out = new Int16Array(query.length);
  for (let i = 0; i < query.length; i += 1) out[i] = quantizeValue(query[i], scale);
  return out;
}

function predictFromCandidates(refs, query, candidates) {
  const q = quantizeQuery(query, refs.scale);
  const distances = [Infinity, Infinity, Infinity, Infinity, Infinity];
  const indexes = [-1, -1, -1, -1, -1];
  const total = candidates === null ? refs.count : candidates.length;

  for (let c = 0; c < total; c += 1) {
    const i = candidates === null ? c : candidates[c];
    const base = i * refs.dim;
    let d = 0;
    for (let j = 0; j < refs.dim; j += 1) {
      const diff = q[j] - refs.vectors[base + j];
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
    if (refs.labels[indexes[i]] === 1) fraudCount += 1;
  }

  const score = Math.round((fraudCount / 5) * 10000) / 10000;
  return { score, approved: score < 0.6, distances, indexes };
}

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

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
const candidateCounts = [];
const radii = [];
const keysVisited = [];
const mismatches = [];
let approvedMatch = 0;
let scoreMatch = 0;
let emptyCandidateQueries = 0;
let fallbackFullScans = 0;
const startedAt = Date.now();

for (let i = 0; i < nTests; i += 1) {
  const entry = entries[i];
  const query = vectorize(entry.request);
  const quantizedQuery = quantizeQuery(query, refs.scale);
  const candidateResult = candidateIndexesForVector(index, quantizedQuery, {
    minCandidates: FORCE_MAX_RADIUS ? Number.MAX_SAFE_INTEGER : MIN_CANDIDATES,
    maxRadius: MAX_RADIUS,
  });

  let candidateIndexes = candidateResult.indexes;
  if (FALLBACK_FULL_SCAN_UNDER > 0 && candidateIndexes.length < FALLBACK_FULL_SCAN_UNDER) {
    candidateIndexes = null;
    fallbackFullScans += 1;
  }

  const count = candidateIndexes === null ? refs.count : candidateIndexes.length;
  candidateCounts.push(count);
  radii.push(candidateResult.radius);
  keysVisited.push(candidateResult.keysVisited);

  if (candidateIndexes !== null && count === 0) {
    emptyCandidateQueries += 1;
    continue;
  }

  const result = predictFromCandidates(refs, query, candidateIndexes);

  if (result.approved === entry.expected_approved) approvedMatch += 1;
  if (Math.abs(result.score - entry.expected_fraud_score) < 1e-9) scoreMatch += 1;
  else if (mismatches.length < 20) {
    mismatches.push({
      idx: i,
      id: entry.request.id,
      expected_approved: entry.expected_approved,
      actual_approved: result.approved,
      expected_score: entry.expected_fraud_score,
      actual_score: result.score,
      candidates: count,
      radius: candidateResult.radius,
      keys_visited: candidateResult.keysVisited,
      indexes: result.indexes,
      distances: result.distances,
    });
  }
}

candidateCounts.sort((a, b) => a - b);
radii.sort((a, b) => a - b);
keysVisited.sort((a, b) => a - b);
const elapsedMs = Date.now() - startedAt;
const totalCandidates = candidateCounts.reduce((sum, n) => sum + n, 0);

console.log(`\nResults (${nTests} tests):`);
console.log(`  approved_match=${approvedMatch}/${nTests}`);
console.log(`  score_match=${scoreMatch}/${nTests}`);
console.log(`  empty_candidate_queries=${emptyCandidateQueries}`);
console.log(`  fallback_full_scans=${fallbackFullScans}`);
console.log(`  elapsed=${elapsedMs}ms  rate=${(nTests / (elapsedMs / 1000)).toFixed(1)} tests/s`);
console.log('  candidates:');
console.log(`    avg=${(totalCandidates / Math.max(1, nTests)).toFixed(1)}`);
console.log(`    p50=${percentile(candidateCounts, 0.50)}`);
console.log(`    p95=${percentile(candidateCounts, 0.95)}`);
console.log(`    max=${candidateCounts[candidateCounts.length - 1] ?? 0}`);
console.log('  probe:');
console.log(`    radius_p95=${percentile(radii, 0.95)}`);
console.log(`    keys_visited_p95=${percentile(keysVisited, 0.95)}`);

if (mismatches.length > 0) {
  console.log('\nFirst score mismatches:');
  for (const mismatch of mismatches) {
    console.log(`  [${mismatch.idx}] ${mismatch.id}: expected approved=${mismatch.expected_approved} score=${mismatch.expected_score}, got approved=${mismatch.actual_approved} score=${mismatch.actual_score}`);
    console.log(`    candidates=${mismatch.candidates} radius=${mismatch.radius} keys=${mismatch.keys_visited}`);
    console.log(`    top-5 idxs=[${mismatch.indexes.join(', ')}]`);
    console.log(`    top-5 dists=[${mismatch.distances.join(', ')}]`);
  }
}

if (approvedMatch !== nTests || scoreMatch !== nTests) {
  process.exitCode = 1;
}
