import fs from 'node:fs';
import { vectorize } from '../src/vectorizer.js';
import { loadNativeKnn } from '../src/native-knn.js';
import { loadBinaryReferences } from '../src/knn-classifier.js';
import {
  buildMultiGridIndex,
  candidateGroupsForMultiGrid,
  packMultiGridForNative,
  MULTI_GRID_FEATURE_SETS,
} from '../src/multi-grid-index.js';
import { quantizeValue } from './build-references-bin.js';

const argv = process.argv.slice(2);
const BIN_PATH = argv.includes('--bin') ? argv[argv.indexOf('--bin') + 1] : 'resources/references.bin';
const TEST_LIMIT = argv.includes('--tests') ? +argv[argv.indexOf('--tests') + 1] : 200;
const PER_GRID_LIST = (argv.includes('--per-grids')
  ? argv[argv.indexOf('--per-grids') + 1]
  : '10000,7500,5000,2500,1000'
).split(',').map((n) => +n);
const RADIUS_LIST = (argv.includes('--radii')
  ? argv[argv.indexOf('--radii') + 1]
  : '1,0'
).split(',').map((n) => +n);
const REPORT_MISMATCHES = argv.includes('--mismatches');
const MAX_MISMATCH_LIST = 20;

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

const native = loadNativeKnn();
console.log(`Loading references from ${BIN_PATH}...`);
const refs = loadBinaryReferences(BIN_PATH);
console.log(`  ${refs.count} refs`);
const featureSets = MULTI_GRID_FEATURE_SETS;
console.log(`Building ${featureSets.length} grid indexes...`);
const multiIndex = buildMultiGridIndex(refs, featureSets);
const packed = packMultiGridForNative(multiIndex);

const data = JSON.parse(fs.readFileSync('test/test-data.json', 'utf8'));
const entries = data.entries;
const nTests = Math.min(entries.length, TEST_LIMIT);
const queries = new Array(nTests);
const expectedScores = new Array(nTests);
const expectedApproved = new Array(nTests);
for (let i = 0; i < nTests; i += 1) {
  queries[i] = quantizeQuery(vectorize(entries[i].request), refs.scale);
  expectedScores[i] = entries[i].expected_fraud_score;
  expectedApproved[i] = entries[i].expected_approved;
}

const seenScratch = new Uint8Array(refs.count);

function runOne(perGrid, radius) {
  let totalMs = 0;
  let scoreMatch = 0;
  let approvedMatch = 0;
  const uniqueCounts = new Array(nTests);
  const mismatches = [];

  for (let i = 0; i < nTests; i += 1) {
    // Get unique candidate count via JS dedupe groups (cheap relative to scan; exact count).
    const { groups } = candidateGroupsForMultiGrid(multiIndex, queries[i], {
      perGridMinCandidates: perGrid,
      maxRadius: radius,
    });
    const seen = new Uint8Array(refs.count);
    let unique = 0;
    for (const g of groups) {
      for (let k = 0; k < g.length; k += 1) {
        const v = g[k];
        if (seen[v] === 0) { seen[v] = 1; unique += 1; }
      }
    }
    uniqueCounts[i] = unique;

    const t0 = Date.now();
    let bucket;
    try {
      bucket = native.multiGridKnnBucket(
        queries[i],
        refs.vectors,
        refs.labels,
        packed.bucketTables,
        packed.postingsList,
        packed.featuresList,
        refs.scale,
        perGrid,
        radius,
        seenScratch,
      );
    } catch (err) {
      mismatches.push({ i, id: entries[i].request.id, error: err.message, unique });
      totalMs += Date.now() - t0;
      continue;
    }
    totalMs += Date.now() - t0;
    const score = bucketToScore(bucket);
    const approved = score < 0.6;
    if (Math.abs(score - expectedScores[i]) < 1e-9) scoreMatch += 1;
    else if (mismatches.length < MAX_MISMATCH_LIST) {
      mismatches.push({
        i,
        id: entries[i].request.id,
        expected_score: expectedScores[i],
        actual_score: score,
        expected_approved: expectedApproved[i],
        actual_approved: approved,
        unique,
      });
    }
    if (approved === expectedApproved[i]) approvedMatch += 1;
  }

  uniqueCounts.sort((a, b) => a - b);
  const totalUnique = uniqueCounts.reduce((s, n) => s + n, 0);
  return {
    perGrid,
    radius,
    totalMs,
    qps: nTests / (totalMs / 1000),
    msPerQuery: totalMs / nTests,
    scoreMatch,
    approvedMatch,
    avgUnique: totalUnique / Math.max(1, nTests),
    p50Unique: percentile(uniqueCounts, 0.5),
    p95Unique: percentile(uniqueCounts, 0.95),
    maxUnique: uniqueCounts[uniqueCounts.length - 1] ?? 0,
    mismatches,
  };
}

const rows = [];
for (const radius of RADIUS_LIST) {
  for (const perGrid of PER_GRID_LIST) {
    process.stdout.write(`run perGrid=${perGrid} radius=${radius} ... `);
    const res = runOne(perGrid, radius);
    process.stdout.write(`score=${res.scoreMatch}/${nTests} approved=${res.approvedMatch}/${nTests} ms_avg=${res.msPerQuery.toFixed(2)} avg_unique=${res.avgUnique.toFixed(0)}\n`);
    rows.push(res);
  }
}

console.log('\n| perGrid | radius | score_match | approved_match | ms/query | qps | avg_unique | p95_unique | max_unique |');
console.log('|--:|--:|--:|--:|--:|--:|--:|--:|--:|');
for (const r of rows) {
  console.log(`| ${r.perGrid} | ${r.radius} | ${r.scoreMatch}/${nTests} | ${r.approvedMatch}/${nTests} | ${r.msPerQuery.toFixed(2)} | ${r.qps.toFixed(1)} | ${r.avgUnique.toFixed(0)} | ${r.p95Unique} | ${r.maxUnique} |`);
}

if (REPORT_MISMATCHES) {
  for (const r of rows) {
    if (r.mismatches.length === 0) continue;
    console.log(`\nMismatches for perGrid=${r.perGrid} radius=${r.radius}:`);
    for (const m of r.mismatches) {
      if (m.error) {
        console.log(`  [${m.i}] ${m.id}: error=${m.error} unique=${m.unique}`);
      } else {
        console.log(`  [${m.i}] ${m.id}: expected approved=${m.expected_approved} score=${m.expected_score}, got approved=${m.actual_approved} score=${m.actual_score}, unique=${m.unique}`);
      }
    }
  }
}
