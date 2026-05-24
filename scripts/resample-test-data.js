// Resample requests/buckets from test/test-data.requests.ndjson into a new
// dataset with a target fraud_rate and edge_rate. Ground truth (refs) is
// unchanged, so bucket labels stay correct — we just stratify and sample.
//
// Usage:
//   node scripts/resample-test-data.js \
//     --out-prefix test/robustness/highfraud \
//     --fraud-rate 0.55 --edge-rate 0.015 --total 54100 --seed 7

import fs from 'node:fs';

const args = (() => {
  const o = { total: 54100, fraudRate: 0.47, edgeRate: 0.012, seed: 1, outPrefix: 'test/robustness/sample' };
  const a = process.argv.slice(2);
  for (let i = 0; i < a.length; i += 2) {
    const k = a[i].replace(/^--/, '');
    const v = a[i + 1];
    if (k === 'total') o.total = +v;
    else if (k === 'fraud-rate') o.fraudRate = +v;
    else if (k === 'edge-rate') o.edgeRate = +v;
    else if (k === 'seed') o.seed = +v;
    else if (k === 'out-prefix') o.outPrefix = v;
    else throw new Error(`unknown arg: ${k}`);
  }
  return o;
})();

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(args.seed);

const requests = fs.readFileSync('test/test-data.requests.ndjson', 'utf8').trim().split('\n');
const buckets = fs.readFileSync('test/test-data.expected-buckets.ndjson', 'utf8').trim().split('\n').map(Number);
if (requests.length !== buckets.length) throw new Error('length mismatch');

// Stratify indices by class.
const idxLegitClear = []; // bucket 0
const idxLegitGray = [];  // bucket 1,2
const idxFraudGray = [];  // bucket 3,4
const idxFraudClear = []; // bucket 5
for (let i = 0; i < buckets.length; i++) {
  const b = buckets[i];
  if (b === 0) idxLegitClear.push(i);
  else if (b <= 2) idxLegitGray.push(i);
  else if (b <= 4) idxFraudGray.push(i);
  else idxFraudClear.push(i);
}

// Targets.
const N = args.total;
const Nfraud = Math.round(N * args.fraudRate);
const Nlegit = N - Nfraud;
const Nedge = Math.round(N * args.edgeRate);
// Split edges between fraud-edge (3,4) and legit-edge (1,2) proportionally.
const NedgeFraud = Math.round(Nedge * (idxFraudGray.length / (idxFraudGray.length + idxLegitGray.length)));
const NedgeLegit = Nedge - NedgeFraud;
const NfraudClear = Nfraud - NedgeFraud;
const NlegitClear = Nlegit - NedgeLegit;

// Sample with replacement to hit arbitrary totals.
function sample(pool, n) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(pool[Math.floor(rand() * pool.length)]);
  return out;
}

const chosen = [
  ...sample(idxLegitClear, NlegitClear),
  ...sample(idxLegitGray, NedgeLegit),
  ...sample(idxFraudGray, NedgeFraud),
  ...sample(idxFraudClear, NfraudClear),
];
// Shuffle.
for (let i = chosen.length - 1; i > 0; i--) {
  const j = Math.floor(rand() * (i + 1));
  [chosen[i], chosen[j]] = [chosen[j], chosen[i]];
}

const outReqs = `${args.outPrefix}.requests.ndjson`;
const outBuckets = `${args.outPrefix}.expected-buckets.ndjson`;
const reqOut = fs.createWriteStream(outReqs);
const bkOut = fs.createWriteStream(outBuckets);
for (const i of chosen) {
  reqOut.write(requests[i] + '\n');
  bkOut.write(buckets[i] + '\n');
}
reqOut.end();
bkOut.end();

const finalCounts = chosen.reduce((acc, i) => { acc[buckets[i]] = (acc[buckets[i]] || 0) + 1; return acc; }, {});
const fraudFinal = (finalCounts[3] || 0) + (finalCounts[4] || 0) + (finalCounts[5] || 0);
const edgeFinal = (finalCounts[2] || 0) + (finalCounts[3] || 0);
process.stderr.write(`generated ${outReqs}\n`);
process.stderr.write(`  total=${chosen.length} fraud=${fraudFinal} (${(fraudFinal/chosen.length).toFixed(4)}) edge=${edgeFinal} (${(edgeFinal/chosen.length).toFixed(4)})\n`);
process.stderr.write(`  buckets: ${JSON.stringify(finalCounts)}\n`);
