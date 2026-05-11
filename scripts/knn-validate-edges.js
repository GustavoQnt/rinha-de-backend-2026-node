import fs from 'node:fs';
import zlib from 'node:zlib';
import { vectorize } from '../src/vectorizer.js';

console.log('Loading refs...');
let t0 = Date.now();
const refsRaw = JSON.parse(zlib.gunzipSync(fs.readFileSync('resources/references.json.gz')).toString('utf8'));
console.log(`  ${refsRaw.length} refs in ${Date.now() - t0}ms`);

const N_REFS = refsRaw.length;
const refVecs = new Float64Array(N_REFS * 14);
const refLabels = new Uint8Array(N_REFS);
for (let i = 0; i < N_REFS; i += 1) {
  const r = refsRaw[i];
  for (let j = 0; j < 14; j += 1) refVecs[i * 14 + j] = r.vector[j];
  refLabels[i] = r.label === 'fraud' ? 1 : 0;
}

const data = JSON.parse(fs.readFileSync('test/test-data.json', 'utf8'));

// Filter edge cases: expected_fraud_score in {0.4, 0.6}
const edges = data.entries.filter((e) => e.expected_fraud_score === 0.4 || e.expected_fraud_score === 0.6);
console.log(`${edges.length} edge entries`);

function knn(query) {
  const dists = [Infinity, Infinity, Infinity, Infinity, Infinity];
  const idxs = [-1, -1, -1, -1, -1];
  for (let i = 0; i < N_REFS; i += 1) {
    const base = i * 14;
    let d = 0;
    for (let j = 0; j < 14; j += 1) {
      const diff = query[j] - refVecs[base + j];
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
  let f = 0;
  for (let k = 0; k < 5; k += 1) if (refLabels[idxs[k]] === 1) f += 1;
  const score = Math.round((f / 5) * 10000) / 10000;
  return { score, approved: score < 0.6, idxs, dists };
}

let appM = 0, appX = 0, scM = 0, scX = 0;
const tStart = Date.now();
for (let i = 0; i < edges.length; i += 1) {
  const e = edges[i];
  const q = vectorize(e.request);
  const r = knn(q);
  if (r.approved === e.expected_approved) appM += 1; else appX += 1;
  if (r.score === e.expected_fraud_score) scM += 1; else scX += 1;

  if (appX <= 5 && r.approved !== e.expected_approved) {
    console.log(`  MISMATCH [${i}] ${e.request.id}: expected approved=${e.expected_approved} score=${e.expected_fraud_score} got approved=${r.approved} score=${r.score}`);
  }
}
const ms = Date.now() - tStart;
console.log(`\nEdge results: approved_match=${appM} mismatch=${appX} score_match=${scM} mismatch=${scX} elapsed=${ms}ms`);
