import fs from 'node:fs';
import { classifyFraudBucket, RESPONSE_BODIES } from '../src/classifier.js';

const data = JSON.parse(fs.readFileSync('test/test-data.json', 'utf8'));
const bodies = data.entries.map((e) => JSON.stringify(e.request));

function summarize(samples) {
  const sorted = Array.from(samples).sort((a, b) => a - b);
  const n = sorted.length;
  const sum = sorted.reduce((a, b) => a + b, 0);
  return {
    n,
    mean_ns: +(sum / n).toFixed(0),
    p50_ns: +sorted[Math.floor(n * 0.5)].toFixed(0),
    p95_ns: +sorted[Math.floor(n * 0.95)].toFixed(0),
    p99_ns: +sorted[Math.floor(n * 0.99)].toFixed(0),
    p999_ns: +sorted[Math.floor(n * 0.999)].toFixed(0),
  };
}

const N = 500_000;

// Warm
let sink = 0;
for (let i = 0; i < 100_000; i += 1) {
  const obj = JSON.parse(bodies[i % bodies.length]);
  sink += classifyFraudBucket(obj);
}

// 1) JSON.parse alone
{
  const t = new Float64Array(N);
  for (let i = 0; i < N; i += 1) {
    const body = bodies[i % bodies.length];
    const t0 = process.hrtime.bigint();
    const obj = JSON.parse(body);
    const t1 = process.hrtime.bigint();
    sink += obj.transaction.amount > 0 ? 1 : 0;
    t[i] = Number(t1 - t0);
  }
  console.log('JSON.parse:', JSON.stringify(summarize(t)));
}

// 2) classify alone (already parsed)
{
  const objs = bodies.slice(0, 10000).map((b) => JSON.parse(b));
  const t = new Float64Array(N);
  for (let i = 0; i < N; i += 1) {
    const obj = objs[i % objs.length];
    const t0 = process.hrtime.bigint();
    sink += classifyFraudBucket(obj);
    const t1 = process.hrtime.bigint();
    t[i] = Number(t1 - t0);
  }
  console.log('classify:  ', JSON.stringify(summarize(t)));
}

// 3) parse + classify + lookup body
{
  const t = new Float64Array(N);
  let bodySink = 0;
  for (let i = 0; i < N; i += 1) {
    const body = bodies[i % bodies.length];
    const t0 = process.hrtime.bigint();
    const obj = JSON.parse(body);
    const bucket = classifyFraudBucket(obj);
    const out = RESPONSE_BODIES[bucket];
    const t1 = process.hrtime.bigint();
    bodySink += out.length;
    t[i] = Number(t1 - t0);
  }
  console.log('full path: ', JSON.stringify(summarize(t)));
  console.log('sink:', sink, bodySink);
}
