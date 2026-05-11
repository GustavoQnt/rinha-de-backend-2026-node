import fs from 'node:fs';
import { classifyFraudBucket } from '../src/classifier.js';

const data = JSON.parse(fs.readFileSync('test/test-data.json', 'utf8'));
const requests = data.entries.map((e) => e.request);

function warm(iterations) {
  let sink = 0;
  for (let i = 0; i < iterations; i += 1) {
    sink += classifyFraudBucket(requests[i % requests.length]);
  }
  return sink;
}

function measure(iterations) {
  const samples = new Float64Array(iterations);
  let sink = 0;
  for (let i = 0; i < iterations; i += 1) {
    const req = requests[i % requests.length];
    const t0 = process.hrtime.bigint();
    sink += classifyFraudBucket(req);
    const t1 = process.hrtime.bigint();
    samples[i] = Number(t1 - t0);
  }
  return { samples, sink };
}

function summarize(samples) {
  const sorted = Array.from(samples).sort((a, b) => a - b);
  const n = sorted.length;
  const sum = sorted.reduce((a, b) => a + b, 0);
  return {
    n,
    mean_ns: +(sum / n).toFixed(2),
    p50_ns: +sorted[Math.floor(n * 0.5)].toFixed(2),
    p95_ns: +sorted[Math.floor(n * 0.95)].toFixed(2),
    p99_ns: +sorted[Math.floor(n * 0.99)].toFixed(2),
    p999_ns: +sorted[Math.floor(n * 0.999)].toFixed(2),
    max_ns: +sorted[n - 1].toFixed(2),
  };
}

console.log(`Warming with ${requests.length} unique requests...`);
warm(200000);

console.log('Measuring 1,000,000 calls...');
const { samples, sink } = measure(1_000_000);

const stats = summarize(samples);
console.log(JSON.stringify({ ...stats, sink_check: sink % 7 }, null, 2));

const totalUs = (samples.reduce((a, b) => a + b, 0)) / 1000;
console.log(`Total CPU time: ${totalUs.toFixed(0)}us across ${samples.length} calls`);
console.log(`Throughput: ${Math.round(samples.length / (totalUs / 1_000_000))} calls/sec single-thread`);
