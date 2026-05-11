import fs from 'node:fs';
import zlib from 'node:zlib';

console.log('Decompressing references.json.gz ...');
const t0 = Date.now();
const buf = fs.readFileSync('resources/references.json.gz');
const json = zlib.gunzipSync(buf).toString('utf8');
console.log(`Decompressed in ${Date.now() - t0}ms, ${json.length} chars`);

const t1 = Date.now();
const refs = JSON.parse(json);
console.log(`Parsed in ${Date.now() - t1}ms`);

console.log(`Total references: ${refs.length}`);

if (refs.length === 0) process.exit(1);

const first = refs[0];
console.log('First record:', JSON.stringify(first));
console.log('Vector dim:', first.vector.length);

const labelCounts = {};
const dimMin = new Array(first.vector.length).fill(Infinity);
const dimMax = new Array(first.vector.length).fill(-Infinity);
const dimNeg1Count = new Array(first.vector.length).fill(0);

for (const r of refs) {
  labelCounts[r.label] = (labelCounts[r.label] || 0) + 1;
  for (let j = 0; j < r.vector.length; j += 1) {
    const v = r.vector[j];
    if (v === -1) dimNeg1Count[j] += 1;
    if (v < dimMin[j]) dimMin[j] = v;
    if (v > dimMax[j]) dimMax[j] = v;
  }
}

console.log('Label counts:', labelCounts);
console.log('Per-dim ranges:');
for (let j = 0; j < dimMin.length; j += 1) {
  console.log(`  dim ${j}: [${dimMin[j]}, ${dimMax[j]}]  neg1=${dimNeg1Count[j]}`);
}
