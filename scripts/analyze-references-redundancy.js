/**
 * Phase 1 redundancy analysis on resources/references.bin (R26B v1).
 *
 * Reports:
 *   - exact-duplicate vectors (same 14D i16 tuple)
 *   - label conflicts on the same vector (fraud vs legit)
 *   - fraud/legit distribution overall and on the unique-vector set
 *   - duplication histogram (how many vectors appear N times)
 *   - top-N most duplicated vectors (sanity)
 *   - reduction headroom from pure dedup (keep 1 representative per unique vector)
 *
 * Does NOT touch test/test-data.json — pure analysis of the reference dataset.
 *
 * Usage:
 *   node scripts/analyze-references-redundancy.js [--input resources/references.bin] [--top 10]
 */

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const MAGIC_REFS = 'R26B';
const REFS_HEADER_BYTES = 24;
const DIM = 14;
const VECTOR_BYTES = DIM * 2;

function parseArgs(argv) {
  const opts = { input: 'resources/references.bin', top: 10 };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '--input') opts.input = argv[++i];
    else if (a === '--top') opts.top = Number(argv[++i]);
    else throw new Error(`unknown argument: ${a}`);
  }
  return opts;
}

function readHeader(buf) {
  if (buf.length < REFS_HEADER_BYTES) throw new Error('references.bin too small');
  const magic = buf.toString('ascii', 0, 4);
  if (magic !== MAGIC_REFS) throw new Error(`bad magic: ${magic}`);
  const version = buf.readUInt32LE(4);
  const count = buf.readUInt32LE(8);
  const dim = buf.readUInt32LE(12);
  const scale = buf.readUInt32LE(16);
  if (dim !== DIM) throw new Error(`unexpected dim: ${dim}`);
  return { version, count, dim, scale };
}

function bucket(n) {
  if (n === 1) return '1';
  if (n === 2) return '2';
  if (n <= 5) return '3-5';
  if (n <= 10) return '6-10';
  if (n <= 100) return '11-100';
  if (n <= 1000) return '101-1000';
  return '1001+';
}

function decodeVector(buf, off) {
  const v = new Array(DIM);
  for (let j = 0; j < DIM; j += 1) v[j] = buf.readInt16LE(off + j * 2);
  return v;
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  const input = path.resolve(process.cwd(), opts.input);
  const t0 = Date.now();
  console.error(`reading ${input} ...`);
  const buf = fs.readFileSync(input);
  const { version, count, dim, scale } = readHeader(buf);
  const vectorsStart = REFS_HEADER_BYTES;
  const labelsStart = vectorsStart + count * VECTOR_BYTES;
  if (buf.length !== labelsStart + count) throw new Error('unexpected references.bin size');
  console.error(`refs: count=${count} dim=${dim} scale=${scale} version=${version}`);

  // Map: vector key (28-byte latin1 string) -> { count, fraud, legit, firstIdx }
  // 28 bytes per key; latin1 keeps bytes 1:1, faster than hex/base64.
  const map = new Map();
  let totalFraud = 0;
  let totalLegit = 0;

  const progressEvery = 250_000;
  for (let i = 0; i < count; i += 1) {
    const off = vectorsStart + i * VECTOR_BYTES;
    const key = buf.toString('latin1', off, off + VECTOR_BYTES);
    const label = buf[labelsStart + i];
    if (label === 1) totalFraud += 1; else totalLegit += 1;

    const entry = map.get(key);
    if (entry === undefined) {
      map.set(key, { count: 1, fraud: label === 1 ? 1 : 0, legit: label === 0 ? 1 : 0, firstIdx: i });
    } else {
      entry.count += 1;
      if (label === 1) entry.fraud += 1; else entry.legit += 1;
    }

    if ((i + 1) % progressEvery === 0) {
      const seconds = ((Date.now() - t0) / 1000).toFixed(1);
      console.error(`scanned ${i + 1}/${count} unique=${map.size} in ${seconds}s`);
    }
  }

  // Aggregate.
  const histogram = new Map();
  let labelConflicts = 0;
  let uniqueFraud = 0;
  let uniqueLegit = 0;
  let uniqueTotal = 0;
  const top = []; // top-N by count
  const topN = opts.top;

  for (const [, entry] of map) {
    uniqueTotal += 1;
    const b = bucket(entry.count);
    histogram.set(b, (histogram.get(b) || 0) + 1);
    if (entry.fraud > 0 && entry.legit > 0) labelConflicts += 1;
    // Majority label of unique entry: tie -> count as fraud (matches knn tie-break? not relevant here, just stats)
    if (entry.fraud >= entry.legit) uniqueFraud += 1; else uniqueLegit += 1;

    // Track top-N most duplicated.
    if (top.length < topN) {
      top.push(entry);
      top.sort((a, b) => b.count - a.count);
    } else if (entry.count > top[top.length - 1].count) {
      top[top.length - 1] = entry;
      top.sort((a, b) => b.count - a.count);
    }
  }

  const seconds = ((Date.now() - t0) / 1000).toFixed(1);
  console.error(`scan done in ${seconds}s`);
  console.error('');

  const summary = {
    input: opts.input,
    total_count: count,
    unique_count: uniqueTotal,
    dedup_ratio: +(count / uniqueTotal).toFixed(4),
    reduction_factor: +(uniqueTotal / count).toFixed(6),
    fraud_count_total: totalFraud,
    legit_count_total: totalLegit,
    fraud_rate_total: +(totalFraud / count).toFixed(6),
    fraud_count_unique_majority: uniqueFraud,
    legit_count_unique_majority: uniqueLegit,
    fraud_rate_unique: +(uniqueFraud / uniqueTotal).toFixed(6),
    label_conflicts_count: labelConflicts,
    label_conflicts_rate: +(labelConflicts / uniqueTotal).toFixed(6),
    duplication_histogram: Object.fromEntries(
      ['1', '2', '3-5', '6-10', '11-100', '101-1000', '1001+']
        .map((k) => [k, histogram.get(k) || 0])
    ),
    top_duplicates: top.map((e) => ({
      count: e.count,
      fraud: e.fraud,
      legit: e.legit,
      first_idx: e.firstIdx,
      vector: decodeVector(buf, vectorsStart + e.firstIdx * VECTOR_BYTES),
    })),
  };

  console.log(JSON.stringify(summary, null, 2));
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  try { main(); } catch (err) { console.error(err); process.exitCode = 1; }
}
