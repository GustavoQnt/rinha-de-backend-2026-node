/**
 * Extract one JSON object per line from `test/test-data.json` containing only the
 * `request` field plus the case `id`, so the Rust validator can stream them through
 * `json::parse_request` without re-implementing top-level test-suite parsing.
 *
 * Output: `test/test-data.requests.ndjson` — each line is a `request` object with
 * the original `id` preserved (already inside `request.id`).
 *
 * Re-run only when `test/test-data.json` changes.
 */

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const DEFAULT_INPUT = 'test/test-data.json';
const DEFAULT_OUTPUT = 'test/test-data.requests.ndjson';
const DEFAULT_BUCKETS_OUTPUT = 'test/test-data.expected-buckets.ndjson';

function parseArgs(argv) {
  const opts = { input: DEFAULT_INPUT, output: DEFAULT_OUTPUT, bucketsOutput: DEFAULT_BUCKETS_OUTPUT };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '--input') opts.input = argv[++i];
    else if (a === '--output') opts.output = argv[++i];
    else if (a === '--buckets-output') opts.bucketsOutput = argv[++i];
    else throw new Error(`unknown argument: ${a}`);
  }
  return opts;
}

function bucketFromScore(score) {
  if (score === 0) return 0;
  if (score === 0.2) return 1;
  if (score === 0.4) return 2;
  if (score === 0.6) return 3;
  if (score === 0.8) return 4;
  if (score === 1) return 5;
  throw new Error(`unexpected expected_fraud_score: ${score}`);
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const raw = fs.readFileSync(opts.input, 'utf8');
  const data = JSON.parse(raw);
  if (!Array.isArray(data.entries)) throw new Error('test-data.json missing .entries array');

  fs.mkdirSync(path.dirname(opts.output), { recursive: true });
  fs.mkdirSync(path.dirname(opts.bucketsOutput), { recursive: true });
  const tmp = `${opts.output}.tmp`;
  const bucketsTmp = `${opts.bucketsOutput}.tmp`;
  const fd = fs.openSync(tmp, 'w');
  const bucketsFd = fs.openSync(bucketsTmp, 'w');
  try {
    for (const entry of data.entries) {
      fs.writeSync(fd, JSON.stringify(entry.request) + '\n');
      fs.writeSync(bucketsFd, `${bucketFromScore(entry.expected_fraud_score)}\n`);
    }
  } finally {
    fs.closeSync(fd);
    fs.closeSync(bucketsFd);
  }
  fs.renameSync(tmp, opts.output);
  fs.renameSync(bucketsTmp, opts.bucketsOutput);
  console.error(`wrote ${data.entries.length} requests to ${opts.output}`);
  console.error(`wrote ${data.entries.length} expected buckets to ${opts.bucketsOutput}`);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}
