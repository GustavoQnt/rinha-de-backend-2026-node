/**
 * Parity check: compare Rust server output against Node oracle on test-data.json.
 *
 * Usage:
 *   node scripts/parity-rust-server.js [--tests N] [--port PORT] [--edge] [--expected] [--dry-run] [--spawn] [--build]
 *                                     [--ivf resources/references.ivf.bin] [--nprobe 64]
 *
 * By default, expects the Rust server already running. With --spawn, starts the
 * release binary and sets REFS_PATH so it loads the same references.bin as the
 * Node oracle. With --build, runs cargo build --release first.
 */

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { loadBinaryReferences, predictBinaryKnnBucket } from '../src/knn-classifier.js';
import { vectorize } from '../src/vectorizer.js';
import { compareRustResult } from './parity-rust-server-compare.js';
import { selectParityEntries } from './parity-rust-server-selection.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');

// ---------- CLI args ----------
const args = process.argv.slice(2);
let maxTests = Infinity;
let port = 8080;
let shouldSpawn = false;
let shouldBuild = false;
let edgeOnly = false;
let dryRun = false;
let comparisonMode = 'oracle';
let ivfPath = null;
let ivfNprobe = null;
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--tests' && args[i + 1]) maxTests = parseInt(args[++i], 10);
  if (args[i] === '--port' && args[i + 1]) port = parseInt(args[++i], 10);
  if (args[i] === '--spawn') shouldSpawn = true;
  if (args[i] === '--build') shouldBuild = true;
  if (args[i] === '--edge') edgeOnly = true;
  if (args[i] === '--dry-run') dryRun = true;
  if (args[i] === '--expected') comparisonMode = 'expected';
  if (args[i] === '--ivf' && args[i + 1]) ivfPath = path.resolve(ROOT, args[++i]);
  if (args[i] === '--nprobe' && args[i + 1]) ivfNprobe = parseInt(args[++i], 10);
}

const refsPath = path.join(ROOT, 'resources', 'references.bin');
const manifestPath = path.join(ROOT, 'native', 'rinha-server', 'Cargo.toml');
const binPath = path.join(
  ROOT,
  'native',
  'rinha-server',
  'target',
  'release',
  process.platform === 'win32' ? 'rinha-server.exe' : 'rinha-server',
);

function getReady() {
  return new Promise((resolve, reject) => {
    const req = http.request(
      { hostname: '127.0.0.1', port, path: '/ready', method: 'GET', timeout: 1000 },
      (res) => {
        res.resume();
        res.on('end', () => {
          if (res.statusCode === 200) resolve();
          else reject(new Error(`/ready returned HTTP ${res.statusCode}`));
        });
      },
    );
    req.on('timeout', () => {
      req.destroy(new Error('/ready timeout'));
    });
    req.on('error', reject);
    req.end();
  });
}

async function waitForReady(deadlineMs = 15000) {
  const deadline = Date.now() + deadlineMs;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      await getReady();
      return;
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  throw lastError ?? new Error('server did not become ready');
}

if (shouldBuild && !dryRun) {
  const result = spawnSync(
    'cargo',
    ['build', '--release', '--manifest-path', manifestPath],
    { cwd: ROOT, stdio: 'inherit' },
  );
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

let child = null;
function cleanupChild() {
  if (child && !child.killed) {
    child.kill();
  }
}

if (shouldSpawn && !dryRun) {
  const childEnv = { ...process.env, PORT: String(port), REFS_PATH: refsPath };
  if (ivfPath) childEnv.IVF_PATH = ivfPath;
  if (ivfNprobe != null) childEnv.IVF_NPROBE = String(ivfNprobe);

  child = spawn(binPath, [], {
    cwd: ROOT,
    env: childEnv,
    stdio: ['ignore', 'ignore', 'pipe'],
  });

  child.stderr.on('data', (chunk) => {
    process.stderr.write(`[rust] ${chunk}`);
  });

  process.on('exit', cleanupChild);
  process.on('SIGINT', () => {
    cleanupChild();
    process.exit(130);
  });
  process.on('SIGTERM', () => {
    cleanupChild();
    process.exit(143);
  });

  await waitForReady();
}

// ---------- Load test data ----------
const testDataPath = path.join(ROOT, 'test', 'test-data.json');
const testData = JSON.parse(fs.readFileSync(testDataPath, 'utf8'));
const entries = selectParityEntries(testData.entries, { edgeOnly, maxTests });
console.log(`testing ${entries.length} entries against http://localhost:${port}\n`);

if (dryRun) {
  for (const entry of entries) {
    console.log(`${entry.request.id} ${entry.expected_fraud_score}`);
  }
  cleanupChild();
  process.exit(0);
}

// ---------- Load Node oracle ----------
let refs = null;
if (comparisonMode === 'oracle') {
  refs = loadBinaryReferences(refsPath);
  console.log(`oracle refs: count=${refs.count}, scale=${refs.scale}`);
} else {
  console.log('comparison: expected fixture scores');
}

// ---------- HTTP helper ----------
function postFraudScore(payload) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload);
    const req = http.request(
      { hostname: '127.0.0.1', port, path: '/fraud-score', method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } },
      (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          if (res.statusCode !== 200) {
            reject(new Error(`HTTP ${res.statusCode}: ${data}`));
            return;
          }
          try { resolve(JSON.parse(data)); }
          catch (e) { reject(new Error(`bad JSON: ${data}`)); }
        });
      }
    );
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// ---------- Run parity ----------
let pass = 0;
let fail = 0;
const mismatches = [];

for (const entry of entries) {
  const { request, expected_fraud_score } = entry;

  // Rust server
  let rustResult;
  try {
    rustResult = await postFraudScore(request);
  } catch (e) {
    console.error(`  FAIL ${request.id}: request error: ${e.message}`);
    fail++;
    mismatches.push({ id: request.id, error: e.message });
    continue;
  }

  const comparison = compareRustResult(entry, rustResult, {
    mode: comparisonMode,
    getOracleBucket(oracleRequest) {
      const vec = vectorize(oracleRequest);
      return predictBinaryKnnBucket(refs, vec).bucket;
    },
  });

  if (comparison.ok) {
    pass++;
  } else {
    fail++;
    const detail = {
      id: request.id,
      expected_fraud_score,
      expected_score: comparison.expectedScore,
      rust_score: rustResult.fraud_score,
      rust_approved: rustResult.approved,
    };
    mismatches.push(detail);
    if (mismatches.length <= 10) {
      console.error(`  MISMATCH ${request.id}: expected=${comparison.expectedScore} rust=${rustResult.fraud_score}`);
    }
  }
}

console.log(`\nresults: ${pass}/${entries.length} pass, ${fail} fail`);

if (mismatches.length > 0) {
  console.error(`\nfirst mismatches:`);
  for (const m of mismatches.slice(0, 20)) console.error(' ', JSON.stringify(m));
  cleanupChild();
  process.exit(1);
} else {
  console.log('PARITY OK');
  cleanupChild();
}
