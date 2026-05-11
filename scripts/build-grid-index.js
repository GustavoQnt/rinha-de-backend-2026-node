import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { loadBinaryReferences } from '../src/knn-classifier.js';
import { buildGridIndex, GRID_FEATURES, saveGridIndex } from '../src/grid-index.js';

const DEFAULT_BIN = 'resources/references.bin';
const DEFAULT_OUTPUT = 'resources/references.grid.bin';
const DEFAULT_META = 'resources/references.grid.meta.json';

function parseArgs(argv) {
  const opts = {
    binPath: DEFAULT_BIN,
    outputPath: DEFAULT_OUTPUT,
    metaPath: DEFAULT_META,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--bin') opts.binPath = argv[++i];
    else if (arg === '--output') opts.outputPath = argv[++i];
    else if (arg === '--meta') opts.metaPath = argv[++i];
    else if (arg === '--no-meta') opts.metaPath = null;
    else if (arg === '--help') opts.help = true;
    else throw new Error(`unknown argument: ${arg}`);
  }

  return opts;
}

function printHelp() {
  console.log(`Usage: node scripts/build-grid-index.js [options]

Options:
  --bin <path>       Source R26B references.bin (default: ${DEFAULT_BIN})
  --output <path>    Output R26G grid index (default: ${DEFAULT_OUTPUT})
  --meta <path>      Output metadata JSON (default: ${DEFAULT_META})
  --no-meta          Do not write metadata JSON
`);
}

function writeMeta(metaPath, meta) {
  if (!metaPath) return;
  fs.mkdirSync(path.dirname(metaPath), { recursive: true });
  fs.writeFileSync(`${metaPath}.tmp`, `${JSON.stringify(meta, null, 2)}\n`);
  fs.renameSync(`${metaPath}.tmp`, metaPath);
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    printHelp();
    return;
  }

  const startedAt = Date.now();
  console.error(`Loading references from ${opts.binPath}...`);
  const refs = loadBinaryReferences(opts.binPath);

  console.error(`Building grid index for ${refs.count} references...`);
  const index = buildGridIndex(refs);
  saveGridIndex(index, opts.outputPath);

  const bytes = fs.statSync(opts.outputPath).size;
  const elapsedMs = Date.now() - startedAt;
  const bucketSizes = new Uint32Array(index.bucketCount);
  let maxBucketSize = 0;
  let nonEmptyBuckets = 0;

  for (let i = 0; i < index.bucketCount; i += 1) {
    const length = index.bucketTable[i * 3 + 2];
    bucketSizes[i] = length;
    if (length > 0) nonEmptyBuckets += 1;
    if (length > maxBucketSize) maxBucketSize = length;
  }

  bucketSizes.sort();
  const p95BucketSize = bucketSizes[Math.floor(bucketSizes.length * 0.95)] ?? 0;
  const meta = {
    format: 'R26G',
    version: 1,
    source: opts.binPath,
    output: opts.outputPath,
    count: index.count,
    scale: index.scale,
    features: GRID_FEATURES,
    bucket_count: index.bucketCount,
    posting_count: index.postingCount,
    bytes,
    elapsed_ms: elapsedMs,
    avg_bucket_size: +(index.postingCount / Math.max(1, index.bucketCount)).toFixed(2),
    p95_bucket_size: p95BucketSize,
    max_bucket_size: maxBucketSize,
    non_empty_buckets: nonEmptyBuckets,
    generated_at: new Date().toISOString(),
  };

  writeMeta(opts.metaPath, meta);
  console.log(JSON.stringify(meta, null, 2));
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
