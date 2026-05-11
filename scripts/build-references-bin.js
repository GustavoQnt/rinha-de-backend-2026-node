import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import zlib from 'node:zlib';

export const MAGIC = 'R26B';
export const VERSION = 1;
export const DIM = 14;
export const DEFAULT_SCALE = 10000;
export const HEADER_BYTES = 24;

const DEFAULT_SOURCE = 'resources/references.json.gz';
const DEFAULT_OUTPUT = 'resources/references.bin';
const DEFAULT_META = 'resources/references.meta.json';
const DEFAULT_EXPECTED_COUNT = 3_000_000;
const VECTOR_BATCH_REFS = 8192;

export function quantizeValue(value, scale = DEFAULT_SCALE) {
  const q = Math.round(value * scale);
  if (q < -32768 || q > 32767) {
    throw new RangeError(`quantized value out of int16 range: value=${value} scale=${scale} q=${q}`);
  }
  return q;
}

export function createReferencesBinHeader({ count, dim = DIM, scale = DEFAULT_SCALE, reserved = 0 }) {
  const header = Buffer.alloc(HEADER_BYTES);
  header.write(MAGIC, 0, 'ascii');
  header.writeUInt32LE(VERSION, 4);
  header.writeUInt32LE(count, 8);
  header.writeUInt32LE(dim, 12);
  header.writeUInt32LE(scale, 16);
  header.writeUInt32LE(reserved, 20);
  return header;
}

export function readReferencesBinHeader(filePath = DEFAULT_OUTPUT) {
  const fd = fs.openSync(filePath, 'r');
  try {
    const header = Buffer.alloc(HEADER_BYTES);
    fs.readSync(fd, header, 0, HEADER_BYTES, 0);
    return {
      magic: header.toString('ascii', 0, 4),
      version: header.readUInt32LE(4),
      count: header.readUInt32LE(8),
      dim: header.readUInt32LE(12),
      scale: header.readUInt32LE(16),
      reserved: header.readUInt32LE(20),
      bytes: fs.statSync(filePath).size,
    };
  } finally {
    fs.closeSync(fd);
  }
}

function labelToByte(label) {
  if (label === 'fraud') return 1;
  if (label === 'legit') return 0;
  throw new Error(`unknown reference label: ${label}`);
}

function writeMetadata(metaPath, meta) {
  if (!metaPath) return;
  fs.mkdirSync(path.dirname(metaPath), { recursive: true });
  fs.writeFileSync(`${metaPath}.tmp`, `${JSON.stringify(meta, null, 2)}\n`);
  fs.renameSync(`${metaPath}.tmp`, metaPath);
}

function createMeta({ source, outputPath, count, scale, fraudCount, legitCount, bytes }) {
  return {
    format: MAGIC,
    version: VERSION,
    dim: DIM,
    scale,
    count,
    labels: {
      fraud: fraudCount,
      legit: legitCount,
    },
    vectors_bytes: count * DIM * 2,
    labels_bytes: count,
    bytes,
    source,
    output: outputPath,
    generated_at: new Date().toISOString(),
  };
}

export async function buildReferencesBinFromRecords(records, options = {}) {
  const outputPath = options.outputPath ?? DEFAULT_OUTPUT;
  const metaPath = options.metaPath ?? DEFAULT_META;
  const source = options.source ?? 'records';
  const scale = options.scale ?? DEFAULT_SCALE;

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  const count = records.length;
  const fd = fs.openSync(`${outputPath}.tmp`, 'w');
  let fraudCount = 0;
  let legitCount = 0;

  try {
    fs.writeSync(fd, createReferencesBinHeader({ count, scale }), 0, HEADER_BYTES, 0);

    const vectorBuffer = Buffer.allocUnsafe(count * DIM * 2);
    let offset = 0;
    const labels = Buffer.allocUnsafe(count);

    for (let i = 0; i < count; i += 1) {
      const record = records[i];
      if (!Array.isArray(record.vector) || record.vector.length !== DIM) {
        throw new Error(`record ${i} has invalid vector length`);
      }

      for (let j = 0; j < DIM; j += 1) {
        vectorBuffer.writeInt16LE(quantizeValue(record.vector[j], scale), offset);
        offset += 2;
      }

      const label = labelToByte(record.label);
      labels[i] = label;
      if (label === 1) fraudCount += 1;
      else legitCount += 1;
    }

    fs.writeSync(fd, vectorBuffer, 0, vectorBuffer.length, HEADER_BYTES);
    fs.writeSync(fd, labels, 0, labels.length, HEADER_BYTES + vectorBuffer.length);
  } finally {
    fs.closeSync(fd);
  }

  fs.renameSync(`${outputPath}.tmp`, outputPath);
  const bytes = fs.statSync(outputPath).size;
  const meta = createMeta({ source, outputPath, count, scale, fraudCount, legitCount, bytes });
  writeMetadata(metaPath, meta);
  return meta;
}

function parseArgs(argv) {
  const opts = {
    source: DEFAULT_SOURCE,
    outputPath: DEFAULT_OUTPUT,
    metaPath: DEFAULT_META,
    scale: DEFAULT_SCALE,
    expectedCount: DEFAULT_EXPECTED_COUNT,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--source') opts.source = argv[++i];
    else if (arg === '--output') opts.outputPath = argv[++i];
    else if (arg === '--meta') opts.metaPath = argv[++i];
    else if (arg === '--scale') opts.scale = Number(argv[++i]);
    else if (arg === '--expected-count') opts.expectedCount = Number(argv[++i]);
    else if (arg === '--no-meta') opts.metaPath = null;
    else if (arg === '--help') opts.help = true;
    else throw new Error(`unknown argument: ${arg}`);
  }

  if (!Number.isInteger(opts.scale) || opts.scale <= 0) {
    throw new Error(`invalid --scale: ${opts.scale}`);
  }
  if (!Number.isInteger(opts.expectedCount) || opts.expectedCount <= 0) {
    throw new Error(`invalid --expected-count: ${opts.expectedCount}`);
  }
  return opts;
}

function printHelp() {
  console.log(`Usage: node scripts/build-references-bin.js [options]

Options:
  --source <path>          Source references.json.gz (default: ${DEFAULT_SOURCE})
  --output <path>          Output R26B binary (default: ${DEFAULT_OUTPUT})
  --meta <path>            Output metadata JSON (default: ${DEFAULT_META})
  --scale <int>            Quantization scale (default: ${DEFAULT_SCALE})
  --expected-count <int>   Initial label buffer size (default: ${DEFAULT_EXPECTED_COUNT})
  --no-meta                Do not write metadata JSON
`);
}

function growLabels(labels) {
  const next = Buffer.allocUnsafe(labels.length * 2);
  labels.copy(next);
  return next;
}

function parseReferenceVector(vectorText, recordIndex) {
  const parts = vectorText.split(',');
  if (parts.length !== DIM) {
    throw new Error(`record ${recordIndex} has ${parts.length} dims; expected ${DIM}`);
  }
  return parts.map((part) => Number(part));
}

export async function buildReferencesBinFromGzip(options) {
  const { source, outputPath, metaPath, scale, expectedCount } = options;
  const tmpPath = `${outputPath}.tmp`;
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  const fd = fs.openSync(tmpPath, 'w');
  let count = 0;
  let fraudCount = 0;
  let legitCount = 0;
  let labels = Buffer.allocUnsafe(expectedCount);
  const vectorBuffer = Buffer.allocUnsafe(VECTOR_BATCH_REFS * DIM * 2);
  let vectorOffset = 0;

  function flushVectors() {
    if (vectorOffset === 0) return;
    fs.writeSync(fd, vectorBuffer, 0, vectorOffset);
    vectorOffset = 0;
  }

  try {
    fs.writeSync(fd, Buffer.alloc(HEADER_BYTES), 0, HEADER_BYTES);

    const stream = fs.createReadStream(source).pipe(zlib.createGunzip());
    stream.setEncoding('utf8');

    let buffer = '';
    const recordRe = /\{\s*"vector"\s*:\s*\[([^\]]+)\]\s*,\s*"label"\s*:\s*"(fraud|legit)"\s*\}/g;
    const startedAt = Date.now();

    for await (const chunk of stream) {
      buffer += chunk;
      recordRe.lastIndex = 0;
      let lastEnd = 0;
      let match;

      while ((match = recordRe.exec(buffer)) !== null) {
        const vector = parseReferenceVector(match[1], count);
        if (vectorOffset + DIM * 2 > vectorBuffer.length) flushVectors();
        for (let j = 0; j < DIM; j += 1) {
          vectorBuffer.writeInt16LE(quantizeValue(vector[j], scale), vectorOffset);
          vectorOffset += 2;
        }

        if (count >= labels.length) labels = growLabels(labels);
        const label = labelToByte(match[2]);
        labels[count] = label;
        if (label === 1) fraudCount += 1;
        else legitCount += 1;

        count += 1;
        lastEnd = recordRe.lastIndex;

        if (count % 250_000 === 0) {
          const seconds = ((Date.now() - startedAt) / 1000).toFixed(1);
          console.error(`packed ${count} references in ${seconds}s`);
        }
      }

      buffer = buffer.slice(lastEnd);
      if (buffer.length > 2_000_000) {
        throw new Error(`parser buffer grew unexpectedly (${buffer.length} bytes)`);
      }
    }

    if (buffer.trim() !== '' && !/^\]?\s*$/.test(buffer.trim())) {
      throw new Error(`unparsed trailing content after ${count} records`);
    }

    flushVectors();
    fs.writeSync(fd, labels, 0, count);
    fs.writeSync(fd, createReferencesBinHeader({ count, scale }), 0, HEADER_BYTES, 0);
  } finally {
    fs.closeSync(fd);
  }

  fs.renameSync(tmpPath, outputPath);
  const bytes = fs.statSync(outputPath).size;
  const meta = createMeta({ source, outputPath, count, scale, fraudCount, legitCount, bytes });
  writeMetadata(metaPath, meta);
  return meta;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  const meta = await buildReferencesBinFromGzip(options);
  console.log(JSON.stringify(meta, null, 2));
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
