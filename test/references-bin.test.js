import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import zlib from 'node:zlib';
import {
  DEFAULT_SCALE,
  DIM,
  HEADER_BYTES,
  buildReferencesBinFromGzip,
  buildReferencesBinFromRecords,
  quantizeValue,
  readReferencesBinHeader,
} from '../scripts/build-references-bin.js';

const tmpDir = path.join(process.cwd(), 'temporary-results', 'references-bin-test');

function cleanTmp() {
  fs.rmSync(tmpDir, { recursive: true, force: true });
  fs.mkdirSync(tmpDir, { recursive: true });
}

test('quantizeValue maps normalized dimensions to signed int16', () => {
  assert.equal(quantizeValue(0), 0);
  assert.equal(quantizeValue(0.12345), 1235);
  assert.equal(quantizeValue(1), DEFAULT_SCALE);
  assert.equal(quantizeValue(-1), -DEFAULT_SCALE);
});

test('buildReferencesBinFromRecords writes stable R26B layout', async () => {
  cleanTmp();

  const outputPath = path.join(tmpDir, 'references.bin');
  const metaPath = path.join(tmpDir, 'references.meta.json');
  const records = [
    {
      vector: [0, 0.5, -1, 1, 0.0001, 0.9999, 0.1234, 0.9876, 0.3333, 1, 0, 0.75, 0.25, 0.6],
      label: 'legit',
    },
    {
      vector: [1, 0, 0.4321, -1, 0.5555, 0.6666, 0.7777, 0.8888, 0.9999, 0, 1, 0.1111, 0.2222, 0.4444],
      label: 'fraud',
    },
  ];

  const meta = await buildReferencesBinFromRecords(records, { outputPath, metaPath, source: 'unit-test' });
  const header = readReferencesBinHeader(outputPath);

  assert.deepEqual(header, {
    magic: 'R26B',
    version: 1,
    count: 2,
    dim: DIM,
    scale: DEFAULT_SCALE,
    reserved: 0,
    bytes: HEADER_BYTES + records.length * DIM * 2 + records.length,
  });
  assert.equal(meta.bytes, header.bytes);
  assert.equal(meta.labels.fraud, 1);
  assert.equal(meta.labels.legit, 1);

  const bytes = fs.readFileSync(outputPath);
  const firstVector = [];
  for (let i = 0; i < DIM; i += 1) {
    firstVector.push(bytes.readInt16LE(HEADER_BYTES + i * 2));
  }
  assert.deepEqual(firstVector, [0, 5000, -10000, 10000, 1, 9999, 1234, 9876, 3333, 10000, 0, 7500, 2500, 6000]);

  const labelsOffset = HEADER_BYTES + records.length * DIM * 2;
  assert.equal(bytes[labelsOffset], 0);
  assert.equal(bytes[labelsOffset + 1], 1);
});

test('buildReferencesBinFromGzip reserves header bytes before streamed vectors', async () => {
  cleanTmp();

  const source = path.join(tmpDir, 'references.json.gz');
  const outputPath = path.join(tmpDir, 'streamed.bin');
  const records = [
    {
      vector: [0, 0.5, -1, 1, 0.0001, 0.9999, 0.1234, 0.9876, 0.3333, 1, 0, 0.75, 0.25, 0.6],
      label: 'legit',
    },
  ];
  fs.writeFileSync(source, zlib.gzipSync(JSON.stringify(records)));

  await buildReferencesBinFromGzip({
    source,
    outputPath,
    metaPath: null,
    scale: DEFAULT_SCALE,
    expectedCount: 1,
  });

  const header = readReferencesBinHeader(outputPath);
  assert.equal(header.bytes, HEADER_BYTES + DIM * 2 + 1);

  const bytes = fs.readFileSync(outputPath);
  assert.equal(bytes.toString('ascii', 0, 4), 'R26B');
  assert.equal(bytes.readInt16LE(HEADER_BYTES), 0);
  assert.equal(bytes.readInt16LE(HEADER_BYTES + 2), 5000);
  assert.equal(bytes[HEADER_BYTES + DIM * 2], 0);
});
