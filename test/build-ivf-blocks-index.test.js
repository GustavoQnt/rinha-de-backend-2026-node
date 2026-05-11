import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { buildIvfBlocksIndex } from '../scripts/build-ivf-blocks-index.js';

const DIM = 14;
const BLOCK_SIZE = 8;
const HEADER_BYTES = 32;
const PAD_LANES_I16 = 0x7FFF;
const PAD_ORIGIN_U32 = 0xFFFFFFFF;
const PAD_LABEL_U8 = 0xFF;

function writeSyntheticReferences(filePath, vectors, labels, scale = 10000) {
  const count = vectors.length;
  const headerBytes = 24;
  const buf = Buffer.alloc(headerBytes + count * DIM * 2 + count);
  buf.write('R26B', 0, 'ascii');
  buf.writeUInt32LE(1, 4);
  buf.writeUInt32LE(count, 8);
  buf.writeUInt32LE(DIM, 12);
  buf.writeUInt32LE(scale, 16);

  let pos = headerBytes;
  for (const vector of vectors) {
    assert.equal(vector.length, DIM);
    for (const value of vector) {
      buf.writeInt16LE(value, pos);
      pos += 2;
    }
  }
  for (let i = 0; i < labels.length; i += 1) {
    buf[pos + i] = labels[i];
  }
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, buf);
}

function makeVec(firstDim) {
  const v = new Array(DIM).fill(0);
  v[0] = firstDim;
  return v;
}

test('R26I v2 header and section sizes match meta', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ivf-blocks-'));
  const input = path.join(tmp, 'references.bin');
  const output = path.join(tmp, 'references.ivf-blocks.bin');
  const metaPath = path.join(tmp, 'references.ivf-blocks.meta.json');

  // 9 refs across 2 distinct regions; with K=2 we expect 1 cluster of 4 and 1 of 5.
  // Block_size=8, so ceil(4/8)=1 and ceil(5/8)=1, total_blocks=2, padded_slots=16-9=7.
  writeSyntheticReferences(input, [
    makeVec(-1000), makeVec(-950), makeVec(-900), makeVec(-880),
    makeVec(900), makeVec(950), makeVec(1000), makeVec(1050), makeVec(1100),
  ], [0, 0, 0, 0, 1, 1, 1, 1, 1]);

  const meta = buildIvfBlocksIndex({
    K: 2,
    input,
    output,
    metaPath,
    centroidStrategy: 'kmeans',
    iterations: 6,
    seed: 42,
    progressEvery: 0,
  });

  assert.equal(meta.format, 'R26I');
  assert.equal(meta.version, 2);
  assert.equal(meta.K, 2);
  assert.equal(meta.dim, DIM);
  assert.equal(meta.count, 9);
  assert.equal(meta.block_size, 8);
  assert.equal(meta.total_blocks, 2);
  assert.equal(meta.padded_slots, 7);

  const bytes = fs.readFileSync(output);
  assert.equal(bytes.toString('ascii', 0, 4), 'R26I');
  assert.equal(bytes.readUInt32LE(4), 2);
  assert.equal(bytes.readUInt32LE(8), 2);
  assert.equal(bytes.readUInt32LE(12), DIM);
  assert.equal(bytes.readUInt32LE(20), 9);
  assert.equal(bytes.readUInt32LE(24), 8);
  assert.equal(bytes.readUInt32LE(28), 2);

  // Section size check.
  const expectedSize =
    HEADER_BYTES +
    meta.K * DIM * 2 +                 // centroids
    (meta.K + 1) * 4 +                 // block_offsets
    meta.total_blocks * 112 * 2 +      // block_vectors
    meta.total_blocks * 8 * 4 +        // block_origin
    meta.total_blocks * 8 +            // block_labels
    meta.count;                         // labels_by_origin
  assert.equal(bytes.length, expectedSize);
});

test('block-SoA layout: real slots match permuted refs, padded slots use sentinels', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ivf-blocks-soa-'));
  const input = path.join(tmp, 'references.bin');
  const output = path.join(tmp, 'references.ivf-blocks.bin');

  // 5 refs all in one cluster (K=1) -> 1 block, 5 real slots, 3 padded.
  // Use distinguishable vectors so we can verify SoA placement.
  const refs = [
    [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
    [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214],
    [301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314],
    [401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414],
    [501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514],
  ];
  writeSyntheticReferences(input, refs, [0, 1, 0, 1, 1]);

  buildIvfBlocksIndex({
    K: 1,
    input,
    output,
    metaPath: null,
    centroidStrategy: 'first',
    iterations: 0,
    seed: 1,
    progressEvery: 0,
  });

  const bytes = fs.readFileSync(output);
  // Skip header (32) + centroids (1×14×2 = 28) + block_offsets (2×4 = 8) = 68
  const blocksStart = HEADER_BYTES + 1 * DIM * 2 + (1 + 1) * 4;
  // Section sizes
  const totalBlocks = 1;
  const blockVectorBytes = totalBlocks * 112 * 2;
  const blockOriginStart = blocksStart + blockVectorBytes;
  const blockLabelsStart = blockOriginStart + totalBlocks * 8 * 4;
  const labelsByOriginStart = blockLabelsStart + totalBlocks * 8;

  // Verify SoA layout: dim d, slot s -> offset = d * 8 + s within the block.
  const blockBase = blocksStart;
  for (let s = 0; s < 5; s += 1) {
    for (let d = 0; d < DIM; d += 1) {
      const off = blockBase + (d * 8 + s) * 2;
      assert.equal(
        bytes.readInt16LE(off),
        refs[s][d],
        `slot ${s} dim ${d}`
      );
    }
  }
  // Padded slots 5,6,7 must be PAD_LANES_I16 in every dim.
  for (let s = 5; s < 8; s += 1) {
    for (let d = 0; d < DIM; d += 1) {
      const off = blockBase + (d * 8 + s) * 2;
      assert.equal(
        bytes.readInt16LE(off),
        PAD_LANES_I16 < 0x8000 ? PAD_LANES_I16 : PAD_LANES_I16 - 0x10000,
        `padded slot ${s} dim ${d}`
      );
    }
  }

  // Origin: 5 real (0..4), 3 padded (u32::MAX).
  for (let s = 0; s < 5; s += 1) {
    assert.equal(bytes.readUInt32LE(blockOriginStart + s * 4), s);
  }
  for (let s = 5; s < 8; s += 1) {
    assert.equal(bytes.readUInt32LE(blockOriginStart + s * 4), PAD_ORIGIN_U32);
  }

  // Labels per slot: 5 real, 3 padded (0xFF).
  const slotLabels = [0, 1, 0, 1, 1];
  for (let s = 0; s < 5; s += 1) {
    assert.equal(bytes[blockLabelsStart + s], slotLabels[s]);
  }
  for (let s = 5; s < 8; s += 1) {
    assert.equal(bytes[blockLabelsStart + s], PAD_LABEL_U8);
  }

  // labels_by_origin: index by original ref position.
  for (let i = 0; i < 5; i += 1) {
    assert.equal(bytes[labelsByOriginStart + i], slotLabels[i]);
  }
});
