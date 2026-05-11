import fs from 'node:fs';
import { readReferencesBinHeader, HEADER_BYTES } from './build-references-bin.js';

const filePath = process.argv[2] ?? 'resources/references.bin';
const header = readReferencesBinHeader(filePath);

if (header.magic !== 'R26B') {
  throw new Error(`invalid magic: ${header.magic}`);
}
if (header.version !== 1) {
  throw new Error(`unsupported version: ${header.version}`);
}

const expectedBytes = HEADER_BYTES + header.count * header.dim * 2 + header.count;
const labelsOffset = HEADER_BYTES + header.count * header.dim * 2;
const fd = fs.openSync(filePath, 'r');
const labelBuffer = Buffer.allocUnsafe(Math.min(header.count, 1_000_000));
let fraud = 0;
let legit = 0;
let read = 0;

try {
  while (read < header.count) {
    const len = Math.min(labelBuffer.length, header.count - read);
    fs.readSync(fd, labelBuffer, 0, len, labelsOffset + read);
    for (let i = 0; i < len; i += 1) {
      if (labelBuffer[i] === 1) fraud += 1;
      else if (labelBuffer[i] === 0) legit += 1;
      else throw new Error(`invalid label byte at ${read + i}: ${labelBuffer[i]}`);
    }
    read += len;
  }
} finally {
  fs.closeSync(fd);
}

console.log(JSON.stringify({
  ...header,
  expected_bytes: expectedBytes,
  bytes_match: header.bytes === expectedBytes,
  labels: { fraud, legit },
}, null, 2));
