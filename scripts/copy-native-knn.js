import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const addonDir = path.join(root, 'native', 'knn-addon');
const releaseDir = path.join(addonDir, 'target', 'release');
const sourceName = process.platform === 'win32'
  ? 'knn_native.dll'
  : process.platform === 'darwin'
    ? 'libknn_native.dylib'
    : 'libknn_native.so';
const source = path.join(releaseDir, sourceName);
const target = path.join(addonDir, 'knn_native.node');

if (!fs.existsSync(source)) {
  throw new Error(`native build output not found: ${source}`);
}

fs.copyFileSync(source, target);
console.log(`copied ${source} -> ${target}`);
