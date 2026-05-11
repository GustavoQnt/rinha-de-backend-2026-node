import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_ADDON_PATH = path.join(rootDir, 'native', 'knn-addon', 'knn_native.node');

export function loadNativeKnn(addonPath = DEFAULT_ADDON_PATH) {
  return require(addonPath);
}
