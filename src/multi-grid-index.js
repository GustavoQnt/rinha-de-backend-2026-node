import { buildGridIndex, candidateIndexesForVector, GRID_FEATURES } from './grid-index.js';

export const MULTI_GRID_FEATURE_SETS = [
  GRID_FEATURES,
  [
    { dim: 0, bins: 8 },
    { dim: 1, bins: 4 },
    { dim: 2, bins: 8 },
    { dim: 3, bins: 4 },
    { dim: 4, bins: 4 },
  ],
  [
    { dim: 0, bins: 8 },
    { dim: 2, bins: 8 },
    { dim: 7, bins: 8 },
    { dim: 8, bins: 8 },
    { dim: 12, bins: 4 },
  ],
  [
    { dim: 5, bins: 4, sentinel: true },
    { dim: 6, bins: 4, sentinel: true },
    { dim: 7, bins: 8 },
    { dim: 12, bins: 4 },
    { dim: 13, bins: 4 },
  ],
  [
    { dim: 9, bins: 2, fixed: true },
    { dim: 10, bins: 2, fixed: true },
    { dim: 11, bins: 2, fixed: true },
    { dim: 12, bins: 4 },
  ],
];

const FLAG_FIXED = 1;
const FLAG_SENTINEL = 2;

export function packFeaturesForNative(features) {
  const out = new Uint32Array(features.length * 4);
  for (let i = 0; i < features.length; i += 1) {
    const f = features[i];
    let flags = 0;
    if (f.fixed) flags |= FLAG_FIXED;
    if (f.sentinel) flags |= FLAG_SENTINEL;
    const radix = f.sentinel ? f.bins + 1 : f.bins;
    out[i * 4] = f.dim;
    out[i * 4 + 1] = f.bins;
    out[i * 4 + 2] = flags;
    out[i * 4 + 3] = radix;
  }
  return out;
}

export function packMultiGridForNative(multiIndex) {
  return {
    bucketTables: multiIndex.indexes.map((g) => g.bucketTable),
    postingsList: multiIndex.indexes.map((g) => g.postings),
    featuresList: multiIndex.featureSets.map((features) => packFeaturesForNative(features)),
  };
}

export function buildMultiGridIndex(refs, featureSets = MULTI_GRID_FEATURE_SETS) {
  return {
    count: refs.count,
    scale: refs.scale,
    featureSets,
    indexes: featureSets.map((features) => buildGridIndex(refs, features)),
  };
}

export function candidateGroupsForMultiGrid(index, vector, options = {}) {
  const perGridMinCandidates = options.perGridMinCandidates ?? options.minCandidates ?? 10_000;
  const maxRadius = options.maxRadius ?? 1;
  const groups = new Array(index.indexes.length);
  const gridResults = new Array(index.indexes.length);
  let totalBeforeDedupe = 0;

  for (let i = 0; i < index.indexes.length; i += 1) {
    const result = candidateIndexesForVector(index.indexes[i], vector, {
      minCandidates: perGridMinCandidates,
      maxRadius,
    });
    const arr = new Uint32Array(result.indexes.length);
    for (let j = 0; j < result.indexes.length; j += 1) arr[j] = result.indexes[j];
    groups[i] = arr;
    gridResults[i] = {
      candidateCount: result.indexes.length,
      radius: result.radius,
      keysVisited: result.keysVisited,
    };
    totalBeforeDedupe += result.indexes.length;
  }

  return { groups, gridResults, totalBeforeDedupe };
}

export function candidateIndexesForMultiGrid(index, vector, options = {}) {
  const perGridMinCandidates = options.perGridMinCandidates ?? options.minCandidates ?? 10_000;
  const maxRadius = options.maxRadius ?? 1;
  const sortCandidates = options.sortCandidates ?? true;
  const seen = new Set();
  const indexes = [];
  const gridResults = [];
  let totalBeforeDedupe = 0;

  for (const grid of index.indexes) {
    const result = candidateIndexesForVector(grid, vector, {
      minCandidates: perGridMinCandidates,
      maxRadius,
    });
    totalBeforeDedupe += result.indexes.length;
    gridResults.push({
      candidateCount: result.indexes.length,
      radius: result.radius,
      keysVisited: result.keysVisited,
    });

    for (const candidate of result.indexes) {
      if (seen.has(candidate)) continue;
      seen.add(candidate);
      indexes.push(candidate);
    }
  }

  if (sortCandidates) indexes.sort((a, b) => a - b);

  return {
    indexes,
    gridResults,
    totalBeforeDedupe,
  };
}
