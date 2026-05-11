import { classifyFraudBucket as classifyHeuristicBucket } from './classifier.js';
import { createBinaryKnnClassifier, predictBinaryKnnBucket } from './knn-classifier.js';

const DEFAULT_CLASSIFIER = 'heuristic';
const DEFAULT_REFERENCES_BIN = 'resources/references.bin';

export function createRuntimeClassifier(options = {}) {
  const classifierName = options.classifierName ?? process.env.CLASSIFIER ?? DEFAULT_CLASSIFIER;

  if (classifierName === 'knn-bin') {
    const referencesBinPath = options.referencesBinPath ?? process.env.REFERENCES_BIN_PATH ?? DEFAULT_REFERENCES_BIN;
    const knn = createBinaryKnnClassifier(referencesBinPath, options.vectorize);

    return {
      name: classifierName,
      classifyFraudBucket: knn.classifyFraudBucket,
      predictVectorBucket(vector) {
        return predictBinaryKnnBucket(knn.refs, vector).bucket;
      },
    };
  }

  if (classifierName !== DEFAULT_CLASSIFIER) {
    throw new Error(`unknown classifier: ${classifierName}`);
  }

  return {
    name: DEFAULT_CLASSIFIER,
    classifyFraudBucket: classifyHeuristicBucket,
  };
}
