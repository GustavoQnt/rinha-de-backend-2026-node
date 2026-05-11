import fs from 'node:fs';
import { classifyFraud } from '../src/classifier.js';

const data = JSON.parse(fs.readFileSync('test/test-data.json', 'utf8'));

let tp = 0;
let tn = 0;
let fp = 0;
let fn = 0;
let errors = 0;

for (const entry of data.entries) {
  try {
    const result = classifyFraud(entry.request);

    if (result.approved === entry.expected_approved) {
      if (result.approved) tn += 1;
      else tp += 1;
    } else if (result.approved) {
      fn += 1;
    } else {
      fp += 1;
    }
  } catch {
    errors += 1;
  }
}

const total = tp + tn + fp + fn + errors;
const failures = fp + fn + errors;
const weightedErrors = fp + fn * 3 + errors * 5;
const failureRate = failures / total;

const result = {
  total,
  tp,
  tn,
  fp,
  fn,
  errors,
  failures,
  failure_rate: +(failureRate * 100).toFixed(4),
  weighted_errors: weightedErrors,
};

console.log(JSON.stringify(result, null, 2));

if (failureRate > 0.15) {
  process.exitCode = 1;
}
