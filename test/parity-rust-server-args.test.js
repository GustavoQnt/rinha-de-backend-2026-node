import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { selectParityEntries } from '../scripts/parity-rust-server-selection.js';

const entries = [
  { request: { id: 'tx-a' }, expected_fraud_score: 0.0 },
  { request: { id: 'tx-b' }, expected_fraud_score: 0.6 },
  { request: { id: 'tx-c' }, expected_fraud_score: 0.2 },
  { request: { id: 'tx-d' }, expected_fraud_score: 0.4 },
  { request: { id: 'tx-e' }, expected_fraud_score: 0.8 },
  { request: { id: 'tx-f' }, expected_fraud_score: 0.6 },
];

describe('parity-rust-server selection', () => {
  it('selects edge cases before applying max test count', () => {
    const selected = selectParityEntries(entries, { edgeOnly: true, maxTests: 2 });
    assert.deepEqual(selected.map((entry) => entry.request.id), ['tx-b', 'tx-d']);
  });
});
