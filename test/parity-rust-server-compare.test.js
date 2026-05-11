import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { compareRustResult } from '../scripts/parity-rust-server-compare.js';

describe('parity-rust-server comparison', () => {
  it('can compare against expected fixture scores without invoking the Node oracle', () => {
    const entry = {
      request: { id: 'tx-edge' },
      expected_fraud_score: 0.6,
      expected_approved: false,
    };
    const rustResult = { fraud_score: 0.6, approved: false };

    const comparison = compareRustResult(entry, rustResult, {
      mode: 'expected',
      getOracleBucket() {
        throw new Error('oracle should not be called in expected mode');
      },
    });

    assert.equal(comparison.ok, true);
    assert.equal(comparison.expectedScore, 0.6);
    assert.equal(comparison.expectedApproved, false);
  });
});
