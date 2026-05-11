import assert from 'node:assert/strict';
import test from 'node:test';
import { classifyFraud } from '../src/classifier.js';

const legitPayload = {
  id: 'tx-legit',
  transaction: {
    amount: 384.88,
    installments: 3,
    requested_at: '2026-03-11T20:23:35Z',
  },
  customer: {
    avg_amount: 769.76,
    tx_count_24h: 3,
    known_merchants: ['MERC-009', 'MERC-001'],
  },
  merchant: {
    id: 'MERC-001',
    mcc: '5912',
    avg_amount: 298.95,
  },
  terminal: {
    is_online: false,
    card_present: true,
    km_from_home: 13.7,
  },
  last_transaction: {
    timestamp: '2026-03-11T14:58:35Z',
    km_from_current: 18.8,
  },
};

const fraudPayload = {
  id: 'tx-fraud',
  transaction: {
    amount: 9505.97,
    installments: 10,
    requested_at: '2026-03-14T05:15:12Z',
  },
  customer: {
    avg_amount: 81.28,
    tx_count_24h: 20,
    known_merchants: ['MERC-008', 'MERC-007', 'MERC-005'],
  },
  merchant: {
    id: 'MERC-068',
    mcc: '7802',
    avg_amount: 54.86,
  },
  terminal: {
    is_online: false,
    card_present: true,
    km_from_home: 952.27,
  },
  last_transaction: null,
};

test('classifier returns challenge response shape', () => {
  const result = classifyFraud(legitPayload);

  assert.equal(typeof result.approved, 'boolean');
  assert.equal(typeof result.fraud_score, 'number');
});

test('classifier approves high-confidence legit transaction', () => {
  assert.deepEqual(classifyFraud(legitPayload), {
    approved: true,
    fraud_score: 0,
  });
});

test('classifier denies high-confidence fraud transaction', () => {
  assert.deepEqual(classifyFraud(fraudPayload), {
    approved: false,
    fraud_score: 1,
  });
});
