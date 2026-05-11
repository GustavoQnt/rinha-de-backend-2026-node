import { vectorize } from '../src/vectorizer.js';

const cases = [
  {
    label: 'doc legit example',
    req: {
      id: 'tx-1329056812',
      transaction: { amount: 41.12, installments: 2, requested_at: '2026-03-11T18:45:53Z' },
      customer: { avg_amount: 82.24, tx_count_24h: 3, known_merchants: ['MERC-003', 'MERC-016'] },
      merchant: { id: 'MERC-016', mcc: '5411', avg_amount: 60.25 },
      terminal: { is_online: false, card_present: true, km_from_home: 29.23 },
      last_transaction: null,
    },
    expected: [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006],
  },
  {
    label: 'doc fraud example',
    req: {
      id: 'tx-3330991687',
      transaction: { amount: 9505.97, installments: 10, requested_at: '2026-03-14T05:15:12Z' },
      customer: { avg_amount: 81.28, tx_count_24h: 20, known_merchants: ['MERC-008', 'MERC-007', 'MERC-005'] },
      merchant: { id: 'MERC-068', mcc: '7802', avg_amount: 54.86 },
      terminal: { is_online: false, card_present: true, km_from_home: 952.27 },
      last_transaction: null,
    },
    expected: [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055],
  },
];

let pass = 0;
let fail = 0;
for (const c of cases) {
  const actual = vectorize(c.req);
  const ok = actual.length === c.expected.length
    && actual.every((v, i) => Math.abs(v - c.expected[i]) < 1e-9);
  if (ok) {
    pass += 1;
    console.log(`PASS ${c.label}`);
  } else {
    fail += 1;
    console.log(`FAIL ${c.label}`);
    console.log(`  expected: [${c.expected.join(', ')}]`);
    console.log(`  actual:   [${actual.join(', ')}]`);
    for (let i = 0; i < 14; i += 1) {
      if (Math.abs(actual[i] - c.expected[i]) >= 1e-9) {
        console.log(`  diff dim ${i}: expected=${c.expected[i]}  actual=${actual[i]}`);
      }
    }
  }
}

console.log(`\n${pass} pass, ${fail} fail`);
process.exitCode = fail === 0 ? 0 : 1;
