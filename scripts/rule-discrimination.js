import fs from 'node:fs';

const data = JSON.parse(fs.readFileSync('test/test-data.json', 'utf8'));

const RULES = {
  amount_2k:        (r) => r.transaction.amount >= 2000,
  amount_1k:        (r) => r.transaction.amount >= 1000,
  amount_5k:        (r) => r.transaction.amount >= 5000,
  ratio_10:         (r) => r.transaction.amount / r.customer.avg_amount >= 10,
  ratio_20:         (r) => r.transaction.amount / r.customer.avg_amount >= 20,
  ratio_5:          (r) => r.transaction.amount / r.customer.avg_amount >= 5,
  installments_6:   (r) => r.transaction.installments >= 6,
  installments_8:   (r) => r.transaction.installments >= 8,
  installments_10:  (r) => r.transaction.installments >= 10,
  tx_count_8:       (r) => r.customer.tx_count_24h >= 8,
  tx_count_15:      (r) => r.customer.tx_count_24h >= 15,
  tx_count_20:      (r) => r.customer.tx_count_24h >= 20,
  merchant_unknown: (r) => !r.customer.known_merchants.includes(r.merchant.id),
  risky_mcc:        (r) => ['7995', '7801', '7802'].includes(r.merchant.mcc),
  km_200:           (r) => r.terminal.km_from_home >= 200,
  km_500:           (r) => r.terminal.km_from_home >= 500,
  km_800:           (r) => r.terminal.km_from_home >= 800,
  km_1000:          (r) => r.terminal.km_from_home >= 1000,
  is_online:        (r) => r.terminal.is_online,
  not_card_present: (r) => !r.terminal.card_present,
  night_hour:       (r) => parseInt(r.transaction.requested_at.slice(11, 13), 10) < 7,
  late_night:       (r) => {
    const h = parseInt(r.transaction.requested_at.slice(11, 13), 10);
    return h >= 0 && h < 5;
  },
  last_close_10m:   (r) => r.last_transaction && (Date.parse(r.transaction.requested_at) - Date.parse(r.last_transaction.timestamp)) / 60000 <= 10,
  last_far_200km:   (r) => r.last_transaction && r.last_transaction.km_from_current >= 200,
  last_far_500km:   (r) => r.last_transaction && r.last_transaction.km_from_current >= 500,
  // combined signals
  big_amount_unknown_merchant: (r) => r.transaction.amount >= 2000 && !r.customer.known_merchants.includes(r.merchant.id),
  far_and_unknown:  (r) => r.terminal.km_from_home >= 500 && !r.customer.known_merchants.includes(r.merchant.id),
  ratio_and_far:    (r) => r.transaction.amount / r.customer.avg_amount >= 10 && r.terminal.km_from_home >= 500,
};

const fraud = data.entries.filter(e => !e.expected_approved);
const legit = data.entries.filter(e => e.expected_approved);

console.log(`Fraud=${fraud.length}  Legit=${legit.length}`);
console.log('rule                              fraud%   legit%   lift');
console.log('-'.repeat(70));

const results = [];
for (const [name, fn] of Object.entries(RULES)) {
  let fHit = 0;
  for (const e of fraud) if (fn(e.request)) fHit += 1;
  let lHit = 0;
  for (const e of legit) if (fn(e.request)) lHit += 1;
  const fRate = fHit / fraud.length;
  const lRate = lHit / legit.length;
  const lift = lRate > 0 ? fRate / lRate : Infinity;
  results.push({ name, fRate, lRate, lift });
}

results.sort((a, b) => b.lift - a.lift);
for (const r of results) {
  console.log(`${r.name.padEnd(33)} ${(r.fRate * 100).toFixed(1).padStart(5)}%  ${(r.lRate * 100).toFixed(1).padStart(5)}%  ${r.lift.toFixed(2).padStart(5)}x`);
}
