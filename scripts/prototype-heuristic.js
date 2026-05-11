import fs from 'node:fs';

const data = JSON.parse(fs.readFileSync('test/test-data.json', 'utf8'));

function makeClassifier(weights, threshold) {
  return function classify(r) {
    const tx = r.transaction;
    const customer = r.customer;
    const merchant = r.merchant;
    const terminal = r.terminal;
    const last = r.last_transaction;

    const amount = tx.amount;
    const ratio = amount / customer.avg_amount;
    const hour = parseInt(tx.requested_at.slice(11, 13), 10);

    let s = 0;
    if (amount >= 2000) s += weights.amount_2k;
    if (ratio >= 10) s += weights.ratio_10;
    if (tx.installments >= 6) s += weights.installments_6;
    if (customer.tx_count_24h >= 8) s += weights.tx_count_8;
    if (!customer.known_merchants.includes(merchant.id)) s += weights.merchant_unknown;
    if (['7995', '7801', '7802'].includes(merchant.mcc)) s += weights.risky_mcc;
    if (!['5411', '5812', '5912', '5311', '7995', '7801', '7802'].includes(merchant.mcc)) s += weights.unknown_mcc;
    if (terminal.km_from_home >= 200) s += weights.km_200;
    if (terminal.is_online) s += weights.is_online;
    if (!terminal.card_present) s += weights.not_card_present;
    if (hour < 7) s += weights.night;
    if (last) {
      const minutes = (Date.parse(tx.requested_at) - Date.parse(last.timestamp)) / 60000;
      if (minutes <= 10) s += weights.last_close;
      if (last.km_from_current >= 200) s += weights.last_far;
    }

    return s < threshold; // approve if true
  };
}

function evaluate(classify) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const e of data.entries) {
    const approved = classify(e.request);
    if (approved === e.expected_approved) {
      if (approved) tn += 1; else tp += 1;
    } else {
      if (approved) fn += 1; else fp += 1;
    }
  }
  const total = tp + tn + fp + fn;
  const failures = fp + fn;
  const weighted = fp + fn * 3;
  return { total, tp, tn, fp, fn, failure_rate: +(failures / total * 100).toFixed(3), weighted };
}

// Original
const baseline = {
  amount_2k: 3, ratio_10: 3, installments_6: 2, tx_count_8: 2,
  merchant_unknown: 2, risky_mcc: 2, unknown_mcc: 1, km_200: 2,
  is_online: 1, not_card_present: 1, night: 1, last_close: 1, last_far: 2,
};

console.log('Baseline (current weights, threshold 5):');
console.log(JSON.stringify(evaluate(makeClassifier(baseline, 5))));

// Variant: zero noisy weights
console.log('\nNoisy=0 (is_online=0, not_card_present=0):');
console.log(JSON.stringify(evaluate(makeClassifier({ ...baseline, is_online: 0, not_card_present: 0 }, 5))));

// Variant: zero noisy + zero unknown_mcc (also noisy)
console.log('\nNoisy=0 + unknown_mcc=0:');
console.log(JSON.stringify(evaluate(makeClassifier({ ...baseline, is_online: 0, not_card_present: 0, unknown_mcc: 0 }, 5))));

// Variant: keep noisy but lower threshold
console.log('\nKeep noisy, threshold 6:');
console.log(JSON.stringify(evaluate(makeClassifier(baseline, 6))));

console.log('\nKeep noisy, threshold 7:');
console.log(JSON.stringify(evaluate(makeClassifier(baseline, 7))));

// Variant: zero noisy, threshold 4
console.log('\nNoisy=0, threshold 4:');
console.log(JSON.stringify(evaluate(makeClassifier({ ...baseline, is_online: 0, not_card_present: 0 }, 4))));

// Variant: zero noisy, threshold 5, increased night weight
console.log('\nNoisy=0, night+2 instead of +1:');
console.log(JSON.stringify(evaluate(makeClassifier({ ...baseline, is_online: 0, not_card_present: 0, night: 2 }, 5))));

// Variant: zero noisy, increase last_close weight
console.log('\nNoisy=0, last_close=2 instead of 1:');
console.log(JSON.stringify(evaluate(makeClassifier({ ...baseline, is_online: 0, not_card_present: 0, last_close: 2 }, 5))));

const noisyZero = { ...baseline, is_online: 0, not_card_present: 0 };

console.log('\n--- Sweep around (Noisy=0, threshold 4) ---');
for (const t of [2, 3, 4, 5]) {
  console.log(`thr=${t}: ${JSON.stringify(evaluate(makeClassifier(noisyZero, t)))}`);
}

console.log('\n--- Threshold 1 and 2 ---');
for (const t of [1, 2]) {
  console.log(`thr=${t}: ${JSON.stringify(evaluate(makeClassifier(noisyZero, t)))}`);
}

console.log('\n--- Best so far: noisy=0, threshold=3 ---');
const best = noisyZero;
console.log(`thr=3: ${JSON.stringify(evaluate(makeClassifier(best, 3)))}`);

console.log('\n--- Adjust weights at threshold 2 ---');
const adjustments2 = [
  ['amount_2k', [1, 2, 3]],
  ['ratio_10', [1, 2, 3]],
  ['installments_6', [1, 2, 3]],
  ['tx_count_8', [1, 2, 3]],
  ['merchant_unknown', [1, 2, 3]],
  ['risky_mcc', [1, 2, 3]],
  ['km_200', [1, 2, 3]],
  ['last_far', [1, 2, 3]],
  ['night', [0, 1, 2]],
  ['unknown_mcc', [0, 1, 2]],
  ['last_close', [0, 1, 2]],
];
for (const [field, vals] of adjustments2) {
  for (const v of vals) {
    if (v === noisyZero[field]) continue;
    const w = { ...noisyZero, [field]: v };
    const r = evaluate(makeClassifier(w, 2));
    if (r.weighted < 1311) {
      console.log(`${field}=${v} (default ${noisyZero[field]}): ${JSON.stringify(r)}`);
    }
  }
}

console.log('\n--- Adjust weights at threshold 3 ---');
const adjustments3 = [
  ['amount_2k', [2, 3, 4, 5]],
  ['ratio_10', [2, 3, 4, 5]],
  ['installments_6', [1, 2, 3]],
  ['tx_count_8', [1, 2, 3]],
  ['merchant_unknown', [1, 2, 3]],
  ['risky_mcc', [1, 2, 3]],
  ['km_200', [1, 2, 3]],
  ['last_far', [1, 2, 3]],
  ['night', [0, 1, 2]],
  ['unknown_mcc', [0, 1, 2]],
  ['last_close', [0, 1, 2]],
];
for (const [field, vals] of adjustments3) {
  for (const v of vals) {
    if (v === best[field]) continue;
    const w = { ...best, [field]: v };
    const r = evaluate(makeClassifier(w, 3));
    if (r.weighted < 1420) {
      console.log(`${field}=${v} (default ${best[field]}): ${JSON.stringify(r)}`);
    }
  }
}

console.log('\n--- Tune individual weights at threshold 4 ---');
const adjustments = [
  ['amount_2k', [2, 3, 4]],
  ['ratio_10', [2, 3, 4]],
  ['installments_6', [1, 2, 3]],
  ['tx_count_8', [1, 2, 3]],
  ['merchant_unknown', [1, 2, 3]],
  ['risky_mcc', [1, 2, 3]],
  ['km_200', [1, 2, 3]],
  ['last_far', [1, 2, 3]],
  ['night', [0, 1, 2]],
  ['unknown_mcc', [0, 1]],
];
for (const [field, vals] of adjustments) {
  for (const v of vals) {
    if (v === noisyZero[field]) continue;
    const w = { ...noisyZero, [field]: v };
    const r = evaluate(makeClassifier(w, 4));
    if (r.weighted < 1500) {
      console.log(`${field}=${v} (default ${noisyZero[field]}): ${JSON.stringify(r)}`);
    }
  }
}
