import fs from 'node:fs';
import { classifyFraudBucket } from '../src/classifier.js';

const data = JSON.parse(fs.readFileSync('test/test-data.json', 'utf8'));

const fpEntries = [];
const fnEntries = [];

for (const entry of data.entries) {
  const bucket = classifyFraudBucket(entry.request);
  const approved = bucket < 3;
  if (approved !== entry.expected_approved) {
    if (approved) fnEntries.push(entry);
    else fpEntries.push({ entry, bucket });
  }
}

console.log(`FP=${fpEntries.length}  FN=${fnEntries.length}`);

function profileGroup(group, label) {
  if (group.length === 0) return;
  const fields = {
    amount_high: 0,           // >= 2000
    amount_ratio_high: 0,     // ratio >= 10
    installments_high: 0,     // >= 6
    tx_count_high: 0,         // >= 8
    merchant_unknown: 0,
    risky_mcc: 0,
    unknown_mcc: 0,
    far_from_home: 0,
    is_online: 0,
    not_card_present: 0,
    night_hour: 0,
    last_close_in_time: 0,
    last_far_distance: 0,
  };
  let bucketCounts = [0, 0, 0, 0, 0, 0];
  for (const item of group) {
    const e = item.entry || item;
    const r = e.request;
    if (r.transaction.amount >= 2000) fields.amount_high += 1;
    if (r.transaction.amount / r.customer.avg_amount >= 10) fields.amount_ratio_high += 1;
    if (r.transaction.installments >= 6) fields.installments_high += 1;
    if (r.customer.tx_count_24h >= 8) fields.tx_count_high += 1;
    if (!r.customer.known_merchants.includes(r.merchant.id)) fields.merchant_unknown += 1;
    if (['7995', '7801', '7802'].includes(r.merchant.mcc)) fields.risky_mcc += 1;
    if (!['5411', '5812', '5912', '5311', '7995', '7801', '7802'].includes(r.merchant.mcc)) fields.unknown_mcc += 1;
    if (r.terminal.km_from_home >= 200) fields.far_from_home += 1;
    if (r.terminal.is_online) fields.is_online += 1;
    if (!r.terminal.card_present) fields.not_card_present += 1;
    const hour = parseInt(r.transaction.requested_at.slice(11, 13), 10);
    if (hour < 7) fields.night_hour += 1;
    if (r.last_transaction) {
      const minutes = (Date.parse(r.transaction.requested_at) - Date.parse(r.last_transaction.timestamp)) / 60000;
      if (minutes <= 10) fields.last_close_in_time += 1;
      if (r.last_transaction.km_from_current >= 200) fields.last_far_distance += 1;
    }
    if (item.bucket !== undefined) bucketCounts[item.bucket] += 1;
  }
  console.log(`\n=== ${label} (n=${group.length}) ===`);
  const total = group.length;
  for (const [k, v] of Object.entries(fields)) {
    console.log(`  ${k.padEnd(24)} ${v}/${total} (${(v / total * 100).toFixed(1)}%)`);
  }
  if (item => item.bucket !== undefined) {
    console.log(`  bucket distribution: ${bucketCounts.join(', ')}`);
  }
}

profileGroup(fpEntries, 'FALSE POSITIVES (legit denied)');
profileGroup(fnEntries, 'FALSE NEGATIVES (fraud approved)');
