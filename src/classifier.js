const RISKY_MCC = new Set(['7995', '7801', '7802']);
const SAFE_MCC = new Set(['5411', '5812', '5912', '5311']);
const MONTH_DAY_OFFSET = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

function hourOf(timestamp) {
  return (timestamp.charCodeAt(11) - 48) * 10 + (timestamp.charCodeAt(12) - 48);
}

function twoDigits(timestamp, index) {
  return (timestamp.charCodeAt(index) - 48) * 10 + (timestamp.charCodeAt(index + 1) - 48);
}

function yearOf(timestamp) {
  return (
    (timestamp.charCodeAt(0) - 48) * 1000
    + (timestamp.charCodeAt(1) - 48) * 100
    + (timestamp.charCodeAt(2) - 48) * 10
    + (timestamp.charCodeAt(3) - 48)
  );
}

function leapDaysBefore(year) {
  const previous = year - 1;
  return Math.floor(previous / 4) - Math.floor(previous / 100) + Math.floor(previous / 400);
}

function epochMinutes(timestamp) {
  const year = yearOf(timestamp);
  const month = twoDigits(timestamp, 5);
  const day = twoDigits(timestamp, 8);
  const hour = twoDigits(timestamp, 11);
  const minute = twoDigits(timestamp, 14);
  const leapDay = month > 2 && ((year & 3) === 0 && (year % 100 !== 0 || year % 400 === 0)) ? 1 : 0;
  const days = (year - 1970) * 365 + leapDaysBefore(year) - leapDaysBefore(1970) + MONTH_DAY_OFFSET[month] + leapDay + day - 1;

  return days * 1440 + hour * 60 + minute;
}

function minutesBetween(current, previous) {
  return epochMinutes(current) - epochMinutes(previous);
}

function merchantKnown(knownMerchants, merchantId) {
  for (let i = 0; i < knownMerchants.length; i += 1) {
    if (knownMerchants[i] === merchantId) return true;
  }
  return false;
}

export const RESPONSE_BODIES = [
  '{"approved":true,"fraud_score":0}',
  '{"approved":true,"fraud_score":0.2}',
  '{"approved":true,"fraud_score":0.4}',
  '{"approved":false,"fraud_score":0.6}',
  '{"approved":false,"fraud_score":0.8}',
  '{"approved":false,"fraud_score":1}',
];

const RESPONSE_OBJECTS = [
  { approved: true, fraud_score: 0 },
  { approved: true, fraud_score: 0.2 },
  { approved: true, fraud_score: 0.4 },
  { approved: false, fraud_score: 0.6 },
  { approved: false, fraud_score: 0.8 },
  { approved: false, fraud_score: 1 },
];

function bucketFor(score) {
  if (score >= 8) return 5;
  if (score >= 5) return 4;
  if (score >= 2) return 3;
  if (score === 1) return 2;
  return 0;
}

export function classifyFraud(payload) {
  return RESPONSE_OBJECTS[classifyFraudBucket(payload)];
}

export function classifyFraudBucket(payload) {
  const tx = payload.transaction;
  const customer = payload.customer;
  const merchant = payload.merchant;
  const terminal = payload.terminal;
  const last = payload.last_transaction;

  let score = 0;

  const amount = tx.amount;
  const amountRatio = amount / customer.avg_amount;
  const hour = hourOf(tx.requested_at);

  if (amount >= 2000) score += 3;
  if (amountRatio >= 10) score += 3;
  if (tx.installments >= 6) score += 2;
  if (customer.tx_count_24h >= 8) score += 2;
  if (!merchantKnown(customer.known_merchants, merchant.id)) score += 2;
  if (RISKY_MCC.has(merchant.mcc)) score += 2;
  if (!SAFE_MCC.has(merchant.mcc) && !RISKY_MCC.has(merchant.mcc)) score += 1;
  if (terminal.km_from_home >= 200) score += 2;
  if (hour < 7) score += 1;

  if (last !== null && last !== undefined) {
    const minutes = minutesBetween(tx.requested_at, last.timestamp);
    if (minutes <= 10) score += 1;
    if (last.km_from_current >= 200) score += 2;
  }

  return bucketFor(score);
}
