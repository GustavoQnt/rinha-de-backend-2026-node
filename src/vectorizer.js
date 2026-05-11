import fs from 'node:fs';

const norm = JSON.parse(fs.readFileSync('resources/normalization.json', 'utf8'));
const mccRisk = JSON.parse(fs.readFileSync('resources/mcc_risk.json', 'utf8'));

const MAX_AMOUNT = norm.max_amount;
const MAX_INSTALLMENTS = norm.max_installments;
const RATIO = norm.amount_vs_avg_ratio;
const MAX_MINUTES = norm.max_minutes;
const MAX_KM = norm.max_km;
const MAX_TX_COUNT = norm.max_tx_count_24h;
const MAX_MERCHANT_AVG = norm.max_merchant_avg_amount;

function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

function round4(x) {
  return Math.round(x * 10000) / 10000;
}

function parseUtcTimestamp(ts) {
  // ISO format YYYY-MM-DDTHH:MM:SSZ
  const year = +ts.slice(0, 4);
  const month = +ts.slice(5, 7);
  const day = +ts.slice(8, 10);
  const hour = +ts.slice(11, 13);
  const minute = +ts.slice(14, 16);
  const second = +ts.slice(17, 19);
  return Date.UTC(year, month - 1, day, hour, minute, second);
}

function utcHourOf(ts) {
  return +ts.slice(11, 13);
}

function utcDayOfWeek(ts) {
  // JS getUTCDay: Sunday=0..Saturday=6
  // Spec: Monday=0..Sunday=6
  const ms = parseUtcTimestamp(ts);
  const jsDow = new Date(ms).getUTCDay();
  return (jsDow + 6) % 7;
}

export function vectorize(req) {
  const tx = req.transaction;
  const customer = req.customer;
  const merchant = req.merchant;
  const terminal = req.terminal;
  const last = req.last_transaction;

  const v = new Array(14);

  v[0] = clamp01(tx.amount / MAX_AMOUNT);
  v[1] = clamp01(tx.installments / MAX_INSTALLMENTS);
  v[2] = clamp01((tx.amount / customer.avg_amount) / RATIO);
  v[3] = utcHourOf(tx.requested_at) / 23;
  v[4] = utcDayOfWeek(tx.requested_at) / 6;

  if (last !== null && last !== undefined) {
    const minutes = (parseUtcTimestamp(tx.requested_at) - parseUtcTimestamp(last.timestamp)) / 60000;
    v[5] = clamp01(minutes / MAX_MINUTES);
    v[6] = clamp01(last.km_from_current / MAX_KM);
  } else {
    v[5] = -1;
    v[6] = -1;
  }

  v[7] = clamp01(terminal.km_from_home / MAX_KM);
  v[8] = clamp01(customer.tx_count_24h / MAX_TX_COUNT);
  v[9] = terminal.is_online ? 1 : 0;
  v[10] = terminal.card_present ? 1 : 0;
  v[11] = customer.known_merchants.includes(merchant.id) ? 0 : 1;
  v[12] = mccRisk[merchant.mcc] !== undefined ? mccRisk[merchant.mcc] : 0.5;
  v[13] = clamp01(merchant.avg_amount / MAX_MERCHANT_AVG);

  // Apply round4 to ALL dims, but preserve -1 sentinel
  for (let i = 0; i < 14; i += 1) {
    if (v[i] !== -1) v[i] = round4(v[i]);
  }

  return v;
}

export { round4, clamp01, parseUtcTimestamp, utcHourOf, utcDayOfWeek };
