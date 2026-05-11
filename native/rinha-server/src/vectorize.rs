const MAX_AMOUNT: f64 = 10000.0;
const MAX_INSTALLMENTS: f64 = 12.0;
const RATIO: f64 = 10.0;
const MAX_MINUTES: f64 = 1440.0;
const MAX_KM: f64 = 1000.0;
const MAX_TX_COUNT: f64 = 20.0;
const MAX_MERCHANT_AVG: f64 = 10000.0;

fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

fn mcc_risk(mcc: &[u8]) -> f64 {
    match mcc {
        b"5411" => 0.15,
        b"5812" => 0.30,
        b"5912" => 0.20,
        b"5944" => 0.45,
        b"7801" => 0.80,
        b"7802" => 0.75,
        b"7995" => 0.85,
        b"4511" => 0.35,
        b"5311" => 0.25,
        b"5999" => 0.50,
        _ => 0.5,
    }
}

fn parse_digits2(s: &[u8]) -> u32 {
    ((s[0] - b'0') as u32) * 10 + (s[1] - b'0') as u32
}

fn parse_digits4(s: &[u8]) -> u32 {
    ((s[0] - b'0') as u32) * 1000
        + ((s[1] - b'0') as u32) * 100
        + ((s[2] - b'0') as u32) * 10
        + (s[3] - b'0') as u32
}

// Days since Unix epoch using Howard Hinnant's civil-to-days algorithm.
fn civil_to_days(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

// Parse ISO UTC timestamp "YYYY-MM-DDTHH:MM:SSZ" to epoch milliseconds.
fn parse_utc_ms(ts: &[u8]) -> i64 {
    let year = parse_digits4(&ts[0..4]) as i64;
    let month = parse_digits2(&ts[5..7]) as i64;
    let day = parse_digits2(&ts[8..10]) as i64;
    let hour = parse_digits2(&ts[11..13]) as i64;
    let minute = parse_digits2(&ts[14..16]) as i64;
    let second = parse_digits2(&ts[17..19]) as i64;
    let days = civil_to_days(year, month, day);
    days * 86_400_000 + hour * 3_600_000 + minute * 60_000 + second * 1_000
}

fn utc_hour(ts: &[u8]) -> f64 {
    parse_digits2(&ts[11..13]) as f64
}

fn utc_dow(ts: &[u8]) -> f64 {
    let year = parse_digits4(&ts[0..4]) as i64;
    let month = parse_digits2(&ts[5..7]) as i64;
    let day = parse_digits2(&ts[8..10]) as i64;
    let days = civil_to_days(year, month, day);
    // (days + 4) % 7: 0=Sunday like JS getUTCDay; epoch (1970-01-01) was Thursday=4.
    let js_dow = (days + 4).rem_euclid(7);
    // Mon=0..Sun=6 per spec
    let spec_dow = (js_dow + 6) % 7;
    spec_dow as f64
}

pub struct VecInput<'a> {
    pub tx_amount: f64,
    pub tx_installments: f64,
    pub tx_requested_at: &'a [u8],
    pub customer_avg_amount: f64,
    pub customer_tx_count_24h: f64,
    pub known_merchants: &'a [&'a [u8]],
    pub merchant_id: &'a [u8],
    pub merchant_mcc: &'a [u8],
    pub merchant_avg_amount: f64,
    pub terminal_km_from_home: f64,
    pub terminal_is_online: bool,
    pub terminal_card_present: bool,
    pub last_tx_timestamp: Option<&'a [u8]>,
    pub last_tx_km_from_current: Option<f64>,
}

pub fn vectorize(inp: &VecInput) -> [f64; 14] {
    let mut v = [0.0f64; 14];

    v[0] = clamp01(inp.tx_amount / MAX_AMOUNT);
    v[1] = clamp01(inp.tx_installments / MAX_INSTALLMENTS);
    // customer.avg_amount == 0 gives Inf/RATIO → clamp to 1. Same as JS.
    v[2] = clamp01((inp.tx_amount / inp.customer_avg_amount) / RATIO);
    v[3] = utc_hour(inp.tx_requested_at) / 23.0;
    v[4] = utc_dow(inp.tx_requested_at) / 6.0;

    if let (Some(ts), Some(km)) = (inp.last_tx_timestamp, inp.last_tx_km_from_current) {
        let minutes = (parse_utc_ms(inp.tx_requested_at) - parse_utc_ms(ts)) as f64 / 60_000.0;
        v[5] = clamp01(minutes / MAX_MINUTES);
        v[6] = clamp01(km / MAX_KM);
    } else {
        v[5] = -1.0;
        v[6] = -1.0;
    }

    v[7] = clamp01(inp.terminal_km_from_home / MAX_KM);
    v[8] = clamp01(inp.customer_tx_count_24h / MAX_TX_COUNT);
    v[9] = if inp.terminal_is_online { 1.0 } else { 0.0 };
    v[10] = if inp.terminal_card_present { 1.0 } else { 0.0 };
    v[11] = if inp.known_merchants.iter().any(|&m| m == inp.merchant_id) {
        0.0
    } else {
        1.0
    };
    v[12] = mcc_risk(inp.merchant_mcc);
    v[13] = clamp01(inp.merchant_avg_amount / MAX_MERCHANT_AVG);

    // round4 to all dims, preserving -1 sentinels.
    for i in 0..14 {
        if v[i] != -1.0 {
            v[i] = round4(v[i]);
        }
    }

    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinels_survive_when_last_tx_null() {
        let km: &[&[u8]] = &[];
        let inp = VecInput {
            tx_amount: 100.0,
            tx_installments: 1.0,
            tx_requested_at: b"2026-03-11T18:45:53Z",
            customer_avg_amount: 100.0,
            customer_tx_count_24h: 1.0,
            known_merchants: km,
            merchant_id: b"MERC-001",
            merchant_mcc: b"5411",
            merchant_avg_amount: 100.0,
            terminal_km_from_home: 10.0,
            terminal_is_online: false,
            terminal_card_present: true,
            last_tx_timestamp: None,
            last_tx_km_from_current: None,
        };
        let v = vectorize(&inp);
        assert_eq!(v[5], -1.0, "v[5] must be sentinel");
        assert_eq!(v[6], -1.0, "v[6] must be sentinel");
    }

    #[test]
    fn dow_wednesday_2026_03_11() {
        // 2026-03-11 was a Wednesday.  JS getUTCDay: Wed=3.  Spec: (3+6)%7=2.
        let dow = utc_dow(b"2026-03-11T18:45:53Z");
        assert_eq!(dow, 2.0, "2026-03-11 should be Wed=2 in Mon-0 system");
    }

    #[test]
    fn round4_precision() {
        assert_eq!(round4(1.0 / 3.0), 0.3333);
        assert_eq!(round4(2.0 / 12.0), 0.1667);
    }
}
