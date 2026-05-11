use crate::vectorize::VecInput;

pub struct ParsedRequest<'a> {
    pub tx_amount: f64,
    pub tx_installments: f64,
    pub tx_requested_at: &'a [u8],
    pub customer_avg_amount: f64,
    pub customer_tx_count_24h: f64,
    pub merchant_id: &'a [u8],
    pub merchant_mcc: &'a [u8],
    pub merchant_avg_amount: f64,
    pub terminal_km_from_home: f64,
    pub terminal_is_online: bool,
    pub terminal_card_present: bool,
    pub last_tx_timestamp: Option<&'a [u8]>,
    pub last_tx_km_from_current: Option<f64>,
    pub known_merchants: Vec<&'a [u8]>,
}

impl<'a> ParsedRequest<'a> {
    pub fn to_vec_input(&'a self) -> VecInput<'a> {
        VecInput {
            tx_amount: self.tx_amount,
            tx_installments: self.tx_installments,
            tx_requested_at: self.tx_requested_at,
            customer_avg_amount: self.customer_avg_amount,
            customer_tx_count_24h: self.customer_tx_count_24h,
            known_merchants: &self.known_merchants,
            merchant_id: self.merchant_id,
            merchant_mcc: self.merchant_mcc,
            merchant_avg_amount: self.merchant_avg_amount,
            terminal_km_from_home: self.terminal_km_from_home,
            terminal_is_online: self.terminal_is_online,
            terminal_card_present: self.terminal_card_present,
            last_tx_timestamp: self.last_tx_timestamp,
            last_tx_km_from_current: self.last_tx_km_from_current,
        }
    }
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.data.len()
            && matches!(self.data[self.pos], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.pos += 1;
        }
    }

    fn consume(&mut self, b: u8) -> bool {
        self.skip_ws();
        if self.pos < self.data.len() && self.data[self.pos] == b {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn read_string(&mut self) -> Option<&'a [u8]> {
        self.skip_ws();
        if self.pos >= self.data.len() || self.data[self.pos] != b'"' {
            return None;
        }
        self.pos += 1;
        let start = self.pos;
        loop {
            if self.pos >= self.data.len() {
                return None;
            }
            match self.data[self.pos] {
                b'"' => {
                    let s = &self.data[start..self.pos];
                    self.pos += 1;
                    return Some(s);
                }
                b'\\' => {
                    self.pos += 2; // skip the escaped character
                }
                _ => {
                    self.pos += 1;
                }
            }
        }
    }

    fn read_number(&mut self) -> Option<f64> {
        self.skip_ws();
        let start = self.pos;
        if self.pos < self.data.len() && self.data[self.pos] == b'-' {
            self.pos += 1;
        }
        while self.pos < self.data.len() && self.data[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos < self.data.len() && self.data[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < self.data.len() && self.data[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if self.pos < self.data.len()
            && (self.data[self.pos] == b'e' || self.data[self.pos] == b'E')
        {
            self.pos += 1;
            if self.pos < self.data.len()
                && (self.data[self.pos] == b'+' || self.data[self.pos] == b'-')
            {
                self.pos += 1;
            }
            while self.pos < self.data.len() && self.data[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        let s = std::str::from_utf8(&self.data[start..self.pos]).ok()?;
        s.parse().ok()
    }

    fn read_bool(&mut self) -> Option<bool> {
        self.skip_ws();
        if self.data[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Some(true)
        } else if self.data[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Some(false)
        } else {
            None
        }
    }

    fn skip_value(&mut self) {
        self.skip_ws();
        if self.pos >= self.data.len() {
            return;
        }
        match self.data[self.pos] {
            b'"' => {
                self.read_string();
            }
            b'{' => {
                self.skip_object();
            }
            b'[' => {
                self.skip_array();
            }
            b't' => {
                self.pos += 4;
            }
            b'f' => {
                self.pos += 5;
            }
            b'n' => {
                self.pos += 4;
            }
            _ => {
                self.read_number();
            }
        }
    }

    fn skip_object(&mut self) {
        self.consume(b'{');
        loop {
            self.skip_ws();
            if self.pos >= self.data.len() || self.data[self.pos] == b'}' {
                self.pos += 1;
                return;
            }
            if self.data[self.pos] == b',' {
                self.pos += 1;
                self.skip_ws();
            }
            if self.pos >= self.data.len() || self.data[self.pos] == b'}' {
                self.pos += 1;
                return;
            }
            self.read_string();
            self.consume(b':');
            self.skip_value();
        }
    }

    fn skip_array(&mut self) {
        self.consume(b'[');
        loop {
            self.skip_ws();
            if self.pos >= self.data.len() || self.data[self.pos] == b']' {
                self.pos += 1;
                return;
            }
            if self.data[self.pos] == b',' {
                self.pos += 1;
            }
            self.skip_value();
        }
    }
}

fn parse_transaction<'a>(cur: &mut Cursor<'a>, out: &mut ParsedRequest<'a>) -> Option<()> {
    cur.consume(b'{');
    loop {
        cur.skip_ws();
        if cur.pos >= cur.data.len() || cur.data[cur.pos] == b'}' {
            cur.pos += 1;
            break;
        }
        if cur.data[cur.pos] == b',' {
            cur.pos += 1;
            cur.skip_ws();
        }
        if cur.pos >= cur.data.len() || cur.data[cur.pos] == b'}' {
            cur.pos += 1;
            break;
        }
        let key = cur.read_string()?;
        cur.consume(b':');
        match key {
            b"amount" => out.tx_amount = cur.read_number()?,
            b"installments" => out.tx_installments = cur.read_number()?,
            b"requested_at" => out.tx_requested_at = cur.read_string()?,
            _ => cur.skip_value(),
        }
    }
    Some(())
}

fn parse_customer<'a>(cur: &mut Cursor<'a>, out: &mut ParsedRequest<'a>) -> Option<()> {
    cur.consume(b'{');
    loop {
        cur.skip_ws();
        if cur.pos >= cur.data.len() || cur.data[cur.pos] == b'}' {
            cur.pos += 1;
            break;
        }
        if cur.data[cur.pos] == b',' {
            cur.pos += 1;
            cur.skip_ws();
        }
        if cur.pos >= cur.data.len() || cur.data[cur.pos] == b'}' {
            cur.pos += 1;
            break;
        }
        let key = cur.read_string()?;
        cur.consume(b':');
        match key {
            b"avg_amount" => out.customer_avg_amount = cur.read_number()?,
            b"tx_count_24h" => out.customer_tx_count_24h = cur.read_number()?,
            b"known_merchants" => {
                out.known_merchants.clear();
                cur.consume(b'[');
                loop {
                    cur.skip_ws();
                    if cur.pos >= cur.data.len() || cur.data[cur.pos] == b']' {
                        cur.pos += 1;
                        break;
                    }
                    if cur.data[cur.pos] == b',' {
                        cur.pos += 1;
                        cur.skip_ws();
                    }
                    if cur.pos >= cur.data.len() || cur.data[cur.pos] == b']' {
                        cur.pos += 1;
                        break;
                    }
                    if let Some(s) = cur.read_string() {
                        out.known_merchants.push(s);
                    }
                }
            }
            _ => cur.skip_value(),
        }
    }
    Some(())
}

fn parse_merchant<'a>(cur: &mut Cursor<'a>, out: &mut ParsedRequest<'a>) -> Option<()> {
    cur.consume(b'{');
    loop {
        cur.skip_ws();
        if cur.pos >= cur.data.len() || cur.data[cur.pos] == b'}' {
            cur.pos += 1;
            break;
        }
        if cur.data[cur.pos] == b',' {
            cur.pos += 1;
            cur.skip_ws();
        }
        if cur.pos >= cur.data.len() || cur.data[cur.pos] == b'}' {
            cur.pos += 1;
            break;
        }
        let key = cur.read_string()?;
        cur.consume(b':');
        match key {
            b"id" => out.merchant_id = cur.read_string()?,
            b"mcc" => out.merchant_mcc = cur.read_string()?,
            b"avg_amount" => out.merchant_avg_amount = cur.read_number()?,
            _ => cur.skip_value(),
        }
    }
    Some(())
}

fn parse_terminal<'a>(cur: &mut Cursor<'a>, out: &mut ParsedRequest<'a>) -> Option<()> {
    cur.consume(b'{');
    loop {
        cur.skip_ws();
        if cur.pos >= cur.data.len() || cur.data[cur.pos] == b'}' {
            cur.pos += 1;
            break;
        }
        if cur.data[cur.pos] == b',' {
            cur.pos += 1;
            cur.skip_ws();
        }
        if cur.pos >= cur.data.len() || cur.data[cur.pos] == b'}' {
            cur.pos += 1;
            break;
        }
        let key = cur.read_string()?;
        cur.consume(b':');
        match key {
            b"km_from_home" => out.terminal_km_from_home = cur.read_number()?,
            b"is_online" => out.terminal_is_online = cur.read_bool()?,
            b"card_present" => out.terminal_card_present = cur.read_bool()?,
            _ => cur.skip_value(),
        }
    }
    Some(())
}

fn parse_last_tx<'a>(cur: &mut Cursor<'a>, out: &mut ParsedRequest<'a>) -> Option<()> {
    cur.skip_ws();
    if cur.data[cur.pos..].starts_with(b"null") {
        cur.pos += 4;
        out.last_tx_timestamp = None;
        out.last_tx_km_from_current = None;
        return Some(());
    }
    cur.consume(b'{');
    loop {
        cur.skip_ws();
        if cur.pos >= cur.data.len() || cur.data[cur.pos] == b'}' {
            cur.pos += 1;
            break;
        }
        if cur.data[cur.pos] == b',' {
            cur.pos += 1;
            cur.skip_ws();
        }
        if cur.pos >= cur.data.len() || cur.data[cur.pos] == b'}' {
            cur.pos += 1;
            break;
        }
        let key = cur.read_string()?;
        cur.consume(b':');
        match key {
            b"timestamp" => out.last_tx_timestamp = Some(cur.read_string()?),
            b"km_from_current" => out.last_tx_km_from_current = Some(cur.read_number()?),
            _ => cur.skip_value(),
        }
    }
    Some(())
}

pub fn parse_request(body: &[u8]) -> Option<ParsedRequest<'_>> {
    let mut cur = Cursor::new(body);
    let mut out = ParsedRequest {
        tx_amount: 0.0,
        tx_installments: 0.0,
        tx_requested_at: b"",
        customer_avg_amount: 1.0, // avoid div-by-zero; test data always has positive avg_amount
        customer_tx_count_24h: 0.0,
        merchant_id: b"",
        merchant_mcc: b"",
        merchant_avg_amount: 0.0,
        terminal_km_from_home: 0.0,
        terminal_is_online: false,
        terminal_card_present: false,
        last_tx_timestamp: None,
        last_tx_km_from_current: None,
        known_merchants: Vec::new(),
    };

    cur.consume(b'{');
    loop {
        cur.skip_ws();
        if cur.pos >= cur.data.len() || cur.data[cur.pos] == b'}' {
            break;
        }
        if cur.data[cur.pos] == b',' {
            cur.pos += 1;
            cur.skip_ws();
        }
        if cur.pos >= cur.data.len() || cur.data[cur.pos] == b'}' {
            break;
        }
        let key = cur.read_string()?;
        cur.consume(b':');
        match key {
            b"transaction" => parse_transaction(&mut cur, &mut out)?,
            b"customer" => parse_customer(&mut cur, &mut out)?,
            b"merchant" => parse_merchant(&mut cur, &mut out)?,
            b"terminal" => parse_terminal(&mut cur, &mut out)?,
            b"last_transaction" => parse_last_tx(&mut cur, &mut out)?,
            _ => cur.skip_value(),
        }
    }

    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAYLOAD_NULL_LAST: &[u8] = br#"{
        "id": "tx-1329056812",
        "transaction": {"amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z"},
        "customer": {"avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"]},
        "merchant": {"id": "MERC-016", "mcc": "5411", "avg_amount": 60.25},
        "terminal": {"is_online": false, "card_present": true, "km_from_home": 29.2331036248},
        "last_transaction": null
    }"#;

    const PAYLOAD_WITH_LAST: &[u8] = br#"{
        "id": "tx-3576980410",
        "transaction": {"amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z"},
        "customer": {"avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009", "MERC-001"]},
        "merchant": {"id": "MERC-001", "mcc": "5912", "avg_amount": 298.95},
        "terminal": {"is_online": false, "card_present": true, "km_from_home": 13.7090520965},
        "last_transaction": {"timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.8626479774}
    }"#;

    #[test]
    fn parses_null_last_tx() {
        let req = parse_request(PAYLOAD_NULL_LAST).expect("parse failed");
        assert_eq!(req.tx_amount, 41.12);
        assert_eq!(req.tx_installments, 2.0);
        assert_eq!(req.tx_requested_at, b"2026-03-11T18:45:53Z");
        assert_eq!(req.customer_avg_amount, 82.24);
        assert_eq!(req.customer_tx_count_24h, 3.0);
        assert_eq!(req.merchant_id, b"MERC-016");
        assert_eq!(req.merchant_mcc, b"5411");
        assert_eq!(req.terminal_is_online, false);
        assert_eq!(req.terminal_card_present, true);
        assert!(req.last_tx_timestamp.is_none());
        assert!(req.last_tx_km_from_current.is_none());
        assert_eq!(req.known_merchants, vec![b"MERC-003", b"MERC-016"]);
    }

    #[test]
    fn parses_with_last_tx() {
        let req = parse_request(PAYLOAD_WITH_LAST).expect("parse failed");
        assert_eq!(req.tx_amount, 384.88);
        assert_eq!(req.last_tx_timestamp, Some(&b"2026-03-11T14:58:35Z"[..]));
        assert!((req.last_tx_km_from_current.unwrap() - 18.8626479774).abs() < 1e-9);
    }
}
