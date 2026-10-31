//! Fuel's L1 controller protocol parser and reply model.
//!
//! This state owns packet framing, controller identity records, console
//! commands and power events. The UART register file and receive pacing live
//! in the shared Ns16550 core.
//!
//! Ported from hw/misc/sgi-l1.c in jamesbraid/qemu-sgi-origami.

use super::*;

#[derive(Clone, Debug)]
pub struct L1Config {
    pub rack: u32,
    pub bay: u32,
    pub node_position: u8,
    pub type_code: u32,
    pub revision: Option<[u32; 3]>,
    pub mac: [u8; 6],
    pub brick_serial: Option<String>,
    pub auto_power: bool,
}
impl Default for L1Config {
    fn default() -> Self {
        Self {
            rack: 1,
            bay: 1,
            node_position: 0,
            type_code: 0x41,
            revision: None,
            mac: [8, 0, 0x69, 0x12, 0x34, 0x56],
            brick_serial: None,
            auto_power: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum L1Event {
    Power(bool),
    Reset,
    SoftReset,
    Nmi,
    Shutdown,
}

pub(super) struct State {
    pub config: L1Config,
    pub queue: Arc<Mutex<VecDeque<u8>>>,
    frame: Vec<u8>,
    in_frame: bool,
    escape: bool,
    overflow: bool,
    pub display: String,
    pub records: [[Vec<u8>; 3]; 16],
    pub actions: Vec<L1Event>,
    pub output: Vec<u8>,
    pub powered: bool,
    console: bool,
    escaped: bool,
    command: Vec<u8>,
    port: u8,
    debug: u32,
    partition: u32,
    two_bit: u32,
    reasons: [u32; 2],
    events: Option<u8>,
    power_down_pending: bool,
}
fn text_field(out: &mut Vec<u8>, text: &[u8]) {
    out.push(1);
    out.extend_from_slice(text);
    out.push(0);
}
pub(super) fn ints(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for v in values {
        out.push(0);
        out.extend_from_slice(&v.to_be_bytes());
    }
    out
}
fn word(p: &[u8]) -> u32 {
    u32::from_be_bytes(p[..4].try_into().unwrap())
}
fn arguments(mut p: &[u8], count: u8) -> Option<Vec<&[u8]>> {
    let mut fields = Vec::new();
    for _ in 0..count {
        let tag = *p.first()?;
        let len = match tag {
            0 => 5,
            1 => p[1..].iter().position(|b| *b == 0)? + 2,
            t if t & 0x80 != 0 => 1 + (t & 0x7f) as usize,
            _ => return None,
        };
        fields.push(p.get(..len)?);
        p = &p[len..];
    }
    if p.is_empty() {
        Some(fields)
    } else {
        None
    }
}
fn area_finish(mut area: Vec<u8>) -> Vec<u8> {
    while (area.len() + 1) % 8 != 0 {
        area.push(0);
    }
    let sum = area.iter().fold(0u8, |s, b| s.wrapping_add(*b));
    area.push(sum.wrapping_neg());
    area
}
fn ia_text(area: &mut Vec<u8>, text: &str) {
    area.push(0xc0 | text.len() as u8);
    area.extend_from_slice(text.as_bytes());
}
fn board(mac: Option<[u8; 6]>) -> Vec<u8> {
    let mut b = vec![0; 6];
    for s in ["SYNTHETIC", "IP34", "EMUN00", "030-1707-003"] {
        ia_text(&mut b, s);
    }
    b.push(0);
    ia_text(&mut b, "-G");
    b.extend_from_slice(&[1, 0]);
    ia_text(&mut b, "00");
    for _ in 0..3 {
        b.extend_from_slice(&[4, 0, 0, 0, 0]);
    }
    if let Some(mac) = mac {
        ia_text(
            &mut b,
            &mac.iter().map(|v| format!("{v:02X}")).collect::<String>(),
        );
    }
    b.push(0xc1);
    // The count includes the checksum and padding, not a separate header.
    b[1] = (b.len() + 1).div_ceil(8) as u8;
    area_finish(b)
}
impl State {
    pub fn new(config: L1Config, queue: Arc<Mutex<VecDeque<u8>>>) -> Self {
        let mut records: [[Vec<u8>; 3]; 16] =
            std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
        let chassis = area_finish(vec![0, 2, 0, 0xc2, b'N', b'A', 0xc2, b'N', b'A', 0xc1]);
        for c in 0..2 {
            records[c][0] = chassis.clone();
            records[c][1] = board(if c == 1 { Some(config.mac) } else { None });
        }
        let powered = config.auto_power;
        let port = config.node_position << 5;
        Self {
            config,
            queue,
            frame: Vec::new(),
            in_frame: false,
            escape: false,
            overflow: false,
            display: "L1 running".into(),
            records,
            actions: Vec::new(),
            output: Vec::new(),
            powered,
            console: powered,
            escaped: false,
            command: Vec::new(),
            port,
            debug: 0,
            partition: 0,
            two_bit: 0,
            reasons: [0; 2],
            events: None,
            power_down_pending: false,
        }
    }
    pub fn reset(&mut self) {
        // Controller settings survive the controller's local boot. The UART and
        // packet parser reset independently of those persistent settings.
        let persistent = (self.debug, self.partition, self.two_bit, self.reasons);
        let events = self.events.filter(|&channel| channel >= 4);
        let records = std::mem::replace(
            &mut self.records,
            std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
        );
        let queue = self.queue.clone();
        queue.lock().clear();
        *self = Self::new(self.config.clone(), queue);
        (self.debug, self.partition, self.two_bit, self.reasons) = persistent;
        self.events = events;
        self.records = records;
    }
    fn emit(&mut self, payload: &[u8]) {
        if payload.len() > 128 {
            crate::dlog!(LogModule::L1, "L1 reply exceeds frame capacity");
            return;
        }
        let bytes = encode(payload);
        let mut queue = self.queue.lock();
        if queue.len() + bytes.len() <= 4096 {
            queue.extend(bytes);
        } else {
            crate::dlog!(LogModule::L1, "L1 receive queue full");
        }
    }
    fn reply(&mut self, channel: u8, code: i32, count: u8, fields: &[u8]) {
        let mut p = vec![0x20 | channel];
        p.extend_from_slice(&code.to_be_bytes());
        p.push(count);
        p.extend_from_slice(fields);
        self.emit(&p);
    }
    pub fn parse(&mut self, mut byte: u8) {
        if byte == 0x7e {
            if self.in_frame
                && !self.escape
                && !self.overflow
                && self.frame.len() >= 3
                && self.frame.iter().fold(0xffff, |v, b| fcs(v, *b)) == 0xf0b8
            {
                let payload = self.frame[..self.frame.len() - 2].to_vec();
                self.request(&payload);
            }
            self.in_frame = true;
            self.escape = false;
            self.overflow = false;
            self.frame.clear();
            return;
        }
        if !self.in_frame || self.overflow {
            return;
        }
        if byte == 0x7d {
            self.escape = true;
            return;
        }
        if self.escape {
            byte ^= 0x20;
            self.escape = false;
        }
        if self.frame.len() == 258 {
            self.overflow = true;
        } else {
            self.frame.push(byte);
        }
    }
    fn eeprom(&mut self, channel: u8, a: &[&[u8]]) {
        if a.len() < 4 || a[..4].iter().any(|v| v[0] != 0) {
            self.reply(channel, -100, 0, &[]);
            return;
        }
        let component = word(&a[0][1..]) as usize;
        let area = word(&a[1][1..]);
        let offset = word(&a[2][1..]) as usize;
        let requested = word(&a[3][1..]) as usize;
        if component >= 16 || self.records[component].iter().all(Vec::is_empty) {
            self.reply(channel, -104, 0, &[]);
            return;
        }
        if area > 4 {
            self.reply(channel, -105, 0, &[]);
            return;
        }
        let index = match area {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => {
                self.reply(channel, -104, 0, &[]);
                return;
            }
        };
        let record = &self.records[component][index];
        if record.is_empty() {
            self.reply(channel, -104, 0, &[]);
            return;
        }
        let checksum_len = if area == 4 {
            record.len().min(64)
        } else {
            record.len()
        };
        if record[..checksum_len]
            .iter()
            .fold(0u8, |s, b| s.wrapping_add(*b))
            != 0
        {
            self.reply(channel, -107, 0, &[]);
            return;
        }
        if offset >= record.len() {
            self.reply(channel, -105, 0, &[]);
            return;
        }
        let len = requested.min(record.len() - offset);
        if len > 116 {
            crate::dlog!(LogModule::L1, "L1 EEPROM reply exceeds frame capacity");
            self.reply(channel, -105, 0, &[]);
            return;
        }
        let mut fields = ints(&[(record.len() - offset - len) as u32]);
        if requested != 0 {
            fields.push(0x80 | len as u8);
            fields.extend_from_slice(&record[offset..offset + len]);
        }
        self.reply(channel, 0, if requested == 0 { 1 } else { 2 }, &fields);
    }
    fn request(&mut self, p: &[u8]) {
        let channel = p[0] & 31;
        if p[0] & 0xe0 == 0x40 {
            self.port = (self.port & 0x60) | channel;
            if self.console && self.powered {
                self.output.extend_from_slice(&p[1..]);
            } else {
                crate::dlog!(LogModule::L1, "L1 console output while operator is in L1 mode");
            }
            return;
        }
        if p[0] & 0xe0 != 0 || p.len() < 7 {
            return;
        }
        let address = word(&p[1..]);
        let cmd = u16::from_be_bytes([p[5], p[6]]);
        let key = address >> 12;
        if key != 0xffff && key != ((self.config.rack << 6) | self.config.bay) {
            crate::dlog!(LogModule::L1, "L1 unsupported destination {address:#x}");
            self.emit(&[0x20 | channel, 255, 255, 255, 255, 0, 0, 0, 8]);
            return;
        }
        let count = if p.len() == 7 { 0 } else { p[7] };
        let raw = if p.len() > 8 { &p[8..] } else { &[] };
        let task = address & 31;
        if task == 5 || task == 21 {
            if self.powered {
                self.emit(&[
                    ((address >> 5) & 31) as u8,
                    raw.get(4).copied().unwrap_or(0),
                ]);
            } else {
                let mut field = Vec::new();
                text_field(
                    &mut field,
                    format!(
                        "WARNING: power on {:03}c{:02} appears off!",
                        self.config.rack, self.config.bay
                    )
                    .as_bytes(),
                );
                self.reply(channel, -1, 1, &field);
            }
            return;
        }
        if task != 6 && task != 3 && task != 18 {
            crate::dlog!(LogModule::L1, "L1 unsupported task {task}");
            return;
        }
        let Some(a) = arguments(raw, count) else {
            self.reply(channel, -100, 0, &[]);
            return;
        };
        if task == 3 {
            if a.len() != 1 || a[0][0] != 1 {
                self.reply(channel, -100, 0, &[]);
                return;
            }
            let text = String::from_utf8_lossy(&a[0][1..a[0].len() - 1]);
            let (code, output) = self.run_line(&text, false);
            let mut fields = Vec::new();
            let mut n = 0;
            for line in output.split("\r\n").filter(|s| !s.is_empty()) {
                if fields.len() + line.len() + 2 > 121 {
                    break;
                }
                text_field(&mut fields, line.as_bytes());
                n += 1;
            }
            self.reply(channel, code, n, &fields);
            return;
        }
        if task == 18 {
            let length = a.iter().take(3).map(|v| v.len()).sum::<usize>().min(121);
            self.reply(channel, cmd as i32, count.min(3), &raw[..length]);
            return;
        }
        let mut values = Vec::new();
        let mut fields = Vec::new();
        let mut field_count = 0;
        let mut error = 0;
        match cmd {
            1 => values.push(self.debug),
            2 => values.push((self.config.rack << 6) | self.config.bay),
            3 => {
                if self.config.type_code == 0 {
                    error = -104;
                } else {
                    values.extend([
                        (self.config.rack << 6) | self.config.bay,
                        self.config.type_code,
                    ]);
                }
            }
            4 => {
                if let Some(serial) = &self.config.brick_serial {
                    text_field(&mut fields, serial.as_bytes());
                    field_count = 1;
                } else {
                    error = -104;
                }
            }
            5 => {
                if let Some(rev) = self.config.revision {
                    values.extend(rev);
                } else {
                    crate::dlog!(LogModule::L1, "L1 revision metadata not configured");
                    error = -104;
                }
            }
            6 => {
                self.eeprom(channel, &a);
                return;
            }
            7 => error = if a.is_empty() { -100 } else { -104 },
            8 => {
                let mac = self
                    .config
                    .mac
                    .iter()
                    .map(|b| format!("{b:02X}"))
                    .collect::<String>();
                text_field(&mut fields, mac.as_bytes());
                field_count = 1;
            }
            9 => values.push(self.partition),
            0xa => values.extend([0, 0]),
            0xb => values.push(self.two_bit),
            0xc => values.extend([0; 8]),
            0xd => values.extend([0; 4]),
            0xe => values.push((self.port >> 5) as u32 & 3),
            0xf => values.extend(self.reasons),
            0x10 => values.extend([1, 1, 1, 1]),
            0x11 => {
                error = -106;
                values.push(1);
            }
            0x1001 | 0x1002 | 0x1006 | 0x1007 | 0x1008 => {
                if a.is_empty() {
                    error = -100;
                } else if a[0][0] != 0 {
                    error = -105;
                } else {
                    let v = word(&a[0][1..]);
                    match cmd {
                        0x1001 => self.debug = v & 0xffff,
                        0x1002 => self.port = ((self.port as u32 & 0x60) | v) as u8 & 0x7f,
                        0x1006 => self.partition = v,
                        0x1007 => self.events = Some(v as u8),
                        _ => self.two_bit = v & 3,
                    }
                }
            }
            0x1003 => {}
            0x1004 | 0x1005 => {
                if a.is_empty() {
                    error = -100;
                } else if a[0][0] != 1 {
                    error = -102;
                } else {
                    self.display = a[0][1..a[0].len() - 1]
                        .iter()
                        .take(72)
                        .map(|b| {
                            if (0x20..=0x7e).contains(b) {
                                *b as char
                            } else {
                                '?'
                            }
                        })
                        .collect();
                }
            }
            0x1009 => {
                if a.len() < 2 {
                    error = -100;
                } else if a[0][0] != 0 || a[1][0] != 0 || word(&a[1][1..]) & 15 > 2 {
                    error = -105;
                }
            }
            0x100a => self.events = None,
            0x100b => {
                if a.len() != 3 {
                    error = -100;
                } else {
                    self.events = None;
                }
            }
            0x100c => {
                if a.len() != 2 {
                    error = -100;
                }
            }
            0x2001 | 0x2005 | 0x2006 => {
                self.reply(channel, 0, 0, &[]);
                match cmd {
                    0x2001 => self.reset_node(),
                    0x2005 => self.power_down(true, false),
                    _ => {
                        self.power_down(false, false);
                        self.power_up(true);
                    }
                }
                return;
            }
            0x2002..=0x2004 => error = -100,
            0x2007 | 0x2008 => error = -108,
            0x4000 => error = -104,
            0x4002 => {
                self.reply(channel, 0, 0, &[]);
                self.reboot();
                return;
            }
            _ => {
                crate::dlog!(LogModule::L1, "L1 unsupported request {cmd:#x}");
                error = -101;
            }
        }
        if !values.is_empty() {
            field_count = values.len() as u8;
            fields = ints(&values);
        }
        self.reply(channel, error, field_count, &fields);
    }
    fn reason(&mut self) {
        self.reasons[1] = self.reasons[0];
        self.reasons[0] = 3;
    }
    fn reset_node(&mut self) {
        if self.powered {
            self.reason();
            self.display = "Reset done".into();
            self.actions.push(L1Event::Reset);
        }
    }
    fn power_down(&mut self, shutdown: bool, confirm: bool) {
        if self.powered {
            if confirm && !self.power_down_pending {
                self.power_down_pending = true;
                self.event(
                    0x8000,
                    0,
                    "Power Down: issue 'pwr d' again to immediately power down.",
                );
                return;
            }
            self.power_down_pending = false;
            self.reason();
            self.powered = false;
            self.display = "Powered Down".into();
            self.actions.push(L1Event::Power(false));
            if shutdown {
                self.actions.push(L1Event::Shutdown);
            }
        }
    }
    fn power_up(&mut self, restart: bool) {
        self.reason();
        self.powered = true;
        self.power_down_pending = false;
        self.port &= 0x60;
        self.console = true;
        self.display = "Powered Up".into();
        self.actions.push(L1Event::Power(true));
        if restart {
            self.actions.push(L1Event::Reset);
        }
    }
    fn reboot(&mut self) {
        self.power_down_pending = false;
        self.console = false;
        self.escaped = false;
        self.command.clear();
        self.frame.clear();
        self.in_frame = false;
        self.port = self.config.node_position << 5;
        self.display = "L1 running".into();
        if self.events.is_some_and(|c| c < 4) {
            self.events = None;
        }
    }
    pub fn event(&mut self, code: u32, number: u32, text: &str) {
        let Some(channel) = self.events else {
            return;
        };
        let mut p = vec![0x40 | (channel & 31)];
        p.extend_from_slice(
            &((self.config.rack << 18) | (self.config.bay << 12) | 4).to_be_bytes(),
        );
        p.extend_from_slice(&code.to_be_bytes());
        p.push(2);
        p.extend(ints(&[number]));
        text_field(
            &mut p,
            format!(
                "\r\n{:03}c{:02} ATTN: {text}\r\n",
                self.config.rack, self.config.bay
            )
            .as_bytes(),
        );
        if p.len() <= 127 {
            self.emit(&p);
        }
    }
    fn run_line(&mut self, line: &str, interactive: bool) -> (i32, String) {
        let words = line.split_whitespace().collect::<Vec<_>>();
        let local = if words.first() == Some(&"*") {
            &words[1..]
        } else {
            &words[..]
        };
        match local {
            ["pwr" | "power", "d" | "down"] => {
                self.power_down(true, interactive);
                (0, String::new())
            }
            ["pwr" | "power", "u" | "up"] => {
                if !self.powered {
                    self.power_up(true);
                }
                (0, String::new())
            }
            ["reset"] => {
                self.reset_node();
                (0, String::new())
            }
            ["softreset"] => {
                self.actions.push(L1Event::SoftReset);
                self.display = "Sft rst done".into();
                (0, String::new())
            }
            ["nmi"] => {
                self.actions.push(L1Event::Nmi);
                self.display = "NMI done".into();
                (0, String::new())
            }
            ["reboot_l1"] => {
                self.reboot();
                (0, String::new())
            }
            ["l1"] => {
                self.escaped = false;
                (0, String::new())
            }
            ["version"] => {
                if let Some(r) = self.config.revision {
                    (
                        0,
                        format!("L1 Controller: Revision {}.{}.{}\r\n", r[0], r[1], r[2]),
                    )
                } else {
                    (
                        -104,
                        "L1 firmware revision metadata not configured\r\n".into(),
                    )
                }
            }
            ["debug"] if words.len() == 1 => (
                0,
                format!("debug switches set to 0x{:04x}\n\r\n", self.debug),
            ),
            ["debug", v] if words.len() == 2 => {
                let n = if let Some(hex) = v.strip_prefix("0x") {
                    u32::from_str_radix(hex, 16)
                } else {
                    v.parse()
                };
                match n {
                    Ok(v) => {
                        self.debug = v & 0xffff;
                        (
                            0,
                            format!("debug switches set to 0x{:04x}\n\r\n", self.debug),
                        )
                    }
                    Err(_) => (-1200, "ERROR: command not found.\n\r\n".into()),
                }
            }
            _ => {
                crate::dlog!(LogModule::L1, "L1 unsupported operator command {line:?}");
                (-1, "ERROR: command not found.\r\n".into())
            }
        }
    }
    fn prompt(&mut self) {
        self.output.extend_from_slice(
            format!("{:03}c{:02}-L1>", self.config.rack, self.config.bay).as_bytes(),
        );
    }
    pub fn console_input(&mut self, byte: u8) {
        if self.console {
            if byte == 0x14 {
                self.console = false;
                self.escaped = true;
                self.output
                    .extend_from_slice(b"escaping to L1 system controller\r\n");
                self.prompt();
            } else if self.powered {
                self.emit(&[self.port & 31, byte]);
            } else {
                crate::dlog!(LogModule::L1, "L1 console input while powered off");
            }
        } else if byte == 4 {
            self.command.clear();
            self.console = true;
        } else if byte == b'\r' || byte == b'\n' {
            let line = String::from_utf8_lossy(&self.command).into_owned();
            self.command.clear();
            let (_, output) = self.run_line(&line, true);
            self.output.extend_from_slice(output.as_bytes());
            if self.escaped {
                self.console = true;
            } else {
                self.prompt();
            }
        } else if byte == 8 || byte == 127 {
            self.command.pop();
        } else if self.command.len() < 255 {
            self.command.push(byte);
        }
    }
}
