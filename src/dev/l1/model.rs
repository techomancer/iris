//! Fuel's L1 controller protocol parser and reply model.
//!
//! This state owns packet framing, controller identity records, console
//! commands and power events. The UART register file and receive pacing live
//! in the shared Ns16550 core.
//!
//! Ported from hw/misc/sgi-l1.c in jamesbraid/qemu-sgi-origami.

use super::*;

/// Firmware revision the model reports for request 0x0005.
///
/// 1.48.1 is the revision a real Fuel's L1 prints. The model reproduces
/// that firmware's replies, not its code, so the revision is a label.
pub const DEFAULT_REVISION: [u32; 3] = [1, 48, 1];

/// Brick serial the model reports for request 0x0004.
///
/// A made-up serial, `EMU` for the emulator, system-board letter `N` and
/// instance 0. The real format is unknown, and the value must not pass as
/// a real SGI serial.
pub const DEFAULT_BRICK_SERIAL: &str = "EMUN00";

/// Parse a "major.minor.patch" revision string, as a `[fuel] l1_revision`
/// override spells it. Each part is a decimal number.
pub fn parse_revision(text: &str) -> Result<[u32; 3], String> {
    let bad = || format!("L1 revision \"{text}\" must be major.minor.patch, each a decimal number");
    let mut parts = text.split('.');
    let mut rev = [0u32; 3];
    for slot in &mut rev {
        let part = parts.next().ok_or_else(bad)?;
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return Err(bad());
        }
        *slot = part.parse().map_err(|_| bad())?;
    }
    if parts.next().is_some() {
        return Err(bad());
    }
    Ok(rev)
}

#[derive(Clone, Debug)]
pub struct L1Config {
    pub rack: u32,
    pub bay: u32,
    pub node_position: u8,
    pub type_code: u32,
    pub revision: Option<[u32; 3]>,
    pub mac: [u8; 6],
    pub brick_serial: Option<String>,
    /// The system serial request 0x0008 answers with. `None` answers with
    /// the station address in hex, as the Fuel's L1 does. It names the
    /// system, so every L1 of a machine is given the same text; the board
    /// serials in the records stay as they are, because the PROM ties its
    /// NVRAM to the node board's serial.
    pub system_serial: Option<String>,
    pub auto_power: bool,
    /// First integer of the port-speed reply (request 0xa). The PROM uses it
    /// to select a display branch; it is not a link speed. Zero or one.
    pub port_speed_first: u32,
    /// Answer the router command-interpreter `brick` request. Only the brick
    /// types in `router_type_name` have a reply.
    pub router: bool,
    /// The CPU configuration the node board's record carries, if any.
    pub cpu_context: Option<CpuContext>,
    /// The node board's identity in component 0's board area. Unset, the
    /// board is the Fuel's system board, which answers as both the node
    /// board and the I/O board, the I/O area carrying the station address.
    /// Set, component 0 names this board and no I/O board record is given,
    /// as QEMU's L1 does for an IP53 node.
    pub node_board: Option<BoardIdentity>,
}

/// A board's identity as its board area names it: product name, part number
/// and revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoardIdentity {
    pub name: &'static str,
    pub part: &'static str,
    pub revision: &'static str,
}
impl BoardIdentity {
    /// The Fuel's system board as the model names it: a placeholder, not a
    /// read record.
    pub const FUEL: Self = Self { name: "IP34", part: "030-1707-003", revision: "-G" };
}

/// The CPU configuration an IP35 node board's EEPROM record carries beside
/// its identity: the processor mode word and the processor and Hub clocks.
/// At start-up the PROM checks the configuration block in its boot flash
/// against these, and when a board record carries none it falls back to a
/// table row chosen by the board word; a block that matches neither draws
/// "CPU configuration values mismatch". The encoding is the one QEMU's L1
/// gives the Tezro and Origin 350 boards, which the stock PROM 6.210
/// accepts; no factory record has been read, so the bytes are a software
/// input, not a capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuContext {
    /// The full 32-bit mode word, as the configuration block holds it.
    pub mode_word: u32,
    pub cpu_mhz: u32,
    pub hub_mhz: u32,
}
impl Default for L1Config {
    fn default() -> Self {
        Self {
            rack: 1,
            bay: 1,
            node_position: 0,
            type_code: 0x41,
            revision: Some(DEFAULT_REVISION),
            mac: [8, 0, 0x69, 0x12, 0x34, 0x56],
            brick_serial: Some(DEFAULT_BRICK_SERIAL.into()),
            system_serial: None,
            auto_power: true,
            port_speed_first: 0,
            router: false,
            cpu_context: None,
            node_board: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum L1Event {
    Power(bool),
    Reset,
    Shutdown,
}

pub(super) struct State {
    pub config: L1Config,
    pub queue: Arc<Mutex<VecDeque<u8>>>,
    frame: Vec<u8>,
    in_frame: bool,
    escape: bool,
    overflow: bool,
    pub records: [[Vec<u8>; 3]; 16],
    pub actions: Vec<L1Event>,
    pub output: Vec<u8>,
    pub powered: bool,
    console: bool,
    escaped: bool,
    command: Vec<u8>,
    port: u8,
    events: Option<u8>,
    power_down_pending: bool,
    /// One reassembly per subchannel.
    router: [Option<RouterAssembly>; 32],
}

/// A router message being reassembled from 0x3001, 0x3002 and 0x3003 chunks.
struct RouterAssembly {
    fragment_count: u8,
    fragments_left: u8,
    first_length: usize,
    expected: usize,
    received: usize,
    data: Vec<u8>,
}
const ROUTER_MAX_FRAGMENT: usize = 0x400;
const ROUTER_CHUNK_DATA: usize = 113;

/// Brick types for which the router interpreter reply exists.
fn router_type_name(type_code: u32) -> Option<&'static str> {
    match type_code {
        0x37 => Some("Chimera Server"),
        0x39 => Some("Chimera WS"),
        0x51 => Some("Rackmount WS"),
        _ => None,
    }
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
fn board(identity: BoardIdentity, mac: Option<[u8; 6]>, cpu: Option<CpuContext>) -> Vec<u8> {
    let mut b = vec![0; 6];
    for s in ["SYNTHETIC", identity.name, "EMUN00", identity.part] {
        ia_text(&mut b, s);
    }
    b.push(0);
    ia_text(&mut b, identity.revision);
    b.extend_from_slice(&[1, 0]);
    ia_text(&mut b, "00");
    if let Some(cpu) = cpu {
        // Format 2 and a one-byte binary 3 select the CPU fields: the Hub
        // mode bits, Config0 to Config3, the Hub and CPU clocks in MHz and
        // three minimum versions. Only Config0 and the clocks are known;
        // the rest are empty fields, not zeros.
        b[0] = 2;
        b.extend_from_slice(&[1, 3, 0]);
        b.push(4);
        b.extend_from_slice(&cpu.mode_word.to_be_bytes());
        b.extend_from_slice(&[0, 0, 0]);
        for mhz in [cpu.hub_mhz, cpu.cpu_mhz] {
            b.push(4);
            b.extend_from_slice(&mhz.to_be_bytes());
        }
        b.extend_from_slice(&[0, 0, 0]);
    }
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
        let identity = config.node_board.unwrap_or(BoardIdentity::FUEL);
        records[0] = [chassis.clone(), board(identity, None, config.cpu_context), Vec::new()];
        if config.node_board.is_none() {
            records[1] = [chassis, board(identity, Some(config.mac), None), Vec::new()];
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
            records,
            actions: Vec::new(),
            output: Vec::new(),
            powered,
            console: powered,
            escaped: false,
            command: Vec::new(),
            port,
            events: None,
            power_down_pending: false,
            router: std::array::from_fn(|_| None),
        }
    }
    pub fn reset(&mut self) {
        // The event subchannel survives the controller's local boot if it is
        // one of the operating system's. The UART and packet parser reset.
        let events = self.events.filter(|&channel| channel >= 4);
        let records = std::mem::replace(
            &mut self.records,
            std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
        );
        let queue = self.queue.clone();
        queue.lock().clear();
        *self = Self::new(self.config.clone(), queue);
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
        if self.config.router
            && p.len() >= 7
            && p[0] & 0xe0 == 0
            && word(&p[1..]) == 0
            && p[5] == 0x30
            && matches!(p[6], 1..=3)
        {
            self.router_chunk(p);
            return;
        }
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
        if task != 6 && task != 3 {
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
        let mut values = Vec::new();
        let mut fields = Vec::new();
        let mut field_count = 0;
        let mut error = 0;
        // Requests that neither the stock PROM nor IRIX sends are refused
        // as unknown, and logged.
        match cmd {
            // No debug switch set.
            1 => values.push(0),
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
            // Without a configured system serial it is the station address
            // as twelve uppercase hexadecimal digits in address order, no
            // separators or OUI added: the text the stock PROM converts
            // straight back into the Ethernet address.
            8 => {
                let serial = self.config.system_serial.clone().unwrap_or_else(|| {
                    self.config.mac.iter().map(|b| format!("{b:02X}")).collect()
                });
                text_field(&mut fields, serial.as_bytes());
                field_count = 1;
            }
            // Not partitioned.
            9 => values.push(0),
            0xa => values.extend([self.config.port_speed_first, 0]),
            // Zero: the PROM's partition check goes no further.
            0xb => values.push(0),
            0xc => values.extend([0; 8]),
            0xe => values.push((self.port >> 5) as u32 & 3),
            0x1007 => {
                if a.is_empty() {
                    error = -100;
                } else if a[0][0] != 0 {
                    error = -105;
                } else {
                    self.events = Some(word(&a[0][1..]) as u8);
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
            // The expander byte, which IRIX's VPro driver asks for, is only
            // on router and I/O bricks: refused on this one.
            0x2007 | 0x2008 => error = -108,
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
    /// The router replies are chunked and reassembled with the same three
    /// commands; a malformed chunk drops the pending message.
    fn router_chunk(&mut self, p: &[u8]) {
        let channel = p[0] & 0x1f;
        let slot = usize::from(channel);
        if p.len() < 10 {
            crate::dlog!(LogModule::L1, "L1 malformed router chunk");
            return;
        }
        let command = u16::from_be_bytes([p[5], p[6]]);
        let payload = &p[10..];
        let fragment_length = u16::from_be_bytes([p[8], p[9]]) as usize;
        match command {
            0x3001 => {
                let fragments = p[7];
                if self.router[slot].take().is_some() {
                    crate::dlog!(LogModule::L1, "L1 replacing pending router request");
                }
                if fragments == 0
                    || fragments >= 128
                    || fragment_length == 0
                    || fragment_length > ROUTER_MAX_FRAGMENT
                    || payload.len() > fragment_length
                {
                    crate::dlog!(LogModule::L1, "L1 unsupported router message start");
                    return;
                }
                self.router[slot] = Some(RouterAssembly {
                    fragment_count: fragments,
                    fragments_left: fragments,
                    first_length: 0,
                    expected: fragment_length,
                    received: payload.len(),
                    data: payload.to_vec(),
                });
            }
            0x3002 => {
                let ok = self.router[slot].as_ref().is_some_and(|a| {
                    a.fragments_left != 0
                        && a.received == a.expected
                        && fragment_length != 0
                        && fragment_length <= ROUTER_MAX_FRAGMENT - a.data.len()
                        && payload.len() <= fragment_length
                });
                let Some(a) = self.router[slot].as_mut().filter(|_| ok) else {
                    self.router[slot] = None;
                    crate::dlog!(LogModule::L1, "L1 invalid router next fragment");
                    return;
                };
                a.expected = fragment_length;
                a.received = payload.len();
                a.data.extend_from_slice(payload);
            }
            _ => {
                let ok = self.router[slot].as_ref().is_some_and(|a| {
                    a.received != a.expected
                        && payload.len() <= a.expected - a.received
                        && payload.len() <= ROUTER_MAX_FRAGMENT - a.data.len()
                });
                let Some(a) = self.router[slot].as_mut().filter(|_| ok) else {
                    self.router[slot] = None;
                    crate::dlog!(LogModule::L1, "L1 router continuation without valid message");
                    return;
                };
                a.received += payload.len();
                a.data.extend_from_slice(payload);
            }
        }
        self.router_complete(channel);
    }
    fn router_complete(&mut self, channel: u8) {
        let slot = usize::from(channel);
        let Some(a) = self.router[slot].as_mut() else { return };
        if a.received != a.expected {
            return;
        }
        if a.first_length == 0 {
            a.first_length = a.data.len();
        }
        a.fragments_left -= 1;
        if a.fragments_left != 0 {
            return;
        }
        let a = self.router[slot].take().unwrap();
        // Only the single-fragment layout has a known descriptor mapping.
        if a.fragment_count == 1 {
            self.router_reply(channel, &a.data[..a.first_length], a.fragment_count);
        } else {
            crate::dlog!(LogModule::L1, "L1 multi-fragment router dispatch is unsupported");
        }
    }
    fn router_reply(&mut self, channel: u8, request: &[u8], fragment_count: u8) {
        let Some(type_name) = router_type_name(self.config.type_code) else {
            crate::dlog!(LogModule::L1, "L1 unsupported or malformed router request");
            return;
        };
        let n = request.len();
        // Fixed header, an all-zero scalar block, then the string descriptor
        // (kind 0x0a, 1) carrying the interpreter command.
        let well_formed = n >= 37
            && usize::from(u16::from_be_bytes([request[0], request[1]])) == n
            && request[2] == 2
            && request[3] == 0
            && request[5] == 0x80 | fragment_count
            && request[6] == 0
            && request[7] == 0
            && word(&request[8..]) == 0x0fff_f003
            && request[16] == 2
            && request[20..28] == [0; 8]
            && request[28] == 0x0a
            && request[29] == 1
            && word(&request[32..]) == 52
            && usize::from(u16::from_be_bytes([request[30], request[31]])) == n - 36
            && request[n - 1] == 0;
        if !well_formed {
            crate::dlog!(LogModule::L1, "L1 unsupported or malformed router request");
            return;
        }
        // The captured form has one trailing space; the plain spelling is
        // accepted too. Anything else is refused.
        if !matches!(&request[36..], b"brick\0" | b"brick \0") {
            crate::dlog!(LogModule::L1, "L1 router command interpreter operation refused");
            return;
        }
        // The node is not partitioned (see request 9).
        let partition = "none";
        let serial = self.config.brick_serial.as_deref().unwrap_or("(null)");
        let mut text = format!(
            "rack: {:03}, slot: {:02}, partition: {partition}, type: {type_name} \
             [?MB flash], serial:{serial}, source: OTHER\n\n",
            self.config.rack, self.config.bay
        )
        .into_bytes();
        text.push(0);
        let total = 36 + text.len();
        let node_id = (self.config.rack << 6) | self.config.bay;
        let mut response = vec![0u8; 36];
        response[..2].copy_from_slice(&(total as u16).to_be_bytes());
        response[2] = 2;
        response[4] = request[4];
        response[5] = 0x81;
        response[6] = 1;
        // The reply goes back to the requesting channel on this node.
        response[8..12].copy_from_slice(&((node_id << 12) | (u32::from(channel) << 5) | 0x15).to_be_bytes());
        response[12..16].copy_from_slice(&((node_id << 12) | 3).to_be_bytes());
        response[16] = 2;
        response[17] = request[17];
        response[18..20].copy_from_slice(&request[18..20]);
        response[28] = 0x0a;
        response[29] = 1;
        response[30..32].copy_from_slice(&(text.len() as u16).to_be_bytes());
        response[32..36].copy_from_slice(&0x34u32.to_be_bytes());
        response.extend_from_slice(&text);
        if total > ROUTER_MAX_FRAGMENT {
            crate::dlog!(LogModule::L1, "L1 router brick response exceeds fragment limit");
            return;
        }
        for (i, part) in response.chunks(ROUTER_CHUNK_DATA).enumerate() {
            let mut chunk = vec![0x20 | channel, 0, 0, 0, 0, 0x30, if i == 0 { 1 } else { 3 }];
            if i == 0 {
                chunk.push(1);
                chunk.extend_from_slice(&(total as u16).to_be_bytes());
            } else {
                chunk.extend_from_slice(&[0, 0, 0]);
            }
            chunk.extend_from_slice(part);
            self.emit(&chunk);
        }
    }
    fn reset_node(&mut self) {
        if self.powered {
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
            self.powered = false;
            self.actions.push(L1Event::Power(false));
            if shutdown {
                self.actions.push(L1Event::Shutdown);
            }
        }
    }
    fn power_up(&mut self, restart: bool) {
        self.powered = true;
        self.power_down_pending = false;
        self.port &= 0x60;
        self.console = true;
        self.actions.push(L1Event::Power(true));
        if restart {
            self.actions.push(L1Event::Reset);
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
            ["l1"] => {
                self.escaped = false;
                (0, String::new())
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
