//! Minimal telnet (RFC 854/855/857/858/1091/856) option negotiation for the
//! emulator's TCP-facing endpoints (serial ports and monitor).
//!
//! The serial-console endpoint wants character-at-a-time input with no client
//! local echo and no CR/LF translation, so on accept the server offers:
//!   WILL ECHO, WILL SGA, WILL BINARY, DO BINARY.
//! Clients that recognize these (BSD telnet, PuTTY raw-telnet, etc.) will then
//! suppress local echo, drop line buffering and stop mangling 0x0D/0x0A.
//!
//! We don't implement the full RFC 1143 "Q method" state machine — we just
//! track the last reply we sent for each option and only emit a new reply if
//! the desired state actually changed. That's enough to avoid the WILL/DO
//! ping-pong loops the naive "always reply" approach causes.

use std::collections::HashMap;
use std::io::{self, Read, Write};

pub const IAC: u8 = 255;
pub const DONT: u8 = 254;
pub const DO: u8 = 253;
pub const WONT: u8 = 252;
pub const WILL: u8 = 251;
pub const SB: u8 = 250;
pub const SE: u8 = 240;

pub const OPT_BINARY: u8 = 0;
pub const OPT_ECHO: u8 = 1;
pub const OPT_SGA: u8 = 3;

#[derive(Debug, Clone, Copy)]
enum State {
    Data,
    Iac,
    Will,
    Wont,
    Do,
    Dont,
    Sb,
    SbIac,
}

pub struct TelnetFilter {
    state: State,
    // Last WILL/WONT we sent for each option (true = WILL, false = WONT).
    we_will: HashMap<u8, bool>,
    // Last DO/DONT we sent for each option (true = DO, false = DONT).
    we_do: HashMap<u8, bool>,
}

impl TelnetFilter {
    /// Character-at-a-time mode: server will echo, suppress go-ahead, and run
    /// the link 8-bit clean. Use for raw serial console endpoints.
    pub fn new() -> Self {
        // Pre-populate with the initial handshake state so the client's
        // confirming replies don't cause us to re-send.
        let mut we_will = HashMap::new();
        we_will.insert(OPT_ECHO, true);
        we_will.insert(OPT_SGA, true);
        we_will.insert(OPT_BINARY, true);
        let mut we_do = HashMap::new();
        we_do.insert(OPT_BINARY, true);
        Self { state: State::Data, we_will, we_do }
    }

    /// Passive / NVT mode: don't initiate anything, decline whatever the
    /// client offers. Leaves the client in default line-buffered + local-echo
    /// behavior — the right fit for a line-oriented endpoint like the
    /// monitor. Still strips inbound IAC sequences and replies politely to
    /// client-initiated negotiation.
    pub fn new_passive() -> Self {
        Self { state: State::Data, we_will: HashMap::new(), we_do: HashMap::new() }
    }

    /// Bytes the server should write immediately after accepting a connection
    /// in character-at-a-time mode. Passive mode sends nothing.
    pub const fn initial_handshake() -> [u8; 12] {
        [
            IAC, WILL, OPT_ECHO,
            IAC, WILL, OPT_SGA,
            IAC, WILL, OPT_BINARY,
            IAC, DO, OPT_BINARY,
        ]
    }

    /// Feed one inbound byte. Returns `Some(data)` if a real data byte fell
    /// out of the state machine. Any required reply bytes are appended to
    /// `out` and must be written back to the client.
    pub fn feed(&mut self, byte: u8, out: &mut Vec<u8>) -> Option<u8> {
        match self.state {
            State::Data => {
                if byte == IAC {
                    self.state = State::Iac;
                    None
                } else {
                    Some(byte)
                }
            }
            State::Iac => match byte {
                IAC => { self.state = State::Data; Some(IAC) }
                WILL => { self.state = State::Will; None }
                WONT => { self.state = State::Wont; None }
                DO => { self.state = State::Do; None }
                DONT => { self.state = State::Dont; None }
                SB => { self.state = State::Sb; None }
                _ => { self.state = State::Data; None }
            },
            State::Will => {
                let desired = byte == OPT_BINARY;
                self.maybe_reply_do(byte, desired, out);
                self.state = State::Data;
                None
            }
            State::Wont => {
                self.maybe_reply_do(byte, false, out);
                self.state = State::Data;
                None
            }
            State::Do => {
                let desired = matches!(byte, OPT_ECHO | OPT_SGA | OPT_BINARY);
                self.maybe_reply_will(byte, desired, out);
                self.state = State::Data;
                None
            }
            State::Dont => {
                self.maybe_reply_will(byte, false, out);
                self.state = State::Data;
                None
            }
            State::Sb => {
                if byte == IAC { self.state = State::SbIac; }
                None
            }
            State::SbIac => {
                if byte == SE { self.state = State::Data; } else { self.state = State::Sb; }
                None
            }
        }
    }

    fn maybe_reply_will(&mut self, opt: u8, desired: bool, out: &mut Vec<u8>) {
        if self.we_will.get(&opt).copied() != Some(desired) {
            out.extend_from_slice(&[IAC, if desired { WILL } else { WONT }, opt]);
            self.we_will.insert(opt, desired);
        }
    }

    fn maybe_reply_do(&mut self, opt: u8, desired: bool, out: &mut Vec<u8>) {
        if self.we_do.get(&opt).copied() != Some(desired) {
            out.extend_from_slice(&[IAC, if desired { DO } else { DONT }, opt]);
            self.we_do.insert(opt, desired);
        }
    }
}

/// Escape one outbound data byte: 0xFF must be doubled so the client doesn't
/// mistake it for the start of a telnet command.
pub fn escape_byte(byte: u8, out: &mut Vec<u8>) {
    if byte == IAC { out.push(IAC); }
    out.push(byte);
}

/// Write adapter that turns bare `\n` into `\r\n` on output. NVT requires the
/// server to emit CRLF for end-of-line, but most of our text code uses bare
/// `\n` via `writeln!`. Tracks the previous byte so a `\r\n` written by the
/// caller passes through unchanged.
pub struct CrlfWriter<W: Write> {
    inner: W,
    last_was_cr: bool,
}

impl<W: Write> CrlfWriter<W> {
    pub fn new(inner: W) -> Self {
        Self { inner, last_was_cr: false }
    }
}

impl<W: Write> Write for CrlfWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut start = 0;
        for (i, &b) in buf.iter().enumerate() {
            if b == b'\n' && !self.last_was_cr {
                if i > start { self.inner.write_all(&buf[start..i])?; }
                self.inner.write_all(b"\r\n")?;
                start = i + 1;
                self.last_was_cr = false;
            } else {
                self.last_was_cr = b == b'\r';
            }
        }
        if start < buf.len() { self.inner.write_all(&buf[start..])?; }
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> { self.inner.flush() }
}

/// Read adapter that owns a duplex byte stream (e.g. a `TcpStream`), strips
/// inbound telnet commands, and writes any negotiation replies back through
/// the same stream. Sends the initial handshake the first time `read` is
/// called.
///
/// Used by line-oriented endpoints (the monitor) that want to plug a
/// `BufReader` on top without thinking about IAC sequences.
pub struct TelnetReader<S: Read + Write> {
    inner: S,
    filter: TelnetFilter,
    handshake: Option<&'static [u8]>,
}

impl<S: Read + Write> TelnetReader<S> {
    /// Character-at-a-time mode. Sends the full WILL/DO handshake on first
    /// read. Use for serial console endpoints.
    pub fn new(inner: S) -> Self {
        const HS: [u8; 12] = TelnetFilter::initial_handshake();
        Self { inner, filter: TelnetFilter::new(), handshake: Some(&HS) }
    }

    /// Passive / NVT mode: no initial offer; just filter inbound IAC. Use for
    /// line-oriented endpoints where the client's default local-echo +
    /// line-edit behavior is exactly what we want.
    pub fn new_passive(inner: S) -> Self {
        Self { inner, filter: TelnetFilter::new_passive(), handshake: None }
    }
}

impl<S: Read + Write> Read for TelnetReader<S> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if let Some(hs) = self.handshake.take() {
            self.inner.write_all(hs)?;
        }
        if buf.is_empty() { return Ok(0); }
        let mut raw = vec![0u8; buf.len()];
        loop {
            let n = self.inner.read(&mut raw)?;
            if n == 0 { return Ok(0); }
            let mut replies = Vec::new();
            let mut out_idx = 0;
            for &b in &raw[..n] {
                if let Some(d) = self.filter.feed(b, &mut replies) {
                    buf[out_idx] = d;
                    out_idx += 1;
                }
            }
            if !replies.is_empty() {
                self.inner.write_all(&replies)?;
            }
            if out_idx > 0 {
                return Ok(out_idx);
            }
            // The chunk was all telnet protocol with no data. Loop and read
            // more; on a blocking socket this is what the caller expects.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_data_passes_through() {
        let mut f = TelnetFilter::new();
        let mut out = Vec::new();
        assert_eq!(f.feed(b'a', &mut out), Some(b'a'));
        assert_eq!(f.feed(b'\r', &mut out), Some(b'\r'));
        assert!(out.is_empty());
    }

    #[test]
    fn iac_iac_yields_ff() {
        let mut f = TelnetFilter::new();
        let mut out = Vec::new();
        assert_eq!(f.feed(IAC, &mut out), None);
        assert_eq!(f.feed(IAC, &mut out), Some(0xFF));
        assert!(out.is_empty());
    }

    #[test]
    fn client_confirming_will_echo_does_not_reply() {
        // After initial_handshake we're WILL ECHO; client replies DO ECHO.
        // We must not send anything in return.
        let mut f = TelnetFilter::new();
        let mut out = Vec::new();
        for &b in &[IAC, DO, OPT_ECHO] { assert_eq!(f.feed(b, &mut out), None); }
        assert!(out.is_empty(), "got unwanted reply: {:?}", out);
    }

    #[test]
    fn client_confirming_do_binary_does_not_reply() {
        let mut f = TelnetFilter::new();
        let mut out = Vec::new();
        for &b in &[IAC, WILL, OPT_BINARY] { assert_eq!(f.feed(b, &mut out), None); }
        assert!(out.is_empty(), "got unwanted reply: {:?}", out);
    }

    #[test]
    fn client_offers_unknown_option_we_decline() {
        let mut f = TelnetFilter::new();
        let mut out = Vec::new();
        // Client offers TERMINAL-TYPE (24). We say DONT.
        for &b in &[IAC, WILL, 24] { assert_eq!(f.feed(b, &mut out), None); }
        assert_eq!(out, vec![IAC, DONT, 24]);
    }

    #[test]
    fn client_asks_us_to_do_unknown_option_we_decline() {
        let mut f = TelnetFilter::new();
        let mut out = Vec::new();
        for &b in &[IAC, DO, 24] { assert_eq!(f.feed(b, &mut out), None); }
        assert_eq!(out, vec![IAC, WONT, 24]);
    }

    #[test]
    fn subnegotiation_swallowed() {
        let mut f = TelnetFilter::new();
        let mut out = Vec::new();
        // IAC SB <opt> <bytes> IAC SE
        let bytes = [IAC, SB, 24, 1, b'a', b'b', IAC, SE, b'X'];
        let mut data = Vec::new();
        for &b in &bytes {
            if let Some(d) = f.feed(b, &mut out) { data.push(d); }
        }
        assert_eq!(data, vec![b'X']);
        assert!(out.is_empty());
    }

    #[test]
    fn iac_inside_subnegotiation_not_terminator() {
        let mut f = TelnetFilter::new();
        let mut out = Vec::new();
        // IAC SB ... IAC IAC ... IAC SE (escaped FF inside SB)
        let bytes = [IAC, SB, 24, IAC, IAC, b'x', IAC, SE, b'Y'];
        let mut data = Vec::new();
        for &b in &bytes {
            if let Some(d) = f.feed(b, &mut out) { data.push(d); }
        }
        assert_eq!(data, vec![b'Y']);
    }

    #[test]
    fn crlf_writer_translates_bare_lf() {
        let mut out = Vec::new();
        {
            let mut w = CrlfWriter::new(&mut out);
            w.write_all(b"hello\nworld\n").unwrap();
        }
        assert_eq!(out, b"hello\r\nworld\r\n");
    }

    #[test]
    fn crlf_writer_passes_through_existing_crlf() {
        let mut out = Vec::new();
        {
            let mut w = CrlfWriter::new(&mut out);
            w.write_all(b"hello\r\nworld\r\n").unwrap();
        }
        assert_eq!(out, b"hello\r\nworld\r\n");
    }

    #[test]
    fn crlf_writer_handles_split_cr_lf_across_writes() {
        let mut out = Vec::new();
        {
            let mut w = CrlfWriter::new(&mut out);
            w.write_all(b"foo\r").unwrap();
            w.write_all(b"\nbar").unwrap();
        }
        assert_eq!(out, b"foo\r\nbar");
    }

    #[test]
    fn escape_byte_doubles_ff() {
        let mut out = Vec::new();
        escape_byte(0xFE, &mut out);
        escape_byte(0xFF, &mut out);
        escape_byte(0x00, &mut out);
        assert_eq!(out, vec![0xFE, 0xFF, 0xFF, 0x00]);
    }
}
