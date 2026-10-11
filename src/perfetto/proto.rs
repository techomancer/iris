//! Minimal protobuf writer for the Perfetto trace format.
//!
//! Hand-written rather than generated: we emit about thirty fields across
//! seven messages, and Perfetto follows protobuf's compatibility rules (field
//! numbers are never renumbered or reused, only added), so a table of
//! constants is all a generator would buy us. The numbers below are from
//! upstream `protos/perfetto/trace/`:
//!
//!   trace_packet.proto, track_event/track_event.proto,
//!   track_event/track_descriptor.proto, track_event/thread_descriptor.proto,
//!   track_event/process_descriptor.proto, track_event/counter_descriptor.proto,
//!   track_event/debug_annotation.proto, interned_data/interned_data.proto
//!
//! A trace file is a serialised `Trace { repeated TracePacket packet = 1; }`.
//! Repeated fields concatenate, so the file is a plain stream of
//! `[0x0A, varint len, TracePacket bytes]` records: no header, no footer, and
//! a file cut short (process exit, crash) still loads up to the last packet.

/// `Trace.packet`.
pub const TRACE_PACKET: u32 = 1;

pub mod packet {
    pub const TIMESTAMP: u32 = 8;
    pub const TRUSTED_PACKET_SEQUENCE_ID: u32 = 10;
    pub const TRACK_EVENT: u32 = 11;
    pub const INTERNED_DATA: u32 = 12;
    pub const SEQUENCE_FLAGS: u32 = 13;
    pub const PREVIOUS_PACKET_DROPPED: u32 = 42;
    pub const TRACK_DESCRIPTOR: u32 = 60;

    pub const SEQ_INCREMENTAL_STATE_CLEARED: u64 = 1;
    pub const SEQ_NEEDS_INCREMENTAL_STATE: u64 = 2;
}

pub mod track_event {
    pub const CATEGORY_IIDS: u32 = 3;
    pub const DEBUG_ANNOTATIONS: u32 = 4;
    pub const TYPE: u32 = 9;
    pub const NAME_IID: u32 = 10;
    pub const TRACK_UUID: u32 = 11;
    pub const COUNTER_VALUE: u32 = 30;

    pub const TYPE_SLICE_BEGIN: u64 = 1;
    pub const TYPE_SLICE_END: u64 = 2;
    pub const TYPE_INSTANT: u64 = 3;
    pub const TYPE_COUNTER: u64 = 4;
}

pub mod debug_annotation {
    pub const NAME_IID: u32 = 1;
    pub const UINT_VALUE: u32 = 3;
}

pub mod track_descriptor {
    pub const UUID: u32 = 1;
    pub const NAME: u32 = 2;
    pub const PROCESS: u32 = 3;
    pub const THREAD: u32 = 4;
    pub const PARENT_UUID: u32 = 5;
    pub const COUNTER: u32 = 8;
}

pub mod process_descriptor {
    pub const PID: u32 = 1;
    pub const PROCESS_NAME: u32 = 6;
}

pub mod thread_descriptor {
    pub const PID: u32 = 1;
    pub const TID: u32 = 2;
    pub const THREAD_NAME: u32 = 5;
}

pub mod interned_data {
    pub const EVENT_CATEGORIES: u32 = 1;
    pub const EVENT_NAMES: u32 = 2;
    pub const DEBUG_ANNOTATION_NAMES: u32 = 3;

    /// `EventCategory`, `EventName` and `DebugAnnotationName` share a layout.
    pub const IID: u32 = 1;
    pub const NAME: u32 = 2;
}

const WIRE_VARINT: u32 = 0;
const WIRE_LEN: u32 = 2;

/// Appends protobuf fields to a byte buffer.
///
/// Nested messages use `begin`/`end`: `begin` reserves a fixed four-byte
/// length (a padded varint, legal protobuf and what Perfetto's own writer
/// does) and `end` patches it, so nothing is encoded twice.
#[derive(Default)]
pub struct ProtoWriter {
    pub buf: Vec<u8>,
}

/// Largest message a four-byte padded length can describe.
const MAX_NESTED: usize = (1 << 28) - 1;

impl ProtoWriter {
    pub fn new() -> Self { Self::default() }

    pub fn clear(&mut self) { self.buf.clear(); }

    #[inline]
    pub fn raw_varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.buf.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.buf.push(v as u8);
    }

    #[inline]
    fn tag(&mut self, field: u32, wire: u32) {
        self.raw_varint(((field << 3) | wire) as u64);
    }

    #[inline]
    pub fn varint(&mut self, field: u32, v: u64) {
        self.tag(field, WIRE_VARINT);
        self.raw_varint(v);
    }

    /// int64 fields are plain two's-complement varints (not zigzag).
    #[inline]
    pub fn int64(&mut self, field: u32, v: i64) {
        self.varint(field, v as u64);
    }

    pub fn bytes(&mut self, field: u32, b: &[u8]) {
        self.tag(field, WIRE_LEN);
        self.raw_varint(b.len() as u64);
        self.buf.extend_from_slice(b);
    }

    pub fn string(&mut self, field: u32, s: &str) {
        self.bytes(field, s.as_bytes());
    }

    /// Opens a nested message; returns the token `end` needs.
    #[inline]
    pub fn begin(&mut self, field: u32) -> usize {
        self.tag(field, WIRE_LEN);
        let at = self.buf.len();
        self.buf.extend_from_slice(&[0x80, 0x80, 0x80, 0x00]);
        at
    }

    #[inline]
    pub fn end(&mut self, at: usize) {
        let len = self.buf.len() - at - 4;
        assert!(len <= MAX_NESTED, "perfetto: nested message too large ({len} bytes)");
        self.buf[at] = (len as u8 & 0x7f) | 0x80;
        self.buf[at + 1] = ((len >> 7) as u8 & 0x7f) | 0x80;
        self.buf[at + 2] = ((len >> 14) as u8 & 0x7f) | 0x80;
        self.buf[at + 3] = (len >> 21) as u8 & 0x7f;
    }

    /// Writes an empty nested message (presence is the information).
    pub fn empty(&mut self, field: u32) {
        self.tag(field, WIRE_LEN);
        self.buf.push(0);
    }
}

#[cfg(test)]
pub mod decode {
    //! Just enough of a protobuf reader for the tests to check what the
    //! writer produced.

    #[derive(Debug, Clone, PartialEq)]
    pub enum Value<'a> {
        Varint(u64),
        Fixed64(u64),
        Bytes(&'a [u8]),
        Fixed32(u32),
    }

    pub fn varint(b: &[u8], pos: &mut usize) -> u64 {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let byte = b[*pos];
            *pos += 1;
            v |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 { return v; }
            shift += 7;
        }
    }

    /// All fields of one message, in order.
    pub fn fields(b: &[u8]) -> Vec<(u32, Value<'_>)> {
        let mut out = Vec::new();
        let mut pos = 0;
        while pos < b.len() {
            let key = varint(b, &mut pos);
            let field = (key >> 3) as u32;
            let v = match key & 7 {
                0 => Value::Varint(varint(b, &mut pos)),
                1 => { let v = u64::from_le_bytes(b[pos..pos + 8].try_into().unwrap()); pos += 8; Value::Fixed64(v) }
                2 => { let n = varint(b, &mut pos) as usize; let v = &b[pos..pos + n]; pos += n; Value::Bytes(v) }
                5 => { let v = u32::from_le_bytes(b[pos..pos + 4].try_into().unwrap()); pos += 4; Value::Fixed32(v) }
                w => panic!("bad wire type {w} at {pos}"),
            };
            out.push((field, v));
        }
        out
    }

    pub fn get<'a>(f: &[(u32, Value<'a>)], field: u32) -> Option<Value<'a>> {
        f.iter().find(|(n, _)| *n == field).map(|(_, v)| v.clone())
    }

    pub fn get_u64(f: &[(u32, Value<'_>)], field: u32) -> Option<u64> {
        match get(f, field)? { Value::Varint(v) => Some(v), _ => None }
    }

    pub fn get_msg<'a>(f: &[(u32, Value<'a>)], field: u32) -> Option<Vec<(u32, Value<'a>)>> {
        match get(f, field)? { Value::Bytes(b) => Some(fields(b)), _ => None }
    }

    pub fn get_str<'a>(f: &[(u32, Value<'a>)], field: u32) -> Option<&'a str> {
        match get(f, field)? { Value::Bytes(b) => std::str::from_utf8(b).ok(), _ => None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::decode::*;

    #[test]
    fn varint_encoding() {
        let mut w = ProtoWriter::new();
        w.raw_varint(0);
        w.raw_varint(1);
        w.raw_varint(127);
        w.raw_varint(128);
        w.raw_varint(300);
        w.raw_varint(u64::MAX);
        assert_eq!(&w.buf[..6], &[0x00, 0x01, 0x7f, 0x80, 0x01, 0xac]);
        assert_eq!(w.buf.len(), 1 + 1 + 1 + 2 + 2 + 10);
        let mut pos = 0;
        for want in [0, 1, 127, 128, 300, u64::MAX] {
            assert_eq!(varint(&w.buf, &mut pos), want);
        }
    }

    #[test]
    fn nested_padded_length_round_trips() {
        let mut w = ProtoWriter::new();
        let outer = w.begin(11);
        w.varint(9, 3);
        let inner = w.begin(4);
        w.varint(1, 7);
        w.end(inner);
        w.string(23, "hello");
        w.end(outer);
        w.empty(8);
        w.int64(30, -5);

        let top = fields(&w.buf);
        assert_eq!(top.len(), 3);
        let ev = get_msg(&top, 11).unwrap();
        assert_eq!(get_u64(&ev, 9), Some(3));
        assert_eq!(get_u64(&get_msg(&ev, 4).unwrap(), 1), Some(7));
        assert_eq!(get_str(&ev, 23), Some("hello"));
        assert_eq!(get(&top, 8), Some(Value::Bytes(&[])));
        assert_eq!(get_u64(&top, 30), Some(-5i64 as u64));
    }
}
