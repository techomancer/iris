//! Inbound IPv4 fragment reassembly (guest datagrams larger than the MTU).

use std::collections::{BTreeMap, HashMap};
use std::net::Ipv4Addr;
use std::time::{Duration, Instant};

use super::packet::Ipv4;

/// Give up on a datagram whose fragments stop arriving.
const REASSEMBLY_TIMEOUT: Duration = Duration::from_secs(5);

/// Reassembly state for one datagram.
struct Partial {
    frags: BTreeMap<usize, Vec<u8>>, // offset -> bytes (deduped)
    total: Option<usize>,            // known once the last fragment arrives
    last: Instant,
}

impl Partial {
    fn add(&mut self, offset: usize, data: &[u8], more: bool) -> Option<Vec<u8>> {
        self.frags.insert(offset, data.to_vec());
        self.last = Instant::now();
        if !more {
            self.total = Some(offset + data.len());
        }
        let total = self.total?;
        let mut out = Vec::with_capacity(total);
        for (&off, d) in &self.frags {
            if off != out.len() {
                return None; // a gap: still waiting on a fragment
            }
            out.extend_from_slice(d);
        }
        (out.len() == total).then_some(out)
    }
}

/// Fragments in flight, keyed by (source, IP id, protocol).
#[derive(Default)]
pub(crate) struct Reassembler {
    partial: HashMap<(Ipv4Addr, u16, u8), Partial>,
}

impl Reassembler {
    /// Add a fragment; returns the whole transport payload once complete.
    pub(crate) fn add(&mut self, ip: &Ipv4) -> Option<Vec<u8>> {
        self.partial.retain(|_, p| p.last.elapsed() < REASSEMBLY_TIMEOUT);
        let key = (ip.src(), ip.id(), ip.proto());
        let whole = self
            .partial
            .entry(key)
            .or_insert_with(|| Partial { frags: BTreeMap::new(), total: None, last: Instant::now() })
            .add(ip.frag_offset(), ip.payload(), ip.more_fragments())?;
        self.partial.remove(&key);
        Some(whole)
    }
}
