//! Minimal ONC RPC portmapper: answers GETPORT for the in-core NFS server.
//!
//! XDR layout of a GETPORT call (big-endian u32s): xid, msg_type=0 (CALL),
//! rpcvers=2, prog=100000, vers, proc=3, cred (flavor, len, body), verf
//! (flavor, len, body), then the mapping: prog, vers, prot, port. The reply
//! is xid, 1 (REPLY), 0 (MSG_ACCEPTED), verf AUTH_NULL, 0 (SUCCESS), port.

pub(crate) const PORTMAP_PORT: u16 = 111;
/// Ports the guest sees for the in-core NFS server and mountd.
pub(crate) const NFS_PORT: u16 = 2049;
pub(crate) const MOUNTD_PORT: u16 = 1234;

pub(crate) const PROG_PORTMAP: u32 = 100000;
pub(crate) const PROG_NFS: u32 = 100003;
pub(crate) const PROG_MOUNTD: u32 = 100005;
pub(crate) const PROC_GETPORT: u32 = 3;

fn word(b: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(i..i + 4)?.try_into().ok()?))
}

/// The protocol number of UDP in a mapping; the servers are UDP only.
pub(crate) const IPPROTO_UDP: u32 = 17;
/// The guest-visible port a GETPORT call asks for, or 0 if unknown. A TCP
/// mapping is unknown: IRIX 6.5.30's mount asks for TCP first and only
/// falls back to UDP when the answer is 0.
pub(crate) fn lookup(call: &[u8]) -> u32 {
    let parse = || -> Option<u32> {
        if word(call, 4)? != 0 || word(call, 8)? != 2 || word(call, 12)? != PROG_PORTMAP
            || word(call, 20)? != PROC_GETPORT {
            return None;
        }
        let cred_len = word(call, 28)? as usize;
        let verf = 32 + cred_len;
        let args = verf + 8 + word(call, verf + 4)? as usize;
        word(call, args + 12)?; // the whole mapping must be there
        if word(call, args + 8)? != IPPROTO_UDP {
            return Some(0);
        }
        Some(match word(call, args)? {
            PROG_NFS => NFS_PORT as u32,
            PROG_MOUNTD => MOUNTD_PORT as u32,
            PROG_PORTMAP => PORTMAP_PORT as u32,
            _ => 0,
        })
    };
    parse().unwrap_or(0)
}

/// The reply to a GETPORT call.
pub(crate) fn reply(call: &[u8]) -> Vec<u8> {
    let xid = word(call, 0).unwrap_or(0);
    let mut r = Vec::with_capacity(28);
    for w in [xid, 1, 0, 0, 0, 0, lookup(call)] {
        r.extend_from_slice(&w.to_be_bytes());
    }
    r
}

#[cfg(test)]
pub(crate) fn getport_call(xid: u32, prog: u32) -> Vec<u8> {
    let mut p = Vec::new();
    for w in [xid, 0, 2, PROG_PORTMAP, 2, PROC_GETPORT, 0, 0, 0, 0, prog, 3, 17, 0] {
        p.extend_from_slice(&u32::to_be_bytes(w));
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn getport_answers_the_in_core_servers() {
        for (prog, port) in [(PROG_NFS, 2049), (PROG_MOUNTD, 1234), (PROG_PORTMAP, 111), (100099, 0)] {
            let r = reply(&getport_call(0xabcd, prog));
            assert_eq!(&r[..4], &0xabcdu32.to_be_bytes());
            assert_eq!(u32::from_be_bytes(r[24..28].try_into().unwrap()), port);
        }
        assert_eq!(lookup(&getport_call(1, PROG_NFS)[..40]), 0, "truncated call");
    }

    #[test]
    fn getport_has_no_tcp_mappings() {
        let mut call = getport_call(1, PROG_MOUNTD);
        call[48..52].copy_from_slice(&6u32.to_be_bytes());
        assert_eq!(lookup(&call), 0);
    }
}
