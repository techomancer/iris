# NFS READDIR replies must respect the client's `count` and fit one UDP datagram

## Symptom

Mounting the in-core NFS server (`src/net/nfsudp.rs`) works, but `ls` on a folder
with many entries (e.g. a real Mac `~/Downloads`) fails on the guest with:

```
NFS2 readdir failed for server 192.168.x.1: Can't decode result
```

A small directory lists fine; only a *large* one fails. The mount, GETATTR,
LOOKUP, and READ all work — so it is specifically the READDIR **reply**.

## Cause

"Can't decode result" is an RPC-client XDR decode failure, not a timeout — the
datagram arrived but the client couldn't parse it. Two reply-size mistakes cause
it, and a big directory trips both at once:

1. **Ignoring the request's `count`.** BSD/SunRPC-derived clients (IRIX's NFS is
   one) size their reply *receive buffer* to the `count` field they sent in the
   READDIR/READDIRPLUS args. If the server returns more directory data than
   `count`, the reply is truncated on the client side and the decode runs off
   the end. The server MUST cap the reply at `count`.
2. **Spilling into IP fragments.** A multi-kilobyte reply fragments into several
   Ethernet frames; any reassembly weakness corrupts the datagram. READDIR is
   designed to be *paged* (the client re-issues with the last cookie), so there
   is no reason to emit a fragmented readdir reply at all.

The original code ignored `count` and used a fixed ~8 KB byte budget, which both
overran a smaller client buffer and forced fragmentation.

## Fix

Page each readdir reply to fit **both** the client's `count` **and** a single
unfragmented UDP datagram, continuing via the cookie:

- **v2** (`nfs2_readdir`): `let limit = count.clamp(512, 1400);` — 1400 keeps the
  whole RPC reply under the 1472-byte single-datagram UDP payload (MTU 1500 − IP
  20 − UDP 8) after the ~36 bytes of reply header + list terminators.
- **v3** (`nfs3_readdir`): read `maxcount` (READDIRPLUS sends `dircount` first,
  then `maxcount`) and cap at `maxcount.clamp(512, 16_000)`. v3 keeps the larger
  budget because its larger client buffers make fragmented readdir replies
  workable, but it still must never exceed the client's stated `maxcount`.

The `budget > 0` guard already guarantees at least one entry per reply, so a tiny
`count` still makes progress; `eof` is set only when the last entry fits.

Regression test: `nfsudp::tests::v2_readdir_pages_within_one_datagram` builds a
200-entry directory, asserts every page is ≤ 1472 bytes, walks the cookie chain,
and checks every name comes back exactly once.

## Watch out

- Any new procedure that returns a variable-length list (e.g. MOUNT EXPORT with
  many exports) has the same single-datagram / client-buffer constraint.
- READ already fragments by design (file data up to rtmax = 32768), so the
  outbound fragmenter (`net.rs::ip_frames_udp`) must stay correct regardless —
  this fix sidesteps fragmentation for *readdir*, it does not remove the need for
  it elsewhere.
