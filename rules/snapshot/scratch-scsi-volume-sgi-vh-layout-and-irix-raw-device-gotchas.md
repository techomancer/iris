# Scratch SCSI volume - SGI VH layout and IRIX raw-device gotchas

**Keywords:** scratch,sgi,vh,volume,header,partition,scsi,raw,dd,iris,irix,phase2.4
**Category:** snapshot

# Scratch SCSI Volume (Phase 2.4)

A SCSI device with `scratch = true` in `iris.toml` is a host-controlled raw block device for file injection/extraction without networking. iris pre-formats it with a minimal SGI Volume Header.

## Partition layout

| Slot | Device node            | Purpose                  | Type        | first_block | nblks       |
|------|------------------------|--------------------------|-------------|-------------|-------------|
| 0    | /dev/rdsk/dks0dNs0     | Payload (host writes)    | PT_RAW=3    | 8           | total - 8   |
| 8    | /dev/rdsk/dks0dNvh     | Volume header itself     | PT_VOLHDR=0 | 0           | 8           |
| 10   | /dev/rdsk/dks0dNvol    | Whole-disk view          | PT_VOLUME=6 | 0           | total       |

Slot 10 (vol) is special - by SGI convention it always starts at sector 0 regardless of first_block. Use slot 0 (s0) for payload reads.

## Host wire format

scratch-write and scratch-read operate on the payload area. offset = 0 means raw-byte 4096 in the underlying file (the first byte after the VH). The CI commands never touch the VH.

```
host: iris CI: scratch-write {host_path: "bundle.tar"}
guest: dd if=/dev/rdsk/dks0d2s0 bs=512 | tar xf -
```

## IRIX gotchas

1. Reads must be sector-aligned. dd bs=64 returns "Read error: I/O error" with no SCSI-level error. Use bs=512 (or any 512-multiple).
2. Writes must be padded to bs. dd bs=512 from a 28-byte file produces "0+1 records in / 0+0 records out" with "Write error: I/O error". Add conv=sync to pad with zeros, plus conv=notrunc if you don't want to truncate the device file:
   `dd if=/tmp/data of=/dev/rdsk/dks0d2s0 bs=512 conv=sync,notrunc`
3. Without a valid VH at sector 0, IRIX creates the device nodes but every read returns I/O error.
4. Checksum is required: vh_csum at offset 0x1F8 must make the sum of all 128 big-endian u32 words equal 0. iris computes this in sgi_vh::fix_csum.

## When to use scratch over NFS

NFS (now served in-process by the NAT, no `unfsd` since June 2026) still requires IRIX networking before any file movement. The scratch volume works at PROM time, single-user, or any other phase.

