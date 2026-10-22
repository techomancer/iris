# NFS fileid 1 panics IRIX 5.3 through imon (fam)

**Symptom:** IRIX 5.3 with `/shared` mounted over our in-core NFS (seen
with v3) panics while a desktop session is logged in, on ordinary file
activity in the share root (`touch`, `rm`, `mkdir`, ... in `/shared`):

```
PANIC: KERNEL FAULT
PC: 0x880175fc ep:ffffcad8       (or 0x8801768c)
EXC code:128, 'Software detected SEGV '
Bad addr: 0x0, cause: 0x10000008<CE=1,EXC=RMISS>   (or 0x8887a1e0)
```

The PC is inside `bcopy`/`ovbcopy`, on a *store*, while the cause says
load miss. Nothing about it points at NFS, and the bad address is not the
store's address. Our own 6.5 guests and a 5.3 install without fam watching
the share never hit it.

**Cause:** the export root had fileid (inode number) **1**. IRIX's inode
monitor `imon`, which `fam` (and so the desktop) uses to watch files, puts
pathname records in its event queue marked by inode 1. `dequeue` in
`imonread` diverts any record whose first word is 1 to `dequeue_pn`, which
takes the second word (here the device number, `0x100000` for the NFS
mount) as the name length and `bcopy`s that much into a MAXPATHLEN buffer
in the u-area. The copy runs off the top of the kernel stack at
`0xffffe000`; the TLB miss on that unmapped page is the "SEGV". Real
filesystems never have an inode 1 (the root is 2, 1 is reserved), which
is why only our server triggered it.

**Fix:** `nfsudp::ROOT_ID` is 2, ids are handed out from 3, and 0 and 1
never appear as fileids (test `root_is_a_dir_with_inode_2_and_no_id_is_0_or_1`).
Any synthetic filesystem we serve to IRIX must follow the same rule.

**How it was found (reusable):**
- `bp add 0x<dumpsys>` catches any panic before the dump is written, with
  the trap frame still at `ep`. `panic()` and `cmn_err(CE_PANIC)` breakpoints
  did not fire: the fault path does not go through them.
- The trap frame (`ecommon` layout: `a0` at +56, `a1` +64, `a2` +72, `ra`
  +272, `BadVAddr` +304, `Cause` +312, `EPC` +320, all 64-bit) gave the real
  copy: destination `0xffffe010`, 8 KB left. Each frame's saved `ra` led up:
  `ovbcopy` ← `dequeue_pn` ← `dequeue` ← `imonread`.
- `dt file <path> 2000000` dumps the last million instructions; the miss
  path (`kmiss` → `kmissnxt` → `longway`) showed the kernel rejecting the
  page-table address, i.e. a bad virtual address, not a TLB bug.
- Breakpoints on the exception vectors with an EPC condition also stop on
  harmless interrupts and refills at that PC; condition on the cause too.
