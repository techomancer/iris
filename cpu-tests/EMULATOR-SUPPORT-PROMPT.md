# Handoff prompt — emulator support for the CPU test suite

Copy everything below the line into a fresh IRIS session. It is self-contained.
When that work lands, `cpu-tests/` can start being built (see `cpu-tests/PLAN.md`).

---

I need four additive features in IRIS to support a bare-metal MIPS CPU test
suite that will live in `cpu-tests/` (see `cpu-tests/PLAN.md` for the full
context — read it first, it explains what the suite is and why each feature
below exists). The suite is a static ELF32-BE binary, loaded either directly,
by the PROM from an SGI volume header, or by the PROM over the network, that
runs with no OS and prints PASS/FAIL over serial.

Read `HACKING.md` and `CLAUDE.md` before touching device or CPU code. In
particular:

- **Endianness lives only at "The Edge."** ELF parsing and TFTP packet
  construction *are* the Edge and may byte-swap there; CPU, bus, and MC code
  must not. Do not introduce `.to_be()` / `.to_le()` into memory or register
  paths.
- **Concurrency is per-device.** Follow the existing `Device` / `BusDevice`
  patterns in `src/traits.rs`; deadlocks live in callbacks *up* to a parent
  device.
- Any new device must round-trip through snapshot save/load — see
  `rules/snapshot/per-device-saveloadsave-round-trip-is-the-regression-net.md`.
- When you confirm a non-obvious behaviour, write it up as a short note under
  `rules/`.

All four features are **additive and default-off**. None of them may change
IRIX boot behaviour or the normal user-facing experience. `cargo test` must
stay green, and normal IRIX 6.5 boot must be unaffected.

The four are independent — land them in any order, ideally as separate commits.
Feature A and B are the ones that unblock the most work.

---

## Feature A — SGI volume-directory writer + an image tool

**Why:** the PROM loads a standalone binary by name out of a disk/CD volume
header: `boot -f dksc(0,<scsi-id>,8)cputest`. IRIS can already write the
partition table but not the *volume directory*, so no image it produces is
bootable by name.

**What exists:** `src/sgi_vh.rs` writes a minimal volume header for the scratch
volume — magic at `0x000`, partition table at `0x138` (`PT_TABLE_OFFSET`),
checksum at `0x1F8` (`CSUM_OFFSET`), with `fix_csum()` making the 128 big-endian
words sum to zero. Those offsets confirm the classic `struct volume_header`
layout, so the gap at `0x048` is the volume directory:

```c
struct volume_header {                 /* offset */
    int   vh_magic;                    /* 0x000  0x0BE5A941 */
    short vh_rootpt, vh_swappt;        /* 0x004 */
    char  vh_bootfile[16];             /* 0x008  default boot file name */
    struct device_parameters vh_dp;    /* 0x018  48 bytes */
    struct volume_directory vh_vd[15]; /* 0x048  15 x 16 = 240 bytes   <-- ADD */
    struct partition_table  vh_pt[16]; /* 0x138  16 x 12 = 192 bytes   exists */
    int   vh_csum;                     /* 0x1F8                       exists */
    int   vh_fill;                     /* 0x1FC */
};
struct volume_directory { char vd_name[8]; int vd_lbn; int vd_nbytes; };
```

`vd_lbn` is a **512-byte block number** from the start of the volume, `vd_nbytes`
the exact file length. On CD-ROM this works because the PROM switches the drive
to 512-byte logical blocks — IRIS already supports that (`src/scsi.rs:174`,
`logical_block_size`).

**Build:**

1. In `src/sgi_vh.rs`, add volume-directory support: a way to place named files
   in the volume header area and record their `lbn`/`nbytes`, with the checksum
   recomputed after every mutation. Names are at most 8 characters, NUL-padded,
   not NUL-terminated at 8. Keep the existing `create_scratch_image()` behaviour
   working unchanged.
2. Add a small image builder that produces a complete raw image from a
   description: volume-header files, partition table entries, and optional raw
   partition contents (partition 7 will later hold an EFS filesystem built by
   `cpu-tests/`).
3. Add a CLI binary (`src/bin/mkvh.rs`, wired into `Cargo.toml` alongside the
   existing `chd_extract` entry) with at least:
   - a build mode: given output path, size, and one or more `name=path` files,
     emit a bootable image;
   - a **dump mode**: parse an existing image's volume header and print magic,
     bootfile, every non-empty voldir entry (name, lbn, nbytes), the partition
     table, and whether the checksum validates.

**Validate the field layout against real media, not against your own writer.**
That is what the dump mode is for. Reference image:

```
~/IRIX 6.5.22 Installation Tools and Overlays (1 of 3).iso
```

`mkvh --dump` on it must reproduce this — these values were read out of that
file, so treat any disagreement as a bug in your parser:

```
magic 0x0BE5A941   rootpt 0  swappt 0  bootfile ''      csum VALID

volume directory @ 0x048            lbn = 512-byte block number
  [0] sgilabel   lbn=32      nbytes=512
  [1] mr         lbn=33      nbytes=19200000
  [2] sash64     lbn=37533   nbytes=266000
  [3] sashARCS   lbn=38053   nbytes=342532

partition table @ 0x138
  [ 7] nblks=1283016  first=48736  type=5  (SYSV)
  [ 8] nblks=48736    first=0      type=0  (VOLHDR)
  [10] nblks=1331776  first=0      type=6  (VOLUME)
```

Two structural facts from that dump that your **writer** must honour, because
both differ from what `src/sgi_vh.rs` does today:

- **Partition 8 must span every voldir file.** On the real CD it is 48736
  blocks (~24 MB), not 8 sectors — it covers all the data the voldir points
  at, and partition 7 begins after it. The current scratch-image code hardcodes
  `VH_SECTORS = 8`, which is right for a volume header with no files in it and
  wrong the moment you add one.
- **The filesystem partition is type 5 (`PT_SYSV`), not type 7 (`PT_EFS`)**,
  even though the filesystem on it is EFS — verified by the EFS superblock at
  block 1 of that partition (`fs_magic` = `0x00072959` at offset `0x1c`).
  Copy what the shipping CD does rather than what the type name suggests.

Also worth knowing while you work on this: `sashARCS` on that CD is **ECOFF**
(`f_magic` 0x0163, `a_magic` OMAGIC, text at `0x10000000`, entry `0x10021020`)
and `sash64` is an **ELF64 MSB `ET_REL`** object the PROM relocates. So both
PROM loader paths are exercised by real media, and `0x10000000` is a load
address this PROM demonstrably accepts.

**Acceptance:**
- Unit tests in `src/sgi_vh.rs` covering voldir round-trip, the 8-char name
  edge case, checksum validity after adding files, and rejection of oversized
  or too-many entries (15 max).
- `mkvh` dump on a genuine SGI CD prints a valid, sensible volume header.
- An image built by `mkvh` containing one ELF, attached as a SCSI disk, boots
  via `boot -f dksc(0,<id>,8)<name>` from the PROM.

---

## Feature B — direct binary load (`--load-elf` + monitor `loadelf`/`loadbin`)

**Why:** rebuilding a disk or CD image on every edit makes the test-writing
loop miserable. Loading an ELF straight into RAM and jumping to its entry point
turns a 30-second cycle into a sub-second one, and it is the only way to run
code before POST.

**Build:**

1. **Monitor commands**, registered in the CPU command table in
   `src/mips_exec.rs` (the table around lines 5880–5935, dispatch in
   `execute_command` below it — follow the shape of the existing `mw` / `jump`
   / `loadsym` commands):
   - `loadelf <path>` — parse a static ELF32 MSB `ET_EXEC` for `EM_MIPS`.
     Reject anything else with a clear message (wrong class, wrong endianness,
     wrong machine, dynamic). For each `PT_LOAD`: copy `p_filesz` bytes to
     `p_vaddr`, zero-fill up to `p_memsz`. Set PC to `e_entry`. Print a summary
     line per segment (vaddr, filesz, memsz, flags) and the entry point.
   - `loadbin <path> <addr>` — raw bytes to a virtual address, no PC change.
2. **CLI flag** `--load-elf FILE` in the `Args` struct in `src/config.rs`
   (the `#[arg(long)]` block, lines ~928–1056), wired in `src/main.rs`.

**Gotchas to handle, not discover later:**

- Write through the normal CPU virtual-address path so KSEG0/KSEG1 mapping,
  the MC address mask, and bank remapping all apply. Do not poke `Memory`
  directly behind the bus.
- **Invalidate stale state for the loaded range**: the L1 I-cache
  (`src/mips_cache_v2.rs`) and, when built with `--features jit`, the JIT block
  cache. A loaded-over region with a live translation will execute the old
  code. Test this explicitly by loading two different binaries at the same
  address in one session, with the JIT enabled.
- RAM addressability before POST: `Machine::new` initialises `addr_mask` to
  `mem_size - 1` and `remap_banks()` updates it when the PROM writes
  MEMCFG0/1. Determine whether a cold-start `--load-elf` before POST actually
  lands in RAM. If it does, document it; if it does not, make the flag load
  after PROM POST (or document "restore a PROM-prompt snapshot first, then
  `loadelf`") rather than silently writing into the void.
- Use a non-`lightning` build when testing — `lightning` implies `opcodefusion`
  and disables breakpoint checks (see the feature comments in `Cargo.toml`).

**Acceptance:** a hand-assembled ELF that writes a few bytes to the SCC and
spins can be loaded and run from both the monitor and the CLI flag; `dis`,
`regs`, and `mem` show the loaded image; re-loading a different binary at the
same address executes the new code with the JIT on.

---

## Feature C — TFTP server for PROM network boot

**Why:** the PROM has a complete BOOTP + TFTP client — verified in the embedded
PROM image (`src/prombin.rs`): `bootp()sashIP24`, `"TFTP: RRQ for file %s to %s"`,
`network(%d)tftp()`, `"BOOTP reply from %s(%s): file %s"`. IRIS answers BOOTP
(`src/net.rs:1679`, dispatch at `src/net.rs:1642`) but serves no TFTP, so
`boot -f bootp()<file>` cannot work. With TFTP, a freshly built test binary
boots straight out of the host build directory with no image at all — and it
exercises the PROM's network stack, which nothing tests today.

**Build:** a read-only RFC 1350 TFTP server inside the existing NAT gateway in
`src/net.rs`, next to `handle_bootp`, listening on UDP 69 at the gateway
address (the virtual stack — not a host socket).

- RRQ only. Reply WRQ with ERROR code 2 (access violation). Octet mode; accept
  and ignore `netascii` case-insensitively or reject it explicitly, your call,
  but be consistent.
- 512-byte DATA blocks, ACK-driven, block numbers starting at 1, final short
  block terminates the transfer (including the zero-length final block when the
  file is an exact multiple of 512).
- Retransmit the last DATA on ACK timeout, with a bounded retry count; drop the
  transfer after that. Handle duplicate ACKs without advancing.
- ERROR code 1 (file not found) for a missing file.
- Serve from a configured root: a `tftp_dir` key in the `[network]` section of
  `iris.toml` plus a `--tftp-dir DIR` flag. **Disabled when unset.**
- Path safety: resolve against the root and refuse anything that escapes it
  (`..`, absolute paths, symlinks out of the root). Read-only, no writes ever.
- Packet fields are network byte order — that is the Edge, so swapping there is
  correct and expected.

**Acceptance:** with `--tftp-dir` pointing at a directory containing an ELF,
`boot -f bootp()<file>` at the PROM prompt loads and runs it. Verify a
not-found file produces a clean PROM error rather than a hang, and that
transfers work for a file that is an exact multiple of 512 bytes.

---

## Feature D — test device: guest console, register dump, exit code

**Why:** the suite is self-checking and prints over the SCC, which is what will
run on real hardware. But for debugging a failing check and for CI, we also
want (a) a deterministic process exit code instead of matching a serial string,
and (b) a full machine-state dump at an arbitrary point, in a stable
machine-readable format. This is also what lets the external `cheritest`
corpus (`../cheritest`, ~530 bare-metal MIPS64 tests) be run against IRIS
later — its convention is a magic CP0 write that makes the simulator dump the
register file.

**Build a small MMIO "test device"**, default-off behind a flag
(`--test-device`, and/or a config key). Follow the `Device`/`BusDevice` pattern
in `src/traits.rs` and give it snapshot save/load like every other device.

Registers (pick a physical address range that is genuinely unused on Indy —
document the choice and why it cannot collide with the real IP22/IP24 memory
map; an unused GIO slot region is a reasonable candidate):

| Register | Access | Behaviour |
|---|---|---|
| `SIGNATURE` | read | A fixed magic value. Lets the guest detect the device and fall back to SCC-only output when it is absent (i.e. on real hardware). |
| `PUTC` | write | Byte to the host: stdout and/or the `--serial-log` sink. Console output without touching the SCC. |
| `DUMP` | write | Dump full machine state to a file (path from the flag, or a default under the CWD). Written value distinguishes dump points so a test can dump more than once. |
| `EXIT` | write | Stop the CPU and terminate the process with the written value as the exit code. |

Dump format: stable and machine-parseable — JSON is fine. Include `gpr[32]`,
`hi`, `lo`, `pc`, the CP0 registers by name (at minimum Status, Cause, EPC,
BadVAddr, Config, PRId, EntryHi/Lo0/Lo1, PageMask, Index, Random, Wired,
Context, XContext, Count, Compare, WatchLo/Hi, LLAddr, TagLo/Hi), `fpr[32]`
plus FCSR and FIR, and a `cpu` field identifying R4400 vs R5000 from PRId.

**Additionally**, behind its own flag (e.g. `--cheritest-dump-hook`, default
off), support the cheritest convention: a write to CP0 register 26 triggers the
same dump. Keep this strictly opt-in — **CP0 26 is ECC on a real R4400
(`cp0_ecc` in `src/mips_core.rs`) and the PROM writes it during cache
initialisation**, so this must never be active in a normal boot.

**Acceptance:** with the device enabled, a tiny guest program can print via
`PUTC`, produce a dump file whose values match what the monitor's `regs` /
`cop0` / `cop1` print, and exit the emulator with a chosen code. With the
device disabled (the default), IRIX 6.5 boots exactly as before and the address
range reads/writes behave as they did previously. Snapshot save/restore with
the device enabled round-trips.

---

## Out of scope

Do not build any part of the test suite itself — no test sources, no linker
script, no image-build scripts under `cpu-tests/`. That work is planned
separately in `cpu-tests/PLAN.md` and starts once these four features exist.
