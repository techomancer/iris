# IRIS CPU test suite — plan

Goal: a **self-checking, bare-metal MIPS test suite** for the CPUs IRIS
emulates (R4400 / MIPS III and R5000 / MIPS IV), delivered as a **bootable SGI
CD image** the PROM loads directly, and runnable headless in CI.

This document is the roadmap. Nothing is built yet.

---

## 1. What already exists (don't rebuild it)

| Thing | Where | Note |
|---|---|---|
| 101 host-side Rust CPU tests | `src/mips_exec_test.rs` | ALU, FPU, TLB, LL/SC, COP1X, FR-bit aliasing, fusion… Runs under `cargo test`. |
| TLB unit tests | `src/mips_tlb_test.rs` | |
| SGI volume-header writer | `src/sgi_vh.rs` | Writes magic + partition table + checksum. **Does not yet write the volume directory** (the `vd[15]` array at offset `0x48`) — that is what makes a disk bootable by name. |
| CD-ROM with 512↔2048 block switching | `src/scsi.rs:174` | Exactly what the PROM/`dksc` needs to read an SGI VH + EFS off a CD. |
| Runtime disc load | `iris-ci cdrom-load <id> <path>` | Swap a freshly built test image without restarting the emulator. |
| Serial drive + capture | `iris-ci serial-send / serial-wait / --serial-log` | The CI result channel. |
| Snapshots | `iris-ci save/restore` (~145 ms restore) | Restore a "sitting at the PROM prompt" snapshot instead of re-POSTing each run. |
| Monitor console (TCP 8888) | `src/mips_exec.rs:5880+` | `regs`, `cop0`, `cop1`, `mem`, `dis`, `bp add`, `step`, `loadsym`, `l1i/l1d/l2`, `tlb dump`. This is the debugger for failing tests. |
| GDB stub | `src/gdb_stub.rs`, `--gdb-port` | `mips-…-gdb` can single-step a failing test binary. |
| YAML test driver | `tools/iris-test`, `tools/tests/*.yaml` | Existing pattern for scripted emulator runs. |

The Rust tests exercise the *interpreter's* opcode handlers from the inside.
The new suite exercises **the whole machine** — decode, JIT, caches, TLB,
exception vectors, CP0 timers — from the guest's point of view, the same way
IRIX does. The two are complementary; neither replaces the other.

---

## 2. Facts established about the boot path (verified against the PROM image)

Strings extracted from the embedded PROM (`src/prombin.rs`, 070-9101-011, 512 KB):

- **The PROM loads both ELF and COFF**: `"binary is ELF"`, `"binary is COFF"`,
  `"Illegal f_magic number 0x%x, expected MIPSELMAGIC or MIPSEBMAGIC"`,
  `"Illegal a_magic number 0x%x, expected OMAGIC"`.
  → Target format: **ELF32, big-endian, static, no dynamic sections.**
- **Boot syntax**: `"boot: boot [-f FILE] [-n] [ARGS]"`, device name `dksc`,
  `"partition(8)"`, `"No volume header on device: %s."`,
  `"Warning: unrecognized volume header type."`
  → `boot -f dksc(0,<scsi-id>,8)<name>` loads a **file named in the volume
  header directory**.
- **Distribution-CD layout the PROM itself expects**: `part(8)sashARCS` and
  `part(7)/stand/ide.` → volume header holds the bootable standalone binary;
  **partition 7 holds an EFS filesystem** with `/stand/...`. That is exactly
  the "EFS CDROM" shape you want.
- **Network boot exists in the PROM**: full BOOTP + TFTP client
  (`bootp()sashIP24`, `"TFTP: RRQ for file %s to %s"`, `network(%d)tftp()`).
  IRIS answers BOOTP (`src/net.rs:1679`) but **has no TFTP server** — adding one
  (~150 lines in `net.rs`) would let `boot -f bootp()cputest.elf` pull the
  freshly-built ELF straight off the host with no image rebuild at all.

Volume header layout (matches `src/sgi_vh.rs` offsets, confirms the struct):

```
0x000 magic 0x0BE5A941 | 0x004 root/swap part | 0x008 bootfile[16]
0x018 device_parameters[48]
0x048 volume_directory vd[15]   -> { char name[8]; int lbn; int nbytes; }   <-- to add
0x138 partition_table  pt[16]   -> { int nblks; int first; int type; }      <-- exists
0x1F8 csum (32-bit BE words sum to 0)                                       <-- exists
```

### 2.1 Verified against real media

Parsed from `~/IRIX 6.5.22 Installation Tools and Overlays (1 of 3).iso`
(the PROM boots this exact CD), so this is measured, not inferred:

```
magic 0x0BE5A941   rootpt 0  swappt 0  bootfile ''      csum VALID

volume directory @ 0x048           lbn is a 512-byte block number
  [0] sgilabel   lbn=32      nbytes=512
  [1] mr         lbn=33      nbytes=19200000     (the miniroot)
  [2] sash64     lbn=37533   nbytes=266000
  [3] sashARCS   lbn=38053   nbytes=342532       <- what an Indy boots

partition table @ 0x138
  [ 7] nblks=1283016  first=48736  type=5  (SYSV)
  [ 8] nblks=48736    first=0      type=0  (VOLHDR)
  [10] nblks=1331776  first=0      type=6  (VOLUME)
```

Three things this settles, two of which contradict the obvious guess:

1. **Partition 8 spans every voldir file.** It is 48736 blocks (~24 MB) here,
   not the 8 sectors `src/sgi_vh.rs` writes today. The volume-header partition
   must cover all the data the voldir points at; partition 7 starts after it.
2. **Partition 7 is type 5 (`PT_SYSV`), not type 7 (`PT_EFS`)** — even though
   the filesystem on it *is* EFS. Confirmed by the superblock at block 1 of
   that partition: `fs_magic` = `0x00072959` (`EFS_MAGIC`) at offset `0x1c`,
   `fs_size` = 1280320 sectors, `fs_ncg` = 160, `fs_cgfsize` = 8000. Copy what
   the real CD does.
3. **Both loader paths are in use on one disc.** `sashARCS` is **ECOFF**
   (`f_magic` 0x0163 = MIPSEBMAGIC, `a_magic` OMAGIC, `F_EXEC`), text at
   **`0x10000000`**, entry `0x10021020`. `sash64` is **ELF64 MSB, `ET_REL`** —
   a relocatable object the PROM relocates, which is what the PROM string
   `"Cannot load and relocate %s."` refers to.

---

## 3. Two CPUs, one binary

R4400 vs R5000 is a **compile-time cargo feature in IRIS** (`r5k`, plus
`r5ksc` / `r5ksc_triton`), not a runtime switch — see `Cargo.toml` and
`rules/perf/hardware-profiles.md`. So the *suite* must run unmodified on both
and decide expectations at runtime:

| | R4400 (default build) | R5000 (`--features r5k`) |
|---|---|---|
| `PRId` | `0x00000440` | `0x00002321` |
| FPU `FIR` | `0x00000500` | `0x00002300` |
| L1 I/D | 16 KB, direct-mapped, 16 B lines | 32 KB, 2-way, 32 B lines |
| `Config.IB/DB` | 0 / 0 | 1 / 1 |
| ISA | MIPS III | MIPS IV (RECIP/RSQRT, MOVF/MOVT, COP1X MADD/LDXC1…, PREF) |
| L2 | present | absent, external, or Triton on-die per feature |

(source: `src/mips_core.rs:348-364`, `src/mips_cache_v2.rs:41-100`,
`src/mips_exec.rs:777-790`)

Design rule: **every check declares which CPUs it applies to.** A MIPS IV
opcode is a *pass* on R4400 only if it raises Reserved Instruction, and a
*pass* on R5000 only if it computes the right answer. That cross-CPU
differential is one of the most valuable things this suite can test, and
nothing tests it today.

---

## 4. Harness design

### 4.1 Result model — self-checking in the guest

Each test computes, compares against a compiled-in expected value, and prints
one line. No host-side parsing of register dumps in the normal path.

```
IRIS CPU TEST  build 2026-08-18  cpu=R4400 prid=0x00000440 fir=0x00000500
alu/dadd_overflow .......... PASS
mem/lwl_all_offsets ........ FAIL  off=3 got=0x0000000012345678 want=0x1234567800000000
...
RESULT: 412 passed, 1 failed, 7 skipped (n/a on this cpu)
IRIS-CPUTEST-DONE rc=1
```

`IRIS-CPUTEST-DONE rc=N` is the token `iris-ci serial-wait` matches on, and the
exit code the CI job reports. Works over serial, works on real hardware,
works from the CD — no emulator-specific hooks required.

Secondary path (debugging + the external `cheritest` corpus): the IRIS **test
device** — a signature register the guest probes for, plus `PUTC`, a full
machine-state `DUMP`, and an `EXIT` that sets the process exit code. Absent on
real hardware, so the suite probes for it and falls back to SCC-only output.
Landed as `--test-device` (`src/testdev.rs`).

### 4.2 Console

Write bytes straight to the Z85C30 through KSEG1 (uncached):

- IOC base `0x1FBD9800` (`src/ioc.rs:15`), SCC at `+0x30..0x3C`
  (`IOC_SERIAL1_CMD/DATA`, `IOC_SERIAL2_CMD/DATA`) → KSEG1 `0xBFBD9830…`.
- Poll TX-empty in RR0 before each byte. ~40 lines of C.
- The PROM has already initialised the console before it hands control over, so
  the test binary does not need to program baud rate / WR registers.

### 4.3 Exceptions

The suite must own the exception vectors, because half the interesting tests
*are* exception tests. `start.S` will:

1. Set `Status` = known state (BEV=0, EXL=0, KX/SX/UX per test needs, CU1 on, FR per test).
2. Install handlers at `0x80000000` (TLB refill), `0x80000080` (XTLB refill),
   `0x80000180` (general) — write, then flush I-cache with `CACHE` ops.
3. Provide `expect_exception(code) { … }` helpers: arm an expectation, run the
   faulting instruction, handler records `Cause.ExcCode` / `Cause.BD` /
   `EPC` / `BadVAddr` and bumps `EPC` past the fault, test asserts the record.

This mirrors how `cheritest` does it (`bev0_handler_install`,
`check_instruction_traps`) — a proven shape, see §7.

### 4.4 Where the binary lives

Static ELF32-BE, `-nostdlib -ffreestanding -mno-abicalls -fno-pic -G0`.
Whole suite is one binary; `.data`/`.bss` inside it; own stack in `.bss`.

Link address: the one address *proven* to work with this PROM is
**`0x10000000`** — that is where the shipping `sashARCS` on the IRIX 6.5.22 CD
puts its text (§2.1). But `0x10000000` is in **kuseg, which is TLB-mapped**,
and a suite whose whole job includes rewriting all 48 TLB entries cannot
safely execute out of a mapped region — the first `tlbwi` test would unmap the
code running it.

So `start.S` **relocates itself into KSEG0 and jumps there** before running a
single test:

1. PROM loads us wherever it likes (`0x10000000`, or a KSEG0 `p_vaddr` if that
   turns out to work — Phase 0 tries the direct route first).
2. Startup copies the image to a fixed KSEG0 address (candidate
   `0x80200000`, 2 MB in, clear of PROM scratch), flushes I-cache, jumps.
3. Everything from there runs unmapped: TLB tests can't pull the rug out, and
   cache tests can switch between the KSEG0 and KSEG1 views of their own data
   deliberately rather than accidentally.

This also makes the suite independent of *how* it was loaded — direct
`--load-elf`, `boot -f dksc(...)`, or `boot -f bootp()` all converge on the
same running layout.

---

## 5. Test inventory

Ordered by (emulator-bug-likelihood × cheapness). Each is a directory of small
`.S`/`.c` files registered in a table.

**alu/** — add/addu/sub/subu/and/or/xor/nor/slt(u)/slti(u)/lui; overflow traps
on `add`/`addi`/`dadd`/`daddi` and *no* trap on the `u` forms; **32-bit ops must
sign-extend bit 31 into a 64-bit register** (the classic MIPS64 emulator bug);
`dadd/dsub/daddi(u)`; `dsll/dsrl/dsra` + `32` variants + variable forms;
shift counts ≥ 32; `sll $0` vs `nop` vs `ssnop`.

**muldiv/** — `mult/multu/div/divu/dmult/dmultu/ddiv/ddivu`, HI/LO writes,
`mfhi/mflo/mthi/mtlo`, division by zero (no trap, HI/LO unpredictable but must
be *stable*), `0x8000000000000000 / -1`, 32-bit results sign-extended.

**mem/** — `lb/lbu/lh/lhu/lw/lwu/ld/sb/sh/sw/sd`; **`lwl/lwr/swl/swr` and
`ldl/ldr/sdl/sdr` at all 4/8 alignments** (big-endian merge semantics — very
high bug density, and `src/mips_exec_test.rs` only covers some cases);
unaligned `lw`/`ld` → AdEL/AdES with correct `BadVAddr`; `ll/sc` and `lld/scd`
including SC-fails-after-intervening-store, SC-fails-after-ERET, and `LLAddr`;
KUSEG/KSEG0/KSEG1/KSEG2/XKPHYS address decoding.

**branch/** — every conditional branch taken/not-taken; **branch-likely
nullification**; `j/jal/jr/jalr` (incl. `jalr` with rd≠31); `$ra` value =
PC+8; delay-slot semantics: load in delay slot, `jal` whose delay slot writes
`$ra`, branch at the end of a 256 MB `j` region; exception *in* a delay slot →
`EPC` = branch address and `Cause.BD` = 1.

**excep/** — `SYSCALL`, `BREAK`, all six `T*`/`T*I` traps, integer overflow,
Reserved Instruction, Coprocessor Unusable (CU1 clear → COP1 access traps;
COP2/COP3 always), address error on fetch, correct **vector selection**
(0x…000 vs 0x…080 vs 0x…180, BEV=1 → 0xBFC00200/…380), `EXL`/`ERL` behaviour,
`ERET` (clears EXL, clears LLbit, no delay slot), nested exception with EXL set.

**cp0/** — per-register writable-bit masks (R4400 vs R5000 differ; `PRId`
read-only; `Config` mostly read-only except K0/CU); `Count`/`Compare` →
IP7 timer interrupt and `Count` running at half clock
(`src/mips_core.rs:589`); `Random` decrement + `Wired` floor; `Context`/
`XContext` auto-fill on TLB miss; `WatchLo/WatchHi` watchpoints on load/store;
`LLAddr`; `TagLo/TagHi`; software interrupts via `Cause.IP0/IP1`; `Status.IM`
masking; interrupt taken in a delay slot.

**tlb/** — `TLBWI`/`TLBWR`/`TLBP`/`TLBR` round-trip over all 48 entries;
every page size 4 KB…16 MB via `PageMask`; global bit; ASID match/mismatch;
`V`/`D` bits → TLB Invalid / TLB Modified with the right `ExcCode`; refill
vector choice (32-bit vs 64-bit mode); EntryHi/EntryLo field masking of
reserved bits; probe miss sets `Index` bit 31.

**cache/** — `Config` geometry must match the real CPU's (16 KB/1-way/16 B vs
32 KB/2-way/32 B); all `CACHE` ops (Index_Invalidate, Index_Writeback_Inv,
Index_Load_Tag, Index_Store_Tag, Hit_Invalidate, Hit_WB_Inv,
Create_Dirty_Excl, Fill) on I, D and (where present) S; **TagLo/TagHi
round-trip through Index_Store_Tag / Index_Load_Tag**; **KSEG0 vs KSEG1 view
coherency** (write cached, read uncached, and the reverse, with the right
flushes); associativity probe by controlled eviction; dirty-line writeback on
eviction. `rules/snapshot/l1d-tag-must-match-the-cache-line-layout.md` shows
this area has already bitten once.

**fpu/** — FR=0 register aliasing vs FR=1 (the odd-register rules) for
`mtc1/mfc1/lwc1/swc1/ldc1/sdc1` and arithmetic; `add/sub/mul/div/sqrt/abs/
neg/mov` in S and D; all conversions (`cvt.s/d/w/l`, `round/trunc/ceil/floor`)
under all four rounding modes; **IEEE edge cases** — signed zeros, infinities,
quiet vs signalling NaN propagation, denormal input/output handling (R4400
punts to software on underflow; see `src/mips_exec.rs:3779`); FCSR
cause/enable/flag bit interaction, trapped vs untrapped exceptions;
all 16 `C.cond.fmt` predicates; `BC1T/F` + `BC1TL/FL` with a delay slot;
FCSR readback after each op.

**mips4/** — R5000 only, and *must raise RI on R4400*: `MOVF/MOVT/MOVN/MOVZ`,
`MOVF.fmt/MOVT.fmt/MOVN.fmt/MOVZ.fmt`, `RECIP.fmt`, `RSQRT.fmt`, COP1X
(`LWXC1/LDXC1/SWXC1/SDXC1/MADD/MSUB/NMADD/NMSUB/PREFX`), `PREF`, multiple FP
condition codes (CC field ≠ 0) and `C.cond.fmt` writing CC1–CC7.

**identity/** — `PRId`, `FIR`, `Config`, TLB size, cache geometry — printed in
the header and asserted against the CPU the build claims to be. Cheap, and it
catches "the r5k feature didn't actually take" mistakes instantly.

Rough size: ~400–500 individual checks, of which ~120 are the high-value
exception/TLB/cache/FP-edge ones.

---

## 6. Where expected values come from (the oracle problem)

This is the part that decides whether the suite is worth anything. A test that
records what IRIS currently does and calls it "expected" just freezes today's
bugs.

Priority order for each expected value:

1. **The manuals.** `docs/R4000_um2.pdf` is already in-repo; the MIPS IV ISA
   spec and the R5000 user manual cover the rest. Everything in §5 except the
   FP numeric edge cases is derivable from the spec by hand.
2. **Host-computed IEEE-754 values** for the FPU vectors: a generator script
   computes expected results with correct rounding on the host and emits a
   C table. Removes hand-arithmetic errors for hundreds of FP cases.
3. **Another implementation** as a cross-check where the spec is ambiguous:
   MAME's MIPS3 core, or QEMU's MIPS64 target.
4. **Real hardware.** If an Indy or Indigo2 is reachable, the same CD boots on
   it and prints the same lines. That is the gold standard, and the CD form
   factor is what makes it possible — worth keeping the suite PROM-loadable
   and free of IRIS-specific hooks for exactly this reason.
5. **Golden signature** (record → review → commit) *only* for things no spec
   pins down (e.g. HI/LO after divide-by-zero). Marked as such in the output
   so nobody mistakes them for architectural requirements.

---

## 7. Prior art reviewed

- **`../cheritest`** (Cambridge BERI/CHERI) — the strongest reference.
  ~530 bare-metal MIPS64 tests: `alu` 100, `branch` 75, `cp0` 94, `fpu` 157,
  `mem` 62, `tlb` 29, `cache` 6. Pattern: a `.s` test leaves results in
  registers, the simulator dumps the register file on a magic `mtc0 $x, $26`,
  and a `.py` file asserts on named registers. Their `macros.s`
  (`BEGIN_TEST`/`END_TEST`, `check_instruction_traps`, `bev0_handler_install`)
  is a proven exception-testing idiom worth imitating.
  Caveats: targets MIPS64r2 + CHERI, so a chunk of the corpus
  (`rdhwr`, `UserLocal`, `Config1-6`, `BadInstr`, paired-single, NaN-2008,
  capability tests) is not R4400/R5000; and it is **BERI HW-SW License /
  Apache-2.0**, while IRIS is deliberately BSD-3-Clause throughout. Keep it
  **out of tree** — reference it as an optional external corpus, don't vendor it.
- **`../mips-cpu`** — MIPS-I teaching CPU in Verilog. Its `test/` uses
  `.asm` + `.check` (expected final register values) + a diff script. No CP0,
  no 64-bit, no FPU. Useful only as a smoke-test shape.
- **`../MIPS-CPU-Test-Cases`** — MIPS32 integer subset for a 5-stage Verilog
  core, checked by eyeballing waveforms, explicitly no CP0. Not applicable.

---

## 8. Toolchain

Nothing MIPS is installed on this host (`gcc-mips-linux-gnu`, `clang`,
`mips64-elf-gcc` all absent; `genisoimage`/`xorriso`/`python3` present).

Recommended: **`binutils-mips-linux-gnu` + `gcc-mips-linux-gnu`** (Debian/Ubuntu
packages, one `apt-get` line locally and in CI). Produces ELF32 MSB directly.
MIPS III/IV instructions are reachable from an o32 toolchain with
`.set mips3` / `.set mips4` / `.set gp64` around 64-bit code, and
`-march=mips3 -mabi=32 -mfp64`-style flags for the C parts. Alternatives:
a bare-metal `mips64-elf` GCC (cleaner, but must be built or fetched), or
MIPSPro on the IRIX guest itself (what `test/dktest` and `test/ib` do today —
fine for userland tools, wrong for bare-metal CPU tests).

---

## 9. Build & run loop

Three ways to get the binary executing, fastest first:

**Tier 0 — direct load (needs a small IRIS change).**
Add `--load-elf FILE` / a monitor `loadbin <file> <addr>` command, then
`jump` + `run`. Sub-second iteration, no PROM, no image. ~100 lines against the
existing monitor command table in `src/mips_exec.rs`. Best debug loop by far;
also the only mode that can run before POST.

**Tier 1 — PROM boot from a scratch disk.**
Write the ELF into the volume-header directory of a small raw image attached
at a spare SCSI ID, `boot -f dksc(0,2,8)cputest`. Needs only the `vd[15]`
writer added to `src/sgi_vh.rs`. Rebuild = write a few hundred KB.

**Tier 2 — the real deliverable: bootable EFS CD.**
```
image/
  block 0        SGI volume header: magic + vd[] + pt[] + csum
                   vd:  "cputest"  -> the standalone ELF (bootable by name)
                   pt7: EFS filesystem   pt8: volume header   pt10: whole volume
  partition 7    EFS: /stand/cputest, /stand/<per-area binaries>, /README
  (optional)     ISO9660 image of the same files for reading on a modern host
```
- `boot -f dksc(0,4,8)cputest` from the PROM — the same path a real SGI
  distribution CD uses (`part(8)sashARCS`).
- IRIX can also `mount -t efs -o ro /dev/dsk/dks0d4s7 /CDROM` and run the
  binaries or read the docs (see `rules/irix/cdrom-changer-eject-no-mediad-remount.md`).
- **EFS writing**: no host tool exists. Options, in order of preference:
  (a) write `tools/mkefs` (Rust or Python) — EFS is a simple extent-based FFS
  derivative, ~600 lines, and Linux's read-only `fs/efs` driver plus IRIX
  itself are two independent ways to validate the output;
  (b) bootstrap once inside the guest with `mkfs_efs` on a scratch volume and
  pull the image out with `iris-ci scratch-read`, using that as the reference
  image to diff `mkefs` against.
- The voldir holds 15 entries of ≤8-char names, so a handful of separately
  bootable per-area binaries also fit without EFS at all — that is the useful
  fallback if `mkefs` slips.

**Optional accelerator — TFTP.** Adding a TFTP server next to the existing
BOOTP handler in `src/net.rs` unlocks `boot -f bootp()cputest.elf`: no image,
no disk, straight from the host build directory. ~150 lines, and it exercises
the PROM's network stack (which nothing currently tests).

---

## 10. CI

`.github/workflows/rust.yml` today runs `cargo build` + `cargo test`. Add a
`cpu-tests` job:

```
apt-get install gcc-mips-linux-gnu binutils-mips-linux-gnu
make -C cpu-tests            # build the ELF + image
matrix:
  cpu:  [r4400, r5000]       # cargo features: "" vs "r5k" (± r5ksc_triton)
  exec: [interp, jit]        # IRIS_JIT=1 --features jit
run: iris --ci --headless … ; iris-ci …; expect "IRIS-CPUTEST-DONE rc=0"
```

Four cells. The interp-vs-JIT axis is the point: a guest-visible ISA suite is
the cleanest possible JIT differential test, and today nothing runs one
automatically. (Note `rules/jit/verify-mode.md`: JIT verify mode is invalid for
blocks containing stores, so a full-machine suite covers ground verify mode
structurally cannot.)

A `--serial-log` file is the artifact on failure; a failing check is then
reproduced locally under the monitor (`bp add <sym>`, `regs`, `cop0`) or GDB.

---

## 11. Proposed layout

```
cpu-tests/
  README.md  PLAN.md  Makefile  toolchain.mk
  harness/   start.S vectors.S console.c testlib.{c,h} cpuid.c link.ld
  tests/     alu/ muldiv/ mem/ branch/ excep/ cp0/ tlb/ cache/ fpu/ mips4/ identity/
  gen/       fpvectors.py            # host-computed IEEE-754 expectations
  image/     mkvh.py mkefs.py mkcd.sh
  run/       run-local.sh matrix.sh cputest.yaml
  expected/  r4400.txt r5000.txt     # golden transcripts, reviewed not recorded
  docs/      oracle.md prom-boot.md efs-format.md
```

---

## 12. Phases — status

| Phase | Deliverable | Status |
|---|---|---|
| **0** | Toolchain, `hello.elf`, load address, boot syntax | **done** — n32 cross GCC, links and relocates to `0x88200000` |
| **1** | Harness + `alu`, `muldiv`, `mem`, `branch` | **done** — 85 tests green |
| **2** | `excep`, `cp0`, `tlb` | **done** |
| **3** | `fpu`, `mips4` with the R4400-must-RI differential | **done** — the differential found two bugs |
| **4** | `cache`; JIT-vs-interp matrix in CI | **done** — `run/matrix.sh`, `.github/workflows/cpu-tests.yml` |
| **5** | Volume header → bootable disk → **EFS CD** | **partly** — `mkvh` image boots through the PROM end to end; the EFS partition is the remaining piece |

### Where Phase 5 stands

`boot -f dksc(0,2,8)cputest` at the PROM prompt loads the ELF out of the volume
header and runs the whole suite — verified, 784 checks, via `run/run-prom.sh`.
That is the mechanism the CD needs, proven on a disk image.

What remains for the CD proper:

1. **An EFS writer.** No host tool exists. `tools/mkefs` (~600 lines) is the
   plan; Linux's read-only `fs/efs` driver and IRIX itself are two independent
   ways to check its output. The fallback is to bootstrap one image inside the
   guest with `mkfs_efs` on a scratch volume and pull it out with
   `iris-ci scratch-read`, then diff against it.
   Remember partition 7 is type **5 (`PT_SYSV`)** on real SGI media, not type 7.
2. **The 2048/512 block-size dance.** `src/scsi.rs` already switches the drive
   between 2048- and 512-byte logical blocks, which is what lets the PROM read
   an SGI volume header off a CD at all. The image just needs writing at the
   size the PROM asks for.
3. **Optional**: an ISO9660 view of the same files so the disc is readable on a
   modern host.

The voldir holds 15 entries of ≤8-char names, so several separately bootable
per-area binaries fit without EFS at all — the useful fallback if `mkefs` slips.

## 13. Decisions made (2026-08-18)

1. **Toolchain** — Debian cross-GCC: `gcc-mips-linux-gnu` +
   `binutils-mips-linux-gnu`, emitting ELF32 MSB. One `apt-get` line locally
   and in CI.
2. **Checking model** — **both**. Self-checking in the guest is the normal
   path (portable to real hardware, works from the CD, no host parsing); a
   machine-state dump hook is the debugging path and the enabler for running
   the external `cheritest` corpus.
3. **Emulator changes** — all of them, done first and separately in another
   session. All four landed: the volume-directory writer and `mkvh`, direct ELF
   load (`--load-elf`, monitor `loadelf`/`loadbin`), a TFTP server in the NAT
   gateway, and the test device (`--test-device`).
4. **cheritest** — external and optional, referenced by path. `cpu-tests/`
   stays BSD-3-Clause like the rest of IRIS.
