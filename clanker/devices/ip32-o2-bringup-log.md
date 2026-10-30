# The O2 (IP32) bring-up log

Status: history, 2026-09 to 2026-10. How the O2 was brought up, gate by gate,
with the dead ends. For the machine as it is, read [docs/o2-ip32.md](../../docs/o2-ip32.md);
the "Today" section below summarises it as of 2026-10-24, and the rest is the
journal, in order.

## Today (read this first)

The O2 runs as a machine like the others: `profile = "o2_ip32"`, configured
by the shared keys (`prom`, `nvram` for the PROM's flash, `banks` as DIMM
kits, `[scsi.N]` on controllers 0 and 1, `serial_port_a`/`b`, `headless`,
`[network]`, `[clock] fixed_mhz`) with `[o2]` holding only `mouse`,
`control_port` and `arcs_fpu`; the old `[o2]` spellings of the rest are
folded in on load (2026-10-24). Started with `iris --config`. Its devices
are on the shared monitor (`monitor_port`, default 8888).

How it is put together now:

- **The bus** (`src/ip32.rs`): RAM, the PROM, GBE and VICE, and two chips
  that claim their own space and decode it, as HPC3 does on the Indigo2:
  `Mace` (`src/dev/mace.rs`: MACE's 8 MB and the PCI windows, with the
  bridge, Ethernet, ISA block, PS/2, UST, audio, serial DMA, UARTs, RTC and
  1-Wire ID) and `CrimeSpace` (`src/dev/crime.rs`: CRIME's registers and
  the rendering engine).
- **Memory** (`src/dev/crime_mem.rs`): the banks in one ppmem object, mapped
  into a window as CRIME's bank controls decode them. The CPU's cache and
  the JIT, and the rendering engine (`RamWindow`), use the window directly.
- **The rendering engine** (`CrimeRe` in `src/dev/o2gfx.rs`) runs on its own
  thread, fed through the shared gfifo standing in for CRIME's interface
  buffer (far deeper than the real 128 entries); a
  full buffer, or a register read while the engine is busy, makes the CPU
  retry.
- **Both CPUs' caches** (`src/cpu/mips_cache_shadow.rs`): the tagless shadow
  model from reset. POST keeps its stack in the first 4 KB of physical memory,
  in the data cache, while it tests the memory under it; the shadow model
  holds that window's lines while POST runs
  ([o2-post-cache-as-ram.md](o2-post-cache-as-ram.md)).

The rest of this document is the bring-up journal, in order. Its middle
chapters drive the machine through a test harness steered by `IRIS_IP32_*`
variables (`IRIS_IP32_SCRIPT`, `_STEPS`, `_DISK`, `_DIS`, `ip32-view` and
so on). That harness has since been removed; the findings stand, the
commands no longer run.

Diagnostics that remain, all off unless set:

| variable | what it does |
|---|---|
| `IRIS_IP32_HOT` | per-register access counters for MACE, GBE and the rendering engine (finding a driver polling a bit nothing sets) |
| `IRIS_IP32_RE_TRACE=N` | keep the last N rendering-engine register writes; save with the control port's `retrace` |
| `IRIS_IP32_MEC_TRACE` | Ethernet frames sent and received, and why a received one was dropped |
| `IRIS_O2_AUDIO_TRACE` | audio output activity, and each time a ring ran dry between sounds |
| `IRIS_O2_AUDIO_DUMP=file` | every sample each output plays, at the guest's rate, before resampling |
| `IRIS_O2_AUDIO_CAPTURE=file` | what the host plays, after resampling and mixing (raw 16-bit stereo) |

## The assessment (2026-09-19)

Assessment written 2026-09-19. Nothing implemented yet. Scope agreed with the
owner: **run the real O2 PROM** (not a synthetic ARCS), serial console only,
graphics explicitly out of scope.

## The PROM images we have

`ip32/ip32prom.rev4.18.bin` and `ip32/ip32prom.rev4.3.bin`, both 512 KiB, both
genuine IP32 firmware. Verified by parsing the section table rather than
trusting the filenames.

Rev 4.18:

| SHDR @ | name | length | version | load |
|---|---|---|---|---|
| 0x000008 | `sloader` | 0x04000 | 1.0 | 0xbfc00000 |
| 0x004008 | `env` | 0x00400 | 1.0 | — |
| 0x004408 | `post1` | 0x04d44 | 1.0 | 0xa0004000 (copied to RAM) |
| 0x009208 | `firmware` | 0x5fffc | **4.18** | 0x81000000 (.rodata +0x48e70) |
| 0x069208 | `version` | 0x00388 | 4.18 | ELF metadata |

SHDR layout, from the bytes: magic at +0, length at +4, type flags at +8,
NUL-padded name at +12, version string at +44, checksum at +52, then load
address and .rodata offset for `firmware`.

4.3 has the same five sections at the same offsets bar the last (0x68008).

## What tracing the reset path already told us

    bfc00000:  b      0xbfc00048        # jump over the SHDR
    bfc00048:  b      0xbfc003a8        # ...to the real entry
    bfc00100:  mtc0   a0,$17            # exception vector stub
    bfc00104:  lui    t8,0xb400
    bfc00108:  sd     a0,624(t8)        # 0xb4000270
    bfc0010c:  lui    t8,0xb400
    bfc00110:  addiu  t8,t8,512
    bfc00118:  sd     zero,0(t8)        # 0xb4000200

`0xb4000000` is uncached KSEG1 for physical `0x14000000` — **CRIME**. Against
NetBSD's `crimereg.h` those two offsets are `CRIME_MEM_ERROR_ECC_REPL` (0x270)
and `CRIME_MEM_CONTROL` (0x200): a memory-error exception handler, installed
before anything else runs.

Two facts fall out immediately:

- CRIME registers are **64-bit** — the PROM uses `sd`/`ld`, not `sw`/`lw`. Our
  bus plumbing must carry 64-bit accesses to this device.
- CRIME is live from the very first exception, so it cannot be stubbed out
  later; it is the first thing to build.

## The machine, from NetBSD's own GENERIC32_IP3x

```
crime0     at mainbus0 addr 0x14000000     memory + interrupt controller
crmfb0     at mainbus0 addr 0x16000000     framebuffer            [out of scope]
mace0      at mainbus0 addr 0x1f000000     I/O ASIC
  macepci0 +0x080000 → pci0 → ahc          Adaptec SCSI on real PCI
  mec0     +0x280000                       ethernet
  mavb0    +0x300000                       audio                  [out of scope]
  macekbc0 +0x320000                       keyboard               [out of scope]
  com0/1   +0x390000 / +0x398000           16550-style serial
  mcclock0 +0x3a0000                       RTC
```

Nothing of this exists in IRIS today: 17 source files mention IP22/IP24, and
`crime`/`mace` appear nowhere.

The serial console is a genuine simplification — `com` is a 16550 derivative,
far simpler than the Indy's Z85C30, and the PROM's built-in `env` section uses
the **same `console` variable** we already handle (`console=g` is its default,
`console=d` selects serial).

## Why the real PROM, and what that costs

NetBSD tries real ARCS first and only falls back to its own emulation:

```c
if (arcbios_init(ARCS_VECTOR) == 1) { ... arcemu_init(...) }
```

and `arcemu` only covers IP6/IP10/IP12/IP20/IP22. **IP32 needs real ARCS**, so
the PROM is not optional if the goal is stock NetBSD — and it is certainly not
optional for IRIX.

The consequence, and it is the important one: **virtio does not shrink this
job.** virtio removes devices the *guest OS* needs; the PROM POSTs the real
machine regardless. `post1` is ~20 KiB of memory test driven through CRIME, and
the firmware section will probe MACE and the PCI bridge. virtio's payoff comes
after the PROM is satisfied — it means we would not also have to emulate `mec`
and the `ahc` SCSI controller well enough for a running system, only well
enough for POST to not reject them.

## Milestone reached (and the check was wrong)

The success message POST prints on the good path is

    <post1> <SizeMEM> bank%d = 0x%0lx (128M)

The `<post1>` tag settles it: **we are executing post1.** The check below —
"trace the PC reaching 0xa0004000" — never fires because post1 runs *in place*
from the PROM window at 0xbfc04408+, not from the address its section header
names. The SHDR load address is where sloader would copy it, not where it first
executes. Worth remembering for any future section.

## Suggested first milestone

Not "boot NetBSD" — too far. Instead: **get `sloader` to hand off to `post1`.**

That needs only:

1. A `MachineType::Ip32` profile: reset at 0xbfc00000, the IP32 physical map,
   PROM image loaded at 0x1fc00000.
2. CRIME as a 64-bit register file at 0x14000000 — enough of `CRIME_MEM_*` and
   the revision/ID register to get through early setup.
3. RAM behind CRIME's bank controls.

Success is observable without any console at all: trace the PC reaching
0xa0004000. That single milestone will also answer the real unknown — how much
of CRIME the memory test actually exercises — which is what determines whether
the rest is weeks or months.

After that, in order: MACE register file → `com0` (first console output, and
the first time the PROM talks to us) → RTC → then decide how much of PCI/`ahc`
POST insists on.

## First milestone: what the PROM actually did

`src/ip32.rs` drives the real `MipsExecutor` against a minimal IP32 bus (RAM at
0, CRIME, a MACE stub, PROM). `cargo test --lib ip32::bringup -- --nocapture`,
with a PROM image present. It executes — and answered the question the
milestone existed to answer, on the first run.

**CRIME registers POST touches** (first-touch order):

| offset | | meaning |
|---|---|---|
| 0x0008 | R, then W | `CRIME_CONTROL` |
| 0x0208 … 0x0240 | W | eight consecutive qwords — the memory bank controls |

So the PROM reads CRIME_CONTROL, writes it, then programs **eight memory banks
at a linear 8-byte stride**. Worth noting against NetBSD's `crimereg.h`, which
names these in an interleaved order (`BANK_CTRL0` 0x208, `CTRL1` 0x218,
`CTRL2` 0x210 …). The PROM writes them straight through, which is the better
guide to what the hardware is.

Nothing else in CRIME was touched: no `TIME`, no interrupt registers, no ECC.
The memory-error handler installed at reset never fired, which is the outcome
we want.

**MACE**: 5 reads, 20 writes, against a stub returning zero — so MACE is needed
earlier than expected, but is evidently satisfied by benign answers for now.

**Where it stops.** At PC 0xbfc05c9c the PROM writes physical 0x40000000 and
then reads a register block at 0x40000080–0x400000dc. From
`mace/macereg.h`:

```c
#define MACE_PCI_NATIVE_VIEW    0x40000000
```

It is enumerating the **PCI bus**, looking for the Adaptec SCSI controller.
With nothing there it takes a bus error, wanders, and is lost by 2 M
instructions (ending in RAM at 0x00780824).

**So the gate is not CRIME — CRIME is nearly free.** Eight bank-control writes
and a control register is all POST asked for. The gate is the MACE PCI host
bridge. That is a much better position than the assessment above assumed, and
it changes the order of work: PCI config space next, not more CRIME.

## Second run: the empty PCI bus works

Adding a MACE PCI host bridge whose config reads return **all ones**, plus a
PCI native-view window that does the same, cleared it completely: **no bus
error, and zero unmapped accesses**. The PROM accepts a bus with nothing on it.

It never used `CONFIG_ADDR`/`CONFIG_DATA` at all — 0 config cycles. It only
touched the native-view window (1 read, 7 writes). So at this stage it is not
enumerating PCI so much as poking at where a device would be.

The bug that mattered was in the stub, not the bridge: returning **0** for an
absent device reads back as vendor ID 0x0000, which firmware takes for a
present-but-broken device. `0xffffffff` is the architectural "nobody home".

## Third and fourth gates: two timers and a 1-Wire chip

With PCI answered, the PROM spun 393,301 times on `MACE + 0x340000` —
`MACE_UST_MSC`, the free-running system timer. Its delay loops are

    ld   v1, 0(0xbf340000)      # now
    daddu v1, v1, a0            # target = now + delay
    1:  ld t0, 0(0xbf340000)
        sltu at, t0, v1
        bnezl at, 1b

so a constant hangs forever. Giving it a counter that advances (currently a
coarse `UST_STRIDE` per read — see the caveat in `ip32.rs`) dropped MACE reads
from 393,307 to 6 and let it run on.

It then went straight to hammering `MACE + 0x310008` —
`MACE_ISA_FLASH_NIC_REG` — 20,331 paired read-modify-writes between delays.

**That register is shared, and the first reading of it was wrong.** It does
carry the Dallas 1-Wire data line (`NIC_DATA` 0x08, `NIC_DEASSERT` 0x04), which
is what the name suggests and what I assumed. But capturing the actual values
written settled it: every write is **0x30**, and

```c
#define MACE_ISA_LED_RED    0x10
#define MACE_ISA_LED_GREEN  0x20
```

It is blinking the front-panel LED, not bit-banging a serial ID. The loop is

    bfc05e70:  lui  s0, 0xbf31          # s0 = 0xbf310008, the LED register
    bfc05e80:  ori  s1, s1, 0xa120      # delay = 0x7a120 = 500000
    bfc05e8c:  jal  0xbfc058b0          # delay(s1)
    bfc05e94:  ld   t7, 0(s0)
    bfc05e98:  xori t6, t7, 0x30        # toggle RED|GREEN
    bfc05e9c:  b    0xbfc05e8c          # forever

a panic blinker at 2 Hz. **Lesson: name the register by the bits that are
actually set, not by the name of the register.** Watching values, not
addresses, is what caught it.

## The real gate: `Error, no SIMM in bank0`

The blinker is entered from `bnez s1, ...` — `s1` is a size, and zero means
failure. Just before it, the PROM prints a string, and the string is in the
image at 0xbfc06d14:

    Error, no SIMM in bank0

So POST's **memory sizing** is what fails. The test itself is at 0xbfc05dc8:
three 64-bit patterns written into RAM and compared against constants held in
the PROM at 0xbfc06f90/f98/fa0, with any mismatch branching to the blinker.

What CRIME is told beforehand is the clue:

    0x0008 <- 0x0
    0x0208 <- 0x100      } all eight bank-control
    0x0210 <- 0x100      } registers get the same
    ...                  } value
    0x0240 <- 0x100

Eight banks, all programmed identically, then probed. Our RAM is flat at
physical 0 and ignores the bank registers entirely, so every bank aliases onto
the same store — which is exactly what a sizing algorithm is designed to detect
and report as "nothing there". ### SizeMEM passes: RAM is at 0x40000000

Tracing back from the failing `bne` gives the whole routine (0xbfc05c80):

    sd t6, 0(s3)          # base + 0
    sd t9, 0(a0)          # base + 0x01fffff8
    sd t1, 0(a1)          # base + 0x02000000
    sd t3, 0(a2)          # base + 0x07fffff8
    ld v1, 0(s3)          # read base + 0 back
    bne v1, t0, fail

It *does* write its patterns. The reason none appeared in RAM is that the
physical address they land on is **0x40000000** — visible all along as the
target of the bus-erroring `sd` at 0xbfc05c9c, which I had read as a PCI probe.

**O2 main memory is based at 0x40000000, not at 0.** This is the "RAM at a
non-zero base address" GXemul's documentation mentions as an O2 trait. It
collides with what `macereg.h` calls `MACE_PCI_NATIVE_VIEW`; memory has the
stronger claim on that address, so the PCI window is unmapped for now.

Moving RAM there, SizeMEM passes outright:

    0x40000000 W ffffffff00000000     0x40000000 R ffffffff00000000
    0x41fffff8 W fe00000701fffff8     0x41fffff8 R fe00000701fffff8
    0x42000000 W fdffffff02000000     0x42000000 R fdffffff02000000
    0x47fffff8 W f800000707fffff8     0x47fffff8 R f800000707fffff8
    0x48000000 W ...                  <- bank 1, at base + 128 MB

Every pattern reads back. MACE traffic drops from 20,334 accesses to **5** —
the panic blinker is gone — and CRIME's bank registers go from `0x100` to
`0x104` for the banks behind bank 0, so bit 2 is something POST sets once a
bank has been sized.

The one remaining unmapped access is `0x48000000 W`: bank 1, past our 128 MB.
That is correct behaviour rather than a bug — the PROM installs a bus-error
handler (`jr s8` at 0xbfc00388, with a CP0 dump behind it) specifically so it
can probe banks that are not populated.

### The lesson from this one

The bus error at 0x40000000 was visible in the *first* run of the evening. I
read it as "the PROM is enumerating PCI" because `macereg.h` has a constant at
that address, and then built a PCI window to satisfy it — which duly absorbed
the writes and made main memory invisible. The address was right there; the
wrong name was attached to it.

Twice in one session the same mistake: `MACE_ISA_FLASH_NIC_REG` read as 1-Wire
when the bits said LED, and 0x40000000 read as PCI when the access pattern said
memory. **Trust the access pattern over the header name.**

## Instrumentation, and then the milestone

Three instruments, each chosen because its absence had already cost a wrong
turn:

1. **PC attribution on every recorded access.** The bus publishes the current
   PC (`PcTap`) and the miss log records it. `0x00000018 W from PC 0xbfc051b4`
   is a one-line diagnosis; `0x00000018 W` alone is a puzzle.
2. **A printf tap.** post1's print routine at 0xbfc04d74 is a stub that emits
   nothing, but the format strings are in the image and the arguments are in
   `a0`-`a3` at the call. Reading them there recovers POST's narrative that the
   hardware never prints.
3. **Stall detection with CP0.** When the PC stops spreading, report the loop
   body *and* Cause/ExcCode/EPC. "Spinning at 0xbfc003a0" says nothing;
   "ExcCode=7 (DBE) EPC=0xbfc051b4" names the faulting instruction.

They paid for themselves immediately. The printf tap gave:

    <post1> <SizeMEM> Entering routine
    <post1> <SizeMEM> ECC off
    <post1> <SizeMEM> Initilaize BANK0
    <post1> <SizeMEM> bank0 = 0x... (128M)
    <post1> <SizeMEM> bank1 = 0x..., no simm installed
    ...
    <post1> <SizeMEM> MEM Size = 0x8000000 bytes

and the stall reporter turned two further hangs into single-line answers:

- `ExcCode=7 (DBE)` at 0xbfc051b4 → miss log → `0x00000018 W`. POST writes a
  walking pattern to **low physical memory** long after sizing RAM at
  0x40000000. Low memory aliases the base of RAM, as it does on the Indy.
- The same again at 0xbfc05088 → `0x00001000 R`, i.e. the alias was too small.

Two more gates, both found and fixed in minutes rather than by disassembly.

### The milestone, literally

    stopped after 112536 steps at PC 0xa0004000  <-- post1 entry
    stalls detected: 0
    unmapped accesses (0):

sloader sizes memory, copies post1 into RAM and jumps to it. No stalls, no
unmapped accesses, and the test now asserts it rather than merely printing it.

What it took, in total: CRIME's control and bank registers, a UST timer that
advances, a PCI bridge answering all-ones, a memory window that absorbs
accesses to unpopulated banks instead of bus-erroring, and a low-memory alias.
No PCI enumeration, no `ahc`, no graphics, no 1-Wire, no console.

## POST completes, and the PROM talks

Letting it run past post1 rather than stopping there: post1 executes, returns
to sloader, and sloader **prints to `com0`**. 522 bytes:

    ^@^A^B ... !"#$%&'()*+,-./0123456789:;<=>?@ABC ... ~^?  (the whole byte range)
    SL-9600-8E>

A character-set sweep — the UART testing itself — followed by **sloader's
prompt**: "SL", 9600 baud, 8 data bits, even parity. It then sits polling
`com0`'s line status for data-ready, which the disassembly confirms directly:

    0xbfc01ea0 -> 0xbf390507      com0 + 0x507, LSR
    0xbfc01e8c -> 0xbf390007      com0 + 0x007, RBR

exactly the MACE register layout implemented here. So **POST is finished and
the machine is waiting for console input.**

Receive is implemented (`Com16550::feed`) and the PROM does consume what it is
given — the queue drains — but neither `?` nor a bare carriage return draws any
response. `SL-9600-8E>` is a *serial loader* prompt, which most likely expects
a download protocol rather than typed commands; that is the next thing to
establish. Set `IRIS_IP32_INPUT` to experiment.

### It should hand off, and it is not

Answered, and the answer is that we are on the wrong path.

**It must hand off.** The `firmware` section (v4.18) exists precisely to be
loaded at 0x81000000, and it is the ARCS monitor — the thing `arcbios_init`
finds and NetBSD requires on IP32. A machine that never runs it has no ARCS and
can boot nothing.

**It never does.** Watching the load region directly:

    firmware section: 0 write(s) to its load region, PC entered it: false

Not copied, not entered. So `SL-9600-8E>` is not a resting place on the way to
booting; it is somewhere else entirely.

**It is post1 that takes us there, not sloader.** Capturing the distinct PCs
leading to the first console byte shows them all inside post1's copy in RAM
(0xa0004xxx–0xa00054xx), calling back into the PROM's serial routines at
0xbfc01xxx. So post1 runs, and then *post1* prints the byte-range self-test and
the prompt and waits.

The prompt itself is assembled from a baud table at 0xbfc01e50 — 4800, 9600,
19200, 38400, 57600, 115200 — around the fragments `\n\rSL-` and `-8E`. "SL"
is **serial loader**: a download/diagnostic mode, not the boot path.

### The hypothesis to test next

Something is selecting that mode. The most likely candidate is what we feed it
where a real machine has persistent state: the run touches MACE + 0x3a3f04
(the RTC/NVRAM region, 2 reads and a write) and gets zeros, because nothing
backs it. An all-zero NVRAM is exactly the sort of thing firmware reads as
"diagnostic mode requested" — and the 1-Wire serial-ID chip, which holds the
machine's identity and is also unimplemented, is a second candidate.

So the next gate is probably **persistent state**: NVRAM behind the RTC at
MACE + 0x3a0000, and the 1-Wire ID chip at `MACE_ISA_FLASH_NIC_REG`. Both are
small, and both are the IP32 counterparts of things IRIS already models for the
Indy (`eeprom_93c56`, `ds1x86`).

## The PROM boots: three gates, and an emulator bug

Resolved 2026-09-19, in one chain. Each gate was found by instrumentation, not
by guessing, and the instrumentation is all still in `bringup`.

### 1. The UART self-test, and why the console printed garbage

post1 was blinking the front-panel LED forever: `s0 = 0xbf310008`, writing
`0x20` (green) and `0x30` (amber) a second apart, with `a0 = 0xf4240` =
1,000,000 µs delays between them. That is a POST failure code, not a hang.

Either side of the blink it calls `0xa0004494` with `a0` = 1 and 2, and that
routine sets `s2` to `0xbf390000` or `0xbf398000` — com0 and com1. It is the
**serial self-test**, and it was failing on both ports.

Decoded with MACE's `(reg << 8) + 7` spacing, it sets `MCR = 0x13`. Bit 4 is
**LOOP**: the test runs the UART in internal loopback, writes 0..254, and
requires that it reads the same sequence back, comparing byte by byte and
returning `index | 0x100` on the first mismatch.

`Com16550` had no loopback, so those 255 bytes went out of the transmitter
instead — which is exactly the mysterious `0x00`..`0xFF` dump that had been
appearing on the console and that `grep` kept skipping as binary. One symptom,
one cause. Implementing `MCR_LOOP` (route THR into the receive queue, and the
modem control outputs to the status inputs) makes the test pass and the garbage
disappear.

### 2. `Index_Store_Tag` was writing cache lines back

With the UART fixed, post1 completed and returned to the PROM — which promptly
fell into its serial loader (`SL-9600-8E>`) and blocked in a `getchar` polling
LSR forever.

The loader is reached from `0xbfc00b34`, which is a `jal` plus `b .` — a
terminal state. The real boot is a `jalr v1` at `0xbfc00b00` that should never
return. It returned.

Inside it, four gates guard the handoff, and the PC trace showed the **first**
one failing at `0xbfc044f4`:

```asm
0xbfc0449c  sd   sp, 4104(t0)     ; save sp at 0xa0001008, before...
0xbfc044ac  or   sp, sp, t1       ; ...switching to the uncached stack
...                               ; (post1 runs here)
0xbfc044e8  ld   t0, 8(t0)        ; the saved sp
0xbfc044f0  xor  t1, 0x20000000, sp   ; flip cached/uncached
0xbfc044f4  bne  t0, t1 -> bail
```

A stack-integrity cookie. It should match trivially. A memory watch on the
cookie showed `0xbfc0449c` executing **twice**, the second time with `sp`
already uncached — so the function had re-entered itself, and the cookie no
longer described the current stack.

Watching the physical address of the saved return address explains why:

```
@787us  0x40000fa4 W 0xbfc044b0   sw ra, 44(sp) — correct
@917us  0x40000fa4 W 0xbfc04498   clobbered, from PC 0xbfc06938
@928us  0x40000fa4 R 0xbfc04498   lw ra, 44(sp) — reads the clobbered value
```

`0xbfc06938` is `cache Index_Store_Tag(PD), 0(t2)`, inside a loop walking every
D-cache index storing an invalid tag — ordinary cache initialisation. Our
`C_IST` wrote the line back first. `Index_Store_Tag` must not: it exists to
install tags over power-up garbage, and writing back sends the line's data to
an address derived from the tag being discarded. Here that address was the
PROM's own stack.

This was an **IRIS bug, not an IP32 one** — shared CPU code that every guest
runs through. The fix and its regression testing are in
[`../pr-drafts/pr-cache-index-store-tag.md`](../pr-drafts/pr-cache-index-store-tag.md).
The L2 path in the same `match` arm already got it right.

### 3. GBE has to answer

With the cookie check passing, the PROM copies the `firmware` section in and
enters it — `98336 write(s) to its load region, PC entered it: true`, after
weeks of `0` and `false`. The firmware then probed `0x16000000`, the graphics
back end, which nothing claimed. A storing stub is enough; we drive the machine
on serial and never read a pixel back.

### Where it stands

The firmware runs and prints. Its current complaint is
`ds2502_init: presence pulse not detected` — the 1-Wire identity chip, which is
modelled but whose line-level handshake the firmware does not accept yet. That
is the next gate, and unlike every gate before it, it is one we already have
the parts for.

## The 1-Wire identity chip, gate by gate

Resolved 2026-09-19. Five separate things were wrong, and each one hid the
next. Worth listing because every one of them was found by making the device
report what it saw, not by reading the datasheet harder.

1. **`DEASSERT` alone drives the line.** `nic_write` required `DATA == 0` as
   well, and the firmware writes `DATA = 1` throughout — so the master never
   appeared to pull the line low and no reset was ever recognised. The bus
   trace settles it: only `0x08` and `0x0c` are ever written, so `DATA` is an
   input here, not a level to drive.

2. **The presence pulse is a window in time, not a one-shot.** It had been
   cleared by the first read. The firmware polls around 150 times, so a pulse
   that vanished after the first look was missed by every later one. It is now
   held from `OW_PRESENCE_DELAY_US` to `+ OW_PRESENCE_LEN_US` and any sample
   inside sees it.

3. **The pulse-width thresholds were inside a population.** Printing the
   histogram of master low-pulse widths made this obvious in one line:

   ```text
   31x4 32x20 | 329x1 356x4 357x4 | 1980x2 | 7220 15954 21570
    write-1        write-0          reset      idle low
   ```

   Three populations, 10x and 5.5x apart. `OW_WRITE0_US` was 30 and
   `OW_RESET_US` 400 — the first sat *below* the write-1 cluster, so every
   write-1 read as a write-0 and the command byte came out as zero. Moved to
   120 and 800, in the middle of each gap. The absolute values run long
   against the 1-Wire spec because the PROM bit-bangs with instruction-count
   delays and our instructions-per-microsecond is a free parameter; what
   matters is that the populations stay separated.

4. **The family code.** With the command decoding, the master read exactly
   eight bits — `1,0,0,0,0,0,0,0`, our `0x01` — and immediately reset. It is a
   **DS2502**, family `0x09`. `0x01` is the DS1990A, and it is the wrong
   answer every 1-Wire example hands you.

5. **`READ ROM` has an end; a memory read does not.** After streaming the ROM
   the device stayed in "sending" forever, so the master's next command was
   read as more read slots and silently lost. The decoded log showed it
   plainly — `reset | cmd 0x33 | reset | cmd 0x33`, with no memory command
   ever arriving, while 49 long pulses went by where two `0x33`s account for
   only 8.

Then the memory read itself. Rather than guess the layout, read the parser:
the error strings live at PROM offsets `0x530f8`/`0x53124`/`0x53150`, and
since the `firmware` section is loaded verbatim at `0x81000000` from PROM
offset `0x9208`, each maps to exactly one `addiu` site. `ds2502_get_eaddr` is
at `0x81005674` and `ds2502_read_ram` at `0x810053d4`, and between them they
say:

- send `0xf0`, `TA1`, `TA2`, accumulating a CRC over those three bytes;
- read one byte and compare it with that CRC — mismatch returns failure
  immediately;
- read `128 - addr` data bytes;
- read **one more byte** and check it as a CRC over the data;
- take the first six bytes, **reversed**, as the Ethernet address.

That trailing data CRC is the part no summary of the datasheet mentions, and
its absence is invisible: the master simply reads one byte past the end of the
reply, gets the floating `0xff`, and rejects the whole transfer. The
instrumentation that caught it was logging how much of the reply the master
consumed before giving up — `master took 1040 of 1032 bits` is an eight-bit
shortfall stated out loud.

With all five fixed the console goes quiet, the master takes `1040 of 1040
bits`, and the firmware moves on.

## The PROM boots to its menu

2026-09-19. Two more gates after the identity chip, and the second one changed
the shape of everything before it.

### PS/2: empty is not the same as dead

`MACE + 0x320000` is the pair of PS/2 ports — keyboard at `+0x00`, mouse at
`+0x20`, each `tx / rx / control / status` at eight-byte spacing. The firmware
was reading `+0x320018` **84,619 times**.

Attributing that to code needed one extra piece of instrumentation. The first
attempt named `0x81004d70` every time, which turns out to be a one-line
accessor — `ld a0, 0(a0)`, return the halves. The PROM reaches almost every
register through those, so the PC says nothing; the *return address* is what
names the caller. `PcTap` now carries both, and the answer came straight out:
`0x8101f9e8`, `0x8101fbf8`, `0x8101fd8c`.

There the loop is explicit — spin up to `20834` times waiting for
`status & 0x08`, `TX_EMPTY`. We drive this machine on serial and nothing is
plugged into either port, so a stub returning zero looked harmless. It is not:
a real controller drains its transmit register whether or not a keyboard is
listening, and reporting zero makes every single byte the PROM sends take the
full timeout. Reporting `TX_EMPTY | CLOCK_SIGNAL`, with `RX_FULL` never set,
is what "port present, nothing attached" actually looks like.

### CRIME's counter had to be a clock

With PS/2 quiet, a third of the run was still going into a delay loop at
`0x81006ff0`:

```asm
addiu a2, zero, 66
mult  a0, a2          ; 66 * microseconds
lui   a1, 0xb400
ori   a1, a1, 0x38    ; CRIME + 0x38
sd    zero, 0(a1)     ; zero the counter
ld    a2, 0(a1)       ; ...and wait for it to reach the deadline
```

`66 * microseconds` is the firmware telling us the rate: CRIME's counter runs
at 66.67 MHz. Ours advanced **once per read**, so a delay cost a fixed number
of polls regardless of how long it was supposed to be. Deriving it from the
same clock as the UST — [`CRIME_TICKS_PER_US`], with a write setting where it
counts from, because the PROM zeroes it before each wait — dropped time in
spin loops from 34% of the run to 1%.

And then this happened:

```text
Cannot connect to keyboard -- check the cable.
Warning: time invalid, resetting clock to epoch.
Initialized tod clock.

                         Running power-on diagnostics...

System Maintenance Menu

1) Start System
2) Install System Software
3) Run Diagnostics
4) Recover System
5) Enter Command Monitor

Option?
```

### The calibration lesson

Fixing the clock broke the 1-Wire, and that is the most useful thing in this
section.

The thresholds had been set from measured pulse widths of `32 / 357 / 1980`
microseconds — four to six times the 1-Wire specification's `6 / 60 / 480`.
That was rationalised as the PROM bit-banging slowly against a free parameter,
and the thresholds were moved out to match. They were really compensating for
CRIME's broken counter stretching every delay the firmware asked for.

With the counter fixed the widths are `8 / 90 / 500`, and the **datasheet
numbers work unmodified**. The tuned constants then put `OW_WRITE0_US` above
the write-0 population, so every command decoded as `0xff` and the identity
chip broke again.

A device model that needs its constants tuned away from the specification is
usually telling you something about the clock, not about the device. It said
so for three sessions and was not listened to.

### What it reports about itself

Taking the menu into the command monitor over the serial port:

```text
> version
VERSION 4.18
O2 R5K/R7K/R10K/R12K
IRIX 6.5.x IP32prom IP32PROM-v4

> hinv -v
                   System: IP32
                Processor: 195 Mhz R5000, with FPU
     Primary I-cache size: 32 Kbytes
     Primary D-cache size: 32 Kbytes
              Memory size: 128 Mbytes

> printenv
console=g
eaddr=08:00:69:12:34:56
ConsoleOut=serial(0)
ConsoleIn=serial(0)
...
```

`eaddr` is the end-to-end proof: that address is not configured anywhere the
PROM can see it. It was programmed into the DS2502's EPROM, and the firmware
bit-banged it off the 1-Wire line one pulse width at a time. The bring-up test
asserts it, along with the menu, the monitor, and the inventory, so any one of
those devices regressing fails the test with the console attached.

## The PCI bus, and where block storage stops being cheap

2026-09-19, started. Three things are done and one decision is open.

### What the PROM looks for

Choosing `1) Start System` makes the PROM probe **bus 0, devices 1, 2 and 3,
function 0, register 0** — a vendor-ID scan and nothing more. An empty bus
answers `0xffffffff` three times and it gives up with `Autoboot failed`.

Putting the controller a real O2 has there — an Adaptec AIC-7880, vendor
`0x9004` device `0x8078` — makes it go much further. It reads the class code,
sizes **all six BARs**, sets the cache line size, enables memory decoding, and
goes straight to the device.

So the config space had to become real: a slot table, and BARs that answer a
size probe. A BAR that stores whatever it is given claims four gigabytes, and
firmware lays the bus out accordingly.

### The window is not where the BAR says

The first attempt mapped PCI memory one-to-one, because the PROM had
programmed BAR1 to `0x80001000`. It bus-errored anyway, and the PROM's own
register dump said why:

```text
Instruction Bus error
  tmp: 81070000 0 81073548 81055888 ba001084 810735d0 ffff 4
```

`0xba001084` is physical `0x1a001084`. `0x80001000` is a **PCI** address; the
CPU reaches it through a window at `0x1a000000`. The translation is
`pci = 0x80000000 + (phys - 0x1a000000)`, and with that the device is where
the firmware looks.

A panic register dump is a gift. It named the address nothing had decoded.

### Byte lanes, and a list of registers that suddenly made sense

With the window mapped, the chip conversation was writes to offsets `0x84`,
`0x91`, `0x93` and `0xbc`. Those are nothing in particular on an AIC-7880.

PCI is little-endian and this CPU is not, so the bridge swaps byte lanes: an
8-bit access at `A` reaches device byte `A ^ 3`. Applying that:

| CPU | Device | Register | What it does |
|---|---|---|---|
| `0x84` | `0x87` | `HCNTRL` | write 1 (chip reset), then 4 (pause), read back |
| `0x91` | `0x92` | `CLRINT` | `0x1f`, clear every interrupt |
| `0x93` | `0x90` | `SCBPTR` | `0,1,2,...,15` |
| `0xbc` | `0xbf` | last SCB byte | `0xff` for each, clearing them |

That is textbook aic7xxx bring-up, and it is the confirmation that both the
window and the swizzle are right. Guessing either one wrong produces a
plausible-looking register list that is entirely fictional.

### Where it stops

The whole conversation, by register:

```text
3a..5d : one write each   sequencer scratch RAM, the driver's configuration
60     : 3 writes         SEQCTL
61     : 1696 writes      SEQRAM
62,63  : 2 writes each    SEQADDR0/1
87     : 2w 1r            HCNTRL
90     : 16 writes        SCBPTR
92     : 1 write          CLRINT
bf     : 16 writes        SCB control
```

**1696 bytes written to `SEQRAM`.** The AIC-7880 has no fixed behaviour to
emulate: the driver downloads a sequencer program into the chip and starts it,
and everything the controller does afterwards is that program running. Two
ways forward, and they are very different pieces of work:

1. **Run the sequencer.** Implement its instruction set and execute the 1696
   bytes the PROM supplies. Faithful, and it would work for any driver — the
   PROM, IRIX and NetBSD all download their own program. It is also an
   instruction set to write and debug.
2. **Emulate above it.** Ignore the downloaded program and implement what the
   driver observes: SCBs, the queue in and out FIFOs, and interrupts, doing
   the SCSI work ourselves. Much less code, and IRIS already has a SCSI target
   model to reuse. But it is a behavioural contract with each driver rather
   than with the hardware, so a driver that uses the chip differently — or a
   different sequencer program — can break it.

NetBSD's `ahc` driver shows what the *host side* expects to see, which is
exactly the contract option 2 needs.

## Emulating above the sequencer

2026-09-20. `src/aic7880.rs` (now `src/dev/aic7880.rs`). The decision was to implement what the driver
*observes* rather than the sequencer's instruction set, and the first half of
that works.

### What is standing up

The driver's bring-up now completes:

```text
chip reset
sequencer paused
sequencer RAM selected
sequencer RAM deselected, 1696 bytes held
sequencer RAM selected
sequencer RAM deselected, 1696 bytes held
sequencer running
queued SCB 1
```

Two things in there cost a bug each.

**`LOADRAM` does not erase the program.** It only points `SEQRAM` accesses at
the program store. The driver sets it a second time to read the download back
and check it, and clearing on the rising edge handed it zeros — the download
"succeeded" and then silently verified as empty. The give-away was a second
"download started" with a final length of zero.

**The SCB window follows `SCBPTR`, and the driver probes its depth.** It walks
`SCBPTR` from 0 to `0x7f` writing `0xff` to SCB offset `0x1f` — `SCB_NEXT`,
set to the end-of-list marker. A part with sixteen SCBs wraps, which is what
tells the driver how many it really has.

### Where it stops

After configuring the chip the driver queues SCB 1, then reads `INTSTAT`
**2318 times**, gets nothing, and resets the chip to try again. We post no
completion, so that part is expected.

What is *not* yet understood is how the command gets to us. The SCB the driver
queued is empty apart from the `0xff` the probe left in it, and nothing sets
up `HADDR`/`HCNT` for a DMA of one. Immediately before the `QINFIFO` write it
writes scratch registers `0x32` and `0x33` to zero, which in this sequencer are
plausibly queue positions — which would mean the real queue lives in host
memory and `QINFIFO` is not the submission path at all.

That is a guess, and guesses are what this project keeps getting caught by.
The next step is the one that worked for the identity chip: **read the
driver's own code**. It is at PC `0x81025870` in the loaded firmware, and
`IRIS_IP32_DIS` will disassemble it.

### The submission contract, read out of the driver

The guess in the previous section was wrong in an interesting way, and reading
the code settled it in twenty minutes. The queue write comes from
`0x8101d84c`, and just before it:

```asm
addiu a0, s5, 12
and   s0, a0, 0x1fffffff      ; virtual -> physical
or    s0, s0, 0x40000000      ; -> the address the device sees
jal   <flush 32 bytes>        ; so the device can see it
...
lbu   v0, 59(sp)              ; the SCB tag
lw    t6, 0(s2)               ; base of a host-memory array
sll   v0, v0, 2
sw    s0, 0(t7)               ; array[tag] = the SCB's bus address
jal   <write QINFIFO>         ; and hand the chip the tag
```

So the SCB is **not** written through the register window at all. It lives in
host memory and the sequencer fetches it. The register window is only used
during initialisation, which is why the SCB we saw queued was empty.

The chip is told where things are through its scratch RAM, and the values are
plainly visible once dumped:

```text
        20: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        30: 00 00 00 00 00 00 00 00 00 00 ff ff 00 78 3d 07
        40: 41 00 01 00 ad 0b 7f 7f 7f 7f 7f 7f 7f 7f 7f 7f
        50: 7f 7f 7f 7f 7f 7f 00 3e 07 41 03 00 ad 0b 00 00
```

Two little-endian addresses in RAM: `0x41073d78` at scratch `0x3d`, and
`0x41073e00` at scratch `0x56`. Dumping both says what they are:

```text
0x41073d78  00000000 410752fc 00000000 00000000   <- SCB bus addresses by tag
0x41073e00  ffffffff ffffffff ffffffff ffffffff   <- the queue-out FIFO
```

`array[1] = 0x410752fc`, exactly the tag the driver queued. The second is 256
bytes of `0xff`, which is the loop the driver runs just before queueing.

And the SCB itself:

```text
0x410752fc  01064010 410741e0 0107532c 00000000
0x4107531c  ...      4107531c 00000000
0x4107532c  12000000 40000000 ...                 <- the CDB
```

`12 00 00 00 40 00` is a SCSI **INQUIRY** with a 64-byte allocation length —
the first command anything sends to a new target.

So the contract, end to end:

1. Scratch `0x3d..0x40` holds the bus address of an array of SCB bus
   addresses, indexed by tag.
2. Scratch `0x56..0x59` holds the bus address of a 256-entry queue-out FIFO in
   host memory, pre-filled with `0xff`.
3. The driver builds a 32-byte SCB in host memory with pointers to its CDB and
   scatter-gather list, stores its bus address in `array[tag]`, flushes, and
   writes the tag to `QINFIFO`.
4. The sequencer is expected to fetch the SCB, run the command, put the tag
   into the host queue-out FIFO, and raise `CMDCMPLT`.

None of that is guessable from the chip's documentation, because none of it is
the chip — it is the program the driver downloaded. That is exactly the cost
option 2 was chosen with, and reading it out of the driver is the price.

### It works: the PROM reads a disk

Implemented, and the whole path runs. In order, and every step of it was
found by watching rather than assumed:

1. **Host memory.** The controller was given the RAM handle directly rather
   than a route back through the bus that owns it.
2. **The low-memory alias.** The driver uses both forms in one breath: the
   SCB's own address is `0x410752fc` while the CDB pointer *inside* it is
   `0x0107532c`, the same bytes. Resolving only one reads the SCB correctly
   and then finds an all-zero command, which is exactly what happened first.
3. **Byte order.** The SCB and its pointers are built with ordinary
   big-endian stores, so they are read that way; the chip's scratch RAM is
   filled a register at a time, least significant byte first, so it is read
   the other way. Both are settled by observation and they sit side by side.
4. **Status write-back.** With commands completing, the PROM still said
   `media not loaded`. The driver pre-fills SCB byte 2 with `0x40` — not a
   valid SCSI status — so it is plainly waiting for somebody to overwrite it.
   Writing the status there made the driver go straight on to `MODE SENSE`
   and then `READ(10)`.

The conversation now, against a 64 MB image with an SGI volume header:

```text
CDB 12 00 00 00 40 00   INQUIRY    -> status 0, 36 bytes into [0x01075250+64]
CDB 1b 00 00 00 01 00   START UNIT -> status 0
CDB 00 00 00 00 00 00   TEST UNIT READY -> status 0
CDB 1a 00 3f 00 fe 00   MODE SENSE -> status 0, 4 bytes into [0x01076160+254]
CDB 28 00 00 00 00 00 00 00 01 00   READ(10) LBA 0
                                   -> status 0, 512 bytes into [0x01076300+512]
```

**512 bytes of our disk image, read by the PROM into its own buffer.** Before
this the same command produced `dks0d1s0: volume header not valid` off a
zeroed image, which was the first proof the read path worked at all.

What the SCB layout is now known to be:

| offset | meaning |
|---|---|
| 0 | control |
| 1 | target/lun — varies per command (`0x06`, `0x0a`) |
| 2 | **SCSI status, written back**; pre-filled `0x40` |
| 3 | tag << 4 |
| 4..7 | bus address of the scatter-gather list |
| 8..11 | bus address of the CDB |

Scatter-gather entries are `(bus address, length)` pairs, the length masked to
24 bits. CDB length is taken from the opcode group rather than from a field,
which is how SCSI defines it and one less piece of layout to guess at.

### The target byte: it was not a target byte

Reported earlier as "byte 1 is target/lun, it varies per command". Both halves
of that were wrong, and two controlled experiments say so.

**It is the CDB length.** Correlating it against the opcode:

```text
byte1 0x06  ->  0x00 TEST UNIT READY, 0x12 INQUIRY,
                0x1a MODE SENSE, 0x1b START STOP UNIT     all six-byte CDBs
byte1 0x0a  ->  0x28 READ(10)                             a ten-byte CDB
```

Six and ten. The reason it was read as a target is that the aic7xxx register
named `SCB_TCL` sits at that offset — trusting a header name over observed
behaviour, for the fourth time in this project.

**And the target is not in the SCB at all.** Booting the same disk as
`dksc(0,1,0)` and as `dksc(0,3,0)` produces byte-identical SCBs, and identical
writes to the chip's whole SCSI block (`0x00..0x1f`):

```text
target 1:  05=00 10=00 11=00 00=01 00=00 0c=20 ... 01=12 02=27 11=a4 01=80
target 3:  05=00 10=00 11=00 00=01 00=00 0c=20 ... 01=12 02=27 11=a4 01=80
```

So where the target is conveyed is **still unknown**, and these runs cannot
answer it: the boot fails before any target-specific I/O happens, so every
command seen is from one bus scan that does not depend on the argument. The
experiment to run is one that gets far enough to address a specific disk —
which needs a volume header the PROM will accept.

Until then one disk answers for every address, which is why the PROM believes
it has found several.

### The write path

`WRITE(6)` and `WRITE(10)` go the other way through the same scatter-gather
list: `gather` collects the bytes out of the buffers the driver listed and
`execute_write` puts them on the image. A write past the end of the image is
refused rather than growing it — a disk that silently gets bigger is not a
disk — and that refusal is a test.

The CDB length now comes from the SCB rather than from the opcode group, with
the opcode as a fallback if the SCB says something impossible.

## NetBSD's bootstrap runs on the emulated O2

2026-09-20.

### The volume header is partition 8, not 0

`boot -f dksc(0,1,0)sash` was answering `media not loaded` because partition 0
is the *filesystem*, and the standalone directory an SGI PROM loads from lives
in **partition 8**. With the path corrected:

```text
> ls dksc(0,1,8)
dksc(0,1,8):
sash
> hinv
                 SCSI Disk: scsi(0)disk(1)
                 SCSI Disk: scsi(0)disk(2)
```

The PROM reads our volume header, lists its directory, and reports the disk in
its inventory. (It lists several because one image still answers for every
target — see above.)

### And it runs a NetBSD bootloader

`ip3xboot` out of NetBSD 11's `base.tgz` is the sgimips bootloader for this
machine. Put in the volume header with `mkvh` and loaded from partition 8:

```text
> boot -f dksc(0,1,8)boot
54688+1408 entry: 0x80002000

NetBSD/sgimips 11.0 Bootstrap, Revision 1.5 (Thu Jul 30 15:23:12 UTC 2026)

devopen: pci(0)scsi(0)disk(1)rdisk(0)partition(0) type scsi file boot
open pci(0)scsi(0)disk(1)rdisk(0)partition(0)boot: Input/output error
```

Third-party code, loaded off an emulated SCSI disk through an emulated PCI
controller, running on the emulated CPU and printing to the emulated UART. It
then asks the PROM to open `partition(0)` — the root filesystem, which the
image does not have yet.

### The 8 MB kernel does not load, and the size is the clue

Loading the IP3x kernel directly rather than through the bootloader gets as
far as an entry point and no further:

```text
> boot -f dksc(0,1,8)netbsd
8056208+128448 entry: 0x80069000
```

Those numbers are right — they are the ELF's `filesz` and its bss. But only
**five** SCSI reads happen in the whole session, all of a single block, and
memory at the entry point is zero afterwards: the kernel runs NOPs from
`0x80069000` upwards until it falls out of RAM. The loader reads the first
block of the file and one near the end, prints the summary, and jumps.

A 54 KB bootloader through the same path loads perfectly, so the loader works.
The PROM also says `not enough space` further down its own output. The
likeliest explanation is a limit on where or how much it will load, and the
next step is to find it rather than assume it.

## NetBSD 11 boots; the remaining gate is one loader bug

2026-09-20.

### The disk, built with the guest that already had the tools

`sgivol -i` on a 4 GB image lays down an SGI volume header with partition 0 as
BSD and partition 8 as the header itself; `sgivol -w boot /usr/mdec/ip3xboot`
installs NetBSD's own sgimips bootloader. `newfs`, then base/etc and the IP3x
kernel out of `nbsd11-sets.iso`. All of it done inside the working NetBSD 11
Indy guest, which already has `sgivol`, `newfs` and the sets — far less
error-prone than reimplementing FFS on the host.

### It boots

```text
NetBSD/sgimips 11.0 Bootstrap, Revision 1.5
NetBSD 11.0 (GENERIC32_IP3x) #0
total memory = 127 MB
mainbus0 (root): SGI-IP32 [SGI, 6], 1 processor
cpu0: MIPS R5000 CPU (0x2321) Rev. 2.1 with built-in FPU
com0: console
macekbc0 at mace0 offset 0x320000: PS2 controller
mcclock0 at mace0 offset 0x3a0000
mec0: Ethernet address 08:00:69:12:34:56
ahc0 at pci0 dev 1 function 0: Adaptec aic7880 Ultra SCSI adapter
ahc0: aic7880: Ultra Single Channel A, SCSI Id=7, 16/253 SCBs
scsibus0 at ahc0: 8 targets, 8 luns per target
```

The Ethernet address is the one in the DS2502, read this time by the kernel's
own driver rather than the PROM's.

Two fixes it needed:

- **The PCI enable bit.** MACE's config mechanism does not insist on it: the
  PROM sets it (`0x80000800`), NetBSD's driver does not (`0x00000800`).
  Requiring it made the whole bus invisible to the kernel, which attached
  `pci0` with nothing on it.
- **A timer interrupt.** IRIS normally arms Count==Compare as an hptimer
  one-shot on a thread this harness does not run, so no time passed inside the
  guest at all — NetBSD sat forever at "waiting 2 seconds for devices to
  settle" with its clock stopped at 1.0000030. The harness now polls
  Count against Compare every [`TIMER_POLL_STEPS`] and raises IP7.

### NetBSD's driver, decoded from its source, and root mounted

`sys/dev/ic/aic7xxx_inline.h` settles it in one function. `ahc_queue_scb` does
**not** write a tag to `QINFIFO`:

```c
ahc->qinfifo[ahc->qinfifonext++] = scb->hscb->tag;
...
ahc_outb(ahc, KERNEL_QINPOS, ahc->qinfifonext);
```

The tag goes into a 256-entry ring in host memory and the host writes only the
new **producer index** to a scratch register. The sequencer consumes from the
ring and fetches the SCB from a flat array indexed by tag. Completion comes
back through a second ring beside the first. The addresses live in scratch:

| scratch | meaning |
|---|---|
| `0x44` | `HSCB_ADDR` — base of the SCB array, stride 64 |
| `0x48` | `SHARED_DATA_ADDR` — `qoutfifo`, with `qinfifo` at +256 |
| `0x4c` | `KERNEL_QINPOS` — the producer index, and the trigger |

and the SCB itself is CDB inline at 0, status at 8, `dataptr` 12, `datacnt`
16, `sgptr` 20, `control` 24, `scsiid` 25, `lun` 26, `cdb_len` 28. The first
scatter-gather segment is carried in the SCB; `sgptr` points at the rest.

Three things then had to be right, and each was wrong first:

- **Interrupts.** NetBSD attaches CRIME as `platform.intr0`, so a device
  interrupt arrives as IP2, and the controller sits on CRIME input 8 — the
  kernel works that out and prints it. Without the path the driver only
  noticed a finished command when its watchdog fired, and said so:
  "Interrupts may not be functioning."
- **Endianness.** The chip is little-endian and NetBSD's driver writes its SCB
  fields with `ahc_htole32`, while the PROM's loader handed its own program
  big-endian words. Reading NetBSD's pointers the wrong way round turns
  `0x0005fe58` into `0x58fe0500` — an address outside memory, which lands
  nowhere and reports success. The symptom was an INQUIRY that returned 36
  bytes of nothing, and a driver attaching a disk whose vendor string was
  binary rubbish.
- **The target nibble.** `SCSIID` is target in the high nibble and our own id
  in the low, which the driver's own card dump states plainly:
  `SCB_SCSIID[0x17]` is target 1 on an adapter at id 7. Reading the wrong
  nibble made one disk answer for all sixty-four addresses, and then, once
  filtered, for none.

With those:

```text
sd0 at scsibus0 target 1 lun 0: <SGI, IRIS EMULATED, 1.0> disk fixed
sd0: 4096 MB, 4096 cyl, 64 head, 32 sec, 512 bytes/sect x 8388608 sectors
boot device: sd0
root on sd0a dumps on sd0b
root file system type: ffs
Enter pathname of shell or RETURN for /bin/sh:
```

NetBSD 11 mounts its root off the emulated controller and starts init.

### The earlier account, kept because the wrong guess is instructive

With the scatter-gather fix in, NetBSD gets further but its `ahc` still times
out. Two things are now known and one guess was wrong.

A first attempt read the register-window writes (`0xa0`..`0xbf`, sixteen each)
as "this driver puts SCBs on the chip". It does not: those writes are the
driver **clearing** all sixteen SCBs during initialisation, and the SCBs are
empty when a tag is queued. Treating a window write as evidence made the
device complete commands from blank SCBs, which is worse than failing. The
discriminator now only reports a window-filled SCB that actually carries
something, and otherwise uses the host-memory path both drivers share.

What is actually wrong is narrower: NetBSD sets the scratch SCB-array pointer,
and we read an SCB from it, but always the *same* stale address
(`0x4107681c`, left over from the PROM) with `control 0x05`. So either the
pointer it writes is not the one at `sram::SCB_ARRAY`, or the indexing differs.
The driver then reports `Infinite interrupt loop, INTSTAT = 0`, which is its
watchdog rather than a clue about the chip.

577 commands and 9.1 MB moved in a session, so the path works for the PROM and
the bootloader; this is specifically NetBSD's convention, and it needs the
same treatment the PROM's got — read it out of the driver.

### Two drivers, two submission paths

With time running, NetBSD probes the bus and times out, and the register
histogram says why: it writes each SCB **through the register window**
(`0xa0`..`0xbf`, sixteen times each) and then hands over a tag. The PROM's
sequencer program fetches SCBs from host memory instead and is handed only a
tag. Same chip, same registers, different contract — because the contract
belongs to the downloaded program, which is exactly the trade option 2 was
chosen with.

The device now records which path a driver used rather than assuming, and the
on-chip layout is captured for decoding.

### The loader bug was ours: a scatter-gather list cut short

**Found and fixed.** `SG_MAX` was 32 — a plausible-looking bound with nothing
behind it. A single 264-block read scatters into **33** pages, so the walk
stopped one entry short, dropped the tail, and reported success.

The symptom was a hole. Mapping every scatter target of the `sashARCS` load:

```text
0x07f85350 +  3248 -> 0x07f86000
0x07f86000 +  4096 -> 0x07f87000        (31 more pages...)
0x07fa4000 +  4096 -> 0x07fa5000
0x07fa6550 +  2736 -> 0x07fa7000   <-- GAP 5456 bytes
```

Exactly 32 entries, then a gap — and `sashARCS`'s entry point,
`0x87fa5fa0`, sits inside it. The PROM read the whole file (438 blocks, every
one of them), printed the right sizes and the right entry address, jumped, and
executed whatever had been in that memory beforehand.

Two instruments found it, and neither existed a day ago: session totals that
survive a chip reset (219322 bytes read, against ~161 KB of loadable
sections — so the file was arriving), and a DMA watch on the entry point
(zero writes — so it was not arriving *there*). "The data is read but not
written where it should be" is a much smaller problem than "the PROM cannot
load files".

With the bound raised:

```text
> boot -f dksc(0,1,8)sashARCS
135632+22592+3216+341792+49040d+4528+6784 entry: 0x87fa5fa0
Standalone Shell SGI Version 6.5 ARCS   Jan 20, 2000 (32 Bit)
sash:
```

IRIX's standalone shell, from SGI's own install media, running on the
emulated O2.

The lesson is the ordinary one: a bound invented for safety became a silent
data-loss bug because the operation still reported success. The bound is now
large enough to be unreachable in practice and exists only so a corrupt list
cannot spin forever — the real terminators are running out of data or hitting
a null entry, and both come first in any healthy list.

### The old account of the loader bug

A 54 KB bootloader loads perfectly. A 340 KB `sashARCS` and an 8 MB kernel do
not: the PROM prints correct sizes and an entry point, jumps, and executes
whatever was already in memory. For `sashARCS` the entry at `0x87fa5fa0`
holds a sparse repeating structure, not code.

The reads themselves look healthy — multi-block, with proper scatter-gather:

```text
CDB 28 00 00 00 92 15 00 00 5f 00     95 blocks
  -> status 0, 48640 bytes into [0x01076970+1680 0x01077000+4096 ... 0x01082000+1904]
```

but only ten of them for a file of 665 blocks, and none covering the entry
point's neighbourhood. So the data that *is* transferred lands correctly and
most of the file is never asked for. That is the next thing to find, and it
gates IRIX as well as booting a kernel without the bootloader.

Worth noting: NetBSD's bootloader loads the 8 MB kernel through the same PROM
without trouble, so the fault is in the PROM's own loader path rather than in
the disk, the controller or the scatter-gather.

### What is left

- The PROM reads the volume header but has not yet been given a bootable one.
  `sash` for IP32 has to come from IRIX media.
- Writes are not implemented; only the read path is exercised.
- `tcl` is not decoded, so every command is answered by the one disk
  regardless of which target it names. That is why the PROM finds a disk at
  more than one address.

### What implementing it needs

- The device needs to reach host memory, which means handing `Aic7880` a way
  to read and write the bus rather than only answering register accesses.
- A SCSI target to run the CDB against, and a disk image behind it. IRIS
  already models SCSI targets for the Indy's WD33C93A; that is the part worth
  reusing rather than rewriting.
- The SCB's field layout beyond the two pointers — data pointer, transfer
  length, scatter-gather count, status — which the same technique will settle:
  the driver reads status back out of the SCB after completion, so the fields
  it reads name themselves.

### The honest state of the trade

Option 2 was chosen knowing it is a contract with what a driver observes
rather than with silicon. This is what that costs: the submission path has to
be read out of each driver instead of looked up. The upside is unchanged —
there is no sequencer instruction set to write — and the parts that are
generic to the chip rather than to the program, which is most of what is above,
did not need the driver at all.

### Diagnostics need a disk

`3) Run Diagnostics` answers `No SystemPartition set`. The IDE suite is a
standalone program loaded from the system disk's volume header, not something
held in the PROM, so running it needs block storage on MACE's PCI bus. That is
the obvious next milestone and a substantial one.

### Still cosmetic

- `Cannot connect to keyboard` — correct; nothing is attached, and the menu
  comes out on serial regardless. Pointing the PROM's console at serial in
  NVRAM would silence it.
- `time invalid, resetting clock to epoch` — the RTC answers, but not with
  anything the firmware accepts as a valid time.

## NetBSD 11 boots to multi-user on the emulated O2

2026-09-20.

```text
NetBSD/sgimips (o2-nbsd) (constty)

login: root
o2-nbsd# uname -a; sysctl hw.model; df -h /
NetBSD o2-nbsd 11.0 NetBSD 11.0 (GENERIC32_IP3x) #0: Thu Jul 30 15:23:12 UTC 2026
    mkrepro@mkrepro.NetBSD.org:/usr/src/sys/arch/sgimips/compile/GENERIC32_IP3x sgimips
hw.model = SGI-IP32
Filesystem     Size   Used  Avail %Cap Mounted on
/dev/sd0a      3.9G   231M   3.5G   7% /
```

`/etc/rc` runs, generates host keys, and reaches a login prompt; root logs in
on the serial console.

### Reproducing it

```bash
IRIS_IP32_SCSI=1 IRIS_IP32_FAST=1 \
IRIS_IP32_DISK=/path/to/nbsd-o2.raw \
IRIS_IP32_INPUT='5\rboot -f dksc(0,1,8)boot\r' \
IRIS_IP32_SCRIPT='Enter pathname=>\r;;# =>exit\r;;CLEAN? [yn]=>y\r;;login:=>root\r' \
IRIS_IP32_CONSOLE=/tmp/o2.log IRIS_IP32_STEPS=40000000000 \
cargo test --release --lib ip32::bringup -- --nocapture
```

The `CLEAN? [yn]` step is there because killing a run leaves the filesystem
dirty and `/etc/rc` stops to ask. A guest shut down properly does not need it.

## Where IRIX stands, and where it stops

The installer works end to end: `2) Install System Software` finds the
emulated CD-ROM, copies the miniroot to the target disk's swap partition, and
boots it.

```text
Copying installation program to disk.
......... 100%
Copy complete
IRIX Release 6.5 IP32 Version 01200452 System V
```

And then it stops, after the banner, every time.

What is ruled out, from the harness rather than from reasoning:

- **Not a fault.** The exception log shows two entries all session, both at
  PROM startup. Nothing after the banner.
- **Not a missing device region.** The run asserts on unmapped accesses and
  does not trip.
- **Not the console.** `console=d` and `nogfxkbd=1` set and persisted in
  flash, with the PROM autoboot disabled, changes nothing.
- **Not the second controller.** Adding the O2's second AIC-7880 at PCI dev 2
  (which is right regardless — CRIME has a separate interrupt line for each)
  changes nothing.

So it is a tight loop on a register that is mapped and never changes. The
`busiest MACE offsets` report names it; that run had not finished when this
work was paused.

One flaw found while looking, and fixed: the console interrupt hack
*overwrote* the whole of `MACE_ISA_INT_STATUS` on every poll rather than
setting and clearing its own bits. Anything a guest wrote there -- another
device's pending bit, or an acknowledgement in progress -- was erased under
it. That is exactly the shape of failure that leaves a kernel spinning, and it
should be re-tested first when this resumes.

### Setting up an IRIX install

The target disk needs an SGI label before the PROM will install to it, because
`fx` only exists inside the miniroot that needs the swap partition to boot:

```bash
mkvh build target.raw --size 4G \
  --part 8:0:0:4096 --part 1:3:4096:262144 \
  --part 0:10:266240:8122368 --part 10:6:0:8388608
# then set vh_swappt to 1 (offset 6, big-endian) and fix the header checksum
```

`vh_swappt` matters: the PROM copies the miniroot to whichever partition it
names, and mkvh leaves it at 0, which produces
"swap partition (0) is not a valid swap area".

### What is still wrong

- `mec0: device timeout`, repeatedly. Ethernet is not emulated; the driver
  attaches because the MAC address is readable and then finds no hardware.
- `/etc/rc.d/mountall` reports a failure — the non-root entries in `fstab`.
- `WARNING: preposterous TOD clock time`. The RTC answers but not with a time
  anything believes, so the clock comes from the filesystem.
- One disk still answers at one address only because it is told to; the PROM's
  SCBs carry no target, so that path answers for anything.

## A shell

```text
Enter pathname of shell or RETURN for /bin/sh:
# df -k
Filesystem      1K-blocks         Used        Avail %Cap Mounted on
root_device       4064590       236756      3624606   7% /
#
```

NetBSD 11 running interactively on the emulated O2, reading its own
filesystem off the emulated controller.

### The console needed interrupts, and a reason to keep talking

A PROM polls; a kernel does not. Two things were missing and each had a
distinct symptom.

**Receive.** Typing at a prompt did nothing: the character sat in the receive
register and nobody looked. `com0` hangs off CRIME input 4, which is the ISA
cascade — NetBSD's handler reads `MACE_ISA_INT_STATUS` and matches each
device's mask against it, so raising CRIME alone is not enough; the
demultiplexer has to find something. The masks are the ones the kernel prints
for itself: "com0 at mace0 offset 0x390000 intr 4 intrmask 0x3f00000".

**Transmit.** The prompt arrived as `Enter pathname o` and stopped — in the
middle of a word. The driver fills the transmit FIFO and waits to be told it
has drained, and our `IIR` always said "nothing pending". A UART that is
always ready but never says so leaves a kernel waiting forever. Implementing
`IIR` properly — receive-data above transmit-empty above none — finished the
sentence.

The `WARNING: preposterous TOD clock time` is still outstanding: the RTC
answers, but not with a time anything believes.

### An accidental half-hour

Worth recording because the instrument caused it. The script matcher called
`com0.output()` — which copies the whole console buffer — on **every
instruction**. Boots that had taken minutes stopped finishing at all; the
first run was killed after half an hour still in the device probe, and it
looked like a guest bug. Checking `bytes_out()` first and only scanning when
the guest has actually said something took the same boot to a shell in under
four minutes.

## Driving the machine: `IRIS_IP32_SCRIPT`

Booting to multi-user means answering three different things in turn — the
PROM's menu, then the bootloader, then init — and each prompt only exists once
the previous answer has been given. Feeding everything at once does not work:
the bytes queue in the UART and the later consumer never sees them, because
the earlier one has already drained the line.

`IRIS_IP32_SCRIPT` takes `wait-for=>type-this` steps separated by `;;`. Each
step arms only after the one before it has fired, and only matches output
produced *since* then, so a prompt that appears twice does not consume two
steps.

```bash
IRIS_IP32_SCRIPT='Enter pathname=>\r;;#=>exit\r'
```

## Instrumentation worth keeping

All of this came out of the harness, and none of it out of reading the PROM
top to bottom. In rough order of how much time each saved:

- **A call tracer.** A call is recognised exactly — an instruction that leaves
  `ra` equal to its own address + 8, which `jal`/`jalr`/`bal` do and an
  `lw ra, n(sp)` epilogue restore does not. Returns pop the matching frame and
  record `v0`. Run-length encoded, and filtered to shallow or slow calls, it
  prints the entire shape of the boot in forty lines, with return values.
- **A memory watchpoint** (`IRIS_IP32_WATCHMEM=lo[:hi]`), logging address,
  direction, value, PC and UST time. It is what turned "the cookie is wrong"
  into "this exact instruction clobbered it". Note the watch sits *after*
  address resolution, so low-memory addresses must be watched at their
  `RAM_BASE` equivalents.
- **Disassembly of spin-loop bodies**, with the registers the loop names. A
  five-instruction loop plus `v0=0xbf340000 v1=0x2c8d` says "waiting for the
  UST to reach 11405" immediately.
- **Loop entry counts.** Time in a loop says it is hot; the number of *entries*
  says whether it is progress or a wedge. Entered 5 times, 99% of the run:
  those are real delays. Entered once: that is the hang.
- **`IRIS_IP32_DIS=start:end`**, reading through the bus so RAM-resident code
  is visible.
- **Report everything, then assert.** The asserts now run at the very end of
  the test. A panic in the middle used to throw away the evidence for itself.

## Two lessons, both already on the list

- **Trust observed access patterns over header names.** `MACE_ISA_FLASH_NIC_REG`
  is named for the 1-Wire chip, and it *is* that — but the writes that had been
  puzzling us were `0x20`/`0x30` a second apart, which is the LED. Same
  register, different bits. The third time this exact mistake has cost time.
- **A wrong emulator can look like a missing device.** Three sessions of this
  were spent looking for IP32 hardware we had not implemented. One of the three
  gates was hardware; one was a UART feature; one was a CPU bug that had been
  in the tree the whole time, affecting every guest.

## Open questions

- ~~How much does `post1` insist on?~~ Answered: CRIME is cheap, MACE PCI is
  the gate. See above.
- ~~What does the PROM expect to find on PCI?~~ Answered: an empty bus is fine.
- What do CRIME's bank-control bits mean? POST writes `0x100` to all eight,
  then `0x104` to the ones behind bank 0. Bit 2 is set once a bank is sized.
- Where does the PCI native view really live, given memory owns 0x40000000?
- ~~Implement `com0` to read POST's messages.~~ **Done, and it does not help
  yet — see below.**
- ~~1-Wire is still unimplemented.~~ Modelled now, and the firmware has
  reached it: `ds2502_init: presence pulse not detected`. The reset/presence
  handshake is the open part.
- ~~`UST_STRIDE` is a bring-up shortcut.~~ Retired: the UST is now a real
  clock, advanced from the instruction count at `UST_TICKS_NUM/UST_TICKS_DEN`
  (one tick ≈ 1 µs at ~100 instructions/µs). It had to become time-based —
  post1 polls it for deadlines, one of them a full second, and 1-Wire is
  decoded by pulse width.
- Does the PROM require a framebuffer to be present even with `console=d`? The
  Indy PROM does not; the O2's `crt_option=1` in the default env suggests it at
  least looks.
- CPU: O2 shipped R5000/RM5200/RM7000/R10000/R12000. We have R5000, which is
  the cheapest target. RM7000 would need extra CP0 state, and the decompiled
  PROM branches on `CP0_PRID`, so it will notice.

## Carried onto upstream main (2026-10-17)

The bring-up was done on a branch from 2026-09-21. On 2026-10-17 its 33 commits
were replayed onto upstream main (b615195), limited to the IP32 files -- the
three that also changed the CPU side say so; those changes (Index_Store_Tag
without a writeback, a JTLB as big as the CPU model) are upstream in their own
form -- and one commit moved it onto upstream's layout:

- `src/aic7880.rs` is now `src/dev/aic7880.rs`, with the other devices.
- Imports name `crate::cpu::...`, where upstream keeps the CPU now.

Nothing else had to change: the executor, TLB, cache and `BusDevice` the
bring-up drives are the same interfaces. On upstream main, `cargo test --lib
ip32` passes 21 (the PROM traced from reset among them) and `aic7880` 10, and
NetBSD 11 boots to multi-user and a root login with the command in "Reproducing
it" above, plus `IRIS_IP32_PROM=<path to ip32prom.rev4.18.bin>` when the PROM
is not at `ip32/` beside the working directory.

The machine it builds: an R5000 (rev 2.1, 48 TLB entries, 32 KB/32 KB L1),
128 MB at `0x40000000`, the real PROM, CRIME and MACE, com0 as the console, and
an AIC-7880 with the disk at target 1 (a second, empty, as on a real O2), and
the Ethernet (below). Not there yet: audio, graphics output, and a real-time
clock time. It is a test harness, not a machine profile: the console
goes to a file (`IRIS_IP32_CONSOLE`), input can be scripted, and
`IRIS_IP32_SERIAL=port` puts com0 on a live TCP console as well -- `telnet
127.0.0.1 port`, the same telnet character mode as the other machines' serial
ports. With `IRIS_IP32_STEPS` large the machine runs until stopped: log in, use
it, `shutdown -p now`, then stop the test.

## Ethernet: MACE's MAC-110 (2026-10-17)

`src/dev/mec.rs`, at MACE + 0x280000 on CRIME interrupt 3, implements what
NetBSD's `mec` driver relies on: 64-bit registers; a 64-entry ring of 128-byte
transmit descriptors (inline data ending at byte 127, then up to three pointers,
each descriptor's word 0 rewritten as its status, the read pointer in
INT_STATUS bits 16-24, TX_EMPTY latched when the ring runs dry); a 16-deep FIFO
of 4 KB receive buffers (two bytes of padding, the frame at offset 34 with its
CRC, the status word written last, RX_THRESHOLD); write-one-to-clear
interrupts; and a QS6612 PHY at MII address 8 that has a 100 Mbit full-duplex
link, as on a real O2 -- NetBSD attaches `qsphy0 at mec0 phy 8`.

`IRIS_IP32_NET=1` puts it behind IRIS's NAT gateway (the Indy's: DHCP, DNS,
port forwards), `IRIS_IP32_FWD=tcp:2263:22,...` forwards host ports on
127.0.0.1. NetBSD 11 on the O2 then takes 192.168.0.2 by DHCP, pings the gateway
and the internet, resolves names, and runs sshd: `ssh -p 2263 root@127.0.0.1`
logs in, and 4 MB copied each way with scp came back identical (about 200 KB/s,
the emulated R5000 doing the encryption), with no interface errors and no
watchdog timeouts.

Related: [`virtio-mmio-design.md`](virtio-mmio-design.md).

## Memory as CRIME banks, and the R10000 (2026-10-17)

Memory used to be one flat block from `0x40000000` that ignored CRIME's bank
controls. It is now `src/dev/crime_mem.rs`: eight banks of 32 or 128 MB,
placed in the 1 GB window by their control registers (`0x208 + 8*i`, base in
32 MB units in bits 0-4, 128 MB in bit 8), and decoded the way POST relies on:

- several controls matching one address: the lowest-numbered bank answers;
- a matching bank with nothing fitted: no error, the read returns what was
  last on the memory bus (zero, since POST clears it before reading);
- a 32 MB bank set to 128 MB: its 32 MB repeat across the range, which is how
  POST tells the sizes apart;
- nothing matching: a bus error.

The low alias at physical 0 is now the first 256 MB of the window, not "as
much as there is RAM": with 1 GB fitted the old rule would have covered CRIME
and MACE.

What is fitted follows SGI's memory installation flier (007-3778-002): four
DIMM pairs (A-D), kits of 32 MB (2x16 MB single-sided, one CRIME bank),
64 MB (2x32 double-sided, two 32 MB banks), 128 MB (2x64 single-sided, one
128 MB bank) and 256 MB (2x128 double-sided, two 128 MB banks); pair A always
fitted, pairs filled in order, no pair larger than the one before it. That is
69 fittings from 32 MB to 1 GB; a unit test runs POST's own sizing procedure
against every one and checks it finds what is fitted, bank by bank.

    IRIS_IP32_DIMMS=256,128,32   the kit in each pair from A, in MB
    IRIS_IP32_RAM_MB=1024        a total, fitted with the fewest kits
                                 (neither: 128 MB). Impossible fittings are
                                 refused with SGI's rule.

**NetBSD sees at most 256 MB.** With 1 GB fitted, POST sizes all eight banks
(the PROM's `hinv` says `Memory size: 1024 Mbytes`) but the PROM's ARCS memory descriptors stop at `0x10000000`, and
NetBSD/sgimips builds its memory map from those alone (32- or 64-bit kernel).
Using the rest takes a kernel that reads the bank controls itself (IRIX does;
OpenBSD is the next thing to try).

### R10000

`IRIS_IP32_CPU=r10000` (or `r10k`) runs the O2 on `R10000O2Cache`, the IP28
model with one difference: **64-byte secondary-cache lines** (Config SB = 0).
With the IP28's 128-byte lines the PROM panicked before printing anything:
`Write TLB miss exception, bad address: 0x0` in `bzero`, called by the SCSI
probe with a NULL buffer. The chain, from breakpoints on the allocator:

- the DMA-buffer allocator aligns to a global the cache setup computes; it read
  `0x1f` on the R5000 and `0xfffff` on the R10000;
- the R10000 path of that setup (firmware `0x01006330`) reads SS and SB from
  Config; for SB = 0 it stores a 64-byte line, but for SB = 1 its branch skips
  the assignment and stores the secondary cache's *size* as the line size;
- 1 MB alignment exhausts the PROM heap, the allocator returns NULL, and the
  first SCSI command clears address 0.

The SB = 1 path cannot work, so the inference is that O2 R10000 modules all
have 64-byte secondary lines and the PROM never met anything else. With that, PROM 4.18 boots NetBSD 11 on the R10000:
`cpu0: MIPS R10000 CPU (0x925) Rev. 2.5`, `1024KB/64B 2-way ... L2`.

`IRIS_IP32_BREAK=pc,...` (new) prints a0-a3, v0, v1, sp and ra the first eight
times each address is reached; `IRIS_IP32_DIS` now takes several ranges.
`machines/o2-netbsd/console.sh` in the workspace defaults to R10000 and 1 GB
(`CPU=`, `RAM=`, `DIMMS=` to change).

## OpenBSD and NetBSD on jitv2 (2026-10-18)

tcache is a run-time switch on the full cache model (`set_tcache_live`): the
PROM keeps its stack in the data cache while it tests memory, so the cache
holds its data until POST has handed over to the firmware, and again whenever
the guest is back inside the PROM (ARCS PowerDown sets the caches up the same
way). Switching it on writes every dirty line back first. O2 memory keeps a
generation counter per 4 KB page, bumped by every write through
`O2Memory::write` (CPU stores, SCSI and Ethernet DMA, the rendering engine),
which is what jitv2 checks compiled code against.

OpenBSD 6.9 boots to ssh in about 150 s on jitv2 (fork stress 95 s, against
191 s interpreted); NetBSD 11's heavier work runs three to five times faster.

## Graphics: GBE, the rendering engine and the keyboard (2026-10-18)

`src/dev/o2gfx.rs` is a plain CPU model, meant as the oracle for a later GPU
version:

- `Gbe`: a 32-bit register file; a beam counter driven by guest time (plus a
  little per read, so a blanking poll always sees it move); the DMA control
  shadows copied straight to their live registers; chip id 1 with a non-zero
  board revision strap (revision 0 makes the PROM load a bit-reversed gamma
  ramp); an idle colour-map FIFO; and `render()`, a pure function of RAM and the
  registers: tile list (16-bit entries, 64 KB tiles of 128 lines x 512 bytes,
  partial right-hand tiles), 8/16/32 bpp, WID types, CMAP, gamma, cursor.
  `vt_crossings()` raises the four vertical-timing interrupts.
- `CrimeRe`: TLB A/B/C tile lookup, points, lines, rectangles (fill,
  opaque/transparent stipple, pixel-transfer copy through a temporary), logic
  ops, plane mask, byte lanes, and the memory transfer engine's clears and
  copies. Every primitive runs when started; status always reads idle.
- MACE PS/2: two real ports driving `dev/ps2.rs`'s keyboard (set 3 for the
  PROM and IRIX, set 2 for OpenBSD, set 1 for NetBSD); interrupts on ISA bits
  9..12, CRIME line 5. MACE reads as revision 2 (bit 4 of 0x1f310000).
- CRIME's ID has to read 0xa1 before the PROM offers a graphics console, and
  it needs a keyboard that answers its probe (0xF5).

With those the PROM draws its graphical console, and OpenBSD's gbe(4) and
NetBSD's crmfb(4) run glass consoles. NetBSD on IP32 prints through the PROM
until crmfb attaches, with the FPU disabled, and the PROM scrolls with
ldc1/sdc1: once its text window fills, the kernel faults inside the firmware.
`[o2] arcs_fpu` / `IRIS_IP32_ARCS_FPU=1` is a labelled test aid that lends it
the FPU meanwhile.

CRIME's interrupts follow the CRIME 1.5 spec: INTSTAT is the level sources
plus the latched edges (30, 24:22, 19:16) plus SOFTINT; writing 0 to a latched
bit in HARDINT clears it.

## The O2 as an iris machine (2026-10-18)

`[machine] profile = "o2_ip32"` builds an O2 from the `[o2]` section (prom,
flash, ram_mb or dimms, disks, serial_port/serial_log, graphics, net,
forwards, arcs_fpu) instead of the Indy/Indigo2 board: `src/ip32_machine.rs`.
The CPU is iris's own thread (with jitv2's compile pool); a batch hook on the
executor switches tcache as above; CRIME's output reaches IP2 through an
interrupt sink the bus calls before CRIME reads and after any non-RAM write
(a 100 us poll alone left acknowledged interrupts up long enough for OpenBSD
to report them as spurious); the MACE clock runs from the host clock, read
live (the 1-Wire ID chip's slots are microseconds). `O2Display` scans GBE out
at 60 Hz for the standard window, whose keys go to the PS/2 keyboard. The
PROM's flash is saved whenever the guest has written to it.

The bring-up harness stays for diagnostics (trap logs, hot PCs and registers,
breakpoints), and `ip32-view` gives it a window through a shared file.

## IRIX 6.5.22 (2026-10-18, in progress)

From the Overlay 1 CD the PROM loads sashARCS, the miniroot copies to the
target disk and its IP32 kernel runs. With GBE's timing interrupts and CRIME's
latching in place it gets past driver initialisation to the SCSI probe, where
IRIX's own AIC-7880 driver (adp78, with Adaptec's HIM) times out on every
command: it uses a third host/sequencer contract the model does not know yet.
With more than 256 MB the kernel faults earlier, walking a corrupt page frame
list (both CPUs); not yet understood.

## Clip IDs (2026-10-19, not modelled)

CRIME can test each pixel against a clip-ID buffer (ClipMode bit 11, enCid;
bit 10 cidMapSel). The spec names the enable bits, buffer type 6 ("tiled
buffer - clipID TLB") for BufMode and the MTE, and `TLB.cid[4]` at RE 0x6e0,
and says nothing else: no register holds the ID a pixel is compared with.

What IRIX 6.5.22 does with it, from RE traces (desktop, demograph, perfly,
gr_osview, gmemusage):

- Maps 8 clip-ID tiles once at boot: `TLB.cid` = 0x106, 0x109, 0x10c, 0x10f,
  0x112, 0x115, 0x118, 0x11b (every third 64 KB tile), the rest invalid.
- Clears it once with the MTE: mode 0x18 (dst 6), FG 0, byte mask all ones,
  DST0 0x01000000, DST1 0x01ff03ff: 256 bytes x 1024 rows, which does not
  fit the framebuffer's 16-tiles-a-row layout.
- Writes ClipMode 0x821 (clip ID + screen mask 0) a few times, but no
  primitive has run with bit 11 set in any trace so far.

So the clear is skipped (the MTE notes "mte buffers src 0 dst 6") and a
primitive drawn with the test on notes "clip-ID test (drawn unclipped)".
When that note appears, capture the trace: it will show what IRIX writes into
the buffer and how the ID is chosen.

## DrawMode bits 8 and 9 (2026-10-19)

The spec's DrawMode table has enLogicOp at bit 8 and enDither at bit 9.
IRIX uses them the other way round: the X server sets bit 9 with LogicOp 3
(COPY) on every primitive, and GL sets bit 8 (dithering is on by default in
OpenGL) with a LogicOp of 0, which as a logic op would be CLEAR and draw
every GL primitive black. `draw::LOGIC_OP` is bit 9 and `draw::DITHER` bit 8.

Dithering matters in the double-pix windows a double-buffered GL visual
gets (BufMode 0x226/0x227: A1R5G5B5 halves): Quake 2 lightmaps every wall
by multiplying the frame buffer (blend 0x2), and at 5 bits a channel the
result fell into flat bands. Triangles into those windows now dither with
a 4 x 4 ordered (Bayer) matrix (`CrimeRe::dither5`); CRIME's own matrix is
not in the spec.

## Blending and the alpha test (2026-10-19)

Blend.func (RE 0x1a8): op 11:8, source factor 7:4, destination factor 3:0,
in the spec's (OpenGL's) order; Blend.constColor (0x1a0) RGBA, R high;
AlphaTest (0x198): function 11:8, reference 7:0. `CrimeRe::blend` follows
the register for triangles and GL lines; the alpha test runs before the
depth test, as in GL. Everything IRIX 6.5.22 drew in the traces so far uses
0x45 (ADD, SRC_ALPHA, ONE_MINUS_SRC_ALPHA), on GL lines (antialiased icon
outlines) and triangles at alpha 255.

## Textures: 32-bit texels and mipmaps (2026-10-19)

The spec leaves texel look-up "TBD"; both were solved with o2gltest
(scratch/o2-bench), whose textures describe themselves: an RGBA8 map with
R = 4x, G = 4y, and a mip pyramid with each level a solid colour.

- **Layout.** One subtile layout for 16- and 32-bit texels, in byte terms
  (`subtile_offset`): byte column (x times texel size) bits 2:0, y 1:0,
  byte column 6:3, y 5:4, y 3:2, byte column 8:7, y 6: one 64 KB tile is
  512 bytes by 128 rows. A larger level's tiles form a grid of TLB
  entries, the next 512-byte column one entry on and the next 128 rows
  four on (`subtile_step`; Quake 2's 256 x 128 RGBA8 skins in entries 80
  and 81, 256 x 256 RGBA8 in 80, 81, 84, 85). Before that rule the right
  half of every 256-wide 32-bit skin read black. 32-bit texels are RGBA8
  with R high (spec figure 7-7).
- **Mip levels** sit at fixed places for their size, not after their
  texture (`mip_level_addr`), counted from 1024 (Texture.mode 17:14 is the
  base level, 4 for 64 x 64; 13:10 the last level below it): 128 at TLB
  entry 88, 64 at entry 92, 32 at 93 + 0x8000, 16 at 92 + 0xc000, 8, 4, 2
  and 1 at 93 + 0xe000, 0xf000, 0xf800, 0xfc00. Larger sizes are guesses.
- **Level of detail** per pixel from the texel steps to the next pixel in x
  and y (spec 7.3.7.5); the magnification filter at LOD <= 0, otherwise the
  minification filter (Texture.mode 9:7), nearest or blended levels as in
  OpenGL.
- **Texture function** (1:0): MODULATE, DECAL and REPLACE as in OpenGL;
  BLEND is drawn as MODULATE until its environment colour is found.

Polygon stipple needs nothing from the engine: IRIX's GL stipples polygons
itself (o2gltest's stippled triangle arrives as runs of lines).

## The demo sweep, and what it found (2026-10-20)

Every program in IRIX's Demos icon catalog, the O2 demos (`/usr/demos/O2`),
GLUT, perfly and the Quakes were run one at a time on the test clone,
with screenshots, window lists, the emulator's notes and the audio dumped
for each (`scratch/o2-sweep`, `sweep.py`). Most demos draw correctly. What
was wrong, and what it turned out to be:

- **16-bit texel types.** `Texture.mode` 23:20 also names RGBA4 (4),
  ALPHA (5), INTENSITY (6), LUMINANCE (7) and LUMINANCE_ALPHA (8), all
  16 bits wide here. RGBA4 has R in the top nibble and LA8 has L in the
  high byte (spec figure 7-7). The spec draws no 16-bit A, I or L texel.
  They are taken to sit where LA8 keeps its channels: L and I high, A low.
  bzflag's luminance font reads cleanly that way. GLQuake's world and
  status bar (RGBA4), buttonfly's labels (LA8), and bzflag's and Macau's
  text (L) had drawn untextured.
- **Texture coordinates run from 1.0 to 2.0.** IRIX's GL hands the engine
  s and t offset by 1.0 (16384 in 1.14). A repeating map hides that. A
  clamped map read its far edge for every pixel: distort's picture came
  out one tan colour, and Quake 3's lightmaps came out black, and its walls
  with them. perfly's hubcap centre, s 1.49, agrees.
- **GL lines without blending.** Whether such a line covers a pixel is up
  to coverage alone. Weighing coverage by the colour's alpha drew none of
  amesh's mesh, whose IRIS GL RGB colours carry alpha 0.
- **MACE UST counts every 960 ns** (33.3 MHz / 32, spec 7.2.2), as IRIX
  converts it. Counting microseconds ran 4% slow against the codec.
- **gr_osview's panel turning yellow** next to mandel is the colour map,
  not the scan-out. mandel is a 12-bit colour-index window (I12), which
  looks up entries 0-4095 directly, the same RAM as the 8-bit windows'
  maps (`cm` << 8). Installing its map overwrites gr_osview's map 14
  until 4Dwm reinstalls that one. The GBE spec describes the same sharing,
  so a real O2 should flash the same way (unconfirmed).
- **Quake 3** dies of SIGBUS in its RoQ cinematic: `move4_32` copies with
  `ldc1` from a 4-byte-aligned buffer. That is an address error on real
  hardware too, so it is a port bug, not the emulator. A user-fault trap
  log in the CPU core found it (a bring-up aid, since removed).
- **Quake 2's sound** ran dry about 45 times a second for a moment each.
  `sfplay` of a steady file at 11025 Hz ran dry 6 times in 18 s, so the
  audio path holds. Quake 2 under emulation does not keep the port fed,
  and `s_mixahead` does not change that.

Instruments added for it: the control port's `pixel X Y` (one scanned-out
point: frame-buffer word, DID, mode-table entry, colour map, overlay) and
`unmapped`; `IRIS_O2_FRAG_PROBE=x,y` (every triangle fragment at a pixel,
with texture coordinates, LOD and level); `IRIS_O2_AUDIO_DUMP` (each
output's samples at the guest's rate); and the "ring ran dry" count in the
audio trace.
