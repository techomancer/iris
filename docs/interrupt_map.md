# IRIS Interrupt Map — SGI Indy (IP24 / Guinness) & Indigo2 (IP22 / Fullhouse)

All information derived from: IRIX `kern/sys/IP22.h`, `kern/ml/IP22.c`, `kern/sys/hpc3.h`,
IOC2 datasheet, MAME `ioc2.cpp`/`ioc2.h`, Linux `arch/mips/include/asm/sgi/ip22.h`, and live
disassembly of a running IRIX kernel (`ip22_gio0/1/2_intr`, `ip22_newportInterrupt`,
`ng1_init`, `setgiovector`, `setlclvector`).

Sections through "IRIS `IocInterrupt` enum → register mapping" describe IP24/Indy's INT3
registers at `IOC_BASE` (`0x1FBD9800`, PBUS PIO channel 6). The **"IP22 fullhouse"** section
near the end covers Indigo2's separate INT2 register block, `PORT_CONFIG`, `EXT_IO`, and the
real (counterintuitive) mechanism Newport's IRIX driver uses to detect vertical retrace.

---

## IOC2 register addresses (via HPC3, phys base `0x1FBD9800`)

| Register         | Phys offset | IRIX symbol          | Width |
|------------------|-------------|----------------------|-------|
| L0_STAT (ISR)    | `+0x00`     | `LIO_0_ISR_ADDR`     | u8    |
| L0_MASK          | `+0x04`     | `LIO_0_MASK_ADDR`    | u8    |
| L1_STAT (ISR)    | `+0x08`     | `LIO_1_ISR_ADDR`     | u8    |
| L1_MASK          | `+0x0C`     | `LIO_1_MASK_ADDR`    | u8    |
| MAP_STAT (ISR)   | `+0x10`     | `LIO_2_3_ISR_OFFSET` | u8    |
| MAP_MASK0        | `+0x14`     | `LIO_2_MASK_ADDR`    | u8    |
| MAP_MASK1        | `+0x18`     | `LIO_3_MASK_ADDR`    | u8    |
| MAP_POL          | `+0x1C`     | —                    | u8    |

---

## L0_STAT / L0_MASK — Local 0 (→ CPU IP2)

| Bit  | Mask   | IRIX symbol   | VECTOR         | Device / signal              |
|------|--------|---------------|----------------|------------------------------|
| 0    | `0x01` | `LIO_FIFO`    | `VECTOR_GIO0`  | **FIFO_FULL_N** pin (active-low); on Indy also = REX3 GFIFO-full (GIO_INT_0 from GFX slot) |
| 1    | `0x02` | `LIO_SCSI_0`  | `VECTOR_SCSI`  | SCSI channel 0 (`SCSI0INT` pin) |
| 2    | `0x04` | `LIO_SCSI_1`  | `VECTOR_SCSI1` | SCSI channel 1 (`SCSI1INT` pin) |
| 3    | `0x08` | `LIO_ENET`    | `VECTOR_ENET`  | Ethernet (`ENET_INT` pin)    |
| 4    | `0x10` | `LIO_GDMA`    | `VECTOR_GDMA`  | MC DMA done (`MC_DMA_DONE` pin) |
| 5    | `0x20` | `LIO_CENTR`   | —              | Parallel port (`CENTR_INT`?) |
| 6    | `0x40` | `LIO_GIO_1`   | `VECTOR_GIO1`  | **GRX_INT_N** pin (active-low); REX3 GIO_INT_1 from GFX slot (graphics interrupt) |
| 7    | `0x80` | `LIO_LIO2`    | `VECTOR_LCL2`  | **MAP_INT0** output: fires when any `MAP_STAT & MAP_MASK0` bit is set |

---

## L1_STAT / L1_MASK — Local 1 (→ CPU IP3)

| Bit  | Mask   | IRIX symbol     | VECTOR          | Device / signal              |
|------|--------|-----------------|-----------------|------------------------------|
| 0    | `0x01` | `LIO_POWER`     | `VECTOR_POWER`  | Front panel power button     |
| 1    | `0x02` | `LIO_ISDN_HSCX` | `VECTOR_ISDN_HSCX` | ISDN HSCX (IP24 only)     |
| 2    | `0x04` | `LIO_ISDN_ISAC` | `VECTOR_ISDN_ISAC` | ISDN ISAC                 |
| 3    | `0x08` | —               | —               | (unused on IP24)             |
| 4    | `0x10` | `LIO_HPC3`      | `VECTOR_HPCDMA` | HPC3 DMA done (`HPC_DMA_DONE` pin) |
| 5    | `0x20` | `LIO_AC`        | `VECTOR_ACFAIL` | AC fail (`AC_FAIL_N` pin, active-low) |
| 6    | `0x40` | `LIO_VIDEO`     | `VECTOR_VIDEO`  | VINO video (`VIDEO_VSYNC_N` pin) |
| 7    | `0x80` | `LIO_GIO_2`     | `VECTOR_GIO2`   | **VERT_RETRACE_N** pin (active-low); REX3 GIO_INT_2 from GFX slot (vertical retrace) |

> **IP24 note**: datasheet `LOCAL1_N<0>` (pin 31) and `LOCAL1_N<2>` (pin 30) are "general
> purpose, reserved in INT2" — in IRIX they appear as `GP0`/`GP2` but are not used by any
> standard driver on Indy. `EISA_ERROR_N` (pin 15) maps to `LIO_EISA_MASK` but EISA is
> IP22-fullhouse only.

---

## MAP_STAT / MAP_MASK0 / MAP_MASK1 — Mappable interrupts

MAP_STAT contains 8 active-low inputs (`MAP_INT_N<7:6,3:0>` pins, polarity selectable via
MAP_POL). When `MAP_STAT & MAP_MASK0 != 0`, it drives **MAP_INT0** → L0 bit 7 (`LIO_LIO2`).
When `MAP_STAT & MAP_MASK1 != 0`, it drives **MAP_INT1** → L1 bit 3 (unused on IP24).

**IP22 fullhouse** bits 6–7 = `LIO_DRAIN0`/`LIO_DRAIN1` (GFX FIFO not-full feedback).  
**IP24 Indy** bits 6–7 = `LIO_GIO_EXP0`/`LIO_GIO_EXP1` (expansion slot interrupts).

| Bit  | Mask   | IRIX symbol (IP24)  | VECTOR            | Device / signal              |
|------|--------|---------------------|-------------------|------------------------------|
| 0    | `0x01` | —                   | —                 | (unused / MAP_INT_N<0>)      |
| 1    | `0x02` | —                   | —                 | (unused / MAP_INT_N<1>)      |
| 2    | `0x04` | —                   | —                 | (unused / MAP_INT_N<2>)      |
| 3    | `0x08` | —                   | —                 | (unused / MAP_INT_N<3>)      |
| 4    | `0x10` | `LIO_KEYBD_MOUSE`   | `VECTOR_KBDMS`    | Keyboard / mouse (Z8530)     |
| 5    | `0x20` | `LIO_DUART`         | `VECTOR_DUART`    | Serial DUART (Z85C30)        |
| 6    | `0x40` | `LIO_GIO_EXP0`      | `VECTOR_GIOEXP0`  | **GIO expansion slot 0** interrupt (u64 board → `u64_giointr`) |
| 7    | `0x80` | `LIO_GIO_EXP1`      | `VECTOR_GIOEXP1`  | GIO expansion slot 1         |

`VECTOR_GIOEXP0 = 22` → `lcl_id = 22/8 = 2`, `level = 22 & 7 = 6` →
`MAP_MASK0 |= (1 << 6)` at driver init (`setlclvector(VECTOR_GIOEXP0, u64_giointr, ...)`).

---

## Interrupt routing summary

```
Device            → IOC2 pin          → Register bit   → CPU cause
─────────────────────────────────────────────────────────────────────
SCSI0             → SCSI0INT          → L0 bit 1       → IP2
SCSI1             → SCSI1INT          → L0 bit 2       → IP2
Ethernet (SEEQ)   → ENET_INT          → L0 bit 3       → IP2
MC DMA            → MC_DMA_DONE       → L0 bit 4       → IP2
Parallel port     → (HPC3)            → L0 bit 5       → IP2
REX3 GFIFO full   → FIFO_FULL_N       → L0 bit 0       → IP2
REX3 GIO_INT_1    → GRX_INT_N         → L0 bit 6       → IP2
REX3 vert retrace → VERT_RETRACE_N    → L1 bit 7       → IP3
HPC3 DMA          → HPC_DMA_DONE      → L1 bit 4       → IP3
AC fail           → AC_FAIL_N         → L1 bit 5       → IP3
VINO              → VIDEO_VSYNC_N     → L1 bit 6       → IP3
Keyboard/mouse    → MAP_INT_N<4>      → MAP bit 4      → MAP_INT0 → L0 bit 7 → IP2
Serial (DUART)    → MAP_INT_N<5>      → MAP bit 5      → MAP_INT0 → L0 bit 7 → IP2
GIO EXP0 (u64)    → MAP_INT_N<6>      → MAP bit 6      → MAP_INT0 → L0 bit 7 → IP2
GIO EXP1          → MAP_INT_N<7>      → MAP bit 7      → MAP_INT0 → L0 bit 7 → IP2
Power button      → (front panel)     → L1 bit 0       → IP3
```

---

## REX3 / Newport interrupt detail

Newport uses **GIO_SLOT_GFX** which on IP24 maps directly to the `giointr[]` vectors
(no fan-out via EXTIO needed — that is IP22 fullhouse only; see that section below for the
fan-out mechanism and, notably, why fullhouse's Newport driver doesn't use GIO_INT_2 at all):

| GIO_INTERRUPT | VECTOR       | IOC2 pin        | Stat register | Bit    | Use              |
|---------------|--------------|-----------------|---------------|--------|------------------|
| GIO_INT_0     | VECTOR_GIO0  | FIFO_FULL_N     | L0_STAT       | bit 0  | GFIFO above threshold |
| GIO_INT_1     | VECTOR_GIO1  | GRX_INT_N       | L0_STAT       | bit 6  | Graphics (XZ/Elan GE11-done only — Newport has no real signal here) |
| GIO_INT_2     | VECTOR_GIO2  | VERT_RETRACE_N  | L1_STAT       | bit 7  | Vertical retrace  |

`setgiogfxvec(0, ...)` / `setgiovector(0, GIO_SLOT_GFX, giogfx_intr, 0)` registers
the GFIFO handler at `VECTOR_GIO0` → L0 bit 0.

On IP22 fullhouse all three GIO interrupt lines are shared between GFX slot and expansion
slots; the EXTIO register disambiguates which slot fired. On IP24 Indy, GFX slot gets the
three dedicated pins and expansion slots route through MAP instead.

---

## u64 board (N64 dev board) interrupt detail

```
u64_init():
  setgiovector(GIO_INTERRUPT_0, GIO_SLOT_0, u64_giointr, controller)
  → IP24 path: setlclvector(VECTOR_GIOEXP0, u64_giointr, ...)
  → MAP_MASK0 |= (1 << 6)   i.e. LIO_GIO_EXP0 = 0x40
```

Interrupt flow when N64 sends RDB packet:
1. N64 writes to RDB port (GIO address `0x1F400000`-range on Indy bus)
2. u64 board asserts `MAP_INT_N<6>` (active-low)
3. IOC2: `MAP_STAT bit 6` set → `MAP_MASK0 bit 6` enables → `MAP_INT0` asserted
4. `MAP_INT0` → `L0_STAT bit 7` (`LIO_LIO2`) set → `L0_MASK bit 7` enables → **IP2**
5. CPU enters IP2 handler → `lcl0_intr` → dispatches via `lcl0vec_tbl[VECTOR_LCL2+1]`
6. `lcl2_intr` reads `MAP_STAT & MAP_MASK0` → bit 6 set → dispatches `lcl2vec_tbl[7]`
7. `u64_giointr` runs, reads RDB register, dispatches to RDB handler

In IRIS: `IocInterrupt::GioExp0` sets `map_stat |= LIO_GIO_EXP0 (0x40)`.
`update_interrupts()` propagates: if `map_stat & map_mask0 != 0` → sets `l0_stat bit 7`
→ if `l0_mask bit 7` set → fires IP2.

---

## IRIS `IocInterrupt` enum → register mapping

| Variant          | Register  | Bit    | Mask   |
|------------------|-----------|--------|--------|
| `Scsi0`          | L0_STAT   | bit 1  | `0x02` |
| `Scsi1`          | L0_STAT   | bit 2  | `0x04` |
| `Ethernet`       | L0_STAT   | bit 3  | `0x08` |
| `McDma`          | L0_STAT   | bit 4  | `0x10` |
| `FifoFull`       | L0_STAT   | bit 0  | `0x01` |
| `Graphics`       | L0_STAT   | bit 6  | `0x40` |
| `VertRetrace`    | L1_STAT   | bit 7  | `0x80` |
| `HpcDma`         | L1_STAT   | bit 4  | `0x10` |
| `AcFail`         | L1_STAT   | bit 5  | `0x20` |
| `Vino`           | L1_STAT   | bit 6  | `0x40` |
| `Gp0`            | L1_STAT   | bit 0  | `0x01` |
| `Gp2`            | L1_STAT   | bit 2  | `0x04` |
| `GioExp0`        | MAP_STAT  | bit 6  | `0x40` |
| `GioExp1`        | MAP_STAT  | bit 7  | `0x80` |

---

## GIO64 slot memory map (IP24 / Indy, from `physical.rs` and `config.rs`)

| Slot           | Phys range                | Size | IRIS constant(s)                    | Occupant on Indy                  |
|----------------|----------------------------|------|--------------------------------------|------------------------------------|
| GFX (slot 0)   | `0x1F000000`–`0x1F3FFFFF` | 4 MB | `NEWPORT_BASE`/`NEWPORT_END`        | Newport/REX3 (or MGRAS IMPACT)     |
| EXP0 (slot 1)  | `0x1F400000`–`0x1F5FFFFF` | 2 MB | `GIO_SLOT0_BASE`/`GIO_SLOT0_END`    | Ultra64 (N64 dev board) if enabled |
| EXP1 (slot 2)  | `0x1F600000`–`0x1F9FFFFF` | 4 MB | `GIO_SLOT1_BASE`                    | second/third MGRAS board (Max IMPACT configs); otherwise unused |

These are the same physical ranges MAME's `gio64.h`/`indy_indigo2.cpp` use for
`GIO64_SLOT_GFX`/`EXP0`/`EXP1`. IP24 Indy wires GFX slot interrupts straight to the three
dedicated `GIO_INT_0/1/2` pins (see REX3/Newport table above); EXP0/EXP1 route through the
MAP mappable-interrupt bank instead (`LIO_GIO_EXP0`/`LIO_GIO_EXP1`, MAP bits 6/7 — see u64
board detail above). This differs from IP22 fullhouse, where all three GIO_INT lines are
shared/fanned-out across GFX + expansion slots via an EXTIO disambiguation register; IP24
doesn't need that because expansion already has its own MAP path.

## REX3 STATUS register — read-clears-latched-interrupt behavior (`rex3.rs`)

`REX3_STATUS` lives at GIO offset `0x1338` (phys `0x1F0F1338` in the GFX slot), aliased
read-only at `REX3_USER_STATUS` (`0x133C`) which does **not** clear on read.

Reading `REX3_STATUS` (not `USER_STATUS`) atomically clears two latched bits and fires the
matching callback to deassert the IOC2 pin, mirroring the MAME `newport.cpp` clear-on-read
behavior:

| Status bit         | Bit | Cleared on `REX3_STATUS` read → callback  | Feeds IOC2 pin  |
|---------------------|-----|--------------------------------------------|-----------------|
| `STATUS_VRINT`      | 5   | `vblank_cb(false)`                          | `GIO_INT_2` → L1 bit 7 → IP3 |
| `STATUS_GFIFO_INT`  | 19  | `fifo_full_cb(false)`                       | `GIO_INT_0` → L0 bit 0 → IP2 |

`STATUS_BFIFO_INT` (bit 18, back-FIFO threshold) is tracked in `status` but is not one of
the two bits cleared by the STATUS read path above.

Practical implication: a driver (or `iris-ci` script) that reads `REX3_USER_STATUS` to poll
for vblank/FIFO state will **not** deassert the interrupt line — only a `REX3_STATUS` read
does that. This matches real hardware/IRIX behavior, where `USER_STATUS` exists specifically
so user-mode graphics libraries can poll status without racing the kernel's interrupt
handler (which reads `STATUS` and clears the pending bit).

---

## IP22 fullhouse (Indigo2) — PORT_CONFIG and EXT_IO

Everything above this section is written for IP24/Indy. Fullhouse (Indigo2) exposes a
*second*, more compact interrupt register block — **INT2** at HPC3 PBUS PIO channel 4
(`HPC3_INT2_ADDR = 0x1FBD9000`), distinct from Indy's INT3 registers at PIO channel 6
(`HPC3_INT3_ADDR = 0x1FBD9880` = `IOC_BASE + 0x80`). Real IRIX picks between them **at
boot, per-machine, based on detected chip revision** — not by board type alone:

```c
#define HPC3_INT_ADDR (is_ioc1() ? HPC3_INT3_ADDR : HPC3_INT2_ADDR)
```

`is_ioc1()` is computed from `HPC3_SYS_ID`'s (`0x1FBD9858`) chip-revision field
(`CHIP_REV_MASK = 0xE0`, `CHIP_IOC1 = 0x20`): it's `0` ("pre-006 fullhouse") whenever the
chip-rev bits aren't `CHIP_IOC1`, which routes to **INT2**. IRIS's fullhouse `sys_id = 0x11`
(`docs/indigo2-ip22.md`) has chip-rev field `0x00 ≠ 0x20`, so `is_ioc1() == 0` and real IRIX
resolves `HPC3_INT_ADDR`/all `LIO_*_ADDR` macros to **INT2** — matching everything below.
Changing `sys_id`'s chip-rev bits would silently break this and point IRIX at INT3 instead.

INT2's registers are the same shape as INT3's (`L0_STAT/MASK`, `L1_STAT/MASK`,
`MAP_STAT/MASK0/MASK1`), packed at 4-byte stride starting `0x1FBD9000`, **plus two
registers INT3 doesn't have** (fullhouse-only, no guinness equivalent):

| Register     | INT2 offset | Index | IRIX symbol         | Width  |
|--------------|-------------|-------|----------------------|--------|
| L0_STAT      | `+0x00`     | 0     | (`LIO_0_ISR_ADDR`)   | u8     |
| L0_MASK      | `+0x04`     | 1     | (`LIO_0_MASK_ADDR`)  | u8     |
| L1_STAT      | `+0x08`     | 2     | (`LIO_1_ISR_ADDR`)   | u8     |
| L1_MASK      | `+0x0C`     | 3     | (`LIO_1_MASK_ADDR`)  | u8     |
| MAP_STAT     | `+0x10`     | 4     | (`LIO_2_3_ISR_ADDR`) | u8     |
| MAP_MASK0    | `+0x14`     | 5     | (`LIO_2_MASK_ADDR`)  | u8     |
| MAP_MASK1    | `+0x18`     | 6     | (`LIO_3_MASK_ADDR`)  | u8     |
| **PORT_CONFIG** | `+0x1C`  | 7     | `PORT_CONFIG`        | u8     |
| TMR_CLR      | `+0x20`     | 8     | —                    | u8, wo |
| PIT ch 0-2/ctl | `+0x30-0x3C` | 12-15 | —                 | u8     |
| **EXT_IO**   | `0x1FBD9900` (= `IOC_BASE + 0x100`, own address, *not* in the INT2 block) | — | `HPC3_EXT_IO_ADDR` | **u16**, read as `uint` |

`IP22BOFF(x) = x | 0x3` (`kern/sys/IP22.h`) is a `_MIPSEB` byte-lane adjustment for
byte-wide registers on the big-endian PROM/kernel build — every `LIO_*_OFFSET` constant in
IRIX source has its low 2 bits set (e.g. `PORT_CONFIG_OFFSET = IP22BOFF(0x1c) = 0x1F`) for
that reason. The **dword index is unaffected** (`offset >> 2` is the same either way), which
is what IRIS's `int2_read8`/`int2_write8` (`ioc.rs`) actually implement.

### PORT_CONFIG (fullhouse only, INT2 index 7)

Controls GIO slot reset and per-slot retrace-clear strobes. No guinness equivalent — Indy's
PIO6 has `MAP_POL` at the same relative offset instead (a genuinely different register).

| Bit | Mask   | Symbol                | Meaning (active-low `_N`)        |
|-----|--------|------------------------|-----------------------------------|
| 0   | `0x01` | `PCON_DMA_SYNC_SEL`    | DMA sync target: 1=slot1, 0=slot0 |
| 1   | `0x02` | `PCON_SG_RESET_N`      | Reset GFX slot                    |
| 2   | `0x04` | `PCON_S0_RESET_N`      | Reset EXP0 slot                   |
| 3   | `0x08` | `PCON_CLR_SG_RETRACE_N`| Clear GFX slot's retrace latch    |
| 4   | `0x10` | `PCON_CLR_S0_RETRACE_N`| Clear EXP0 slot's retrace latch   |

Real IRIX (this kernel tree) never references `PCON_CLR_*`/`PCON_*_RESET_N` from any driver
— these are defined but unused by the version of IRIX in this tree, so their effect (IRIS
clears the shared `L1_STAT` `VERTICAL_RETRACE` bit when a clear strobe is asserted) is
speculative, not confirmed by a live call site.

### EXT_IO / `HPC3_EXT_IO_ADDR` (fullhouse only, `0x1FBD9900`)

Not part of the INT2 block — its own address, one byte-word past guinness's `IOC_SIZE`
(`IOC_BASE + 0x100`). **16 bits wide in hardware** (`HPC3_CFGPIO_DS_16` set on PBUS PIO
channel 6 at boot — `kern/ml/IP22.c`), but IRIX always reads/writes it as a **32-bit `uint`**
(confirmed both in `kern/ml/IP22.c`'s comment and in the live `ip22_gio0/1/2_intr`
disassembly: `lw s1, 0(a0)`). IRIS implements this via `Ioc`'s `BusDevice::read16`/`write16`
(true 16-bit path) plus a zero-extending special case in `read32`/`write32`.

All bits **active-low** (0 = pending/asserted, 1 = idle):

| Bit | Mask     | Symbol            | Meaning                              |
|-----|----------|--------------------|----------------------------------------|
| 0   | `0x0001` | `EXTIO_SG_STAT_0`  | (unused by this IRIX version)          |
| 1   | `0x0002` | `EXTIO_SG_STAT_1`  | (unused)                               |
| 2   | `0x0004` | `EXTIO_S0_STAT_0`  | (unused)                               |
| 3   | `0x0008` | `EXTIO_S0_STAT_1`  | (unused)                               |
| 4   | `0x0010` | `EXTIO_HPC3_BUSERR`| HPC3 bus error                         |
| 5   | `0x0020` | `EXTIO_MC_BUSERR`  | MC bus error                           |
| 6   | `0x0040` | `EXTIO_EISA_BUSERR`| EISA bus error (fullhouse w/ EISA only)|
| 7   | `0x0080` | `EXTIO_GIO_33MHZ`  | GIO bus running at 33MHz               |
| 8   | `0x0100` | `EXTIO_SG_RETRACE` | GFX slot: `GIO_INTERRUPT_2` pending    |
| 9   | `0x0200` | `EXTIO_SG_IRQ_1`   | GFX slot: `GIO_INTERRUPT_1` pending    |
| 10  | `0x0400` | `EXTIO_SG_IRQ_2`   | GFX slot: `GIO_INTERRUPT_0` pending    |
| 11  | `0x0800` | `EXTIO_SG_IRQ_3`   | GFX slot: vid.vsync (not a GIO vector) |
| 12  | `0x1000` | `EXTIO_S0_RETRACE` | EXP0 slot: `GIO_INTERRUPT_2` pending   |
| 13  | `0x2000` | `EXTIO_S0_IRQ_1`   | EXP0 slot: `GIO_INTERRUPT_1` pending   |
| 14  | `0x4000` | `EXTIO_S0_IRQ_2`   | EXP0 slot: `GIO_INTERRUPT_0` pending   |
| 15  | `0x8000` | `EXTIO_S0_IRQ_3`   | EXP0 slot: vid.vsync                   |

`kern/sys/hpc3.h` notes "IP22-006 splits EXTIO into two registers to support 3rd gio slot" —
`EXTIO_S1_IRQ_1/2/3`/`EXTIO_S1_RETRACE` (values `0x0002/0x0004/0x0008/0x0001`) numerically
collide with the `SG_STAT_*`/`S0_STAT_*` bits above, so the 3rd-slot bits must live in a
**second, undocumented register** — no address for it is defined anywhere in
`IP22.c`/`IP26.c`/`IP28.c`. That 3rd slot is gated behind `SPECIAL_GIO_RESET`, described in
`IP22.c` as "only meaningful on specially-modified IP22 systems used by an important medical
equipment-manufacturing customer" — not a configuration IRIS emulates. IRIS's `GioS1*`
`IocInterrupt` variants exist for API symmetry but nothing ever drives their EXT_IO bits.

### The real fan-out mechanism: `ip22_gio0/1/2_intr`

On fullhouse, all 3 GIO slots (GFX, EXP0, EXP1) share the *same 3 physical IOC2 pins* —
`FIFO_FULL_N`→L0 bit 0, `GRX_INT_N`→L0 bit 6, `VERT_RETRACE_N`→L1 bit 7 — confirmed by
`kern/sys/IP22.h`'s `VECTOR_GIO0/1/2 = 0/6/15` being identical constants to IP24's. What
differs is that `setgiovector()` (`kern/ml/IP22.c`, decompiled/disassembly-confirmed against
the live kernel) installs **`ip22_gio{0,1,2}_intr`** — not the caller's handler directly —
for whichever GIO vector any slot has registered a real ISR for. Each fan-out function reads
`HPC3_EXT_IO_ADDR` as a 32-bit word and, in slot order (GFX, then EXP0, then EXP1), checks
whether that slot's bit for this vector is clear; if so it calls
`giovec_tbl[slot][level].isr(giovec_tbl[slot][level].arg, ep)`. Confirmed against
`ip22_gio2_intr`'s disassembly (`andi at, s1, 0x100` / `beq at, zero, →GFX dispatch`, then
`andi v0, s1, 0x1000` / `→EXP0 dispatch`) — matches this table's bit layout exactly.

IRIS implements this fan-out for real: `IocInterrupt::Gio{Sg,S0,S1}{Fifo,Graphics,Retrace}`
(`ioc.rs`) each set the one shared L0/L1 bit *and* clear/set their own EXT_IO bit; `GIO_SLOT_MAP`
picks which variant a given `GioSlot` uses per profile.

### Why Newport retrace is delivered through GIO_INTERRUPT_1, not GIO_INTERRUPT_2

This is the counterintuitive finding of this investigation, confirmed against the live
kernel's disassembly. `ng1_init` (Newport's driver init) on fullhouse only ever calls:

```c
setgiovector(GIO_INTERRUPT_0, slot, ip22_newportInterrupt, board);
setgiovector(GIO_INTERRUPT_1, slot, ip22_newportInterrupt, board);
```

**It never calls `setgiovector(GIO_INTERRUPT_2, ...)` for Newport at all.** `ip22_gio2_intr`
(the retrace fan-out, `VECTOR_GIO2`/L1 bit 7) is installed for *nothing*, and never fires for
Newport — this was reproduced live (`bp` on `ip22_gio2_intr` never hit, even running the
screensaver).

Instead, `ip22_newportInterrupt` — the handler installed for GIO_INTERRUPT_0 (real: GFIFO
above/below, `FIFO_INT_N`) and GIO_INTERRUPT_1 (nominally "graphics", but Newport has no
real GE-done signal; only XZ/Elan drives that pin) — reads REX3's `STATUS` register directly
(offset `0x1338`) on every entry and checks `STATUS & 0x20` (`STATUS_VRINT`, `VV_INT_N` on
the REX3 datasheet). If set, it calls `ip22_newportRetrace` as a plain subroutine call, no
separate vector involved. So on real Indigo2, vertical retrace is noticed as a side effect
of *fifo* interrupt traffic (or occasionally the shared-but-unused graphics vector),
never through its own dedicated GIO_INTERRUPT_2/EXTIO_*_RETRACE path for Newport specifically.

**Practical consequence for IRIS:** IRIS's software GFIFO (`GFIFO_DEPTH = 65536`) essentially
never reaches real hardware's threshold behavior under light draw load (a screensaver,
idle desktop) — by design, it's better to buffer than stall/interrupt at emulated speed —
so `FIFO_INT_N`/`STATUS_GFIFO_INT` rarely if ever fires, and `ip22_newportInterrupt` is
rarely entered, so `STATUS_VRINT` is rarely polled, and the retrace-driven screensaver hangs
even though REX3 is correctly asserting VRINT every frame. Confirmed fix (live-tested):
`GIO_SLOT_MAP`'s fullhouse rows route the GFX/EXP0/EXP1 slots' `vblank_cb` through
`GioSgGraphics`/`GioS0Graphics`/`GioS1Graphics` (the GIO_INTERRUPT_1 fan-out) instead of
`GioSgRetrace`/etc — this forces entry into `ip22_newportInterrupt` on every real vblank,
where it correctly finds `STATUS_VRINT` set and calls `ip22_newportRetrace`. The dedicated
GIO_INTERRUPT_2/`ext_io` retrace-fanout implementation is kept (accurate to real IOC2
behavior, and MAME/guinness/other drivers might legitimately use it), but Newport's own
retrace delivery does not depend on it.

Note: `graphics_cb`/`gfx_drain_cb` (an earlier, incorrect model treating "graphics" and
"fifo-drain" as distinct real Newport signals) were removed from `Rex3` — Newport only has
2 real interrupt pins, `FIFO_INT_N` and `VV_INT_N` (REX3 datasheet); everything above uses
only those two, routed through whichever `IocInterrupt`/EXT_IO variant is appropriate.
