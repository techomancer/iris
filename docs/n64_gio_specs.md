# Nintendo 64 Development Board: GIO Bus & Shared Memory Specifications

This document outlines the low-level hardware specifications, memory addresses, register layouts,
and communication protocols for the original Nintendo 64 Development Board (the double-wide
GIO-bus peripheral card designed for Silicon Graphics Indy workstations).

> **Emulation note**: This hardware is being emulated by two cooperating processes — IRIS (SGI
> Indy) and gopher64 (N64). See `src/dev/ultra64.rs` (IRIS) and gopher64's `src/device/sgi_dev.rs`
> for the implementation. The IPC bridge design is documented at the end of this file.

---

## 1. Physical Addresses & Memory Mapping

### 1.1 RAMROM — the shared SRAM (16 MB)

The central feature of the board is 16 MB of high-speed static RAM called the **RAMROM**. It
sits on the GIO card itself, independent of both the Indy's main RAM and the N64's RDRAM. It
serves two roles simultaneously:

- The **SGI Indy** uses it as a staging area: the host writes compiled game binaries, textures,
  and assets into it over the GIO bus.
- The **N64 CPU/RCP** sees it as its cartridge ROM: when the N64 boots, its PI bus reads from
  RAMROM exactly as if reading a consumer mask-ROM cartridge at `0x10000000`.

The RAMROM arbiter does **not** support dual-porting. Simultaneous access by the Indy and the
N64's PI DMA will stall the bus. Software synchronization via the GIO registers (§3) is required.

### 1.2 SGI Indy Host GIO Space

The GIO64 bus presents the board in three windows:

| GIO Physical Address          | Access | Size  | Description |
| :---                          | :---:  | :---: | :--- |
| `0x1F400000` – `0x1F4FFFFF`  | R/W    | 1 MB  | **Control Register Space** (reset, cart-int, dram-page, gio-int, gio-sync) |
| `0x1F480000` – `0x1F48000F`  | R/W    | 16 B  | **RDB Port** (dual 32-bit data + interrupt clear registers) |
| `0x1F500000` – `0x1F5FFFFE`  | R/W    | 1 MB  | **RAMROM Page Window** (sliding 1 MB view into the 16 MB RAMROM) |

The host can only address 1 MB of RAMROM at a time through the `0x1F500000` window. The 4-bit
page register at `0x1F400A00` bits[23:20] selects which 1 MB block is currently visible.
(SDK manpage erroneously lists this as `0x1F400600`; see §2.4.)

### 1.3 N64 CPU (R4300i) Address Space

From the N64 R4300i's perspective:

| N64 Physical Address          | Access | Description |
| :---                          | :---:  | :--- |
| `0x10000000` – `0x10FFFFFF`  | R      | **RAMROM as cartridge ROM** (full 16 MB, PI Domain 1) |
| `0x18000000`                  | W      | **GIO Interrupt Register** — signal the host *(obsolete SDK 2.0F+)* |
| `0x18000400`                  | W      | **GIO Sync Register** — polling, no interrupt *(obsolete SDK 2.0F+)* |
| `0x18000800`                  | R      | **Cartridge Interrupt Clear** — ACK host→N64 INT1 *(obsolete SDK 2.0F+)* |
| `0xC0000000`                  | R/W    | **RDB Data Register** — R=read Indy's packet, W=send packet to Indy |
| `0xC0000008`                  | W      | **RDB Write Interrupt Clear** — write `0` to ACK write-complete interrupt |
| `0xC000000C`                  | W      | **RDB Read Interrupt Clear** — write `0` to ACK read-complete interrupt |

The `0x18000xxx` registers are in PI Domain 1 upper range. The `0xC0000xxx` RDB registers are
in the SysAD device address space (not PI/cartridge domain).

The N64 accesses the full 16 MB RAMROM directly through the PI bus at `0x10000000`. Bulk
transfers go through the N64's PI DMA engine (CPU configures PI_CART_ADDR, PI_DRAM_ADDR,
PI_WR_LEN, then DMA copies RAMROM → RDRAM). Direct CPU reads from `0x10000000` are also valid
(slower, subject to PI bus timing).

---

## 2. Hardware Registers & Bit Allocation

All host-side registers are in the `0x1F40xxxx` control block.

### 2.1 Product ID Register (`0x1F400000` — Read Only)

Used by the IRIX kernel during boot-probe to identify the N64 board on the GIO bus.

| Bits    | Value  | Description |
| :---    | :---:  | :--- |
| [6:0]   | `0x15` | Fixed board type identifier (`_U64_PRODUCT_ID_VALUE` in kernel headers; kernel masks with `_U64_PRODUCT_ID_MASK = 0x7f`) |
| [7]     | `0`    | Unused / undriven |
| [29:8]  | `0`    | Reserved |
| [30]    | RDB-R  | Set when the R4300i has **read** from its RDB register (hw rev 2+ only) |
| [31]    | RDB-W  | Set when the R4300i has **written** to its RDB register (hw rev 2+ only) |

The base value with no pending RDB events is `0x00000015`. The IRIX autoconfig probe is `exprobe=(r,0xBF400000,4,0x15,0xff)` — reads 4 bytes, masks with `0xff`, expects `0x15`.

The `u64_giointr()` ISR reads this register first on every GIO interrupt to decide which RDB
direction fired (bits 30/31), then dispatches to `Handle_RDB_Incoming()` or `send_write_buffer()`
accordingly. A GIO interrupt with neither bit set is an error ("DID NOT USE THE RDB PORT!!!").

### 2.2 Reset Control Register (`0x1F400400` — Write Only)

Controls the electrical and execution state of the N64 hardware.

| Bit | Description |
| :-- | :--- |
| [1] | **N64 CPU Reset**: asserted (held in reset) when `1`, released when cleared to `0` |
| [2] | **NMI**: armed when set to `1`; the NMI fires on the R4300i when this bit is cleared back to `0` |

### 2.3 Cartridge Interrupt Register (`0x1F400800` — Read/Write)

**Obsolete since SDK 2.0F. Reserved for N64 Disk Drive peripherals; do not use for host communication.**

| Bits  | Description |
| :---  | :--- |
| [5:0] | 6-bit payload delivered to the N64 interrupt handler |

Writing here asserts **INT1** on the N64's R4300i (`CAUSE.IP4`). The interrupt is cleared when
the N64 CPU reads from `0x18000800`.

### 2.4 DRAM Page Control Register (`0x1F400A00` — Read/Write)

Selects which 1 MB page of the 16 MB RAMROM is visible through the `0x1F500000` window.

| Bits    | Description |
| :---    | :--- |
| [23:20] | **Page Select**: 4-bit index (0–15) → maps megabyte `page` into `0x1F500000–0x1F5FFFFE` |

> **Address discrepancy**: The SDK manpage lists this register at `0x1F400600`, but the IRIX
> kernel's own `u64gio.h` struct padding (`fill_3[0x1fc]` after CART_INT at `+0x800`) places it
> at `+0xa00` = `0x1F400A00`. The kernel struct is authoritative; the manpage has a typo.
> `u64_write_ramrom()` confirms: `bdata->board->dram_page_cntrl = (page_start << 20);`

### 2.5 GIO Interrupt Acknowledge Register (`0x1F400C00` — Read Only)

**Obsolete since SDK 2.0F.** Used by the host to drain a pending N64→host GIO interrupt.

| Bits  | Description |
| :---  | :--- |
| [5:0] | 6-bit payload written by the N64 to `0x18000000` |

Reading this register simultaneously de-asserts the GIO interrupt line on the Indy.

### 2.6 GIO Sync Register (`0x1F400E00` / N64 `0x18000400` — Read/Write both sides)

**Obsolete since SDK 2.0F.** Polling-only synchronization register. N64 writes bits[5:0]; host
reads back the same bits. Writing from the N64 side does **not** assert the GIO interrupt line.

### 2.7 RDB Port Registers (`0x1F480000` / N64 `0xC0000000` — Hardware Rev 2+)

The RDB (Remote Debugger) port is the **mandatory communication channel** from SDK 2.0F onward.
It provides two independent 32-bit registers — one per direction — allowing simultaneous
bidirectional data transfer. The IRIX driver (`u64_giointr`) only handles RDB interrupts; a GIO
interrupt with neither RDB bit set is logged as an error.

The registers live in a **separate GIO window** at `0x1F480000`, distinct from the control
registers at `0x1F400000`. On the N64, they appear in the SysAD device address space at
`0xC0000000` (not the PI/cartridge domain).

#### RDB Data Register

| GIO Address      | R4300 Address  | Direction | Description |
| :---             | :---           | :---:     | :--- |
| `0x1F480000` (W) | `0xC0000000` (R) | Indy→N64 | Host writes 32-bit packet; N64 reads it |
| `0x1F480000` (R) | `0xC0000000` (W) | N64→Indy | N64 writes 32-bit packet; host reads it |

Both directions share the same GIO address (`0x1F480000`) and R4300 address (`0xC0000000`);
direction is determined by which side is reading vs. writing. The hardware maintains separate
internal registers for each direction — data is never lost.

#### RDB Write Interrupt Register

| GIO Address      | R4300 Address    | Description |
| :---             | :---             | :--- |
| `0x1F480008` (W) | `0xC0000008` (W) | Write `0` to clear the write-complete interrupt on **this** processor |

Whenever either processor **writes** to its RDB register, a write-complete interrupt is sent to
the **other** processor. The recipient's ISR clears it by writing `0` to its own
`0x1F480008` / `0xC0000008`.

#### RDB Read Interrupt Register

| GIO Address      | R4300 Address    | Description |
| :---             | :---             | :--- |
| `0x1F48000C` (W) | `0xC000000C` (W) | Write `0` to clear the read-complete interrupt on **this** processor |

Whenever either processor **reads** from its RDB register, a read-complete interrupt is sent to
the **other** processor. The recipient clears it by writing `0` to its own
`0x1F48000C` / `0xC000000C`.

#### N64-side RDB interrupt lines

| Event                               | N64 Interrupt | CAUSE bit |
| :---                                | :---          | :---      |
| Indy wrote to RDB (data ready)      | INT3          | IP6       |
| Indy read from RDB (ack, send next) | INT4          | IP7       |

The Indy side uses the GIO slot interrupt (`IocInterrupt::Gp0`); Product ID register bits
distinguish which direction fired (bit 30 = read-ack, bit 31 = write / data ready).

#### RDB Packet Format

Each 32-bit word is an `rdbPacket` as defined in `PR/rdb.h`:

```
[31:26]  type   (6 bits)  — RDB_TYPE_* constant
[25:24]  length (2 bits)  — number of valid data bytes (0–3)
[23:16]  buf[0]           — data byte 0
[15:8]   buf[1]           — data byte 1
[7:0]    buf[2]           — data byte 2
```

Key `RDB_TYPE_*` values (direction **G**ame→**H**ost or **H**ost→**G**ame):
- `GtoH_PRINT` — `osSyncPrintf` output
- `GtoH_FAULT` — fault/crash data
- `GtoH_DATA` / `HtoG_DATA` — hostio bulk data
- `GtoH_DEBUG` / `HtoG_DEBUG` — debugger (rmon) packets
- `GtoH_RAMROM` — game releasing RAMROM access
- `GtoH_READY_FOR_DATA` — game buffer ready
- `HtoG_REQ_RAMROM` / `HtoG_FREE_RAMROM` — host arbitrating RAMROM access

---

## 3. Interrupt Protocols & Signaling

### 3.1 RDB — Primary Communication (SDK 2.0F+, mandatory on rev 2 hardware)

Each 32-bit RDB packet carries 6 bits of type, 2 bits of length, and up to 3 data bytes (§2.7).

**Indy → N64 send:**
1. Indy writes 32-bit packet to `0x1F480000`.
2. N64 receives **INT3** (`CAUSE.IP6`) — "data ready".
3. N64 ISR reads `0xC0000000` (consumes the packet), then writes `0` to `0xC000000C` (clears its read-interrupt).
4. Indy receives GIO interrupt; product_id_reg bit 30 set → "N64 read it, can send next".
5. Indy ISR writes `0` to `0x1F48000C` (clears its read-interrupt), calls `send_write_buffer()`.

**N64 → Indy send:**
1. N64 writes 32-bit packet to `0xC0000000`.
2. Indy receives GIO interrupt; product_id_reg bit 31 set → "N64 wrote data".
3. Indy ISR writes `0` to `0x1F480008` (clears its write-interrupt), reads `0x1F480000`, calls `Handle_RDB_Incoming()`.
4. N64 receives **INT4** (`CAUSE.IP7`) — "Indy read it, can write next".
5. N64 ISR writes `0` to `0xC0000008` (clears its write-interrupt).

**Reset / NMI** are still issued via the Reset Control Register `0x1F400400` (§2.2) — these are
not conveyed over RDB.

### 3.2 Legacy: Cartridge Interrupt / INT1 *(obsolete since SDK 2.0F)*

1. Host writes 6-bit payload to `0x1F400800`.
2. N64 R4300i receives **INT1** → `CAUSE.IP4` set.
3. N64 reads `0x18000800` → interrupt cleared.

Reserved for N64 Disk Drive peripherals in SDK 2.0F+. Do not use for host communication.

### 3.3 Legacy: N64-to-Host GIO Interrupt *(obsolete since SDK 2.0F)*

1. N64 writes 6-bit value to `0x18000000`.
2. GIO slot ISR fires on Indy (`IocInterrupt::Gp0`).
3. Indy reads `0x1F400C00` → payload extracted; interrupt cleared.

### 3.4 Legacy: GIO Sync Polling *(obsolete since SDK 2.0F)*

N64 writes bits[5:0] to `0x18000400`; Indy polls `0x1F400E00`. No interrupt on either side.

---

## 4. Data Transfer Protocol

### 4.1 Indy Writing to RAMROM (typical game-load flow)

1. Host asserts N64 reset via `0x1F400400` bit[1] to hold CPU stopped.
2. Host selects the target 1 MB page: write `(page << 20)` to `0x1F400A00`.
3. Host writes code/data into `0x1F500000–0x1F5FFFFE`.
4. Repeat for each page until all 16 MB are loaded.
5. Host releases N64 reset (clears bit[1] of `0x1F400400`); CPU boots from IPL3 in RAMROM.

### 4.2 N64 Copying RAMROM → RDRAM (PI DMA, the normal execution path)

After the host interrupt fires, the N64 boot stub or game code:
1. Sets `PI_CART_ADDR_REG` = desired RAMROM address (within `0x10000000–0x10FFFFFF`).
2. Sets `PI_DRAM_ADDR_REG` = destination RDRAM address.
3. Writes transfer length to `PI_WR_LEN_REG` → triggers DMA.
4. Polls `PI_STATUS_REG` for `DMA_BUSY` to clear, or waits for `MI_INTR_PI`.
5. RDRAM now holds the data; CPU and RSP execute from RDRAM normally.

### 4.3 N64 Direct CPU Read from RAMROM

Valid but slow (subject to PI bus timing). Typically used only for short register probes or
in early boot before DMA is configured. Games copy to RDRAM before executing.

---

## 5. Emulation IPC Bridge Design

In our emulation, IRIS (Indy) and gopher64 (N64) are separate OS processes. The RAMROM SRAM
is modeled as a **POSIX shared memory region** that both sides can read and write directly.

We target **SDK 2.0I / hardware rev 2** — meaning the RDB port is the primary communication
channel. The legacy GIO interrupt / cart interrupt registers are implemented for completeness but
not used by current SDK code.

### 5.1 Shared Memory Layout (`/iris_n64_bridge`)

The shm region begins with two embedded `raw_sync` Events (auto-reset, semaphore semantics) in a
fixed 256-byte area, followed by the header and the 16 MB RAMROM.

```
[0x000–0x0FF]   Event area (256 bytes)
                  EVT_H2N at offset 0    — Indy signals, N64 waits (RDB Indy→N64 + reset/NMI)
                  EVT_N2H after h2n      — N64 signals, Indy waits (RDB N64→Indy)

[0x100–0x11F]   ShmHeader (current, to be replaced by ring buffers — see below)
                  magic:             u32  0x4E36344D ("N64M")
                  version:           u32  1
                  cart_int_payload:  u32  (legacy 6-bit, Indy→N64)
                  gio_int_payload:   u32  (legacy 6-bit, N64→Indy)
                  gio_sync:          u32  (legacy bits[5:0])
                  n64_reset:         u32  (1 = reset asserted)
                  n64_nmi:           u32  (1 = NMI pending)
                  page_select:       u32  (current 1 MB page, 0–15)
                  cart_int_seq:      u32  (sequence counter, legacy)

[0x120–0x11FFFF] RAMROM (16 MB)
```

**Planned redesign**: Replace the flat header fields with two SPSC ring buffers, one per
direction, each carrying typed 32-bit entries that mirror the real `rdbPacket` format:

```
[31:24]  type     — mirrors RDB_TYPE_* (+ emulation-specific types for reset/NMI)
[23:0]   payload  — 24 bits of data (matches rdbPacket length+buf fields)
```

Additional emulation-specific types (no hardware equivalent):
- `RESET_ASSERT` / `RESET_DEASSERT` — replace the `n64_reset` field
- `NMI` — replaces the `n64_nmi` field

Ring depth 16 entries; head/tail as u32 counters (wrapping mod 16). This eliminates all
sequencing/overwrite races inherent in the flat field approach.

### 5.2 Events

| Event    | Direction  | Fires on |
| :---     | :---       | :--- |
| `EVT_H2N` | Indy → N64 | Indy writes an RDB packet (data ready); reset/NMI state change |
| `EVT_N2H` | N64 → Indy | N64 writes an RDB packet |

These are `raw_sync` auto-reset events (one wake per signal), embedded in the shm region so both
processes can attach to them without a separate named semaphore.

### 5.3 No Dual-Port Arbitration

Matching real hardware: we make no attempt to arbitrate simultaneous Indy and N64 access to the
`ramrom` region. Software on both sides must coordinate RAMROM access via RDB messages
(`HtoG_REQ_RAMROM` / `GtoH_RAMROM` / `HtoG_FREE_RAMROM`) exactly as real SDK code does.
