# Ultra64 Development Board — Hardware Protocol Reference

## Hardware overview

The SGI Indy N64 development board is a double-wide GIO64 card. It connects
the Indy workstation (host) to the N64 cartridge port via a 16 MB SRAM called
RAMROM.

- **Indy → N64 data path**: Indy writes game code/assets into RAMROM via the
  GIO bus. The N64 sees RAMROM as cartridge ROM at PI bus `0x10000000–0x10FFFFFF`.
- **N64 → Indy data path**: via the RDB (Remote Debug) port (see below).
- **N64 RDRAM** (4–8 MB) is private to the N64; the Indy cannot access it directly.

## GIO register map (Indy side, base `0x1F400000`)

| Offset  | Name              | Dir | Notes                                            |
|---------|-------------------|-----|--------------------------------------------------|
| `+0x000`| `PROD_ID`         | R   | `0x15` + bit 30 = `GIO_RDB_READ_INTR`, bit 31 = `GIO_RDB_WRITE_INTR` |
| `+0x400`| `RESET_CTRL`      | W   | bit 1 = `_RESET` (hard), bit 2 = `_NMI`         |
| `+0x800`| `CART_INT`        | R/W | 6-bit payload; write raises N64 `INT1` (IP4)    |
| `+0xA00`| `DRAM_PAGE`       | R/W | bits[23:20] = page select (0–15 × 1 MB windows) |
| `+0xC00`| `GIO_INT_ACK`     | R   | 5-bit N64→Indy interrupt payload; read clears interrupt |
| `+0xE00`| `GIO_SYNC`        | R   | 5-bit polling value from N64, no interrupt       |

## RDB port (Indy side, base `0x1F480000`)

The RDB (Remote Debug) port is a single bidirectional 32-bit register used to
exchange structured packets between the N64 and the Indy.

| Offset | Name                   | Dir      | Notes                                           |
|--------|------------------------|----------|-------------------------------------------------|
| `+0x0` | `GIO_RDB_BASE_REG`     | R/W      | R = read N64's last write; W = send packet to N64 |
| `+0x8` | `GIO_RDB_WRITE_INTR_REG` | W (clr)| Write 0 to clear `PROD_ID` bit 31 and ACK "N64 wrote" |
| `+0xC` | `GIO_RDB_READ_INTR_REG`  | W (clr)| Write 0 to clear `PROD_ID` bit 30 and ACK "Indy read" |

## N64-side control registers (N64 PI bus, via osMapTLBRdb)

`osInitRdb` calls `osMapTLBRdb` which maps virtual `0xC0000000` →
physical `0x80000000` (uncached). Offsets from that virtual base:

| Offset | Name                   | Dir      | Notes                                           |
|--------|------------------------|----------|-------------------------------------------------|
| `+0x0` | `GIO_RDB_BASE_REG`     | R/W      | W = send packet to Indy; R = read Indy's last write |
| `+0x8` | `GIO_RDB_WRITE_INTR_REG` | W (clr)| Write 0 to clear N64 CAUSE IP7 ("Indy wrote")  |
| `+0xC` | `GIO_RDB_READ_INTR_REG`  | W (clr)| Write 0 to clear N64 CAUSE IP6 ("Indy read/ACK'd") |

The N64 kernel driver also uses the GIO interrupt line for general signalling:

| N64 phys addr   | Dir | Notes                                              |
|-----------------|-----|----------------------------------------------------|
| `0x18000000`    | W   | Write 5-bit value → raises Indy `Gp0` interrupt   |
| `0x18000400`    | W   | Write 5-bit value → Indy `GIO_SYNC` polling reg   |
| `0x18000800`    | R   | Read → ACK Indy→N64 `INT1`, clears N64 `CAUSE.IP4`|

## RDB packet format

One u32 per packet:

```
bits[31:26]  type    (6 bits)  — direction + purpose
bits[25:18]  length  (8 bits)  — number of valid data bytes (0–3)
bits[17:0]   data   (18 bits)  — up to 3 bytes, MSB first: [17:10] [9:2] [1:0+padding]
```

The data field packs up to 3 ASCII bytes:
- byte 0 = `(data >> 16) & 0xFF`
- byte 1 = `(data >>  8) & 0xFF`
- byte 2 = `data & 0xFF`

Selected type values (from `PR/rdb.h`):

| Type | Name                  | Direction  |
|------|-----------------------|------------|
| 1    | `GtoH_PRINT`          | N64 → Indy |
| 2    | `GtoH_FAULT`          | N64 → Indy |
| 9    | `GtoH_RAMROM`         | N64 → Indy |
| 13   | `HtoG_LOG_DONE`       | Indy → N64 |
| 14   | `HtoG_DEBUG`          | Indy → N64 |
| 16   | `HtoG_DATA`           | Indy → N64 |
| 18   | `HtoG_REQ_RAMROM`     | Indy → N64 |
| 19   | `HtoG_FREE_RAMROM`    | Indy → N64 |

## RDB roundtrip protocol

### N64 → Indy (e.g. osSyncPrintf)

```
N64 game                          IRIX kernel (u64 driver)
--------                          ------------------------
1. *GIO_RDB_BASE_REG = pkt        → hardware latches pkt, asserts PROD_ID bit 31
                                    (GIO_RDB_WRITE_INTR_BIT), fires GIO interrupt

2.                                2. u64_giointr() reads PROD_ID:
                                     bit 31 set → GIO_RDB_WRITE_INTR_BIT
                                     *GIO_RDB_WRITE_INTR_REG = 0   (clears bit 31)
                                     Handle_RDB_Incoming(*GIO_RDB_BASE_REG)
                                     → dispatches on pkt.type

3. [N64 blocked in write_buf_sema, waiting for Indy to ACK]

4.                                4. After handling the packet, if more data queued,
                                     Indy needs to ACK to unblock N64:
                                     *GIO_RDB_READ_INTR_REG = 0
                                     → hardware asserts CAUSE IP6 on N64

5. N64 IP6 interrupt fires          5.
   *GIO_RDB_READ_INTR_REG = 0       (clears IP6)
   send_write_buffer():
     *GIO_RDB_BASE_REG = next_pkt   → repeat from step 1

   If write_buf empty:
     vsema(&write_buf_sema)          → unblocks next u64_internal_write call
```

The write semaphore (`write_buf_sema`) ensures only one `osSyncPrintf` call is
in-flight at a time. The first packet of each call is sent directly; remaining
packets are queued in `write_buf` and sent one per Indy ACK.

### Indy → N64 (e.g. RAMROM request response)

```
IRIX kernel                       N64 game
-----------                       --------
1. *GIO_RDB_BASE_REG = pkt        → hardware asserts CAUSE IP7 on N64
                                    (GIO_RDB_WRITE_INTR_BIT in N64 view)

2.                                2. N64 IP7 handler:
                                     *GIO_RDB_WRITE_INTR_REG = 0  (clears IP7)
                                     pkt = *GIO_RDB_BASE_REG       (read packet)
                                     → hardware asserts PROD_ID bit 30 on Indy
                                       (GIO_RDB_READ_INTR_BIT)
                                     dispatch on pkt.type

3. u64_giointr() sees bit 30:      3.
   *GIO_RDB_READ_INTR_REG = 0       (clears bit 30)
   send_write_buffer()               → sends next queued Indy→N64 packet
```

### Interrupt summary

| Event                      | Signal on N64          | Signal on Indy         |
|----------------------------|------------------------|------------------------|
| N64 wrote to RDB port      | —                      | `PROD_ID` bit 31, `Gp0` interrupt |
| Indy ACK'd N64's write     | CAUSE IP6              | —                      |
| Indy wrote to RDB port     | CAUSE IP7              | —                      |
| N64 read Indy's write      | —                      | `PROD_ID` bit 30, `Gp0` interrupt |

Both sides use `Gp0` (GIO interrupt 0) as the single interrupt line. The
`PROD_ID` register on the Indy encodes which sub-event fired so the ISR
can dispatch correctly.

## Reset and NMI behavior

### Hard reset (`_U64_RESET_CONTROL_RESET = 0x2`)

Used by `gload` to load a new ROM and cold-boot the N64.

- Deasserts N64 CPU clock and RDRAM refresh. RDRAM contents are lost (DRAM
  without refresh decays within milliseconds).
- PIF performs a full cold-boot sequence (CIC challenge from scratch).
- IPL1 → IPL2 (PIF ROM) → IPL3 (from RAMROM[0x40]) → game entry at `0x80000000`.

### NMI (`_U64_RESET_CONTROL_NMI = 0x4`)

Used by `greset` to reboot the N64 without reloading RAMROM.

- PIF asserts the NMI pin directly to the R4300i. Non-maskable; `IE` bit ignored.
- CPU jumps immediately to `0xBFC00000` with `Status.SR=1`, `Status.ERL=1`,
  `Status.BEV=1`; `ErrorEPC` = pre-NMI PC.
- PIF starts a ~0.5 s timer; if the NMI handler doesn't complete, hard reset fires.
- RDRAM contents are **preserved** across NMI (refresh continues).
- PIF re-runs the CIC challenge for the warm-reset boot word.
- IPL3 sees `reset_type=1` in the boot word, skips RDRAM init, jumps to `0x80000000`.

## RAMROM endianness

RAMROM bytes are stored in **big-endian (N64 file) order** — matching `.z64`
ROM format. When the Indy writes a 32-bit word via GIO bus the MIPS CPU provides
it in host byte order; it must be stored as `to_be()`. Reads must `from_be()`
before returning to the MIPS CPU.

## CIC challenge on reset

IPL2 (running in SP IMEM at `0xA4001000`) spins polling `pif.ram[0x3F]` bit 7,
which is set by `process_ram` when it sees command `0x20` in that byte.

- Cold boot: `pif::init` → `write_mem` → SI event → `process_ram` → `ram[0x3F] = 0x80`.
- Warm reset (NMI): the NMI handler must explicitly write `0x20` to `ram[0x3F]`
  and schedule the SI event, otherwise IPL2 spins forever.
