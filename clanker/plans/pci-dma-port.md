# PCI bus masters: one DMA trait, shared on the O2, per device on the Octane

Status: plan, 2026-10-26. Step 1 done (the trait takes 64-bit bus addresses
and returns a status; the O2 is its first user). Steps 2-6 open.

Scope: how a PCI function reaches host memory and raises interrupts
(`src/pci.rs`: `PciBusDevice`, `PciDma`), the O2's MACE bridge
(`src/dev/mace_pci.rs`) and the Octane's Bridge/Xbridge to come. Sources:
`docs/Octane/SGI-Bridge_ASIC_Specification-*.pdf` (chapters 3.3, 3.8, 3.9,
4, 11, appendix A) and `SGI-Xbridge_ASIC_Specification-*.pdf` (1.5, 7.3,
appendix C). Both are scans: read them with the PDF reader, not
`pdftotext`. Bridge document page N is PDF page N+14; Xbridge pages match.

## 1. The rule

A PCI function only ever calls the `PciDma` it was handed by
`connect_dma`, and drives the `IrqLine` it was handed by
`connect_interrupt`. It never learns whether the DMA object is its own or
shared, or how its interrupt reaches the CPU. The bridge decides both when
it attaches the function.

- **O2:** one shared `HostMemory` for every PCI function (MACE's views are
  fixed), one `IrqLine` per slot pin (already so).
- **Octane:** one port per device holding `(bridge, device number)`, because
  the Bridge's translation depends on which device is mastering.

## 2. What the Octane's Bridge does

### DMA (PCI bus address to Crosstalk)

The PCI memory space is cut into fixed, global regions:

| PCI address | Bridge behaviour |
|---|---|
| 0 - 0x3fff_ffff | local: no response, except 0x3fff_0000-0x3fff_ffff, which flushes the mastering device's buffers |
| 0x4000_0000 - 0x7fff_ffff | page mapped: index into a global ATE table (Bridge: 128 internal entries, then up to 64K in external SSRAM, 4K or 16K pages; Xbridge: 512 entries, 8 MB, the rest of the GB direct mapped) |
| 0x8000_0000 - 0xffff_ffff | 32-bit direct: one global register (`DIR_OFF` 16:0, `DIR_ADD512` 17, `DIR_W_ID` 23:20; the CPU reads it as the word at widget offset 0x84) places the 2 GB in Crosstalk space |
| 0x1_0000_0000 - 0x0fff_ffff_ffff_ffff | unused, no response |
| 0x1000_0000_0000_0000 and up | 64-bit direct (DAC masters only): bits 47:0 pass through; 63:60 target widget, 59 prefetch, 58 precise, 57 virtual request, 56 barrier, 55:48 remote map (Bridge) / 55 swap (Xbridge) |

There is no region reserved for 32-bit devices: a single-address-cycle
master simply cannot leave the low 4 GB, so it reaches memory through the
2 GB direct window or the ATEs. `DIR_ADD512` exists because system memory
starts at 0x2000_0000.

Per device, `Device(x)` (one per device, 0..7) supplies the attributes:
`SWAP_DIRECT` 19, `SWAP_PMU` 20, `PREFETCH` 18, `PRECISE` 17, `COHERENT`
16, `BARRIER` 15, `GBR` 14, `DEV_SIZE` 22 (32/64-bit), write gathering
23/24. In 64-bit direct space some come from the address instead; in page
space from the ATE (Xbridge also takes swap from the ATE).

ATE (8 bytes): 63:48 remote map, 47:12 upper address, 11:8 target ID,
4 barrier, 3 prefetch, 2 precise, 1 coherent, 0 valid.

The device number is the bus request/grant pair that won arbitration, not
the configuration slot (usually the same). The translation is therefore a
function of (bus address, global bridge state, mastering device), and the
result is (target widget, 48-bit Crosstalk address, attributes). For main
memory the target is Heart, which decodes the Crosstalk address to RAM.

### PIO (CPU to PCI)

- Widget space, per device: a window of 2 MB (devices 0-1) or 1 MB (2-7)
  at widget offsets 0x20_0000..0xbf_ffff. `DEV_OFF` (11:0) supplies the
  upper PCI address bits, `DEV_IO_MEM` (12) memory or I/O space,
  `DEV_SWAP` (13) the byte swap.
- Global big windows: the low 1 GB of PCI memory at widget offset
  0x4000_0000 (32-bit targets) and again at 0x8000_0000 (64-bit targets);
  PCI I/O space from widget offset 4 GB. Swapped by the global `MEM_SWAP`
  / `IO_SWAP` control bits.
- Configuration space at widget offset 0x2_0000 + device * 0x1000, type 1
  at 0x2_8000.
- Heart gives the CPU each widget through a small (16 MB at
  0x1000_0000 + id * 16 MB), medium (2 GB at 0x8_0000_0000 + id * 2 GB) and
  large (64 GB at id * 64 GB) window. The large windows end at
  0xff_ffff_ffff: the Octane fits the 40-bit `SparseDeviceMap` exactly.

### Interrupts

Eight pins, INT_N[7:0], wired on the board; several devices may share one,
one device may drive several. On each pin change the Bridge sends a
Crosstalk write to the address and bit in `Interrupt(x) Host` (a set; a
clear only if `INT_MODE` asks). `INT_DEV` names the device a pin flushes
buffers for; it does not route. Error interrupts use `HOST_ERR_FIELD`.
So the chain is device, pin, host bit: the port's interrupt half drives the
pins the board wires to that device.

### Byte swap (appendix A)

"Swap" keeps byte addresses (address invariant): in a byte-addressed model
it is a plain copy. "No swap" keeps byte lanes: T5 byte 0 lands on PCI
byte 3 of a 32-bit bus (`^3`), presumably `^7` on a 64-bit one. This is the
same `^3`/`^7` question `docs/o2/aic7880.md` §12.1 leaves open for MACE.

## 3. What the O2's MACE does

Fixed views, no per-device state (`mace_pci.rs`): bus addresses 0-1 GB the
swapped view, 1-2 GB the native view, both onto CRIME memory; 2 GB and up
not host memory. Today both views deliver bytes in address order and the
AIC-7880 model compensates per driver (`dma_read32` vs `dma_read32_le`,
`complete_him` at `p ^ 3`); see `docs/o2/aic7880.md` §3.3. MACE's own
engines (MEC, audio, serial DMA) address CRIME physical memory, not PCI
space, and today hold `O2Memory` directly.

## 4. The trait

```rust
pub enum PciDmaStatus { Ok, MasterAbort }

pub trait PciDma: Send + Sync {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus;
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus;
}
```

- `addr` is a PCI bus address, 64 bits for DAC masters. A 32-bit master
  passes `u32 as u64`.
- `MasterAbort`: some byte had no target (reads 0, writes dropped), as
  today's `false`. On the Octane it also latches the bridge's error state
  and may raise its error interrupt; on the O2 nothing more happens yet.
  Room for `TargetAbort` / parity later.
- Slices stay: a bridge and memory model can copy a run instead of a byte.

Interrupts stay `connect_interrupt(IrqLine)`. On the Octane the line a
device gets drives a pin input of the Bridge's pin controller, which ORs
its sources and sends set/clear packets to Heart.

## 5. Steps

1. **Done.** `PciDma` takes `u64` addresses and returns `PciDmaStatus`;
   `HostMemory` and the AIC-7880 follow. No behaviour change.
2. **Done (2026-10-26).** `O2Memory` has no lock: it is shared as a plain
   `Arc`, byte ops take `&self`, the bank decode is atomics, and only
   `set_ctrl` rebuilds serialise. `HostMemory` decodes once per 4 KiB page
   and copies the page with one `write`/`read_into` (one jitv2 generation
   bump per page, not per byte).
3. MACE's internal engines (MEC, audio, serial DMA) take a shared
   `Arc<dyn PciDma>`-like memory view onto CRIME physical addresses instead
   of `Arc<O2Memory>` (a second fixed view, not the PCI one; a
   separate trait or the same one with CRIME addresses, decide then).
4. The O2's views as the spec has them: settle `^3` vs `^7` and why a
   swapped native view broke OpenBSD, then move the AIC's per-driver word
   order into `HostMemory` if the bridge really does it.
5. Octane: `BridgePort { bridge: Arc<Bridge>, dev: u8 }` implementing
   `PciDma` (regions, `DIR_*`, ATEs, `Device(x)` swap) and handing the
   translated `(widget, xtalk addr)` to a Crosstalk target (Heart for RAM).
   `Bridge::attach(dev, function)` calls `connect_dma(port)` and
   `connect_interrupt(pin line)` per the board's wiring.
6. Bridge PIO as a `BusDevice` in the decode table behind Heart's widget
   windows: per-device windows, big windows, configuration space.

## 6. Open

- No Heart specification in `docs/Octane`: Heart's Crosstalk-to-memory
  decode and its interrupt register are taken from appendix C of the Xbridge
  spec and the Octane technical report until one turns up.
- Which Octane board revision has Bridge and which Xbridge, and whether the
  model needs both.
- Prefetch, precise, write gathering and read response buffers matter only
  for timing and for software that reads the buffer registers; model the
  registers, not the buffering, unless a driver depends on it.
