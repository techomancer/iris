# PCI passthrough: a host PCI device in an emulated slot

Status: plan, 2026-11-17. Scoping only; nothing implemented. Linux/VFIO only.

Scope: give a guest a real PCI function of the host (through Linux VFIO) in a
slot of an emulated bus, so the guest's own driver runs the real chip. Covers
what the PCI layer (`src/pci/`), the three hosts (`mace_pci`, `bridge` for
Bridge/XBridge, `pic`) and `PciToPciBridge` already give, what is missing, and
the order to build it in. Layer description: `docs/pci.md`; invariants:
`clanker/devices/pci-layer-design.md`.

## 1. Why

- Real SGI-era cards under their real IRIX drivers (QLogic FC, LSI SCSI,
  Tigon/BCM570x, OHCI, audio), with real tapes, disks and loops behind them.
  Legacy PCI cards sit behind a PCIe-to-PCI adapter on the host.
- A reference for our models: a tap on a passed-through card records every
  configuration and BAR access with the real chip's answers. That is the
  oracle `bcm5701`, `isp2200`, `ice1724`, `ohci` and `isp12160` lack today.
- A test of the hosts themselves: a real master exercises the Bridge's
  windows, ATEs and swap bits in ways our own cards never will.

**Drivers are ours to write** (decided 2026-11-17). The cards we pass
through will mostly run guest drivers we write, so a driver can be made to
fit the passthrough: word-value DMA only (no swap bit, no byte-swapping
aperture, no 64-bit device bit), 32-bit bus addresses, INTx, and an explicit
cache flush after reading code in. The limits in 3.2 to 3.5 are therefore
rules for those drivers first, and obstacles only for stock drivers.
Byte-swapping apertures stay unimplemented until something needs them.

## 2. What the layer already gives

| Need | State |
|---|---|
| A function in a slot, or behind a bridge | `PciBus::attach`; `PciToPciBridge` forwards BARs, pins `(pin + d) mod 4`, DMA is the bridge's. Cards follow the `nvme::Card` pattern (`open`, `attach(bus, device)`, optional `bridge = true`). |
| Configuration reads from the chip | `PciFunction::config_dword` override. |
| BAR access with the host's byte order already applied | `bar_read`/`bar_write` get the offset and the word the device sees. A real BAR is an mmap; the access is one volatile load or store. `ByteOrder` needs no change. |
| Interrupt pin to host output | `IrqLine` from `state().irq()`, any thread may drive it; the combiner serialises. |
| RST# | `PciFunction::reset`, also reached by secondary bus reset and the Bridge's reset pins. Maps to `VFIO_DEVICE_RESET`. |
| BAR sizes known only at run time | `PciConfig::new` takes `[Bar; 6]` by value. |

## 3. What is missing

### 3.1 Configuration writes below a dword, and outside the header

`PciFunction::config_write` is offered whole-dword writes only; a narrower
write always takes the header's path and never reaches the chip. A real
card's registers from 0x40 up (capabilities, vendor registers) need the write
as the guest made it, at its width: a read-modify-write on our side would
acknowledge write-one-to-clear bits the guest did not touch.

Add one hook, offered first for any offset at or above 0x40:
`fn config_write_raw(&self, off: usize, width: usize, val: u32) -> bool`
(default false). The header (0x00-0x3f) stays emulated: the guest's BAR
values, command and interrupt line are guest facts, not host facts. The
passthrough function mirrors only the command register's I/O, memory and
master bits to the chip, and never the BARs (VFIO keeps the host's).

Reads: vendor/device/revision/class/subsystem and the capability list come
from the chip at construction; 0x40 and up are read live. Hide MSI, MSI-X and
PCI Express capabilities by unlinking them from the list the guest sees (no
SGI host has MSI; a guest that enabled it would silence the pin).

### 3.2 DMA is pull today; an IOMMU needs push

`PciDma` is called by the function for each transfer. A real chip masters
the bus itself: the host IOMMU must already hold the translation the guest's
bridge would apply. Each host needs to *publish* its DMA view for a requester
and say when it changes:

```rust
/// One run of PCI bus addresses and the guest memory it reaches.
struct DmaRun { bus: u64, len: u64, host: *mut u8, placement: Placement }
trait PciDmaView { fn runs(&self) -> Vec<DmaRun>; fn on_change(&self, f: ...); }
// PciHost gains: fn dma_view(&self, requester: u8) -> Option<Arc<dyn PciDmaView>>
```

Sources of change per host:

| Host | View | Changes on |
|---|---|---|
| MACE | Two fixed 1 GB views of the same memory (0 and 0x4000_0000), nothing above 2 GB | CRIME bank control writes (`O2Memory::decode_gen`) |
| Bridge | 2 GB direct window at 0x8000_0000; 128 internal ATEs at 0x4000_0000 (4K or 16K pages) | Direct Map register, `Device(x)` swap and 64-bit bits, each ATE write, page size bit, HEART memory configuration |
| XBridge | As Bridge with 512 ATEs, the swapped alias at 0x6000_0000, per-ATE swap | The same |
| PIC | Per-slot, see `src/dev/pic.rs` | To be read when its turn comes |

The 64-bit direct space (bus address bits 63:60 name the widget) cannot be
mapped: host IOMMUs give 48 bits or fewer of IOVA. A guest driver that asks
for 64-bit DMA addresses will fault on the host. Report it; do not work
around it.

### 3.3 Words, not bytes: which placements map

PCI carries words; it has no byte order. A real card on a little-endian host
stores the word on AD[31:0] as a host `u32`. Guest RAM is `u32` words in
host order (guest byte `o` is at host offset `o ^ 3`, `src/ppmem/ppmem.rs`).
So the PCI word *is* the guest's memory word with no copy, exactly when the
host bridge's rule is "word values":

| Bridge placement (`src/dev/bridge.rs`) | Set by | Through an IOMMU |
|---|---|---|
| `Word`: PCI word = memory word | swap clear, 32-bit device (the reset state) | **maps directly** |
| `Identity`: byte stream | swap bit set (`Device(x)` direct/PMU swap, XBridge ATE swap or alias, bit 55) | cannot map: needs every word reversed |
| `Doubleword` | swap clear, 64-bit device bit | cannot map: halves of each doubleword trade places |

A run with a placement that cannot map is left out of the IOMMU, with one
loud message naming the device and the register that asked for it. The
transfer then faults on the host (the kernel logs an I/O page fault) instead
of silently corrupting memory.

**MACE is the open case.** The MACE specification calls the low gigabyte
byte-swapped and the next one native, which in words is byte stream at 0 and
word values at 0x4000_0000. `HostMemory` instead hands both views over in
address order, because every driver we run needed that with our emulated
cards. Either the native view really is a byte stream, or our cards carry a
compensating error. A real card settles it: if a passed-through card works
through the native view mapped as `Word`, the emulated cards are wrong, not
the spec. Do the Octane first; treat the O2 as an experiment that also fixes
`mace_pci`.

### 3.4 Writes the emulator does not see

A real master's writes bypass `bump_gen`, so jitv2 would keep compiled code
for a page a disk read has just replaced. Options, in order of preference;
measure before choosing:

1. Rely on the guest: an OS flushes the instruction cache after reading text
   in, because real DMA does not update caches either. Needs checking that
   jitv2 invalidates on the guest's `CACHE` operations and not only on
   generation counters.
2. On each interrupt from the card, bump the generation of every page that
   holds compiled code. The decoded-word snapshot compare
   (`clanker/jitv2/`, churn avoidance) makes a false alarm cheap.
3. IOMMU dirty tracking (iommufd, needs hardware support; not on Zen 1).

The interpreter with no tcache needs nothing. The O2's `dirty` page list
(`crime_mem.rs`) needs the same look.

### 3.5 Interrupts without KVM

VFIO signals INTx on an eventfd and masks the pin until told otherwise. A
thread waits on it and raises the `IrqLine`. The guest's acknowledge is not
visible, so do what QEMU does without KVM: on the next BAR access by the
guest, lower the line and unmask; if the chip still asserts, it fires again.
A slow timer does the same in case the driver acknowledges through
configuration space only. A card with no INTx pin (pin register 0) gets no
line; converting MSI to a level is not planned.

### 3.6 Snapshots

A real chip's state cannot be saved. A machine with a passthrough function
refuses `snapshot save` with a message naming it, and `PciBus::load_state`
already refuses a mismatched function set. Check `iris-ci` paths that
snapshot implicitly.

### 3.7 Limits to state in the docs, not solve

- BARs must fit the host's CPU windows: 32 MB on MACE's low window, 1 or 2 MB
  per device window and the 1 GB alias on the Bridge. `Bar::Mem64` sizes stay
  under 4 GiB and decode below 4 GiB only.
- Expansion ROMs keep reading zero.
- Guest time is not wall time; a chip's own time-outs run at real speed. A
  guest stopped in the monitor leaves the chip running.
- Every function in the host's IOMMU group goes to iris together (all cards
  behind one PCIe-to-PCI adapter).
- All guest RAM is pinned while the device is open (`RLIMIT_MEMLOCK`).

## 4. Host side

New module `src/dev/vfio/` (`#[cfg(target_os = "linux")]`, Cargo feature
`pci-passthrough`, off by default), hand-written ioctls over `libc` (already
a dependency): the type 1 container API (`VFIO_GET_API_VERSION`,
`VFIO_SET_IOMMU`, `VFIO_IOMMU_MAP_DMA`/`UNMAP_DMA`), group and device fds,
region info, configuration region `pread`/`pwrite`, BAR mmap, `VFIO_DEVICE_
SET_IRQS`, `VFIO_DEVICE_RESET`. `pci/` imports none of it (layer invariant).

Configuration, shaped like `[nvme]`:

```toml
[[pci_passthrough]]
host = "0000:06:00.0"   # the host function, already bound to vfio-pci
# board = 0             # Octane: which [[xio_board]]
# slot = 1              # PCI device number; unset: first free
# bridge = false        # behind a PciToPciBridge
# trace = false         # log every configuration and BAR access
```

O2: the option slot (device 3), exclusive with `[nvme]`. Octane: a slot of a
`pci-cage` or `pci-shoehorn`. Tezro/Fuel: later.

Host facts, this workstation (2026-11-17): Threadripper 2990WX, kernel
7.2.11, `/sys/kernel/iommu_groups` is **empty**, so the IOMMU is off (firmware
setting, then `amd_iommu=on` if the kernel still does not enable it). Nothing
can be tested until that is on. No-IOMMU VFIO mode is no substitute: the
chip would master host physical addresses. Candidates already in the box: the
second Intel I211 (04:00.0 or 06:00.0, whichever is idle) and the X-Fi
(09:00.0); neither has an IRIX driver, so they serve the BSD/Linux guests and
the plumbing. An IRIX target needs a card IRIX knows, on a PCIe-to-PCI
adapter.

## 5. Order of work

Each step lands on its own and leaves the tree working.

| Step | What | Needs hardware |
|---|---|---|
| 0 | Host: IOMMU on, pick the first device, bind it to `vfio-pci`, device node owned by the user. Record the recipe in `clanker/build/`. | yes (reboot, sudo) |
| 1 | Layer: `config_write_raw`; a capability-list filter helper on `PciConfig`; the snapshot refusal. Unit tests with `pci::testing::TestCard`. No VFIO. | no |
| 2 | `src/dev/vfio`: open, regions, configuration, BAR mmap, reset. `HostFunction: PciFunction`, the `[[pci_passthrough]]` section, `pci status` line. No interrupt, no DMA. Proof: the PROM and `hinv` see the card; `pci config` matches `lspci -x`. | yes |
| 3 | `trace = true`: the access tap, through `devlog`. Useful from here on even if nothing later lands. | yes |
| 4 | INTx: eventfd thread, lower-and-unmask on BAR access, timer fallback. Proof: a driver that polls no memory (or a card whose driver only does PIO) takes interrupts. | yes |
| 5 | `PciDmaView` on the Bridge: direct window and ATEs, `Word` runs mapped, others refused loudly; map/unmap on every change source in 3.2. The jitv2 answer from 3.4. Proof: a disk or NIC under a guest driver on the Octane. | yes |
| 6 | MACE view and the 3.3 experiment; fix `HostMemory` if the real card says so. | yes |
| 7 | XBridge and PIC views; behind `PciToPciBridge` (already the bridge's DMA, so only the requester passes through). | yes |
| 8 | Human docs: `docs/pci.md` (a passthrough section), `FEATURES.md`, `HELP.md`, `debug.md` for the monitor output. | no |

Steps 1 and 8 need no device. Steps 2 to 4 need a device but no working
DMA, so they can start on whatever card is idle.

## 6. Open questions

- Which card first, and for which guest. Decides whether step 5 is tested
  under IRIX or a BSD.
- MACE's two views in words (3.3).
- Does jitv2 invalidate on guest `CACHE` operations (3.4)?
- Does any driver of interest set a swap bit or the 64-bit device bit for its
  DMA? IRIX asks the Bridge per mapping for byte stream or word values; a
  driver that asks for a byte stream gets `Identity` and cannot be passed
  through without a copy. The tap in step 3 plus a `Device(x)` trace answers
  it per driver.
- Host adapters: a PCIe-to-PCI bridge reports its own requester ID for the
  cards behind it; one IOMMU group, one container. Not expected to matter,
  not yet seen.

## 7. Rejected

- **A shadow copy of guest RAM in byte-stream order, synchronised at BAR
  writes and interrupts.** No trap marks when a chip reads or writes memory;
  page-level two-way copies lose updates when the guest and the chip write
  the same page (descriptor rings). Would make `Identity` work, at the cost
  of correctness nobody could argue for.
- **Guest RAM in guest byte order.** Makes byte-stream DMA map and word DMA
  not; breaks the bit-container rule (`HACKING.md`) for nothing.
- **VFIO no-IOMMU mode.** Only for a card that never masters the bus.
- **Class-level passthrough instead** (host SCSI generic for tapes, a tap
  device for a NIC). Worth having for its own reasons, but it runs our chip
  model, not the real chip, and gives no reference traces.
