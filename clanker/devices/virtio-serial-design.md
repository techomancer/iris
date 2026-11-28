# virtio-serial: one core, a transport per bus (design and gotchas)

Status: implemented 2026-11-01 for IP22/IP24/IP28 (mmio, GIO slot) and IP30
(pci, Bridge slot 7 on INTB; slot 4 until 2026-11-16). IP32 transport
exists, not wired. User docs: `docs/dev/virtio-serial.md`. Since 2026-11-12 also on the Tezro (IP53: IO9
PIC bus 1, device 1, `fbcbedde`; the `slot` setting does not apply there).

- **Octane interrupt pins.** Bridge pin 4 is the IOC3's Super I/O interrupt,
  so a function at slot 4 on INTA shares it. Linux's Bridge driver gives each
  (slot, pin) its own IRQ and points the pin at whichever it set up last, so
  loading virtio_console took the serial console's interrupts and the console
  died; OpenBSD's `ioc` takes the SuperIO pin from the first empty slot. Pins
  6 and 7 are the power switch and AC fail; free are pin 3 (the RAD raises
  none) and pin 5. virtio-serial is slot 7 on INTB (7 ^ 4 = 3), virtio-9p
  slot 5 on INTA.

## Layers (`src/dev/virtio/`)

- `queue.rs` split virtqueue over `GuestMemory`; `mod.rs` `VirtioCore`
  (status/feature negotiation, queues, ISR, service thread); `console.rs`
  device; `mmio.rs` / `pci.rs` transports; `backend.rs` host ends.
- `GuestMemory` is the only thing a machine implements for DMA:
  `BusMemory(Arc<Physical>)` on IP22, `PciMemory(Arc<dyn PciDma>)` on PCI.

## Rules that cost time

- **Servicing is off the CPU thread.** A doorbell sets a bit and wakes the
  `virtio` thread (`VirtioCore::start`); `notify` only services inline when no
  thread runs (tests). Backend writes (sockets) can block briefly, so they must
  not run in the CPU's store handler.
- **Level IRQ under `irq_lock`.** The line level is recomputed from `isr`
  under a dedicated lock so two threads cannot leave a stale level. Never call
  `IrqLine::set` while holding the core's `inner` lock.
- **Byte order.** Rings and config are LE (virtio 1.x). `GuestMemory` carries
  memory-order bytes (`physical::read_bytes/write_bytes`), so the device does
  `from_le_bytes` and never swaps words. MMIO registers are numbers (`lw` of
  magic = `0x74726976`); MMIO config space is bytes, wide reads big-endian.
  PCI BAR accesses arrive in device order (the bridge did the lanes).
- **TX used length is 0.** `used.len` counts bytes the device wrote into
  *writable* buffers; a transmit buffer is read-only, so it is 0. A test that
  expects the byte count there is wrong.
- **Queue count vs features.** `num_queues` is fixed at the maximum (2 + 2 per
  port); queues beyond what was negotiated are inert (`Console::role` returns
  `None`). The notified mask is a `u64`, hence at most 31 ports.
- **PCI capability list is read-only** (`PciConfig::with_capabilities`);
  `status` bit 4 is set when dword 13 is non-zero.
- **Slot clashes** are validated in `MachineConfig::validate_virtio_serial`
  and re-checked in `Machine::new` (`check_virtio_slot_free`).

## Open

- **IP32:** needs the CRIME input a third MACE PCI slot drives. Add to
  `crime_input` in `dev/mace_pci.rs`, attach in `ip32.rs`, drop the O2 error in
  `validate_virtio_serial`.
- **Snapshots:** `VirtioCore` has no `Saveable`. The Octane refuses to save or
  load a snapshot with virtio-serial fitted (`084ed672`): the Bridge would
  restore the function's enabled PCI header over a device the load reset.
- **IP22 interrupt:** Indy uses `GioExp0/1`; Indigo2/IP28 use the
  `GIO_SLOT_MAP` `retrace` entry (`GioS0/S1Graphics`). Untested with a guest.
