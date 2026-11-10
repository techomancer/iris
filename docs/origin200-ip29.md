# SGI Origin 200 (IP29 System Board, IP27 software platform)

The Origin 200 is the IP29 System Board on the IP27 software platform: one
Hub node, an R10000 or R12000, a Bridge joined straight to the Hub's XIO port
(no Crossbow), two ISP1040 SCSI controllers and an IOC3. The machine runs
headless on the IOC3's first serial port.

## Configuration

Copy [`origin200.toml.example`](../origin200.toml.example) to `origin200.toml`,
provide the PROM and disk named there, and run `iris --config origin200.toml`.
The sample selects `machine.profile = "origin200_ip29"`.

- `machine.cpu` is `r10000_180` (180 MHz, 1 MiB secondary cache, the default
  module; `r10000` names it) or `r12000_270` (270 MHz, 4 MiB; `r12000` names
  it). One processor is built. The R10000 module runs on the Octane
  R10000's cache type (1 MiB, 128-byte lines); the R12000 module has a 4 MiB
  type of its own, the module's size. Both use the Octane part's line size,
  not the module's. Only the 180 MHz
  module has a one-wire record (`PIMM_1XT5_1MB`); no inventory gives the R12000
  module's, so its part is blank.
- `banks` are the four DIMM-pair banks in MB (64, 128, 256 or 512), filled
  from bank 0. A bank of 128 MB or less is one run at its 512 MiB stride; a
  larger one is four runs at 128 MiB intervals. A 512 MB bank 0 holds the boot
  flash's reset address: cached references reach the bank, and the reset
  fetch, which is uncached, reaches the flash. Memory that is not fitted
  answers every access with a bus error, logged under `hub`.
- `[origin200] flash` keeps the 1 MiB node boot flash, where the PROM stores
  its environment log. Unset, the flash starts fresh every run.
  `controller_store` does the same for the module controller's store.
- `prom` takes the stock IP27 PROM as a raw payload or SGI's distribution
  container. The flash configuration block is patched for the product and
  processor module before the PROM runs.

The PROM runs its diagnostics and reaches the System Maintenance Menu and the
Command Monitor on the first serial port, and an installed IRIX 6.5.30 disk
boots to a serial login.

## Limits

- One node and one processor. A second module and the GIGAchannel cabinet
  are not built. The [Origin 2000](origin2000-ip27.md) and
  [Onyx2](onyx2-ip27.md) desksides are separate profiles.
- Cached accesses use node-relative addresses, which is right for the node
  number zero a one-node machine has.
- The module controller's NMI output and the Hub's per-processor enable and
  NMI stores are logged, not delivered.
- Program and erase of the node flash finish inside the write that starts
  them.
- The Ethernet transceiver is a DP83840A model with a synthetic partner; the
  link comes up two seconds after the first status read.
- The Bridge answers at fifteen widget numbers, every one but the Hub's own.
  Which numbers the hardware answers is not established.
- The node keeps Hub revision 3 and the IP29 board record 030-1244-001
  revision H for the R12000 module too; real 225 and 270 MHz machines report
  Hub revision 5 and board 030-1389-001.
- Memory is the banks the configuration gives. The R12000 module's 512 MB
  total is two 256 MB banks on a real machine; here it is whatever
  `banks` says (`[256, 256]` for two, `[512]` for one), and nothing prefers
  the split.
- A 256 MB bank is refused with the 180 MHz R10000 module, as the PROM
  disables it there.
- The node flash and the uncached Hub-special views have no JIT page
  generation, so a jitv2 build does not compile code from them.
- Snapshots are not supported.

Unmodelled register accesses are visible with `log hub on` in the monitor.
