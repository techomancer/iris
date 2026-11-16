# Multi-node machines: design note

How iris builds SGI's node machines, and what is left to do. Read this before
changing the IP27 or IP35 code. The traps that cost real boots are in
[sn-node-traps.md](../devices/sn-node-traps.md).

## What a node machine is

SGI's NUMA machines are built from nodes. A node is a node-controller chip, its
RAM, its processors and an NI port to one other node. Every processor reaches
every node's memory and registers through a node number in the physical
address.

- Origin 200, Origin 2000 and Onyx2 use the Hub chip on node board IP27.
- Origin 3000/3200, Onyx 3000, Fuel and Tezro use the Bedrock chip on boards
  IP35, IP34 and IP53.

On the Indy, O2 and Octane one memory controller (MC, CRIME, HEART) owns the
machine's `Physical` decode table and rewrites it when the guest reprograms
the controller. The node machines follow the same shape: each chip owns its
node's memory and remaps it through the vocabulary the other controllers use.

## The levels

Each module's doc says which level it is.

| Level | Name | Examples |
|---|---|---|
| Product | what the user picks | Origin 200, Origin 2000, Onyx2, Fuel, Tezro |
| Board module | `ipNN_machine`, one board | `ip29_machine` (Origin 200), `ip27_machine` (Origin 2000, Onyx2), `ip34_machine` (Fuel), `ip35_machine` (Origin 3000, Onyx 3000), `ip53_machine` (Tezro) |
| Family | `ip27`, `ip35`: code shared by a family's boards | the node bus, node construction |
| Shared | `sn`: code shared by both families | `sn::Node`, `sn::Machine` |
| Chip | `hub`, `bedrock` | register files, the chips' units |

Imports run board, family, `sn`, chip. Families never name their boards.
Origin and Onyx versions of one board differ only in attached graphics and
midplane, so they share a board module.

`ipNN` is the family's shared code; `ipNN_machine` is one board, as `ip30` and
`ip30_machine` already are. The machine-level types are `sn::Machine<C, F>`
and one data-form `sn::BoardIo`.

## Names

- **`sn`** is IRIX's own name for the scalable-node layer. SN0 and SN1 are
  SGI's generation names, but iris names code by board and family, so no
  `SN0`/`SN1` identifiers remain except quoted firmware labels.
- **Chips** stay `hub` and `bedrock`. IRIX calls the Bedrock a "hub" too; the
  `bedrock` module doc says so. Chip-specific names carry `hub_` or
  `bedrock_`, platform-specific names `ip27_` or `ip35_`; the shared address
  model is unprefixed. The node shift is per chip: `hub::NODE_SHIFT`,
  `bedrock::NODE_SHIFT`.
- **Verbs.** `build_` makes and returns a subassembly (`build_crossbow`,
  `build_baseio`, a board's `build_io`). `attach_` adds a device to something
  that exists (`attach_vpro`, `attach_ir3`, `attach_mediaio`, `attach_flash`).
  `connect_` joins two existing things (`connect_peer`). There is no `wire_`,
  `fit_` or `plug`.
- **Widgets.** XIO devices are widgets: `Xbow::attach_widget(widget, device)`,
  and the PXBOW takes `attach_widget` too.
- **Presence words.** Memory in a bank is *installed*. A processor, board or
  brick in a slot is *populated*. A device or card on a bus, widget or slot is
  *attached*. *Present* is only for what a register reports as PRESENT. Nothing
  there is *empty* (`NoMemory` for a bank slot).
- **SGI terms only** for the interconnect: NI, II, PI, MD, LB, XB, link, port.
  Not "face", "cable", "fabric", "stretch" or "piece". A name says what a thing
  is, and no tuple stands in for a named type.
- CPU modules are named by part and system clock (`R14000_500_SYSCLK_200`).

## How a node works

**One `Physical` per node.** The chip owns it and rewrites it when the guest
changes the node number, bank sizes or alias registers. The remap runs inside
one `World::stop_others`; the leader lock is not re-entrant, so remaps never
nest.

**Bank placement.** `BankMapping { base, end, bank, src }` says bank `bank`
answers `base..end` and `base` reaches the bank as address `src`.
`Physical::map_banks(old, new)` swaps the mappings a controller placed last
time for new ones; `Physical` keeps no record, so the controller derives `old`
from the register values it is replacing. HEART, CRIME and the SN chips all
use it. `map_banks_of` applies the far node's lists in the same call, so a
range moving between own and far memory is never briefly the default. IP22
shares the `Entry` vocabulary but not `map_banks`: its mirrored window and
read-zero slots do not fit linear mappings.

**The node layer (`sn`).**

- `UncachedSpace` is the R10000 uncached attribute (address bits 58:57),
  decoded once at the processor bus entry. `UncachedBus` holds four views
  indexed by it; this is the processor's one `dyn` hop.
- `Node<C>` is the node as the far NI sees it: number, cached alias, published
  banks, `Physical`, boot flash and the chip as a `Weak`. The far node holds it
  strongly without holding the chip, so there is no `Arc` cycle.
- `NodeDecode<C>` is the node-field decode above `Physical`. It holds the own
  `Arc<Node<C>>`, the peer (`OnceLock`) and the remap.
- `NodeBoard<C, F>` is one node board in either family: chip, flash, uncached
  bus, first CPU. Family builders are inherent impls (`NodeBoard<Hub, NodeFlash>`
  in `ip27.rs`, `NodeBoard<Bedrock, Flash>` in `ip35/node.rs`). Nothing generic
  calls into `C` or `F`, so there is no trait.
- `connect_peer` joins two nodes through the NI port: directed, once, before
  the machine runs. A single-node machine never touches the peer path.
- `NodeQualifiedView<C, const IO: bool>` serves node-qualified HSPEC and I/O
  references. The space is a constant, so there is no per-access branch.

**The machine.** `sn::Machine<C, F>` owns the nodes, one L1 per brick on IP35,
one `TimerManager`, and a `BoardIo` (the product's devices as data, started in
order and stopped in reverse). `workstation::build` does the one dispatch on
the product profile and passes the board's two functions to the machine.

**Products.** A product is `node_specs(cfg)` (a `NodeSpec` per node: CPU module,
memory, L1, flash record) plus a short `build_io(cfg, nodes, ..)` over shared
part functions (`attach_ioc3`, `attach_usb`, `attach_vpro`, `build_baseio`).
Tables exist only where boards differ in values. There are no board traits and
no per-product unit structs.

**Hot paths.** The decode is split once, at chip entry. Only node-machine
processors split JIT code pages by cached and uncached fetch. A register access
must cost no more than on the single-controller machines; the bench counts
host instructions per access (`(I(2N) - I(N)) / N`) and holds steady on a
loaded host. Take timing from Cargo's `release` profile only; a thin-LTO build
misleads.

**Generated registers.** The Hub and Bedrock register files are copies of
generated files from a separate register-definition project, refreshed by a
plain copy and never edited here.

## Tests

- A test that needs only a node builds it with the function the machine calls
  (`NodeBoard::<Bedrock, Flash>::build`).
- A test that needs a product's devices builds the whole machine with the one
  generic `with_machine` in `sn/tests.rs` (headless, network off, under
  `big_stack`) and reaches devices the way a guest does, through
  `machine.nodes[0].bus`. Host-side `#[cfg(test)]` handles exist only for
  ports with no guest path.
- Each IP35 product test asserts that its flash record's mode word matches its
  `CpuModule`, which the PROM's early check also requires.
- One test per behaviour where IP27 and IP35 share it; family tests only where
  the hardware differs.

## Known limitations and approximations

- A compatibility-segment (KSEG1) uncached reference is attribute 3
  (memory) on hardware. iris sends it to the attribute 0 decode. The two agree
  wherever the model answers, because the Hub-special space repeats node memory
  and the boot flash where the reset vector and PROM references land.
- No Hub specification is public. Most Hub register behaviour is a software
  expectation (what the PROM and IRIX do) or an approximation, and each Hub
  module doc names its own. Failing an access to a node number no reachable
  Hub holds, instead of timing out, is an approximation; a processor rendezvous
  hold is not modelled.
- The Bedrock's local register window shows PI1's registers to a processor on
  PI1 when processors sit on both PIs. That the chip answers so is an
  approximation pending a hardware read.
- A 512 MiB bank 0 covers the boot flash's reset address. Cached references
  reach the bank; the uncached reset fetch reaches the flash through the
  Hub-special view. That hardware does the same is inferred.
- The low 128 KiB of the Hub-special memory view is not private to each
  processor of a pair.
- Reset values of the node number, region present and the Hub memory
  configuration are the model's own: a value read from a booted machine is what
  firmware left there.
- The L1's wiring to the Bedrock's UART interrupt and the processor register
  page is inferred from code, not read from a document.
- Host address space for node memory is reserved with an inaccessible mapping
  (`mmap` `PROT_NONE`/`MAP_NORESERVE`, a `VirtualAlloc2` placeholder on
  Windows). Only Linux has been run; macOS and Windows follow from reading the
  host code.
- Every machine stops devices in the reverse of start order. The Tezro now
  stops its V12 graphics last instead of second.

## Follow-ups

- **Interrupt delivery.** The path from widgets into the node chips
  (`PacketSender`, `Bedrock::interrupt_sender`, inbound packets, the Odyssey
  and PIC paths) is more complicated than it needs to be. Simplify it.
- **Clippy.** A clippy pass over files outside the node-machine code.
- **`Workstation` defaults.** The Indy, O2 and Octane still carry boilerplate
  `Workstation` bodies; give them defaults like the node machines, and make
  `ModuleController` an enum.
- **NIC serial marker.** Ethernet serial values are unchanged from before the
  cleanup. They need a neutral marker built on the machine identity.
- **I-brick FC head-F reset.** The I-brick's reset and head-F reset both skip
  the Fibre Channel card (slot 1, 64-bit DMA) while the Fuel resets its SCSI.
  Check that this is right and that both paths agree.
- **Intermittent IOC3 miss.** The Origin 3200 PROM once failed to find the
  I-brick's IOC3, about once in 14 boots. Not reproduced or explained.
- **Fit cleanup elsewhere.** "fit"/"fitted" wording remains in code outside the
  node machines; the same word rules apply.
