# SGI Octane (IP30)

The Octane is SGI's 1997 workstation: one or two R10000s on HEART (the memory
controller and interrupt hub), a Crossbow switch joining the widgets, the
onboard Bridge (XIO to PCI) carrying two QLogic ISP1040 SCSI controllers, the
IOC3 I/O chip (serial ports, keyboard and mouse, the clock, Ethernet) and RAD
audio, and a graphics board on one of the XIO ports. IRIS emulates a
single-processor Octane with a 195 MHz R10000.

With the stock IP30 PROM (the 879,312-byte image, SHA-256 `ee4722ca6f9b821f…`)
and an installed IRIX 6.5.30 disk, the machine boots to a serial login.
`hinv` reports the processor, HEART revision D, 128 MB, Crossbow 1.3, both
QL1040B controllers with the disk at unit 1 on controller 0, both IOC3 serial
ports, the parallel port, `ef0` and the RAD audio processor. With the Solid
IMPACT board fitted (the default) and a blank NVRAM, the PROM comes up on
the graphics console and IRIX starts X to the login screen and a desktop.
The guest pings the NAT gateway, fetches files from its TFTP server and
accepts telnet logins through a port forward. Sound played in IRIX (for
example `playaiff`) comes out of the host's audio device; `no_audio` turns it
off. `reboot` comes back to the login prompt. jitv2 builds run it too.

iris-gui runs the Octane as well: choose "SGI Octane (IP30)" as the platform.
The editor fixes the CPU at the R10000, offers the three IMPACT boards or no
graphics board, limits banks to 64 to 512 MB filled from bank 0, and asks for the IP30
PROM image and an optional flash file. The Indy and Indigo2 NVRAM seeding and
Ethernet-address row do not apply and are skipped.

## Configure the machine

An Octane is set up with the shared keys; `[octane]` holds what only an
Octane has.

```toml
prom  = "IP30prom.bin"        # the stock IP30 PROM (not built in)
nvram = "nvram.bin"           # the DS1685 clock's NVRAM: the PROM's settings
banks = [128]                 # MB in each HEART bank
headless = true               # no window; console=d in the NVRAM for a serial console
serial_log = "serial.log"
nat_subnet = "192.0.2.0/24"

[machine]
profile = "octane_ip30"
cpu = "r10000"

[octane]
flash = "flash-ip30.bin"      # keep the PROM's flash records between runs

[graphics]
# board = "maximpact"         # "solidimpact" (default), "highimpact", "maximpact", "v6", "v8", "v10", "v12" or "none"

[[port_forward]]
proto = "tcp"
host_port = 2323
guest_port = 23

[scsi.1]
path = "irix6530.raw"
overlay = true
```

Paths are relative to the directory iris runs in. Every key is listed in
[FEATURES.md](../FEATURES.md). `machine.cpu` must be `r10000`. `[clock]
fixed_mhz` sets CP0 Count; unset it is 97.5 MHz, half of the R10000's
195 MHz pipeline clock.

## PROM and flash

The PROM image is not built in and is not shipped with iris: `prom` must name
the stock IP30 PROM, at most 1 MiB. It is loaded into the boot flash, a 1 MiB
part with the Intel/Sharp command set on the Bridge's first chip select,
where the PROM also keeps its persistent records. `[octane] flash` names a
file holding the whole flash, so those records survive from run to run; it
is created from `prom` when missing or empty, and an existing file of
exactly 1 MiB is used as it stands (delete it after changing `prom`). A
file of any other size is refused with an error and left untouched. Without it the flash starts from
`prom` every run.

## Memory

`banks` lists the megabytes in each of HEART's four banks (each a matched
pair of DIMMs). Valid sizes are 64, 128, 256 and 512; fill the banks from the
first with no gap, 2 GB at most. The shared default `[128, 128]` is 256 MB.
The PROM sizes the banks itself; IRIX has been booted with 128 MB.

## NVRAM

The PROM keeps its environment (`console`, `AutoLoad`, boot paths) in the
NVRAM of IOC3's DS1685 clock, including its extended RAM. `nvram` names the
file it is kept in (512 bytes). A blank NVRAM makes the PROM pick the graphics
console; set `console=d` (or start from an image that has it) for the serial
console.

## Disks

The `[scsi.N]` tables list the devices on the two ISP1040s: `N` 1-7 is the
id on the internal bus (controller 0), `N` = 16 + id that id on the external
bus (controller 1), or use `controller = 1`. IRIX expects its system disk at
id 1 on controller 0. Raw images and `overlay = true` work as on the other
machines ([STORAGE.md](../STORAGE.md)). The controller's own firmware is
stored and never run; SCSI commands go to iris's shared target model. The
monitor's `scsi` and `cow` commands reach controller 0 (`scsi1`, `cow1` the
external bus).

## Console, keyboard and mouse

The console is IOC3's first serial port on TCP 127.0.0.1 (`serial_port_a`,
8880 unless set), and `serial_log` copies what it sends to a file; the second
port is `serial_port_b`. The PROM's `console` variable picks between the
serial and graphics consoles; `headless` only removes the window.

A PS/2 keyboard and mouse are always plugged into IOC3's two PS/2 ports;
the iris window (or iris-gui) and the monitor's `ps2` commands type and
move them. The PROM and IRIX run the keyboard in scan code set 3. How the
controller works and what each driver does with it is in
[Octane/ioc3-ps2.md](Octane/ioc3-ps2.md).

The front lightbar is two IOC3 general-purpose pins: pin 0 the white power
LED, pin 1 the red fault LED. The GUI's front LED shows them, green for white
and red for red.

## Graphics

The graphics board sits on widget 11: the same raster chips as the Indigo2
IP28's IMPACT boards behind the Octane's XIO host interface (the HQ4), with
interrupts and DMA through HEART. `[graphics] board` (or `--graphics`) picks
the board, as on the other machines:

| Value | The PROM's name | GEs / REs | Texture |
|---|---|---|---|
| `solidimpact` (default) | SI | 1 / 1 | none |
| `highimpact` | SI with texture option | 1 / 1 | 4 TRAMs |
| `maximpact` | MXI | 2 / 2 | 4 TRAMs per RE |
| `v6` (also `vpro:v6`) | V6 | (VPro, BUZZ A) | in its 32 MB board memory |
| `v8` (also `vpro:v8`) | V8 | (VPro, BUZZ A) | in its 128 MB board memory |
| `v10` (also `vpro:v10`) | V10 | (VPro, BUZZ B) | in its 32 MB board memory |
| `v12` (also `vpro:v12`) | V12 | (VPro, BUZZ B) | in its 128 MB board memory |

The two textured boards are new and less tested than the SI. Its screen is the iris window
and what iris-ci screenshots show; the keyboard and mouse are IOC3's PS/2
ports. `board = "none"` leaves the port empty, and the machine
then runs on its serial console with no window.

`v6`, `v8`, `v10` and `v12` are the four Octane VPro boards on the same
port: a port of the VPro (Odyssey) model in jamesbraid/qemu-sgi-origami
(`src/dev/odyssey`), with the X server and OpenGL drawn by the model's
scalar rasteriser on the board's own thread. Each answers its board's
widget revision, memory size and banks and the identity record on its
NIC (`ODY32`, `ODY128`, `ODY32VERSIONB`, `ODY128VERSIONB`). The PROM draws
its startup screen on any of them, but IRIX needs an install with the
VPro kernel driver and X server module: a disk installed with an IMPACT
board finds no screen on a VPro. SGI's upgrade kit (007-4210-003) asks
for PROM 4.0 and IRIX 6.5.10 for the V6 and V8, and PROM 4.5, IRIX 6.5.10
and Crossbow 1.4 for the V10 and V12; with a V10 or V12 the machine
presents the Octane2's Crossbow (revision code 5, 1.4 in hinv), and iris
warns at start-up when the PROM image is older than the board needs.

## Networking

The IOC3 Ethernet (an ICS1890 PHY at MII address 1) is on iris's NAT
gateway, with the usual settings: `nat_subnet`, `[nfs]`, `[[port_forward]]`,
`[network] mac` and `tftp_dir`. The gateway answers on host .1 of
`nat_subnet`, or, until the first packet routes, on whatever address in its
/24 the guest asks for: an IRIX disk configured as 192.0.2.2 with a default
route of 192.0.2.3 works with `nat_subnet = "192.0.2.0/24"` unchanged. When
`[network] mac` is unset the station address is 08:00:69:12:34:56; set your
own if more than one Octane shares a network. `[network] mode = "off"`
unplugs the Ethernet; `pcap` bridges it onto a host interface instead (see
[NETWORKING.md](../NETWORKING.md)), with no NAT services.

## Inspecting the machine

The monitor (`monitor_port`) has, besides the usual CPU and device commands:

- `ip30 memcfg | bridge | gfx | xbow | unmapped`: the memory banks as HEART
  places them, the Bridge's interrupt routing and device registers, the
  graphics widget's link and identification, the Crossbow accesses it
  refused, and physical accesses that failed (the first 64, with the PC).
- `heart status`: HEART's interrupt bits, mask, counter and bank descriptors;
  `heart unsupported`: HEART registers a guest touched that are not modelled.
- `scsi notes`: what the ISP1040 driver did that the model noted.

`IRIS_IP30_TRACE=1` prints failed accesses, unsupported HEART registers and
HEART's bank mapping as they happen.

## Threads

Serial input is the one thing that polls. IOC3 runs one receive thread per
fitted serial port, which looks for host input every 2 ms, and an NVRAM
writer. It arms its deadlines (serial receive and transmit delays, the timer
interrupt, the NVRAM flush) as `hptimer` one-shots, so they fire without the
guest touching the chip. RAD audio is paced by `hptimer` too and
has no thread of its own. Each ISP1040 runs a request-ring thread, started
with the machine and joined when it stops. The Ethernet has a receive pump
and the network backend (`ip30-eth-rx`, `ip30-eth-net`). A power-off from the
guest (the DS1685 clock powering the machine down) reaches the machine
through its event channel. `HACKING.md` lists the thread names.

## What is approximated or not modelled

- One processor only.
- HEART: the answer to the PROM's bank-size probe is an approximation; ECC
  error injection is not modelled (a mode write that asks for it is
  recorded); the 16 KiB at physical 0 are the first 16 KiB of SDRAM,
  which the processor reaches only there (an uncached access to
  0x2000_0000..+16 KiB is a bus error, as the stock IDE memory diagnostics
  expect; cached accesses are the next item). Whether bus masters reach
  that range is not known; the DMA view does.
- A cached access to 0x2000_0000..+16 KiB reaches the bytes. On an Octane2
  (HEART rev F) cached and uncached reads there both take a data bus
  error; here only uncached accesses do, because the processor's cache
  reads memory straight from the ppmem window, which maps the whole 32 MiB
  unit at 0x2000_0000. Leaving the 16 KiB out of the window would put every
  cached access in the first 64 MiB of SDRAM on the slower bus path (the
  window's fast-path flag covers 64 MiB at a time). No supported guest
  touches the range.
- The PROM's cached stores to its own data in the flash are kept in a
  per-line copy, as a write-back cache would hold them; they are never
  written back.
- Crossbow link training is not modelled: a fitted link that is reset comes
  back alive at once.
- The Bridge's processor access to its long translated window and peer DMA
  between widgets other than HEART are not modelled.
- ISP1040: the firmware never runs; some mailbox commands are accepted
  without effect and noted.
- RAD audio input records silence; only the analog output reaches the host.
- Board identity records carry made-up barcodes and serial numbers.
- No snapshots.

## Sources

The device models are ported from the author's QEMU models in
[Origami](https://github.com/jamesbraid/origami) (the
[sgi-origami branch](https://github.com/jamesbraid/qemu-sgi-origami)). The boot flash (`src/dev/sflash.rs`)
derives from MAME's
[intelfsh](https://github.com/mamedev/mame/blob/master/src/devices/machine/intelfsh.cpp).
The physical map is in [Octane/ip30-address-map.md](Octane/ip30-address-map.md);
the Bridge, Xbridge and Crossbow specifications are in [Octane/](Octane/).
