---
name: irix-kernel-debug
description: >-
  Debug the IRIX kernel and PROM running inside IRIS from the monitor: load
  kernel symbols, make boot verbose, read a panic (kernel log, exception
  frame, backtrace), put conditional breakpoints on kernel functions, and walk
  device discovery (SCSI probe to root mount) to find where the guest and the
  emulated hardware disagree. Use when IRIX panics, hangs during boot, does
  not find a device, or a driver misbehaves.
---

# Debugging the IRIX kernel from the monitor

When the guest misbehaves, the question is usually "what did IRIX expect from
the hardware, and what did the emulator give it?" The monitor answers both
sides: it can stop the guest kernel anywhere and inspect it, and each device
has its own status and trace commands.

Read `debug.md` for the commands; this skill is the workflow. Use the
`irix-guest-loop` skill to boot the guest and copy files.

## 1. Build and symbols

- Use a build with breakpoints: plain `--release` or `developer`, not
  `lightning` (it strips breakpoint checks and the traceback). `developer`
  adds the undo buffer and instruction tracing, and starts with the CPU
  paused.
- Get the kernel's symbols from the kernel that actually boots. Copy `/unix`
  out of the guest (`cp /unix /shared/` over telnet, or from the disk image)
  and list it on the host with an `nm` that understands MIPS ELF
  (`mips-linux-gnu-nm`, `llvm-nm`): `nm unix > unix.map`. `loadsym` reads
  `address type name` lines. A kernel relinked after installing drivers or
  patches (`autoconfig`) needs a new map.
- `loadsym unix.map`, then `sym <name>` or `sym <addr>` to check it took. Load
  a PROM map too if you have one; names in both (`bcopy`) become ambiguous
  and are picked as `name:N`.
- IRIX source in `~/gits/irixhistory` (if present; see the user's notes)
  explains what a function does. It is not the exact build you are running:
  trust the symbols and the disassembly (`dis`) over source line numbers.
  Never copy SGI source text into notes, skills or code.

## 2. Make boot talk

- At the PROM prompt, `setenv showconfig 1` turns on the kernel's verbose
  configuration output: drivers report what they probe and attach. It is a
  real kernel option (read from the PROM environment at boot).
  `unsetenv showconfig` turns it off again.
- Other PROM variables that look tempting are not verbosity switches.
  `diagmode` is only used by standalone diagnostics. `kdebug` tells the
  kernel a debugger (symmon) is present and makes it call into it, so do not
  set it without symmon.
- Very early kernel messages go through the PROM's own console output
  (the kernel prints through the debugger if one is loaded, through its own
  console once that is up, and through the PROM before that). Watch the
  serial console (port 8881) or the PROM's text console for them.
- Everything the kernel printed is kept in its console ring: `putbuf` reads
  it at any time, including after a hang.
- The IRIS side has its own traces: `scsi debug on`, `scsi wdt`, `log scsi
  on`, `pdma dump on`, `net debug ...` (most need `developer`). Run them at
  the same time as the guest-side breakpoints so the two timelines can be
  matched.

## 3. Reading a panic or a hang

1. `status` says whether the CPU is running and why it last stopped.
2. `putbuf` shows the panic message, including the exception frame address
   (`ep`) when the kernel took an exception.
3. `eframe <ep>` decodes the frame: registers, Cause and Status, EPC, and a
   backtrace of the interrupted context. Use `eframe <ep> 64` for 64-bit
   kernels (IP28) if the default guess is wrong.
4. `bt` for the current stack, `dt 50` for the last instructions executed
   (not in `lightning`), `dis <addr> 20` around the faulting PC,
   `translate <addr>` and `tlb dump` for address problems.
5. For a hang: `stop`, then `regs`, `bt`, `dt 200`. If the PC sits in a
   polling loop, find what it polls (`dis` the loop, `mem` the address) and
   ask the device: its `status` command, `ioc status` and `hpc3 status` for
   interrupts that never arrive. `locks` shows whether an emulator thread is
   stuck instead.
6. `ex <class> on` (`addr`, `bus`, `tlb`, `ri`, ...) stops at the first
   exception of that class instead of waiting for the panic.

## 4. Breakpoints on the kernel

```text
> loadsym unix.map
> b wd93_inq
> b scsi_device_update if a0 != 0
> bp ignore 1 5            # skip the first five hits
> c
> wait 60000               # returns when a breakpoint hits
> regs
> finish                   # run to the return; v0 holds the result
```

- Arguments are in `a0`-`a3` at the function entry. Further arguments are on
  the stack for o32 kernels, and in registers 8-11 for 64-bit kernels (the
  monitor names those `t0`-`t3`, its o32 names). After `finish`, the return
  value is in `v0`.
- Conditions can use registers, `c0_<name>` and symbols:
  `b panic`, `b cmn_err if a0 == 3`.
- Physical breakpoints (`pr`, `pw`) catch DMA targets and device-visible
  memory that no virtual address in the current context maps.
- `bp save kernel.bp` / `bp load kernel.bp` keep a set across restarts.
- A breakpoint stops the guest's clock-driven world too: timeouts in the
  guest may fire after you continue. That is an artefact of stopping, not
  the bug.

## 5. Walking device discovery (SCSI example)

When IRIX boots but does not see a disk, or fails to mount root, follow the
device from probe to mount and find the first step that goes wrong. Break
on each stage in order, continue, and check the state named:

| Stage | Break on | Check |
|---|---|---|
| controller setup | `wd93_earlyinit`, `wd93edtinit` | reached at all; the controller's registers in `scsi regs` match what the driver just wrote |
| INQUIRY | `wd93_inq` | the command reaches the emulated target (`scsi wdt`), and the driver records success; after `finish`, the result |
| command completion | `wd93intr` | interrupts arrive and the driver reads the status the target sent |
| device registration | `scsi_device_update`, `scsi_dev_add` | called for the target; this is what creates its hardware graph entries |
| open | `dkscopen` | the disk open finds the inquiry data recorded earlier |
| root mount | `devinit`, `devavail` | the root device path opens; failure here prints the "device not available" message |

If INQUIRY succeeds on the wire but registration never happens, the driver
rejected the reply: compare the inquiry bytes IRIS returned with what the
driver checks (peripheral qualifier, device type, removable bit).

The same pattern works for other devices: break on the driver's probe/attach
entry, check what it read from the hardware, then follow the first
divergence into the emulator's device code.

## 6. Recording what you learn

- Emulator bugs found this way get a note in the matching `clanker/`
  category (and an `INDEX.md` line).
- Guest-side facts that cost time to find (what a driver expects of a
  register, a PROM variable's real effect) also go in `clanker/`, in your own
  words.
