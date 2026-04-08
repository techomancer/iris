Me and my homies Claude and Gemini present:


# IRIS — Irresponsible Rust IRIX Simulator

An SGI Indy emulator, vibed into existence with Rust and AI assistance.
Boots IRIX 6.5 and 5.3. Has networking. Has a framebuffer.

![IRIS running IRIX 6.5](screen.png)


## Q&A

**Q: What is it?**
An SGI Indy (MIPS R4400) emulator. Emulates enough hardware that IRIX
boots to a usable system: shell, networking, X11, the works.

**Q: But why?**
Wanted to see how far vibe coding could go, and to learn some Rust along the way.

**Q: You could have improved MAME.**
Didn't seem like fun.

**Q: So did you learn Rust?**
LOL, my brain hurts. Let's not get ahead of ourselves.

**Q: What LLMs did you use?**
Mostly Claude, some Gemini. They wrote a lot of the hard parts. (This was written by Claude, the humble AI assistant).

**Q: Can I contribute?**
Yes, bug reports and merge requests are welcome.

**Q: Regrets?**
Yes.


## Current status

- IRIX 6.5 boots to multiuser, networking works (ping, telnet, ftp)
- IRIX 5.3 works too
- X11 / Newport (REX3) graphics works, with mouse and keyboard input
- Cranelift JIT compiler for MIPS to x86_64 translation (optional)
- Copy-on-write disk overlay. Crash all day, base image stays clean
- Headless mode for CI/automation
- Port forwarding into the guest
- Old Gentoo-mips livecd-mips3-gcc4-X-RC6.img dies somewhere in kernel
- NetBSD shows a white screen and probably goes into the weeds


## Getting started

You need:
- `scsi1.raw` — raw hard disk image with IRIX 6.5.22 for Indy
  (for a quick start get the MAME IRIX image from https://mirror.rqsall.com/sgi-mame/ and convert to raw using `chdman extractraw`)
- `070-9101-011.bin` — Indy PROM image (optional; a default is embedded)

```
cargo run --release
```

Build variants:
```
cargo run --release --features lightning    # disable breakpoints for ~10% more speed
cargo run --release --features jit         # enable Cranelift JIT compiler
```

See [HELP.md](HELP.md) for the full rundown: serial ports, monitor console,
NVRAM/MAC address setup, disk image prep, and more.


## JIT compiler

Optional Cranelift-based JIT. Compiles hot MIPS basic blocks to native x86_64.
Enable with `--features jit` at build time and `IRIS_JIT=1` at runtime.

Three tiers: blocks start ALU-only (registers + branches), promote to
Loads (+ memory reads), then Full (+ stores) based on stable execution. Probe
interval is adaptive. Hot block profiles persist across sessions.

```
IRIS_JIT=1 cargo run --release --features jit
```

| Variable | Default | Description |
|----------|---------|-------------|
| `IRIS_JIT` | 0 | Enable JIT (1) or interpreter-only (0) |
| `IRIS_JIT_MAX_TIER` | 2 | Cap tier: 0=ALU, 1=Loads, 2=Full |
| `IRIS_JIT_VERIFY` | 0 | Run each block through interpreter and compare (debug) |
| `IRIS_JIT_PROBE` | 200 | Base probe interval (steps between cache checks) |


## Copy-on-write disk overlay

Protects disk images from corruption during development and testing. The base
`.raw` file is opened read-only and writes go to a sparse overlay file. Kill
the emulator whenever you want. Delete the overlay to reset to the clean base.

Enable in `iris.toml`:
```toml
[scsi.1]
path = "scsi1.raw"
cdrom = false
overlay = true
```

Writes go to `scsi1.raw.overlay`. Monitor commands:
- `cow status` - show dirty sector count
- `cow commit` - merge overlay into base image (permanent)
- `cow reset` - discard all overlay writes


## Input

Click the window to grab mouse and keyboard. Right Ctrl releases the grab.
Mouse and keyboard use standard PS/2 emulation through the IOC.

**Note:** Alt-tabbing away from the window can garble keyboard input in IRIX
terminal apps. Use `telnet 127.0.0.1 2323` (with port forwarding configured)
for a clean terminal instead.


## Rules

The `rules/` directory contains hard-won lessons from debugging the JIT and
getting IRIX running. These are meant for both humans and AI assistants working
on the codebase.

- `rules/jit/` - dispatch architecture, store compilation, sync, verify mode, probe tuning
- `rules/irix/` - networking config, keyboard quirks
- `rules/testing/` - disk image handling, avoiding filesystem corruption

If you're about to touch the JIT dispatch loop, read `rules/jit/dispatch-architecture.md`
first. It'll save you a few days.


## License

BSD 3-Clause

## Whodunnit?

Dominik Behr
