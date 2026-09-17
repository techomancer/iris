Me and my homies Claude and Gemini present:


# IRIS — Irresponsible Rust IRIX Simulator

An SGI Indy / Indigo2 emulator, vibed into existence with Rust and AI assistance.
Boots IRIX 6.5 and 5.3. Has networking. Has a framebuffer.

![IRIS running IRIX 6.5](screen.png)

**Status snapshot:**

- **Indy IP24** — primary daily-driver; IRIX desktop, X11, networking all work.
  R4400 (default) or R5000, picked at runtime.
- **Indigo2 IP22** — boots IRIX with its own PROM (embedded fallback), both SCSI
  controllers and the fullhouse interrupt layout; framebuffer/desktop path still
  maturing (use `console=d` + serial for debugging; see
  [docs/indigo2-ip22.md](docs/indigo2-ip22.md)).

Pre-built binaries and the Mac App Store GUI are available at
[danifunker/iris releases](https://github.com/danifunker/iris/releases) (upstream packaging).
For latest code, build from source from upstream [techomancer/iris](https://github.com/techomancer/iris). Report bugs and issues in the upstream repository.


## Q&A

**Q: What is it?**

**A:** An SGI Indy (MIPS R4400) emulator. Emulates enough hardware that IRIX
boots to a usable system: shell, networking, X11, the works.

**Q: But why?**

**A:** Wanted to see how far vibe coding could go, and to learn some Rust along the way.

**Q: You could have improved MAME.**

**A:** Didn't seem like fun.

**Q: So did you learn Rust?**

**A:** LOL, my brain hurts. Let's not get ahead of ourselves.

**Q: What LLMs did you use?**

**A:** Mostly Claude, some Gemini. They wrote a lot of the hard parts. (This was written by Claude, the humble AI assistant).

**Q: Can I contribute?**

**A:** Yes, bug reports and merge requests are welcome.

**Q: Regrets?**

**A:** Yes.


## Current status

- IRIX 6.5 boots to multiuser, networking works (ping, telnet, ftp, rsh, NFS, XDMCP)
- IRIX 5.3 works too
- **Indy IP24:** X11 / Newport (REX3) graphics works, with mouse and keyboard input
  (IntelliMouse wheel included), HAL2 audio, and IndyCam video-in through VINO
- **Indigo2 IP22:** hardware emulation + serial boot (see [docs/indigo2-ip22.md](docs/indigo2-ip22.md)); GUI framebuffer still maturing
- R4400 or R5000 CPU, selected per machine at runtime
- Cranelift JIT compiler for MIPS to host code (`jitv2`, optional, experimental),
  plus a REX3 draw pipeline of 400+ precompiled specialised draw functions and an
  optional REX3 shader JIT (`rex-jit`)
- Copy-on-write disk overlay, and CHD images with MAME-style `.diff.chd` sidecars.
  Crash all day, base image stays clean
- Hot-swappable CD-ROM with runtime disc switching
- Snapshots: save, restore, in-memory rollback, content-addressed dedup, HTTP push/pull
- Built-in NAT gateway with DHCP, host-DNS forwarding, port forwarding, an
  in-process NFSv2/v3 server, TFTP for PROM network boot, and FTP/XDMCP helpers;
  or PCAP bridging onto a real LAN
- Headless mode and a CI control socket (`iris-ci`) for automation
- Optional egui front-end (`iris-gui`) with machine management and a benchmark tab
- DaynaPort SCSI/Link Ethernet and the N64 development board (Ultra64), both opt-in
- Other guests: Linux (Debian 7, Gentoo), NetBSD and OpenBSD have had SCSI,
  interrupt and timer fixes land for them. They are not regularly tested, so
  expect rough edges


## Getting started

Super easy mode -> Thanks to Dani we have Windows/Mac/Linux builds at https://github.com/danifunker/iris/releases
So if you dont feel comfortable building it yourself, please head there. Also, he submitted IRIS-GUI to Mac App Store!

You need:
- A hard-disk image with IRIX 6.5.22 (or 5.3) for Indy. To produce one, follow
  [rules/irix/irix-install.md](rules/irix/irix-install.md) (install from the
  original media CDs into an empty CHD/raw disk).
- `070-9101-011.bin` — Indy PROM image (optional; a default is embedded, and so
  is an Indigo2 one)

Now, if you feel like typing some commands in console. Sync the project and:

```
cargo run --release
```

The project pins a nightly toolchain (`rust-toolchain.toml`); rustup picks it up
automatically.

Build variants:
```
cargo run --release --features lightning,rex-jit     # recommended for best speed
cargo run --release --features lightning             # disable emulator breakpoints for a little bit more speed
cargo run --release --features rex-jit               # enable REX3 graphics JIT compiler
cargo run --release --features jitv2,rex-jit         # MIPS JIT v2 (experimental; see "JIT compilers")
cargo run --release --features idle-pause            # park the CPU thread while the guest idles instead of spinning a host core
cargo run --release --features ci_clock              # synthetic deterministic CP0 Count clock (CI/snapshot validator only; loses realtime desktop timing)
cargo run --release --features chd                   # mount .chd disk/CD-ROM images directly (via libchdman-rs); off by default to keep builds light
cargo run --release --features camera                # use a host camera as the IndyCam video source (AVFoundation / V4L / MediaFoundation). See [vino] in iris.toml.
cargo run --release --features pcap                  # bridge guest networking onto a real host interface via libpcap instead of the built-in NAT gateway. See [network] in iris.toml.
cargo run --release --features daynaport             # DaynaPort SCSI/Link: Ethernet over the SCSI bus, selectable per SCSI id. Needs a guest driver. See docs/daynaport.md.
cargo run --release --features ultra64               # N64 development board in GIO slot 0, bridged to an external gopher64. See HELP.md.
cargo run -p iris-gui --release                      # the egui front-end, see iris-gui-README.md
```

`lightning` and `developer` are mutually exclusive, and `lightning` implies the
interpreter's `opcodefusion`. The emulator prints the features it was built with
at startup.

<details>
<summary>Diagnostic and experimental features</summary>

| Feature | What it does |
|---|---|
| `developer` | Undo buffer, execution trace, extra monitor commands; CPU starts paused. Also `--profile developer`. |
| `developer_ip7` | CP0 Compare / timer delivery stats and debug prints |
| `developerx` | Break into the monitor on IBE/DBE/ADEL/ADES/TLB errors |
| `rexdiag` | REX3 per-GO activity bits and dispatch counters. **On by default**; drop with `--no-default-features` to measure their cost |
| `llstats` | Per-address LL/SC reservation histogram (`ll stats`); lightning-compatible |
| `fetchverify` | Check every executed instruction word against memory (stale-code detector); lightning-compatible |
| `opcodefusion` | Interpreter branch+NOP, LUI+ORI/ADDIU and address-calc+load/store fusion. Breakpoints on a fused second instruction never fire |
| `tlbstats` / `tlbcheck` | TLB translation counters / full JTLB consistency check after every TLB write |
| `jitstats` | Counts how far each load/store gets through the JIT inline-memory checks |
| `instr_stats` | Per-opcode decode/execute counters (interpreter only; refused with `jitv2`) |
| `ppmem` | Host-MMU-backed physical memory ([docs/ppmem-design.md](docs/ppmem-design.md)) |
| `tcache` / `tcache_verify` | Transparent cache on top of `ppmem` ([docs/tcache-design.md](docs/tcache-design.md)) / its self-check |
| `jitv2_lockstep` | Verify every JIT instruction against the interpreter (implies `developer`) |
| `jitv2_smc_check` | Report writes into the page the CPU is executing (run with `j2 inline_mem off`) |
| `jitv2_opcodefusion` | jitv2 LUI+ORI/ADDIU and branch+NOP fusion (off by default; see "JIT compilers") |
| `j2wp` | jitv2 whole-page compile instead of one function per entry point (not production-ready) |
| `jitv2_corpus_dump` | Dump compile-request pages to `jitv2_corpus/` |
| `debug_cache` | Track one cache line across all operations |
| `mips4` | Lets jitv2 compile MIPS IV opcodes (otherwise they run in the interpreter). The interpreter enables MIPS IV from the runtime CPU model on its own |
| `tlbvmap` | Vestigial; the vmap TLB fast path is always on |
| `r5k` | Vestigial for CPU selection (the CPU is a runtime setting) |
| `r5ksc`, `r5ksc_triton` | Refuse to build: no working R5000 secondary-cache model yet (`rules/testing/r5k-l1i-cache-bugs.md`) |

</details>

### CHD image support (`--features chd`)

Off by default. When enabled, IRIS can mount `.chd` hard-disk and CD-ROM
images directly without first extracting to raw. Compressed parent CHDs
stay untouched — writes go to a MAME-style `.diff.chd` sidecar.

```
cargo build --release --features chd
```

Without this feature, attempting to mount a `.chd` path returns an
`Unsupported` error; raw images and COW overlays continue to work as
before.

The CHD backend (`libchdman-rs` >= 0.288.8) and the MAME CHD core it vendors
are BSD-3-Clause licensed, so enabling this feature keeps IRIS fully
BSD-3-Clause (see `LICENSE-libchdman-rs.txt`).

See [HELP.md](HELP.md) for the full rundown: serial ports, monitor console,
NVRAM/MAC address setup, disk image prep, and more.

**Windows 11:** full build/launch guide in [wsl/README.md](wsl/README.md).


## PCAP bridged networking (`--features pcap`)

By default IRIS gives the guest networking through a built-in software NAT
gateway (DHCP/DNS/TCP/UDP routing + port forwarding). As an alternative you can
bridge the guest's raw Ethernet frames directly onto a real host interface. The
guest then appears as an independent L2 host on your physical LAN and can be
pinged from other machines and can use your real DHCP and DNS servers.

### Library and licensing

The `pcap` crate links the generic `wpcap` import library on Windows (NOT a
driver-specific one), so IRIS is not tied to any single provider. You can
build/link against **the BSD-licensed WinPcap Developer Pack** as well as Npcap.
IRIS links dynamically and never bundles the driver, so the runtime driver's
license (e.g. Npcap's redistribution terms) does not attach to IRIS.

To point the linker at the WinPcap Developer Pack SDK on Windows:
```
set LIBPCAP_LIBDIR=C:\path\to\WpdPack\Lib\x64
cargo build --release --features pcap
```

On Linux/macOS you need the libpcap headers and library (e.g. `libpcap-dev` on
Debian/Ubuntu, or the macOS system libpcap).

### Enable PCAP mode

1. **Build** with `--features pcap`:
   ```
   cargo build --release --features chd,pcap
   ```

2. **Configure** in `iris.toml` (or pass CLI flags):
   ```toml
   [network]
   mode = "pcap"
   pcap_interface = "1"    # 1-based index (recommended), or exact name, or omit to auto-pick
   ```

   On Windows, if you prefer the full device name (`\Device\NPF_{GUID}`), use a
   TOML *single-quoted* literal string (backslashes are escape characters in
   `"double-quoted"` strings):
   ```toml
   pcap_interface = '\Device\NPF_{8D30ACAE-AC0F-4E05-BF89-F35AD7950663}'
   ```

3. **List interfaces**:
   ```
   iris --list-net-interfaces
   ```
   Or from the monitor console:
   ```
   net interfaces
   ```

Alternatively specify on the command line (the index form works here too):
```
./target/release/iris --net-mode pcap --pcap-interface 1
./target/release/iris --net-mode pcap --pcap-interface eth0
```

Caveats:
- Requires elevated privileges to open a raw capture: root or `CAP_NET_RAW`
  on Linux, root on macOS, Administrator + a WinPcap-compatible driver
  (WinPcap or Npcap) on Windows.
- No NAT services (DHCP/DNS/NFS/port-forward) are provided in PCAP mode — the
  guest uses the real network's services. Configure IRIX networking for your
  LAN accordingly.
- Wired bridges work best. Many Wi-Fi access points reject the guest's extra
  MAC address, so bridging onto a wireless interface may not pass traffic.
- The guest needs a MAC address in NVRAM. IRIS writes `[network] mac` (default
  `08:00:69:12:34:56`) into a blank slot before boot; give each emulated machine
  on the same LAN its own (see `rules/irix/networking.md`).

Without `--features pcap`, selecting `mode = "pcap"` logs a warning and falls
back to the NAT gateway, and `--list-net-interfaces` reports that the feature
is missing.


## DaynaPort SCSI/Link (`--features daynaport`)

A SCSI-attached Ethernet adapter (SCSI type 3, Processor) selectable on any
SCSI id — a second network path for the guest that goes over the SCSI bus
instead of the onboard SEEQ. Off by default, because it is only useful with a
guest driver; IRIX has none in the box (see
[irixdayna](https://github.com/techomancer/irixdayna), where it appears as
`dp0`).

```toml
[scsi.3]
kind = "daynaport"            # default "disk"; "cdrom" / cdrom = true unchanged
mac  = "00:80:19:12:34:56"    # optional; default derived from the SCSI id
subnet = "192.168.10.0/24"    # optional; this target's own NAT subnet
```

Each DaynaPort runs its own NAT gateway (or PCAP bridge, in a `--features pcap`
build) on its own subnet, so `dp0` and `ec0` never share a network. `scsi dayna`
in the monitor shows its MAC, addresses and counters. Full protocol and
verification notes: [docs/daynaport.md](docs/daynaport.md).


## Emulated CPU

R4400 (the default) or R5000, chosen per machine at runtime — not at build time.
Both cache models are compiled into every binary and the machine picks between
them when it starts, so there is no separate R5000 build or download.

| | R4400 | R5000 |
|---|---|---|
| L1 I/D | 16KB direct-mapped, 16B lines | 32KB 2-way, 32B lines |
| Secondary cache | 1MB unified L2 | none (Config `SC=1`) |
| ISA | MIPS III | MIPS IV |
| PRId / FPU FIR | `0x00000440` / `0x00000500` | `0x00002321` / `0x00002300` |

This is a different machine to the guest, not a speed knob: IRIX reads PRId and
configures itself from it, and an R4400 raises Reserved Instruction on the MIPS IV
opcodes an R5000 executes.

Which one is faster depends on the engine, so don't compare scores across CPUs.
On the bare-metal suite the R5000 is about 17% slower under the interpreter —
2-way associativity means probing both ways on every fetch, read and write — but
about 10% faster under jitv2, where the larger cache and 32-byte lines pay off
and the probe is not on the critical path.

Pick it in the GUI (Machine menu, or the General tab of the configuration area),
in a config file, or on the command line:

```
cargo run --release -- --cpu r5000        # or r4400, the default
```

```toml
[machine]
cpu = "r5000"
```

A snapshot records the CPU it was taken on and refuses to restore onto the other
one, since the captured state assumes that machine.


## JIT compilers

### MIPS JIT v2 (`--features jitv2`) — experimental

A physical-page compiler built on Cranelift, with memory-resident
registers and no speculation — compiled code is either exactly
equivalent to the interpreter or it never gets published. Not the default
engine yet. Enabled automatically at runtime once the feature is compiled in.
See `rules/jitv2/jit-v2-design.md` for the full design and `HACKING.md`'s
JIT v2 section for tuning. (The original speculative, tiered MIPS JIT was
removed in August 2026; jitv2 replaces it.)

```
cargo run --release --features jitv2,rex-jit
```

What it does today: compiles on a pool of background threads (`[jitv2]
threads` / `--jitv2-threads`, default 1), inlines L1 data-cache loads and stores
for both CPU models, detects self-modifying code through per-page generation
counters, and falls back to the interpreter for anything it has no emitter for.

Extra features: `jitv2_lockstep` (cross-checks every compiled instruction
against the interpreter — slow, diagnostic only), `jitv2_smc_check`,
`jitv2_corpus_dump` (dumps compile-request page snapshots to `jitv2_corpus/`
instead of compiling, for building an offline test corpus), `j2wp` (whole-page
compile, experimental), and `jitv2_opcodefusion` (LUI+ORI/ADDIU and
branch/jump+NOP delay-slot fusion, jitv2's counterparts to the interpreter's
`opcodefusion` — OFF by default, unlike the interpreter's own fusion, due to a
history of live-boot bugs; see
`rules/jitv2/jitv2_lui_fusion_foreign_delay_slot_hazard.md`). Developer tools:
`jitv2_analyze`, `jitv2_verify` and `jitv2_pcp_dump` binaries, and the `j2`
monitor command.

### REX3 drawing and the graphics JIT (`--features rex-jit`)

Every build draws through one generic REX3 draw routine that is specialised
ahead of time into 400+ native draw functions (`src/rex3_shaders.rs`, generated
by `tools/gen_rex3_shaders.py` from a corpus of the DrawMode0/DrawMode1/clip
combinations the IRIX desktop actually uses). Most desktop drawing already runs
through one of those, with no JIT involved.

`rex-jit` adds a Cranelift JIT for combinations outside that corpus. It compiles
a specialised native "shader" per unique draw mode, inlining the entire draw
loop — coordinate stepping, clipping, shade DDA, pattern advance — into a single
function. Shaders compile in the background on first use and share the dispatch
table with the precompiled set; the profile of modes seen persists across
sessions (`~/.iris/rex-jit-profile.bin`) for instant warm-up on next boot. `rex jit status|list|on|off` in the
monitor inspects and controls it.

```
cargo run --release --features rex-jit
```

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
- `cow commit [id]` - merge overlay into base image (permanent)
- `cow reset [id]` - discard all overlay writes

CHD images (`--features chd`) get the same protection automatically: writes go
to a `.diff.chd` sidecar. `iris-ci chd-sync` (or `iris-ci quit --sync-chd`, or
the GUI's "Commit changes to disk") folds the diff back into the base.


## Snapshots and rollback

Capture the full machine state — RAM, every device, plus the COW overlay — into
`saves/<name>/`, and restore it later. CPU, MC, IOC, HPC3, REX3, RTC, EEPROM,
SCSI controller, and the Seeq Ethernet chip all round-trip. Current schema
version is 3: postcard-encoded binary device state plus content-addressable
chunked RAM under `saves/.cas/`. A second snapshot taken from the same parent
adds **zero bytes** to disk for any RAM region that didn't change — same
storage model as Docker layers. A snapshot records which CPU it was taken on
(R4400/R5000) and refuses to restore onto the other.

From the interactive monitor (`telnet 127.0.0.1 8888`):
```
save base/desktop          # writes saves/base/desktop/
load base/desktop          # restore everything (RAM, devices, disk overlay)
```

From `iris-ci` (the wrapper; see "CI control socket and `iris-ci`"):
```bash
iris-ci save base/desktop
iris-ci restore base/desktop          # full disk-backed reload (~150 ms cold)
iris-ci rollback                      # in-memory rewind to last restore (~40 ms)
iris-ci diff base/desktop tests/grep  # what changed: devices, RAM chunks, COW sectors
iris-ci validate base/desktop -n 1000000  # bit-deterministic re-execution check (build with --features ci_clock)
iris-ci tree                          # snapshot parent-chain hierarchy
iris-ci gc                            # sweep CAS chunks no kept snapshot references
iris-ci pull http://reg/snapshots/base   # fetch a snapshot from another machine
```

Two restore tiers:
- **`restore <name>`** — full disk-backed reload. ~150 ms. Use after a hard
  reset or to switch to a different snapshot.
- **`rollback`** — in-memory rewind to the last `restore` checkpoint. ~40 ms,
  no disk I/O. Use this in tight inner test loops where you keep returning to
  the same starting state.

Reflinks are used on APFS / btrfs / xfs so capturing a snapshot of a 4 GB disk
image takes <10 ms and uses ~18 MB of actual disk.

See [CHANGELOG.md](CHANGELOG.md) for the full feature set, and `rules/snapshot/`
for the format and its gotchas.


## CI control socket and `iris-ci`

`--ci` enables a control socket for headless automation, plus a
small in-process serial backend so the harness can drive the IRIX console
directly. The default is the Unix socket `/tmp/iris.sock`; on Windows it is TCP
`127.0.0.1:19851`. `--ci` implies `--headless` unless you add `--ci-display`,
and `--serial-log FILE` keeps a transcript of the IRIX console.

```
cargo run --release --features lightning -- --ci
```

`cargo build` produces a companion binary, `iris-ci`, that's the **canonical
way** to drive the socket. Don't bother with raw `nc` + JSON unless you're
debugging the wrapper itself.

```bash
# In one terminal: launch iris (Newport window opens; --ci adds a control channel)
./target/release/iris --ci

# In another terminal: drive it
./target/release/iris-ci boot          # PROM menu → IRIS console login (one cmd)
./target/release/iris-ci login         # send root + dismiss vt100 prompt + wait #
./target/release/iris-ci run 'ls /'    # send shell command, get stdout + exit code
./target/release/iris-ci save base/multiuser
./target/release/iris-ci put localfile.tar   # copy file into guest, no bs=512 math
./target/release/iris-ci get /tmp/out --to ./out.tar
./target/release/iris-ci diff base mutated   # per-device + chunk + cow-sector deltas
./target/release/iris-ci tree
./target/release/iris-ci script tests/scenario.iris   # batch-run a sequence of cmds
./target/release/iris-ci cdrom-load 4 disc2.iso       # swap a CD at runtime
./target/release/iris-ci rtc-save                     # persist NVRAM
./target/release/iris-ci quit --sync-chd              # fold CHD diffs, then exit
```

Run `iris-ci --help` for the full list, or `iris-ci <subcmd> --help` for any
subcommand. Every operation has a typed clap arg — no JSON quoting, no
hand-managed timeouts.

For automation that doesn't want to depend on `iris-ci`, the underlying socket
protocol is newline-delimited JSON; `cmd` and `args` per request, `{ok, data,
error}` per response. See `src/ci.rs` for the dispatch table.


## Scratch volume — file injection without networking

A SCSI device with `scratch = true` is a host-controlled raw block device for
pushing files into the guest (and pulling artifacts back out) without bringing
up NFS or anything else. iris pre-formats the underlying file with a minimal
SGI Volume Header on first run, and exposes it inside IRIX as
`/dev/rdsk/dks0d2s0`.

Enable in `iris.toml`:
```toml
[scsi.2]
path    = "scratch.raw"
cdrom   = false
overlay = false
scratch = true
size_mb = 64
```

With `iris-ci` (recommended):
```bash
iris-ci put localfile.tar                 # copies host file into the guest
iris-ci get /tmp/output.log --to ./out.log  # pulls a guest file out
```

`iris-ci put`/`get` handle the IRIX `dd bs=512` sector-alignment quirk
transparently — they compute the right block count from the host file size,
issue the right `dd` recipe to the guest, and truncate to the original byte
length on the receiving end.

Manual/raw paths (if you want to drive `dd` yourself):
- Reads MUST use `bs=512` (or any 512-multiple); `bs=64` returns "I/O error".
- Writes must be padded to `bs`; add `conv=sync` for short inputs.
- Inside IRIX: `dd if=/dev/rdsk/dks0d2s0 bs=512 | tar xf -`


## Input

Click the window to grab mouse and keyboard. In the `iris` window Right Ctrl
releases the grab; in `iris-gui` it is Ctrl+Alt (Option+Command on macOS), with
Ctrl+Alt+Esc as a fallback. Mouse and keyboard use standard PS/2 emulation
through the IOC, including an IntelliMouse scroll wheel. Keys are sent by
physical position, so set IRIX's `keybd` to your layout.

**Note:** Alt-tabbing away from the window can garble keyboard input in IRIX
terminal apps. Use `telnet 127.0.0.1 2323` (with port forwarding configured)
for a clean terminal instead.


## Testing and benchmarking

Two bare-metal MIPS suites run on the emulated CPU with no operating system in
the way. They answer different questions and neither replaces the other.

**`cpu-tests/`** — is this instruction correct? ~250 self-checking tests over
ALU, FPU, TLB, caches, exceptions and the MIPS IV additions, one instruction at
a time with clean state. The expectations are validated on real Indys (R4400
and R5000, both passing every check); see
[cpu-tests/README.md](cpu-tests/README.md).

```sh
sudo apt-get install gcc-mips-linux-gnu binutils-mips-linux-gnu   # or: make -C cpu-tests toolchain-local
make -C cpu-tests && make -C cpu-tests run
cpu-tests/run/matrix.sh                 # R4400/R5000 x interp/jitv2
```

**`bench/`** — how fast is this build, and is it still right after ten million
of them? 46 kernels covering integer, FPU, the cache hierarchy, image and video
editing inner loops, compression, and the emulator-only paths (TLB refill,
exception round trip, cache maintenance, uncached I/O). Every kernel checksums
its result against a golden value computed by building the same C natively, so
each run reports an accuracy percentage next to its throughput — and per-kernel
**guest instructions per host second**, which is directly comparable between the
interpreter and jitv2.

```sh
cargo build --release --bin iris-bench
./target/release/iris-bench run         # ~60 s; --quick for about half that
./target/release/iris-bench run --quick

make -C bench && make -C bench hostbench            # only to change the suite
./target/release/iris-bench matrix      # builds and runs every CPU x engine cell
./target/release/iris-bench host        # the same kernels, natively, for the ratio
```

`run` needs **no MIPS toolchain and no build step**: a known-good guest binary
is checked in at `bench/prebuilt/` and linked into `iris`, and the run happens
in-process on a headless machine the emulator builds for itself. That is also
what the GUI's **Benchmark tab** does, on every platform and inside the App
Store sandbox — one button, and the accuracy score sits next to the speed.

Includes Dhrystone 2.1 (DMIPS) and LINPACK 100x100 (MFLOPS), so an emulated
Indy can be put next to published figures for a real one, plus a Whetstone mix
(reported in passes/s — see bench/README.md for why not MWIPS). See
[bench/README.md](bench/README.md) — and
[rules/testing/benchmark-suite-gotchas.md](rules/testing/benchmark-suite-gotchas.md)
before adding a kernel.

`bench/irix/` is the other half: real workloads under a booted IRIX (filesystem,
buffer cache, IRIX's own tools, X on REX3), driven over `iris-ci`.


## Rules

The `rules/` directory contains hard-won lessons from debugging the JIT and
getting IRIX running. These are meant for both humans and AI assistants working
on the codebase.

- `rules/jitv2/` - jitv2 compiler design, codegen gotchas, delay slots, fusion hazards, lockstep
- `rules/irix/` - the IRIX install guide, networking config, NFS, VINO/IndyCam, keyboard quirks, csh + scratch raw-device gotchas
- `rules/testing/` - cpu-tests/bench harness gotchas, CPU-model findings, disk image handling, benchmark-kernel gotchas
- `rules/snapshot/` - snapshot binary format, scratch-volume conventions, round-trip tests, CI overlay paths, **iris-ci as the canonical CI interface**
- `rules/rex3/` - REX3 drawing engine findings (CID match, blending, FIFO batching)
- `rules/gui/` - iris-gui threading, input capture, keyboard layouts, Windows crash diagnostics
- `rules/macos/` - App Store / sandbox constraints
- `rules/perf/` - idle park, REX3 thread parking, first benchmark numbers
- `rules/scsi/` - WD33C93A behaviour under OpenBSD and Linux, DaynaPort
- `rules/build/` - dependency-upgrade and platform build gotchas

If you're about to touch the jitv2 compiler, read `rules/jitv2/jit-v2-design.md`
first. It'll save you a few days.


## License

BSD 3-Clause (`LICENSE`).

The optional `--features chd` build links `libchdman-rs` (>= 0.288.8), which —
along with the MAME CHD core it vendors — is also BSD 3-Clause, so CHD builds
stay fully BSD 3-Clause. See `LICENSE-libchdman-rs.txt` for that third-party
notice.

## Whodunnit?

Dominik Behr and contributors


## Contribution policy

We have no problems with LLM generated code. In fact most of IRIS is made with LLMs.
But that doesn't mean we don't do proper software engineering. So lets keep PRs small and reasonable to review. One issue/fix per PR, preferably in one commit, since LLM code churn doesn't help with clarity. Lets keep this bisectable too.


