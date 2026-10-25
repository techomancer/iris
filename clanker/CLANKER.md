# Agent instructions — IRIS

`CLAUDE.md` and `AGENTS.md` at the repository root are links to this file.
Paths below are relative to the repository root.

IRIS is an SGI Indy (IP24) and Indigo2 (IP22/IP28) emulator written in Rust, with an
R4400, R5000, or R10000 CPU selected at runtime. It boots IRIX 6.5 and 5.3 to a usable
system (shell, networking, X11). It is **not** cycle-accurate
— IRIX doesn't need it and accuracy would only make it slower.

## Two kinds of documentation

The repository keeps documents for people and documents for agents apart.
Know which one you are reading and which one you are writing.

**For people** (the repository root and `docs/`): `README.md`, `HELP.md`,
`HACKING.md`, `debug.md`, `FEATURES.md`, `STORAGE.md`, `NETWORKING.md`,
`TESTING.md`, `iris-gui-README.md`, `CHANGELOG.md`, and in `docs/` the
per-device guides, the design records, the references we aggregated from
datasheets, SGI sources, MAME and the BSDs (`interrupt_map.md`,
`docs/Indy/*.md`), and the datasheets themselves (PDFs). They describe the
emulator as it is, in prose a newcomer can follow. When you change behaviour
they describe, update them, and keep them readable: no session logs, no
"I tried X", no agent shorthand.

**For agents** (`clanker/`): findings, gotchas, investigation logs,
measurements, plans and task briefs, the backlog, skills and the MCP server.
They may be terse, dated and partly superseded. They are working notes, not
user documentation: do not link users to them as a manual, and do not put
working notes into `docs/` or the root guides.

If you are unsure where something goes: if a user would read it to *use* or
*understand* IRIS, it is a human document; if it records how a problem was
found or what to watch out for while changing code, it goes in `clanker/`.

## Read these first

- `HACKING.md` — architecture: data path/endianness, the bus and device
  traits, threads and locks, the GFIFO, jitv2 concepts. **Read before
  touching device or CPU code.**
- `debug.md` — the monitor console and every other way to look inside a
  running machine (GDB stub, MCP server, traces, perfetto). **Read before
  debugging a live guest.**
- `HELP.md` — running it: configuration, serial ports, NVRAM/MAC, networking,
  disk images.
- `README.md` — overview and status; `FEATURES.md` — every Cargo feature,
  config key, CLI option and environment variable.
- `iris-gui-README.md` — the optional egui front-end (`-p iris-gui`).
- `CHANGELOG.md` — what changed, by area.
- `docs/` — per-device notes, design records and datasheets (see above).
- `clanker/INDEX.md` — every agent note by category, each tagged rule /
  design / howto / measure / open / history / plan. Check it before
  re-deriving a gotcha.
- `ignore/rules` — local unpublished notes, when present.

## clanker/ layout

| Path | Contents |
|---|---|
| `clanker/INDEX.md` | categorized index of every note |
| `clanker/cpu/`, `jitv2/` | CPU architecture findings; the MIPS JIT |
| `clanker/devices/`, `scsi/`, `net/`, `video/` | device models, storage, the NAT stack and its services, VINO/IndyCam |
| `clanker/rex3/`, `gr2/`, `mgras/`, `odyssey/`, `hostgl/` | graphics boards and host GL (`odyssey/` is references only) |
| `clanker/gui/`, `macos/` | front-ends and platform packaging |
| `clanker/snapshot/`, `perf/`, `testing/`, `build/` | snapshots and `iris-ci`, host performance, test harnesses, builds |
| `clanker/irix/` | the guest: install guide (`irix-install.md`), network setup, shell quirks |
| `clanker/plans/` | plans, task briefs and reviews; each has its own status line |
| `clanker/TODO.md` | the feature-completion backlog |
| `clanker/skills/` | agent skills (see below) |
| `clanker/iris_mcp.py` | MCP server exposing the monitor as tools |

`rules/` is a link to `clanker/`, so old references still resolve. Write new
references as `clanker/...`.

### Writing notes

- When you confirm a non-obvious fix or behaviour, write a short markdown note
  in the right `clanker/<category>/` directory so the next session does not
  relearn it. Name the file after the rule (`sc-succeeds-on-llbit-alone.md`),
  start with a `# Title` that states it, then the symptom, the cause, and the
  fix or rule.
- Add the note to `clanker/INDEX.md` with its kind. When a note stops being
  true, update its status line (and its kind in the index) rather than
  leaving it to mislead; delete it only if nothing of value remains.
- Hardware facts for GR2 and IMPACT go in the local `ignore/gr2` and
  `ignore/gr4` headers, not in `clanker/`.
- Never copy SGI source text, comments or prose. Interface names, offsets and
  bit positions are fine.

## Keeping the human guides current

- **Monitor commands**: when you add, remove or change a monitor command
  (`register_commands` / `execute_command` on any `Device`, or `j2`
  subcommands), update `debug.md` in the same change. It is the reference
  people and agents use; a command missing from it does not exist for them.
  Update `clanker/iris_mcp.py` too if an MCP tool wraps the command.
- **Architecture**: when you change a trait in `src/traits.rs`, the threading
  of a device, the locking scheme, the GFIFO, or how jitv2 publishes and
  invalidates code, update `HACKING.md`.
- **Configuration and features**: `FEATURES.md` and `HELP.md`.

## Inspecting a running machine

Start with `debug.md`. In short:

- The monitor listens on `127.0.0.1:8888` (`monitor_port`). `tools/mon.py
  "cmd" ...` runs commands from a shell; `help` lists what the running build
  registered.
- `clanker/iris_mcp.py` wraps the monitor as MCP tools (registers, memory,
  breakpoints, stepping, disassembly, TLB, symbols, IRIX eframes and kernel
  log, plus `run_command` for anything else). It needs `pip install mcp`.
  Register it with your agent, for example in `.mcp.json`:

  ```json
  { "mcpServers": { "iris": { "type": "stdio", "command": "python3",
      "args": ["clanker/iris_mcp.py"] } } }
  ```

  Use an absolute path if your agent does not start servers from the
  repository root. It connects lazily and retries, so iris can be started
  after the agent.
- `iris-ci` drives a running emulator over its CI socket (snapshots,
  scripted console input, file transfer, screenshots). Prefer it to ad-hoc
  serial poking. See `STORAGE.md` and `clanker/snapshot/`.
- Guest shell over telnet: `tools/irix-sh.py` (see the `irix-guest-loop`
  skill).

## Skills

`clanker/skills/` holds skills in the `SKILL.md` format:

| Skill | Use it for |
|---|---|
| `irix-guest-loop` | running commands in the IRIX guest: boot, telnet, NFS share, autologin, MIPSpro builds, library testing, shutdown |
| `irix-gfx-loop` | graphics debugging on Newport (REX3), GR2 and IMPACT: test programs, monitor traces, pixel checks, recordings, offline replay, input injection |
| `irix-inst-packaging` | building and testing IRIX inst packages, tardists and patches |
| `irix-kernel-debug` | debugging the guest kernel from the monitor: symbols, boot verbosity, panics, kernel breakpoints, device discovery |

The repository already links every skill into `.claude/skills/` (Claude
Code) and `.agents/skills/` (Gemini CLI and other agents that read that
directory), so agents working in this checkout find them with no setup.

Skills are **linked, never copied**, so an edit in `clanker/skills/` reaches
every agent at once and the copies cannot drift apart. Link the skill's
directory, not its `SKILL.md`. When you add a skill, add both project links
in the same change. From the repository root:

```sh
# a new skill, project level (relative links, committed)
ln -s ../../clanker/skills/<name> .claude/skills/<name>
ln -s ../../clanker/skills/<name> .agents/skills/<name>

# every project on this machine (user level, absolute, not committed);
# use your agent's own skills directory
ln -s "$PWD/clanker/skills/<name>" ~/.claude/skills/<name>
ln -s "$PWD/clanker/skills/<name>" ~/.gemini/config/skills/<name>
```

Project-level links are relative, so they survive moving the checkout;
user-level links are absolute and must be redone if it moves. Check a link
with `ls -l`: it must point at a directory containing `SKILL.md`.

Edit skills in `clanker/skills/`, never through a link or a copy in an agent
directory. Keep them free of passwords, personal paths and machine-specific
setup.

**Windows.** The links in this repository (`CLAUDE.md`, `AGENTS.md`, `rules`,
the skill links) need `git config core.symlinks true` and Developer Mode (or
an elevated shell) when cloning. Without that, Git writes each link as a
small text file holding its target path. `CLAUDE.md` and `AGENTS.md` then
contain only `clanker/CLANKER.md`: open that file. The skill links become
plain files, so agents will not find the skills; point your agent at
`clanker/skills/` directly or re-clone with symlinks enabled.

## Build & run

```
cargo run --release                                       # interpreter
cargo run --release --features lightning,gpujit           # recommended for speed (gpujit = rex-jit,gr4-jit,gr2-jit)
cargo run --release --features jitv2,gpujit               # enable MIPS JIT v2 (experimental)
cargo run --release -- --cpu r5000                        # CPU is a runtime choice, not a feature
cargo run -p iris-gui --release                           # GUI front-end
```

The toolchain is pinned to nightly (`rust-toolchain.toml`). `lightning` and
`developer` are mutually exclusive; `r5ksc`/`r5ksc_triton` deliberately fail to
build.

Binaries: `iris` (the emulator), `iris-ci` (CI/automation socket client),
`iris-bench` (benchmark driver), `coffdump`, `mkvh` (SGI volume headers),
`chd-extract` (`chd`), `jitv2-analyze`/`jitv2-verify`/`jitv2-pcp-dump`
(`jitv2`), and `iris-gui` in the workspace. Feature flags are documented in
`FEATURES.md`.

## Testing and benchmarking

- `cargo test --workspace` — unit tests (per-device snapshot round trips, CPU,
  TLB, REX3; add `--features jitv2` for the JIT equivalence tests).
- `cpu-tests/` — bare-metal MIPS III/IV correctness suite. "Is this instruction
  right", one instruction at a time. `make -C cpu-tests run`. It does not
  exit 0 on a healthy tree: compare with
  `clanker/testing/cpu-tests-known-failure-baseline.md`.
- `bench/` — bare-metal benchmark suite. "How fast is this build, and is it
  still right after ten million of them." Reports throughput, guest MIPS and an
  accuracy score per kernel; `iris-bench matrix` sweeps R4400/R5000 x
  interpreter/jitv2. Read `clanker/testing/benchmark-suite-gotchas.md` before
  adding a kernel — the accuracy check catches endianness, uninitialised
  memory and out-of-bounds reads, and every one of those has already happened.
- Both share `cpu-tests/harness` (toolchain probe, SCC console, startup and
  exception dispatch). Changing those files affects both suites.

## Hard invariants (from HACKING.md)

- **Endianness lives only at "The Edge."** Host `u32`/`u64` are bit-containers;
  byte-swapping happens at PROM/disk I/O via `swap_on_load`, never in CPU/bus/MC
  logic. **Do not suggest `.to_be()` / `.to_le()` for memory or register code.**
- **Concurrency is per-device.** CPU, the graphics boards, SCSI, and ethernet
  run on their own threads and lock their own state. Deadlocks live in
  callbacks *up* to a parent device (e.g. SCSI → HPC3) — be careful there.
- **A full GFIFO is back-pressure, not an error.** A device write that returns
  `BUS_BUSY` must be retried, by the CPU and by test harnesses alike.

## Automation & CI

- Install IRIX only from original media (see `clanker/irix/irix-install.md`).
  Never use a pre-built MAME CHD as a shortcut.
- Disk images in local configs are often written directly. Never kill iris
  while IRIX runs; shut the guest down, or use a copy-on-write overlay.
- Normal `Machine::stop` (GUI Stop/Quit and guest power-off) saves NVRAM and
  the Indigo2 motherboard EEPROM. Before forced termination, run `rtc save`
  (Indy) or `nveeprom save` (Indigo2) from the monitor to retain PROM changes.
- Whole-page compilation is the only jitv2 implementation; `j2wp` is an alias.
  Persistent compiled-page reuse is opt-in via `[jitv2] cache` / `cache_dir`.
- CHD, camera, DaynaPort, Ultra64, IP28, and ppmem are unconditional in core;
  do not use their retired Cargo features in build commands.
