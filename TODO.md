# Feature-completion backlog

Reviewed against repository code through `862d0fe` on **2026-10-07**.
The previous scratch list is preserved unchanged in [TODO_archive.md](TODO_archive.md).

The target is functional completeness for the existing Indy IP24, Indigo2 IP22,
and Indigo2 IMPACT IP28 profiles, their runtime CPU/graphics choices, and the
features exposed by the CLI and iris-gui. Cycle-accurate R10000 execution, new
machine families, and instruction-level GE7/GE11 microcode emulation are separate
projects. High-level graphics emulation is the current implementation strategy.

**Implementation** identifies a gap visible in current source. **Validation**
identifies behavior that needs an observed guest/host test before it can be
considered complete. **Triage** retains an old report that must be reproduced
before treating it as a current bug. Items below are proposed work, not fixes
performed by the documentation review. Each checkbox includes its completion
condition; update the linked guide and changelog when it lands.

## First: complete restore and CPU correctness

- [ ] **Implementation — GR2 snapshots and rollback.** Serialize RE3 VRAM,
  raster registers, HQ2 parser/geometry/context state, and pending work at a
  defined quiescent boundary. Current `Saveable` only saves selected registers,
  microcode, SRAM, and display tables. Add equivalent support to the in-memory
  checkpoint path. Complete when a restored X/irisGL scene, framebuffer hash,
  and subsequent drawing match uninterrupted execution.
  Sources: [GR2 save/load](src/dev/gr2/mod.rs),
  [machine snapshots/checkpoints](src/machine.rs).
- [ ] **Implementation — IMPACT snapshots and rollback.** `Mgras::save_state`
  returns an empty table and `load_state` is a no-op. Include HQ3/GE state,
  ERAM contexts, RSS registers, tiled pixel memory, TE1 texture memory, DCB
  tables, and interrupt state; include the same board state in checkpoints.
  Complete when save/load and rollback survive context eviction, textured
  drawing, and guest resolution changes with matching output.
  Sources: [MGRAS save/load](src/dev/mgras/mod.rs),
  [board design](rules/mgras/DESIGN.md).
- [ ] **Implementation/validation — snapshot every active device.** Audit
  HAL2, VINO/CDMC, DaynaPort, Ultra64, host networking, and HostGL against the
  state actually captured in `Machine`. Define reset/reconnect behavior for
  external camera/audio/network/bridge/GL resources. Host sockets and GPU
  objects cannot simply be serialized. Complete when each supported device
  either resumes correctly or has an explicit, enforced restore limitation.
  Source: [snapshot and checkpoint manifests](src/machine.rs).
- [ ] **Implementation — remaining architectural FPU differences.** Recheck
  remaining hardware-backed discrepancies and add ordinary arithmetic
  rounding/Overflow/Inexact coverage: the common FP arithmetic helper explicitly
  omits Overflow/Inexact calculation. Condition-code, trapped-result/Flag,
  Cause, denormal/FS, and signaling-NaN fixes are already implemented; rerun
  their guest checks rather than treating the old findings as still open.
  Check R4400 unsupported COP1 exception encodings as well. Complete when
  interpreter and JIT agree with independent R4400/R5000 oracle expectations
  and CI limits reflect the actual remaining failures.
  Sources: [findings](cpu-tests/docs/findings.md),
  [current baselines and historical runs](cpu-tests/docs/status.md).
- [ ] **Implementation — architectural WatchLo/WatchHi exceptions.** CP0
  registers are stored/read back, but no access checks use them to raise
  `EXC_WATCH`. Implement each CPU model's supported match/access semantics,
  including translation and JIT inline-memory paths. Complete when guest
  watch exceptions have correct Cause/EPC/BD behavior and do not depend on
  monitor/GDB breakpoints.
  Sources: [CP0 registers](src/cpu/mips_core.rs),
  [execution and memory paths](src/cpu/mips_exec.rs),
  [JIT memory guards](src/cpu/jitv2/codegen.rs).

## CPU, cache, JIT, and timing

- [ ] **Validation — R10000 coverage.** Extend the CPU/PROM oracle matrix
  beyond the current R4400/R5000 CI baselines. Exercise 44-bit addresses,
  64-entry TLBs, shadow CACHE/tag/data/ECC operations, MIPS IV, FR modes, and
  exception behavior in both engines. Complete when IP28 PROM diagnostics,
  IRIX boot, and guest CPU tests have recorded, reproducible expectations.
  Sources: [R10000 shadow cache](src/cpu/mips_cache_shadow.rs),
  [CPU suite](cpu-tests/README.md), [IP28 guide](docs/indigo2-ip28.md).
- [ ] **Implementation — R5000 secondary cache, if kept as a supported target.**
  `r5ksc` deliberately refuses to compile because its L1I behavior fails
  tests. Fix the cache tests and validate a real supported Indy SC layout
  before exposing it. Decide whether to retain or remove `r5ksc_triton`;
  that O2 variant is outside the current machine scope. Complete when every
  advertised cache variant works or is clearly retired.
  Sources: [build guards](src/lib.rs),
  [L1I investigation](rules/testing/r5k-l1i-cache-bugs.md).
- [ ] **Validation — JIT opcode fusion.** The optional `jitv2_opcodefusion`
  remains off by default after live boot/shutdown failures. Reproduce remaining
  foreign-delay-slot/page-boundary cases and compare interpreter/JIT behavior
  before changing its default. Complete when the fusion-specific corpus,
  exception tests, and real guest boot/shutdown matrix pass.
  Source: [fusion hazard](rules/jitv2/jitv2_lui_fusion_foreign_delay_slot_hazard.md).
- [ ] **Implementation/validation — writes to executing JIT pages.** Audit
  ordinary stores, masked stores, DMA, and SC/SCD against generation updates
  and currently executing regions. `atomics.rs` explicitly notes that SC to
  its own code page is only checked after region exit. Complete when the
  chosen architectural visibility rules are documented and tested with
  non-vacuous self-modifying-code cases.
  Sources: [atomics](src/cpu/jitv2/atomics.rs),
  [physical pages](src/cpu/jitv2/jitv2.rs), [RAM](src/ppmem/ppmem.rs).
- [ ] **Implementation — resolve `j2 min-calls`.** The monitor stores and
  reports this legacy setting, but whole-page dispatch ignores it. Either
  implement an arrival threshold with a measured benefit or retire the
  misleading control. Complete when monitor behavior and help agree.
  Sources: [setting](src/cpu/jitv2/jitv2.rs), [monitor](src/cpu/mips_exec.rs).
- [ ] **Validation — JIT pool and persistent cache lifecycle.** Exercise
  multiworker compilation, arena exhaustion/flush, stop/start, reset, restore,
  cross-CPU/FR/host-architecture cache rejection, and interrupted cache writes.
  Complete when stress runs show no stale execution, deadlocks, or acceptance
  of incompatible cached objects. Existing unit tests and historical cache
  benchmarks alone do not close this item.
  Sources: [cache design](docs/jitv2-persistent-cache.md),
  [compile pool](src/cpu/jitv2/jitv2.rs).
- [ ] **Triage — interrupt/reset and R5000 IRIX 5.3 reports.** Reproduce the
  archived IP7-after-reset/reboot report and Xsgi crash before making a fix.
  Test Compare acknowledgement/deadlines, PIT-driven ticks, idle-pause wakeups,
  repeated warm resets, and R5000 5.3 desktop startup. Complete when the old
  reports have reproduction evidence or an explicit resolved/not-reproduced
  result. Source: [original reports](TODO_archive.md).

## Graphics

- [ ] **Implementation — IMPACT buffer formats and 3D state.** Complete
  DRBpointers' second field (`[19:10]`), per-window front/back selection and
  swaps, 36-bit pixel formats, 12-bit color/CI, Z/stencil, the raw RDRAM
  window, and dual raster-engine behavior for the relevant board variants.
  Complete when independent guest diagnostics and depth/stencil/double-buffer
  workloads pass on Solid, High, and Maximum configurations.
  Sources: [recorded gaps](rules/mgras/DESIGN.md),
  [pixel memory](src/dev/mgras/pixmem.rs), [RSS](src/dev/mgras/rss.rs).
- [ ] **Validation/implementation — IMPACT GL and textures.** Build a token
  and visual coverage matrix for context switching/eviction, geometry,
  clipping, lighting, fog, texture download/formats/filtering/LOD, readback,
  and state shared with X. Implement gaps discovered by captured workloads;
  do not treat recent OKR fixes as exhaustive GL coverage. Complete when
  traces replay and framebuffer/readback comparisons match expected output.
  Sources: [GE HLE](src/dev/mgras/gl.rs), [TE1](src/dev/mgras/te1.rs),
  [record/replay](src/dev/mgras/record.rs).
- [ ] **Implementation/validation — GR2 command and display fidelity.**
  Inventory HQ2 commands that still take the unimplemented-command path;
  prioritize tokens used by shipped X/irisGL software. Resolve the source's
  unverified pixel/context/display-layout assumptions with captured commands
  and hardware references. Complete when XZ/Extreme PROM, X, context,
  pixel-transfer, and 3D workloads pass with independent visual/readback checks.
  Sources: [HQ2](src/dev/gr2/hq2.rs), [GL](src/dev/gr2/gl.rs),
  [RE3](src/dev/gr2/re3.rs), [compositor](src/dev/gr2/gr2comp.rs).
- [ ] **Triage/validation — Newport cursor and primitive edge cases.**
  Reproduce the archived cursor-hotspot and block skip-first/last concerns.
  Compare generic, precompiled, and `rex-jit` paths with an independent
  expected image, including overlays/CID, logic ops, stipples, AA lines,
  blending, and dithering. Complete when old concerns are closed with evidence
  and all drawing paths agree with the intended hardware semantics.
  Sources: [archive](TODO_archive.md), [REX3](src/dev/ng1/rex3.rs),
  [generic drawing](src/dev/ng1/rex3_generic.rs).
- [ ] **Validation — display modes and frontends.** Check guest-programmed
  IMPACT timings, GR2 scanout, Newport presets/dual heads, screenshots, GUI
  capture, resize/fullscreen, HiDPI, cursor/input alignment, and display-off/on.
  Complete when supported dimensions reach CLI, GUI, and capture without
  clipping or stale rows. Custom physical IMPACT VFO compatibility remains
  a separate hardware question.
  Sources: [display interface](src/gfx_display.rs),
  [IMPACT sizing](src/dev/mgras/frame.rs), [GUI capture](iris-gui/src/framebuffer.rs).

## Audio, video, and peripheral registers

- [ ] **Implementation — HAL2 input and remaining modes.** Codec B currently
  writes silence; quad output discards rear channels; AES uses an internal
  loopback rather than host digital I/O. Define and implement supported host
  input/quad/AES behavior, endian handling, CTRL2 gain/mute effects, recovered
  AES clock, and timestamp DMA. Complete when guest diagnostics and channel
  recordings verify the advertised modes, or unsupported physical modes are
  explicitly scoped out. Sources: [HAL2](src/dev/hal2.rs), [guide](docs/hal2.md).
- [ ] **Validation — audio lifecycle.** Check boot chimes, IRIX playback,
  rate changes, underruns, enable/disable, stop/start, host device changes, and
  restoration on supported hosts. Complete when sound is independently heard
  or recorded with correct duration/pitch and no repeated stale buffers.
  Source: [HAL2 pacing/output](src/dev/hal2.rs).
- [ ] **Implementation/validation — VINO/IndyCam capture.** Finish the
  existing checklist: observe IRIX 5.3/6.5 capture, both channel routes,
  repeated-start I2C, CDMC controls, interlacing/diagonal artifacts, and
  non-640×480 descriptor geometry. Distinguish the modeled CDMC controls from
  remaining hardware-register fidelity and the black-hole HPC1 workaround.
  Complete when captured guest frames match the selected source in real apps.
  Sources: [checklist](rules/irix/vino-verification-checklist.md),
  [capture investigation](rules/irix/vino-capture-on-6.5-progress.md),
  [VINO](src/dev/vino.rs), [host sources](src/video_source.rs).
- [ ] **Implementation — serial interrupt vector modification.** Implement
  Z85C30 WR9 Status Affects Vector rather than only reporting its TODO.
  Validate channel interrupts, baud/framing expectations, and reset behavior.
  Complete when SCC diagnostics see the expected vector values.
  Source: [SCC](src/dev/z85c30.rs).
- [ ] **Implementation/validation — fullhouse peripheral gaps.** Resolve the
  logged-only extended PX register, EXTIO bus errors, and relevant EISA
  interrupt paths. Audit MC narrow-width accesses that currently panic,
  using hardware behavior to decide whether to return a bus error or a value.
  Complete when diagnostic access sequences behave architecturally and cannot
  abort the host. Sources: [HPC3](src/dev/hpc3.rs), [IOC](src/dev/ioc.rs),
  [MC](src/dev/mc.rs), [IP22 gaps](docs/indigo2-ip22.md).

## Storage, networking, and bridges

- [ ] **Triage/validation — SCSI media and IRIX 5.3 sense handling.** Reproduce
  the archived CD-ROM sense report. Exercise empty tray, eject/reinsert,
  changer switching, check conditions, DMA/backpressure, both IP22 controllers,
  raw disks, COW overlays, and CHD sidecars. Complete when install and recovery
  workflows preserve media state and expected sense results across engines.
  Sources: [archive](TODO_archive.md), [SCSI](src/scsi.rs),
  [controller](src/dev/wd33c93a.rs).
- [ ] **Validation — disk durability and snapshot coordination.** Check
  interrupted writes, flush/failure paths, snapshot/load with COW and CHD,
  parent/sidecar mismatches, and restore while DMA is pending. Complete when
  guest filesystem checks pass and base images remain intact in overlay modes.
  Sources: [storage](src/scsi.rs), [snapshots](src/machine.rs),
  [COW/CHD design](docs/cow-chd-sync-plan.md).
- [ ] **Triage — Ethernet completion interrupt race.** Investigate the FIXME
  in the SEEQ/NAT control path independently of the already-fixed RX-active
  refusal bug. Complete when a captured reproducer establishes the race or
  the stale FIXME is removed with a justified explanation.
  Sources: [network control](src/net/mod.rs),
  [distinct fixed RX issue](rules/irix/hpc3-rx-channel-refused-while-active.md).
- [ ] **Validation — network services and PCAP.** Stress TCP/UDP forwards,
  DNS fallback, DHCP, TFTP boot, NFSv2/v3, reconnects, and stop/reset with live
  traffic. Check PCAP capture permissions/installation on each host and record
  supported interface constraints. Document NFSv2's 32-bit size limit rather
  than claiming files over 4 GiB work through v2. Complete when service tests
  and guest transfers verify data and lifecycle behavior.
  Sources: [NAT/services](src/net/mod.rs), [NFS](src/net/nfsudp.rs),
  [PCAP](src/net/net_pcap.rs), [GUI capture access](iris-gui/src/capture_access.rs).
- [ ] **Validation/implementation — DaynaPort and Ultra64 bridges.** Verify
  DaynaPort command/status fidelity (including placeholder counters) and
  transfers under load. Test Ultra64 with its actual gopher64 peer: protocol
  mismatch, RAMROM paging, reset/NMI, bidirectional interrupts/RDB, peer loss,
  restart, and shared-memory cleanup. Complete when guest tools perform real
  transfers and recover from peer failures on the advertised hosts.
  Sources: [DaynaPort](src/dev/daynaport.rs), [Ultra64](src/dev/ultra64.rs),
  [bridge protocol](src/ultra_proto.rs).

## Host services, GUI, and delivery

- [ ] **Implementation — HostGL platform and extension coverage.** macOS CGL
  is the only backend; Linux/Windows builds register no GL service. Either add
  those backends or keep availability explicit. Inventory unavailable GL
  entry points and finish SGIS detail/sharpen/filter query result lengths,
  which currently report unsupported. Complete when guest capability queries,
  marshalling/readback, contexts/drawables, and representative GL apps behave
  consistently with the exposed extension set.
  Sources: [HostGL](iris-hostgl/src/lib.rs), [executor](iris-hostgl/src/exec.rs),
  [entry points](iris-hostgl/src/calls.rs).
- [ ] **Validation — hostcall ABI and lifecycle.** Exercise o32/n32/n64
  guest callers, endianness, paged/COW pointers, error paths, and machine
  stop/reset. For HostGL include fonts, pixel-store state, accum buffers,
  multiple contexts, and presentation. Complete when results are checked in
  the guest and visible output is verified; registration/build success is
  insufficient. Sources: [hostcall](iris-hostcall/src/lib.rs),
  [HostGL tests](iris-hostgl/src/tests.rs).
- [ ] **Validation — GUI hardware/configuration parity.** Check all supported
  profile/CPU/board combinations, configuration import/export/migration,
  persistent cache settings, configurable monitor/serial ports, multiple
  machines, safe-stop/halt, and battery-backed save failures. Complete when
  the selected controls reach the machine and Stop/Quit preserves expected
  state without attaching consoles to another instance. Resolve process-wide
  debug/JIT environment leakage between starts: false/blank config values
  currently preserve variables set by a previous machine Start, so an unchecked
  option can remain active. Define explicit override/clear behavior and verify
  switching between machines.
  Sources: [GUI guide](iris-gui-README.md), [worker](iris-gui/src/handle.rs),
  [configuration](iris-gui/src/config_ui.rs), [safe stop](iris-gui/src/safe_stop.rs).
- [ ] **Validation — host and release matrix.** Observe macOS/Linux/Windows
  GUI and CLI behavior, x86_64/aarch64/RISC-V builds where supported, Windows
  ppmem mapping, installers, AppImage, and App Store bookmark/sandbox paths.
  Include GDB step/break/watch behavior in debug builds and expected disabled
  debugging in lightning/fusion builds. Complete when build results and
  live smoke results are recorded separately with remaining limitations.
  Sources: [Windows ppmem](src/ppmem/map_windows.rs),
  [GDB](src/gdb_stub.rs), [packaging workflows](.github/workflows),
  [sandbox integration](iris-gui/src/macos_sandbox.rs).
- [ ] **Maintenance — prevent documentation drift.** Check build-feature
  examples, source links, command help, configuration defaults, repository
  URLs, and changelog landing dates whenever those contracts change. Keep
  historical measurements labeled and tie CPU baseline changes to actual
  oracle results. Complete when review/CI catches stale active examples and
  every new user-facing capability includes its limitations and validation.
  Sources: [README](README.md), [FEATURES](FEATURES.md), [HELP](HELP.md), [HACKING](HACKING.md),
  [CHANGELOG](CHANGELOG.md).
