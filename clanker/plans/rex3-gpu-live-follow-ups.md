# REX3 live GPU: what the first PR leaves out

Status: plan, 2026-11-10. The first upstream PR carries the experimental live
renderer in a lean form (`rex-gpu`, opt-in through `IRIS_REX3_GPU`). The work
below exists on the `feature/rex3-vulkan-live` branch and is deferred to
separate PRs so each can be reviewed, and accepted or declined, on its own.

## 1. Hardened recovery and lifecycle (`a1043e11`, `cb219ad`)

The lean PR recovers from a GPU failure by replaying its journal and leaking
the wgpu objects. The hardened branch adds:

- a process-wide GPU lease with quarantine, so a failed or still-retiring
  device can never be re-created in the same process;
- a retirement worker that destroys a healthy backend off the processor
  thread, with a grace deadline after which the lease is quarantined;
- `GpuStop`: a machine-stop deadline (500 ms per map wait, 4 s overall)
  instead of the lean 10 s map timeout, so a hung GPU cannot stall stop;
- deferred machine stop (`DeferredStopJob`, the stop reaper,
  `NewportStopIntent`, `gfx_display::DisplayStop`), which this needs in
  `machine.rs`, `gfx_display.rs`, `workstation.rs` and `mips_exec.rs`;
- about 9,400 lines of fault-injection recovery tests
  (`rex3_gpu_recovery_tests.rs`) and public-API integration tests
  (`tests/rex3_gpu_public_api.rs`, `rex3_public_fields.rs`,
  `rex3_screenshot_public_api.rs`, `gfx_display_legacy.rs`).

It conflicts heavily with upstream `machine.rs` after 2026-11-07; rebase it
after the lean PR lands, and split the machine-stop part from the GPU part.

## 2. wgpu / wgpu-core 30.0.1 fork (`third_party/`)

A vendored copy whose queue destructor reports device loss instead of
panicking, so the retirement worker can quarantine instead of aborting; plus
an `iris-test-fault` hook (`rex-gpu-test-hooks`). Needed only by item 1.
Offer it as its own PR, or upstream the destructor change to wgpu instead.

## 3. Unrelated hardening found along the way

Each is independent of the GPU work and should be its own PR:

- O2 and Octane teardown hardening (`63fb31b5`);
- monitor: keep slow client connections (`3c5e16d4`);
- CPU/monitor/GDB stub quiescence and `wd33c93a` changes inside `a1043e11`;
- the non-GPU findings in that branch's `FINDINGS.md` (topology-aware
  affinity, a Windows HostGL backend, an x86-64-v3 build tier, the Windows
  COW parent-directory sync fix).

## 4. Open questions for the renderer itself

- **Shared wgpu layer.** Upstream's `src/wgpu` (`Gpu`, `shader::Source`,
  `Batch`, 32×32 coarse-tile binning) postdates `rex-gpu` and uses the same
  one-writer-per-tile rule. Porting REX3 onto it would drop a second device
  and pipeline set; `clanker/plans/wgpu-raster.md` currently leaves Newport
  out of scope.
- **Tiny-draw gate.** The lean tree measures ~1.54x; the hardened tree was
  at 0.96x, apparently because of its per-register framebuffer-access gate.
  Re-measure when item 1 is rebased, and keep that gate off the per-register
  path if possible. See [the parallelization plan](rex3-gpu-parallelization.md).
- **Real workload.** No IRIX X/GL Newport capture has been replayed on the
  GPU yet; the IRIX 6.5.3 guest with the Newport subsystem installed is the
  next step ([guest capture note](../rex3/guest-synthetic-capture-and-replay-divergence.md)).
- **Backends.** Vulkan is measured on this tree. D3D12 passed correctness on
  the hardened tree (tiny 0.947x there) and has not been re-measured; Metal
  is untested.
