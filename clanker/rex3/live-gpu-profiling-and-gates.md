# Live REX3 telemetry passes A/A, but the shader misses both speed gates

**Status, 2026-11-06:** local opt-in prototype and final release harness on
RTX 5070 / NVIDIA 617.42 / Vulkan, default 1,024-draw batches. Upstream remains
replay-only; these synthetic results do not make the live renderer ready or
justify submitting it.

**Superseded measurements, 2026-11-10:** the barrier shader measured below was
replaced by a tile-binned shader with a tile-major GPU plane; the raster gate
now passes by about 2x and the tiny gate sits at its threshold. See
[the parallelization plan](../plans/rex3-gpu-parallelization.md) for current
numbers. The harness instructions here are unchanged, except that
`REX3_GPU_BATCH_DRAWS` now defaults to 16,384.

## Run and interpret

`rex-gpu` is required. `REX3_GPU_TELEMETRY=1` enables stage accounting;
`REX3_GPU_TIMESTAMPS=1` also enables telemetry and requests optional GPU
timestamps. Both accept only `0` or `1`, and neither enables GPU rendering.
For an experimental live run, set `IRIS_REX3_GPU=vulkan` as well; the report
is printed when the processor stops. For supported modes, CPU observation,
fallback, and display boundaries still require synchronization. See
`FEATURES.md` for the controls and their full semantics.

If Cargo is installed in the usual user profile but not on `PATH`, first run
`$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"`.

For example, with the normal emulator configuration:

```powershell
$env:IRIS_REX3_GPU = 'vulkan'
$env:REX3_GPU_TELEMETRY = '1'
$env:REX3_GPU_TIMESTAMPS = '0'
cargo run --release --features rex-gpu
```

Set `REX3_GPU_TIMESTAMPS=1` before launch for timestamp diagnostics.

From the repository root in PowerShell, with a conformant Vulkan adapter:

```powershell
Remove-Item Env:\IRIS_REX3_GPU, Env:\IRIS_NO_JIT -ErrorAction SilentlyContinue
$env:REX3_GPU_BACKEND = 'vulkan'
$env:REX3_GPU_BATCH_DRAWS = '1024'
$env:REX3_GPU_AA_MAX_ROUNDS = '64'
$env:REX3_GPU_AA_SAMPLE_MS = '600'
$env:REX3_GPU_TELEMETRY = '0'
$env:REX3_GPU_TIMESTAMPS = '0'

# Timestamp-free adaptive overhead A/A; the test toggles telemetry itself.
cargo test --release --features 'rex-gpu,rex-jit' gpu_live_gfifo_instrumentation_noise -- --ignored --nocapture --test-threads=1

# Five-sample speed gate; it always enables interval telemetry.
cargo test --release --features 'rex-gpu,rex-jit' gpu_live_gfifo_performance_gate -- --ignored --nocapture --test-threads=1

# Bounded timestamp check, followed by a diagnostic, not an acceptance run.
cargo test --release --features rex-gpu gpu_timestamp_readback_is_bounded -- --ignored --nocapture --test-threads=1
$env:REX3_GPU_TIMESTAMPS = '1'
cargo test --release --features 'rex-gpu,rex-jit' gpu_live_gfifo_performance_gate -- --ignored --nocapture --test-threads=1
```

The gate creates its own GPU board, so `IRIS_REX3_GPU` must be unset; with
`rex-jit`, `IRIS_NO_JIT` must also be unset for the generated CPU comparison.
The gate includes upload and final drain, reports each GPU sample's interval
delta, verifies handled draws and errors, and compares its seeded full
framebuffer against an untimed CPU replay. It returns a failing test status
when a speed threshold fails. Timestamp-on profiling can perturb performance;
use the timestamps-off gate for acceptance.

The adaptive A/A test uses fixed work per engine after warmup and balanced
off/on/on/off or on/off/off/on blocks. It checks after 16 rounds and every
eight thereafter. The default maximum is 48; the maximum may be a multiple of
eight in `16..=128`, and sample time may be `100..=2000` ms. A **PASS** requires
the simultaneous-look-adjusted telemetry interval inside ±3% and order shift
below 3%. The GPU-off A/A interval must cover zero and lie inside ±3%, with
median absolute pair change and half-shift both below 3%. An interval entirely
below -3% is **FAIL** only with usable controls; otherwise the maximum gives
**INCONCLUSIVE** and a failing test status. CPU A/A and telemetry half-shift
are diagnostic; a large half-shift questions the common-median assumption.
The sign intervals assume independent blocks with a common median. Even an
apparent speedup above +3% is inconclusive; do not treat a noisy run as a pass.

Read timestamp support, counts, discarded samples, skipped submissions, and
errors alongside the `GPU delta`. Unsupported/unavailable timestamps mean no
timestamp data; `readback_transfer: false` means transfer timing is absent
even if compute timing is available. Neither means zero time. The bounded
test can return without measuring when support is unavailable. A requested
stage with no usable samples is inconclusive, and profiling errors invalidate
timestamp totals.

At most 64 submissions are retained between readbacks, including a slot
reserved for the final readback; skipped submissions make GPU totals
incomplete. Divide `gpu_compute.nanoseconds` by its usable `count` for a
sampled mean, not by all submissions. The transfer timestamp measures the
GPU buffer-to-readback copy, not the upload. Drain before both snapshots to
attribute samples to an interval; deltas reject resets, different engines,
and timestamp support/error changes. Match/enqueue times sample one in 256
attempts, not every draw. GPU compute, host stages, and device wait may
overlap; do not add them as an elapsed-time breakdown.

On the tested Windows toolchain, the pre-existing libchdman
`__std_max_element_4i` link failure (`LNK2019`/`LNK1120`) required a
disposable test-only compatibility object. Supply its absolute, space-free
path with `$env:RUSTFLAGS = '-C link-arg=<object-path>'`. It is not committed
and is not a fix for the underlying toolchain issue.

## Final synthetic measurements

The overhead gate used 8-pixel GFIFO draws, timestamps off, 64 maximum rounds,
and 600 ms/sample; it stopped **PASS** at round 48. Telemetry on/off paired
median was **-0.959%** with a
**[-2.863%, +1.006%]** 99.79% per-look interval under the simultaneous-look
correction; nominal simultaneous coverage over planned looks is at least
97.87% per metric. GPU-off A/A interval was **[-2.040%, +1.456%]** and order
shift **0.308%**. CPU A/A was diagnostic. Other deliberately noisy runs were
**INCONCLUSIVE**. This bounds telemetry overhead in the stable run, not the
extra overhead of GPU timestamp profiling.

| Timestamps off, five-sample medians | CPU | Vulkan | Vulkan/CPU | Gate |
|---|---:|---:|---:|---:|
| Raster, 1,280-pixel spans | 1,007.7 Mpx/s | 893.9 Mpx/s | **0.887x** | >= 1.20x: fail |
| Tiny, 8-pixel spans | 78.3 Mpx/s | 11.8 Mpx/s | **0.151x** | >= 0.95x: fail |

All intended GPU draws were handled, with zero fallback/errors, and all ten
full-frame comparisons passed. Seventeen telemetry statistical/accounting
tests passed, as did matcher/telemetry units, the bounded timestamp test,
both Vulkan differential tests with timestamps enabled and disabled, and
seeded-frame validation with JIT enabled and `IRIS_NO_JIT=1`. Independent
review found no significant remaining instrumentation issues.

| Timestamps on, median us per 1,024-draw batch | Raster | Tiny |
|---|---:|---:|
| GPU compute | 1,421.06 | 669.32 |
| Host resource creation | 49.53 | 22.36 |
| Host command encoding | 53.03 | 42.94 |
| Host submit | 28.93 | 34.09 |
| Host device wait | 1,284.51 | 616.27 |

These are bounded diagnostics, not complete totals. The sum of the separately
reported resource creation, encoding, and submit medians is about 9% of
raster compute and 15% of tiny compute. Eliminating those host stages alone
cannot plausibly meet the gates:
the displayed ratios require roughly **1.35x** and **6.29x** further GPU
throughput, respectively. The previously quoted **1.41x** raster gap used the
superseded `0.849x` baseline; the corrected baseline is `0.887x`.

## Next decision

The shader processes an ordered batch in one workgroup, serializing draws with
barriers. Instrumentation and private synthetic guest capture are complete;
the captures are functional evidence, not live timing data. Actual desktop
capture is externally blocked by a missing licensed guest driver. The retained
HOSTRW state fix is validated: current interpreter, counter-proven Cranelift,
and Vulkan agree on `pixelclip`'s `host_shifter = 0xff000001`, full RGB/aux
planes, and reads. All three pass all hashes and reads for 9/14 captures:
all seven interpreter-origin cases and compiled-origin `barsB` and
`longfills`. The other five are old compiled-origin recordings whose mark and
final hashes fail on all engines because they contain pre-fix live state. See
[guest capture and replay validation](guest-synthetic-capture-and-replay-divergence.md);
raw captures stay private.

Recapture those five cases after the fix without waiving hashes, then establish
a non-empty independently generated synthetic/post-fix corpus and test
device-loss/readback recovery. After those prerequisites, try conflict-free
epochs or order-preserving per-tile parallelism, keeping overlapping draws and
observation/fallback boundaries ordered. Split dispatches where a global
barrier is required, including cross-tile read/write hazards. Repeat the corpus
and performance gates before reconsidering submission. Do not replace wgpu
with native Vulkan unless measurements identify material overhead specific to
wgpu.
