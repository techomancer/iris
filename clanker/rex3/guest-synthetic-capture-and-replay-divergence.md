# Guest REX3 scanout works; five old compiled captures need recapture

**Status, 2026-11-06:** Private, ROM-free direct-MMIO guest workloads were
captured with the live REX3 CPU renderer in compiled and interpreter modes.
All 14 captures passed guest pixel reads and independent host screenshot
checks. They are synthetic functional probes, not desktop traffic or
performance measurements. The raw recordings, traces, and images remain
private and must not be copied into the repository.
The retained HOSTRW state fix is validated; five old compiled-origin captures
still need recapture before they can be used as strict replay evidence.

## Desktop blocker and scanout

Both available IRIX images lack the licensed Newport kernel driver:
`hinv` lists no graphics, `gfxinfo` reports no boards,
`/var/sysgen/boot/ng1.a` and `/hw/gfx` are absent, and Xsgi cannot find a
screen. This is a guest setup blocker, not a CI display-black bug. Obtain an
authorized matching driver from appropriate media, install it only into a new
disposable overlay, run the matching autoconfig, reboot, and verify `hinv`,
`gfxinfo`, `/hw/gfx`, Xsgi, and X connectivity before real desktop capture.

Scanout itself works: two independent color-bar frames differed in all
128,000 test pixels and each contained 112,000 nonblack pixels. No game or
game-ROM-derived content was captured. Normal machine firmware was used to
boot and configure scanout; the pre-clear splash captures were deleted.

## Workloads and Vulkan replay coverage

Seven cases, including both color-bar frames, were each captured once per
CPU-renderer mode. Each recording has initial, explicit mark, and final hash
checkpoints. The GPU was not enabled during guest capture; the counts below
come from subsequent Vulkan replay. Handled/fallback counts include READ GOs
as well as drawing GOs.

| Case | Draws / reads | Vulkan handled / fallback GOs | Shape |
|---|---:|---:|---|
| bars A and B, each | 8 / 8 | 8 / 8 | Eight color bars |
| windowlike | 9 / 6 | 9 / 6 | Ordered overlapping window-like blocks |
| longfills | 33 / 3 | 33 / 3 | Includes 32 spans of 1,200 pixels |
| tiny | 289 / 4 | 289 / 4 | Includes 288 8x8 blocks |
| pixelclip | 67 / 6 | 2 / 71 | Clipped blocks and 64 1x1 HOSTW writes |
| mixed | 32 / 6 | 23 / 15 | RGB24, CI8, and HOSTW operations |

The matcher handles ordinary RGB24 fills and spans. READ, HOSTW,
SMASK/non-default CID clipping-mode, and CI8 operations fall back; pending GPU
work must first be fully synchronized to the CPU plane, with a full-plane
upload when GPU work resumes. None of these cases reached the 1,024-draw batch
limit. Recorded ordering, coverage, and flushes are functional evidence, not
live GFIFO timing or a representative desktop workload distribution.

## Pre-fix replay evidence

Before the fix, all three checkpoints passed in the following numbers of
cases; all recorded bus reads matched. Compiled replay was separately verified
with compiled GO counters, not merely by selecting the JIT in a test build.
This historical matrix does not describe the current engines.

| Recording origin | Interpreter replay | Compiled CPU replay | Vulkan replay |
|---|---:|---:|---:|
| Live compiled | 2/7 | 6/7 | 2/7 |
| Live interpreter | 7/7 | 2/7 | 7/7 |

In the pre-fix sanitized `pixelclip` diagnostic, the entire 2048x1024 RGB and
auxiliary planes and all six reads matched. After normalizing the
time-dependent STATUS and VC2 working-cursor-Y fields excluded from recorder
hashes, the only differing state was `context.host_shifter`:

| Live compiled | Compiled replay | Interpreter / GPU replay |
|---:|---:|---:|
| `0x0000000000000000` | `0x00000000ff000002` | `0x00000000ff000001` |

## Post-fix validation and corpus gate

The retained fix makes Cranelift persist `host_shifter`, aligns HOSTRW word
boundaries and PIO idle-cursor retirement, and corrects compiled offscreen
HOSTR reads. Current interpreter, counter-proven Cranelift, and Vulkan agree
on `host_shifter = 0x00000000ff000001`, both entire RGB/auxiliary planes, and
the reads in the sanitized `pixelclip` diagnostic. The compiled verifier
proved every GO was compiled, with zero interpreter fallbacks.

All three checkpoints and all reads now pass in these counts for each engine:

| Recording origin | Interpreter replay | Compiled CPU replay | Vulkan replay |
|---|---:|---:|---:|
| Live compiled | 2/7 | 2/7 | 2/7 |
| Live interpreter | 7/7 | 7/7 | 7/7 |

Thus 9/14 private captures pass on all three engines: all seven
interpreter-origin cases, plus compiled-origin `barsB` and `longfills`.
Historical compiled-origin `barsA`, `windowlike`, `tiny`, `pixelclip`, and
`mixed` still fail the same mark and final hashes on every current engine,
with zero differing reads, because they recorded old live compiled state.
They require fresh post-fix recapture and are not corpus candidates. No hashes
were waived; matching pixels and reads alone do not excuse failed checkpoints.

Recapture those five cases privately, then generate independent synthetic
post-fix corpus cases for overlapping window blocks, 1,200-pixel spans, 8x8
blocks, 8-pixel spans, single pixels, HOSTW/HOSTR residual state and reads,
clipping, CI8/mixed fallback, and 1,024-draw batch/fence boundaries. Require
every hash and bus read to match; do not use private raw captures as
repository assets.
