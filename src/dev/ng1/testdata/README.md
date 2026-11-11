# Synthetic REX3 replay corpus

These recordings and their generator are distributed under the repository's
[BSD 3-Clause license](../../../../LICENSE). They were independently authored
as register programs, not extracted or transformed from guest recordings,
images, framebuffer snapshots, binaries, or private pixel data. Each case
starts with a fresh power-on board and explicitly zeroed RGB and auxiliary
planes. The dimensions exercise shapes observed in synthetic guest probes;
the colors, positions, ordering, and data here are newly chosen. This is not a
representative desktop workload or a performance benchmark.

`../rex3_corpus.rs` is the source of truth. It sends its programs through the
repository's interpreter, checks independently specified reads and pixels,
then emits canonical text `REX3REC` records and BLAKE3 state checkpoints.
`manifest.toml` identifies the generator, license, features, per-case intent,
coverage tags, reads, checkpoint indices, dispatch expectations, fallback
reasons, checksums, and the required `output_anchor` name. The first, explicit
mark(s), and final checkpoint are mandatory. The recordings and manifest are
generated; this README and `.rex3-corpus-root` are not. Repository attributes
keep generated files' canonical LF line endings on every checkout.

| Case | Intent |
|---|---|
| `overlap_windows` | Ordered overlapping RGB24 blocks |
| `long_spans` | 32 spans of 1,200 pixels |
| `tiny_tiles` | 8x8 blocks, 8-pixel spans and single pixels |
| `hostrw_residual` | HOSTW/HOSTR reads and residual packed-CI8 shifter state |
| `dma_hostrw` | Packed-CI8 bulk upload, bulk read and primed scalar DMA read |
| `clip_ci8_mixed` | SMASK0 clipping and CI8 fallback between RGB24 fills |
| `batch_1025_fence` | 1,025 draws, with a surviving first-batch pixel and an overlapping final draw, followed by an XMAP display fence |

Run the mandatory CPU gate with
`cargo test -p iris --lib --features gputrace,rex-jit,rexdiag rex3_replay_testdata -- --nocapture`.
It binds record order, reads and mark positions to the authored program even
if the manifest is edited, verifies every checkpoint and deterministic read,
proves Cranelift shader dispatch with GO counters, and compares full planes
and context against the interpreter. `rex3_corpus_regeneration_is_deterministic`
generates each file twice in memory and compares both versions with the
checked-in bytes.
Run it and the manifest/discovery negative tests with
`cargo test -p iris --lib --features gputrace,rex-jit,rexdiag rex3_corpus_`.

To intentionally regenerate, set `REX3_CORPUS_OUT` to an **existing relative**
directory containing a pre-existing, regular, non-link `.rex3-corpus-root`
file. The checked-in corpus already has one. For a fresh scratch output,
create the directory and anchor yourself **before** running the generator; it
never creates or overwrites the anchor. For example, in PowerShell:

```powershell
New-Item -ItemType Directory -Path 'target\rex3-corpus-output'
New-Item -ItemType File -Path 'target\rex3-corpus-output\.rex3-corpus-root'
$env:REX3_CORPUS_OUT = 'target\rex3-corpus-output'
cargo test -p iris --lib --features gputrace,rex-jit,rexdiag rex3_corpus_generate -- --ignored --nocapture
```

Review the changes and rerun the determinism and strict CPU tests. Path
components from the checkout root to the output directory must not be links or
Windows reparse points; aliases to or leading to the checkout are allowed. The
trusted root is opened and pinned before resolving the relative path. Windows
traverses from that original handle without reopening checkout ancestors; Unix
also checks the pinned root against its canonical path. Windows requires usable
file IDs for the root, traversed directories and anchor, preferring 128-bit IDs
and falling back to validated 64-bit IDs only when the wider query is unsupported.
The existing anchor is opened without following links, identity-checked, and
held before generation until all output writes finish. On Windows its handle
denies deletion/rename, keeping the directory nonempty even against an
attribute-only reparse attempt; operations use verified directory handles.
On Unix, directory-relative descriptors keep writes anchored even if a name is
unlinked or renamed. Existing output symlinks and reparse points are rejected;
hard-linked destinations are replaced without modifying their other links.
Generated files are atomically replaced individually, not as a set. Ordinary
tests never write this directory.

Discovery checks the complete set of names before opening any contents.
Manifest and README files are limited to 64 KiB each, and recordings to
256 KiB each; metadata and a bounded read both enforce the limits. Symlinks,
reparse points, and non-regular entries are rejected. The ordinary negative
tests use in-memory readers. To additionally exercise real filesystem links
and replacement, explicitly select a writable build directory:

```sh
REX3_CORPUS_IO_DIR=target cargo test -p iris --lib --features gputrace,rex-jit,rexdiag rex3_corpus_filesystem_security -- --ignored --nocapture
```

The test provisions its own anchor before invoking the generator, validates the
real corpus through checkout-root and checkout-ancestor aliases, and rejects
links beneath the root. On Windows it retargets a checkout junction after the
root is opened and checks that reads and writes stay on the original handle.
It also uses `icacls` on a test-created directory to deny listing a checkout
parent, verifies access beneath it, and restores the ACL. A separate process
opens the guarded directory with `FILE_WRITE_ATTRIBUTES` and attempts to make
it a junction:
the attempt fails while the anchor exists, including after guard release, and
succeeds after anchor removal when junctions are supported. If junctions are
unavailable, the affected checks are reported as such. Windows without file
symlink privilege tests junctions instead; Linux CI exercises file and
directory symlinks when the configured job runs.

GPU validation is optional and explicitly ignored unless selected:
`cargo test -p iris --lib --features rex-gpu,rex-jit,rexdiag
rex3_corpus_gpu_ -- --ignored --nocapture --test-threads=1`. It requires an actual
integrated/discrete adapter and the default batch size; the test then pins the
batch to the 1,024 draws these recordings were authored for. It verifies
handled/fallback counts and reasons, interpreter fallback, state equivalence,
and the batch-full and display-fence flushes. The batch case preserves pixel
(6,5) from the first 1,024 draws while draw 1,025 overwrites (5,5); dropping
the first batch fails both the pixel probe and checkpoints. A test-only
observer compares both framebuffer planes at the `gfifo_fence` publication,
before any later idle flush, and requires completed display-fence readback.
An injected early publication proves that stale pixels fail even if a later
idle flush repairs the final state. An unavailable adapter is not reported as
a successful GPU validation.

The DMA case uses one 64-bit word per transfer; BUS_BUSY retry is covered by
separate mocked replay tests, not by a full real GFIFO. Multiword DMA resume,
screen-to-screen, blending, dither, lines, patterns, Z/CID and many other
modes are not corpus coverage.
