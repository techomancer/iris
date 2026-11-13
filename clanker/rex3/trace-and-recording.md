# REX3 trace and recording (rex3_trace.rs, rex3_rec.rs)

Behind the `gputrace` Cargo feature (on with `developer`; add
`--features gputrace` to a plain or `lightning` build). Without it the bus
wrappers call bus_* directly and none of this is compiled.

Same monitor syntax as gr2/mgras:
  rex trace <file> [gfifo,cpu|all]   rex trace off|flush   IRIS_REX3_TRACE=<file>
  rex rec <file>[.txt]               rex rec mark|off      IRIS_REX3_REC=<file>
  REX3_REPLAY=<file> cargo test -p iris --lib --features gputrace rex3_replay_file -- --ignored --nocapture
    (REX3_REPLAY_ENGINE=interp|jit|gpu, REX3_REPLAY_TRACE, REX3_REPLAY_NOCHECK;
     gpu requires rex-gpu)
`rex buslog` is gone; the GF trace lines replace rex3.log (bulk-transfer
lines keep the same word-count/first/last text).

Where the cost is:
  - BusDevice methods are thin wrappers over the bus_* inherent methods: with
    gputrace, one relaxed load of `tap`, zero when nothing runs. Calls between bus methods
    (read64 -> read32, write64 -> write32, dma_read64 -> read64) use bus_*,
    so a 64-bit access is recorded once, as itself. Keep it that way.
  - GF trace lines are formatted on the processor thread; the CPU thread only
    traces accesses that never enter the GFIFO (CONFIG, DCB, reads, DMA).
  - Recording holds `rec` across the access and its record, on the CPU and
    MC-DMA threads: that is what makes record order == GFIFO order with two
    producers. The processor thread must never take `rec` (the checkpoint
    path waits for it to drain while holding the lock). Lock order rec -> trace.
  - TraceWriter (src/trace_writer.rs) does the file I/O on its own thread;
    producers only append to memory.

Recordings open with an S record (save_state + context bits snapshots skip
+ RLE framebuffers), so they can start mid-session. Text recordings without
S replay on a power-on board.

The checkpoint hash leaves out what changes with time, not with bus traffic:
  - STATUS (retrace and FIFO interrupt flags);
  - VC2 WORKING_CURSOR_Y, which the refresh thread copies from CURSOR_Y_LOC
    every frame.
Plain (non-GO) STATUS reads are not recorded at all. If a replay diverges
only at checkpoints, check both time-dependent fields and bus-driven residual
state; matching pixels and reads do not imply matching context. The private
[guest capture](guest-synthetic-capture-and-replay-divergence.md) exposed a
pre-fix `host_shifter` mismatch despite identical planes and reads. Current
engines agree on the diagnostic after the HOSTRW fix, but five old compiled
recordings still fail their mark and final hashes and require recapture. Do
not waive hashes. Strict replay now fails on a differing bus read as well as a
checkpoint mismatch; inspect its summaries for the affected offsets. A test
build selecting `jit` may still use the interpreter; prove compiled GO
dispatch with counters when claiming compiled replay.

The required [synthetic corpus](../../src/dev/ng1/testdata/README.md) is
independently generated from zeroed power-on boards, not private recordings.
Its strict CPU test validates the manifest, reads and every checkpoint and
requires actual Cranelift dispatch; the hardware-gated GPU test asserts
handled and fallback counts and compares full planes synchronously at display
fence publication, requiring completed GPU readback before later idle flushes.
The first batch leaves an observable pixel as well as an overlapping pixel;
dropping the batch or publishing its fence early is negatively tested. Corpus
discovery rejects unknown names before bounded reads, and the generator uses
no-follow handling below a pinned checkout root and per-file atomic replacement.
On Windows traversal uses the initially opened root handle, not a second
canonical pathname or ancestor-listing handles. Valid IDs are required for
the root, traversed directories and anchor (128-bit when supported; otherwise
64-bit). A pre-existing `.rex3-corpus-root` regular-file anchor is
identity-checked and held for the complete generation window, not generated.
On Windows its no-delete handle keeps the output directory nonempty despite
attribute-only reparse access, and I/O is directory-handle-relative; Unix uses
descriptor-relative I/O. Checkout-ancestor symlinks/junctions remain valid.
The ignored
one-file test above is optional and does not substitute for the corpus; its
documented command is checked against the libtest listing. BUS_BUSY retries
are separately tested with a mock bus; corpus DMA transfers currently use one
word.

Known window: an access that loaded tap==0 just before `rex rec` raised the
bit, and was then descheduled until after the start snapshot, is missing
from the recording. Not seen in practice.
