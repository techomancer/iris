REX3 trace and recording (rex3_trace.rs, rex3_rec.rs)

Behind the `gputrace` Cargo feature (on with `developer`; add
`--features gputrace` to a plain or `lightning` build). Without it the bus
wrappers call bus_* directly and none of this is compiled.

Same monitor syntax as gr2/mgras:
  rex trace <file> [gfifo,cpu|all]   rex trace off|flush   IRIS_REX3_TRACE=<file>
  rex rec <file>[.txt]               rex rec mark|off      IRIS_REX3_REC=<file>
  REX3_REPLAY=<file> cargo test --release rex3_replay_file -- --nocapture
    (REX3_REPLAY_ENGINE=interp|jit, REX3_REPLAY_TRACE, REX3_REPLAY_NOCHECK)
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
only at checkpoints, look for another register some thread other than the
bus updates before suspecting the draw path.

Known window: an access that loaded tap==0 just before `rex rec` raised the
bit, and was then descheduled until after the start snapshot, is missing
from the recording. Not seen in practice.
