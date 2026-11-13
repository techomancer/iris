# VPro trace and recording (trace.rs, record.rs)

Status: design, 2026-11-02. Implemented; monitor commands in `debug.rs`.

Same monitor syntax as gr2/mgras/rex:

    vpro trace <file> [fifo,draw,cpu,dma|all]   vpro trace flush|off   vpro trace (state)
    vpro rec <file>                           vpro rec mark|off      vpro rec (state)
    IRIS_ODSY_TRACE=<file>  IRIS_ODSY_REC=<file>   from power-on
    VPRO_REPLAY=<file> cargo test --release --lib odyssey::golden::replay -- --nocapture

## The trace

Behind `gputrace` (on with `developer`). `Trace::on(mask)` is
`cfg!(feature = "gputrace") && mask & what != 0`, so without the feature
every hook, its formatting included, compiles away; with it a hook costs one
relaxed load when nothing is traced.

Where the hooks are, and which thread writes:

| class | hook | thread |
|---|---|---|
| `fifo` | `Board::cmd_dispatch` (every completed word, before it runs: tag, name, operands), `cmd_start` for the two sync words, the completions in `cmd_word` (copy rect, pattern load, image box, image burst, matrix load, rect, query), `cmd_report` (unknown words) | board (`odyssey-work`) |
| `gl` | `Board::gl_command` (every op into the shared context), `gl_draw` (after the raster unit took or ran it), `gl_raster_ops` (clears) | board |
| `cpu` | `BoardLocks::taken`: every bus access that is not a command store, front registers included; a repeat of one (dir, offset, size, value) extends a run, closed by the next line of any class or a flush | processors |
| `dma` | `Board::dma_descriptor`, once the header is read | board |

The sink is a `Mutex<TraceSink>` behind `Board::trace` (an `Arc` the
`BoardLocks` share). It is a leaf lock: nothing is taken while it is held,
so a hook can run on any thread. The display thread flushes it every frame,
at most twice a second (`Odyssey::frame`); `vpro trace flush` forces it.

Names come from `trace::tag_name` (register-select registers, GL-class
words, the parser's own words), `op_name` (the shared context's ops) and
`offset_name` (modelled aperture offsets). A word with no name prints as
`?` with its hex; a program word as `PROG class 0xNN`. Add a name when a
word gets a meaning; the `words.md` findings are checkable this way.

## The recording

`RecSlot` holds the recorder: an `AtomicBool` the hot paths check and a
`Mutex<Option<Recorder>>`. Three parties share it: the bus path
(`BoardLocks::taken` and the command-store branch of `bus_write`), the DMA
path (`RecordingMem`, on the board's thread) and the monitor. Guest memory
is always wrapped in `RecordingMem`, so a recording can start at any time.

Lock order: the processors' lock (`submit`), then the recorder. The bus
path records under `submit`, which is what keeps the records in the order
the stores were taken with two processors; the DMA path takes the recorder
alone; the monitor quiesces (holds `submit`) and then takes it. The old
order (recorder, then `submit`, in `Odyssey::write`) would deadlock
against a monitor that quiesces first, which is why the record moved into
`BoardLocks`.

Marks (`H` records, format version 2) hold `plane_hash_bytes` with the
board idle. Every recording opens with one, written at `power_on` after
`Board::reset` (the depth plane at all ones), and a replay of a recording
that opens with a mark resets its board first. The first live check caught
exactly this: colour, stencil, id and overlay replayed identically and the
depth plane was zero against the live board's 0xffffff, because the replay
started from `Board::new` and the live board from `reset`. Recordings
without marks (version 1) replay on the bare board their checkpoints were
taken on, so `mirrors.hashes` and `perfly.hashes` stay valid.
`vpro rec mark` and `vpro rec
off` add one; `vpro rec mark` and `vpro rec
off` add one. The replay (`golden/replay.rs`) checks each mark as it is
reached and lists marks as checkpoints beside the every-500,000-records
ones. A first mark a fresh board does not reach means the recording started
mid-session (`Report::mid_session`): its marks are then reported, not
failed, and its blessed checkpoints pin the model's determinism under those
stores, not the live screen. Only a power-on recording (`IRIS_ODSY_REC`)
reproduces what the guest saw.

What a `vpro rec` start cannot catch: a front-register access on another
processor that checked `active()` just before the slot was set. Front
registers are not in the hash and do not reach the planes, so a replay is
not moved by it.
