# Host calls hold the CPU thread: long work goes to the GL thread

**Keywords:** hostgl,hostcall,audio,crackle,hal2,clock,ticks,batch,ENOTYET,OP_BATCH_ASYNC

**Symptom.** Sound crackles under a busy host-GL program (a demo's loud bass
notes, Quake), while the same program's sound alone plays clean. HAL2
captures (`IRIS_HAL2_CAPTURE`) show short runs of silence, ~20 a second.

**Cause.** A host call (syscall 3000) is trapped in `exec_syscall` and
answered on the thread that runs the emulated CPU. Until it returns the
machine stands still: no instruction, no interrupt. Replaying a big command
buffer took 2-5 ms; IRIX took its 1 kHz clock tick late about 15 times a
second (99% of the late ticks fell inside host calls), its audio driver
refilled the 202-frame DMA ring late, and HAL2 played gaps. It is the
host-GL side of `clanker/mgras/guest-gl-benchmarks-miss-board-time.md`:
whatever keeps the guest from taking its ticks costs it time and sound.

Loosening HAL2 instead does not work: letting codec A read further ahead of
the polled DMA position plays slots IRIX has not refilled yet (audio at
0.6x speed, "robot" voice).

**Fix (iris-hostgl/src/threaded.rs).** The GL service runs on its own thread,
which owns every context:

- `OP_BATCH_ASYNC`: the buffer is read on the CPU thread (a missing page is
  still a need-page answer), queued, and answered at once. Only buffers that
  read nothing outside themselves may be sent this way; one that tries is
  cut short and logged.
- Every other call goes through the same queue and the CPU thread waits, so
  calls run in order. It lends the GL thread its `GuestMemory` for the call.
- A call from a client with buffers still queued is answered `ENOTYET`
  (0x3001) rather than held behind them; the library `sginap(0)`s and asks
  again. Waiting there would move the stall to the frame's swap.

**Rules that follow.**
- Never add a host call that does milliseconds of work on the CPU thread.
  Queue it, or answer `ENOTYET` and let the guest come back.
- A library must not send `OP_BATCH_ASYNC` for a buffer holding by-reference
  data (client arrays, pixel pointers): the guest may change that memory
  before the buffer runs. The guest library flushes those with `OP_BATCH`.
- `ENOTYET` is only ever answered to a client that has sent
  `OP_BATCH_ASYNC`, so older libraries never see it.
- The thread wants a host core beside the CPU thread; on a host short of
  cores (a 4-vCPU VM) it costs frame rate. `IRIS_HOSTGL_INLINE=1` puts the
  service back on the CPU thread for comparison.

**Measuring.** `ioc ticks` shows how late IRIX takes its clock; HAL2 capture
plus a count of zero runs shows dropouts.
