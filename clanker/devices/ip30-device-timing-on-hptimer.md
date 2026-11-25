# How the Octane's devices pace themselves: `Timer` deadlines, a few threads

Status: design (read the code before relying on details).

**What to know.** The Octane's devices poll only for serial input. Timed work is
a [`Timer`](machine-clock-and-timer.md) holding a deadline in machine time. When
it fires, the device does the same catch-up a register access would, so an idle
guest still makes progress. Threads exist only where something blocks.

- IOC3 (`src/dev/ioc3/`, [docs/dev/ioc3.md](../../docs/dev/ioc3.md)): one
  receive thread per serial port feeds host input into the FIFO. One `Timer`
  per slot covers the SDMA receive timer, the transmit delay, the
  interrupt-output pulse and the DS1685's NVRAM debounce. Those are also
  evaluated lazily on every register access. A callback takes the chip's gate,
  so `stop` waits one out. The DS1685's writer thread takes the image from the
  timer so file I/O stays off the shared timer thread. All are joined in
  `stop`.
- RAD1 (`src/dev/rad1.rs`): status completions run on one `Timer` on the
  machine clock. The output pump is a separate recurring 10 ms host timer,
  present only while the analog output runs with host audio on. `start` arms
  both and `stop` removes them and waits out a callback in flight.
- ISP1040 (`src/dev/isp1040.rs`): its ring thread starts in `start` and is
  joined in `stop`. Three `Timer`s (the absent-target selection timeout and
  each bus's reset hold-off) wake that thread when they fall due; they run on
  the machine clock, so they stand still while the machine is stopped.

**Why the ISP1040 has a thread.** Running SCSI synchronously inside one
mailbox store stalled the CPU thread for up to about 2 s, and IRIX printed
"hasn't seen a scheduler clock interrupt" warnings. With the ring on its own
thread the CPU keeps ticking; those warnings went away.

**The 2 ms receive sleep is a compromise.** `SerialBackend` has no blocking
read and no wake, so each receive thread parks for `RX_IDLE` (2 ms) between
polls. That bounds input latency and costs a wakeup per port.

**How to apply.** For a new Octane device, build a `Timer`, arm it with
`set_due` and reuse the catch-up that register accesses already run. Add a thread only for
something that blocks or takes milliseconds, and join it in `stop` so nothing
touches guest memory or the interrupt line after the machine stops. Do not run
slow host work, file or SCSI, on the timer thread or the CPU thread. If
`SerialBackend` ever gains a blocking read or a wake, replace the 2 ms sleep.
