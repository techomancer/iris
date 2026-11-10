# How the Octane's devices pace themselves: `hptimer` one-shots, a few threads

Status: design (read the code before relying on details).

**What to know.** The Octane's devices poll only for serial input. Timed work is a
one-shot or recurring `hptimer` that does the same catch-up a register access
would, so an idle guest still makes progress. Threads exist only where
something blocks.

- IOC3 (`src/dev/ioc3/`, [docs/dev/ioc3.md](../../docs/dev/ioc3.md)): one
  receive thread per serial port feeds host input into the FIFO. One-shot
  timers, one slot each, cover the SDMA receive timer, the transmit delay, the
  interrupt-output pulse and the DS1685's NVRAM debounce. Those are also
  evaluated lazily on every register access. The DS1685's writer thread takes
  the image from the timer so file I/O stays off the shared timer thread. All
  are joined in `stop`.
- RAD1 (`src/dev/rad1.rs`): a recurring 10 ms timer only while the analog
  output runs with host audio on, plus one-shots for status completions.
  `start` arms them and `stop` removes them and waits out a callback in
  flight.
- ISP1040 (`src/dev/isp1040.rs`): its ring thread starts in `start` and is
  joined in `stop`.

**Why the ISP1040 has a thread.** Running SCSI synchronously inside one
mailbox store stalled the CPU thread for up to about 2 s, and IRIX printed
"hasn't seen a scheduler clock interrupt" warnings. With the ring on its own
thread the CPU keeps ticking; those warnings went away.

**The 2 ms receive sleep is a compromise.** `SerialBackend` has no blocking
read and no wake, so each receive thread parks for `RX_IDLE` (2 ms) between
polls. That bounds input latency and costs a wakeup per port.

**How to apply.** For a new Octane device, arm an `hptimer` deadline and
reuse the catch-up that register accesses already run. Add a thread only for
something that blocks or takes milliseconds, and join it in `stop` so nothing
touches guest memory or the interrupt line after the machine stops. Do not run
slow host work, file or SCSI, on the timer thread or the CPU thread. If
`SerialBackend` ever gains a blocking read or a wake, replace the 2 ms sleep.
