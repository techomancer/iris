# IRIX calibrates its microsecond delay loop cold, so JIT runs make every delay short

Status: open (measured 2026-10-28 on IP30, IRIX 6.5.22; no fix chosen).

**Symptom.** Some Octane boots print
`ALERT: /hw/node/xtalk/15/pci/3: RAD UST not incrementing` before
"The system is coming up". In the guest's SYSLOG it showed up on 5 of about
62 boots. Two of four jitv2 test boots raised it. The alert is cosmetic: the driver prints
it and goes on, and audio works.

**What the driver checks.** It reads the RAD UST counter (BAR0 0x30), spins a
10 µs kernel delay, reads it again and alerts if it moved by fewer than 3
ticks. The kernel converts with 960 ns per tick (it divides by 0x3c0), so
`UST_NS_PER_TICK` in `src/dev/rad1.rs` must stay 960. The counter itself is
fine. The model enables it on 0x0e then 0x0a, and it counts.

**Cause.** The kernel's microsecond delay is a counted spin loop (11 taken
`bgtz` to self per outer pass). The kernel calibrates it once, early in boot,
against the HEART counter, which runs on host time. The figure kept is the
nanoseconds per outer pass, stored as a word at 0xffffffffffffc4b8 (read it
with `mem 0xffffffffffffc4b8 1 w`). At that point the loop runs cold:

| build | calibrated ns/pass | the same loop later, quiet guest |
|---|---|---|
| interpreter (default features) | 567 (0x237) | ~610 measured in user mode |
| jitv2 | 1148–1267 (0x47c–0x4f3) | ~180–300 (9 passes in 1.6–2.7 µs) |

Under jitv2 every kernel `us_delay` therefore lasts about one fifth of what
was asked. For the RAD check, "10 µs" took 1.6–2.7 µs of host time, which is
2 or 3 UST ticks: right at the threshold, hence one boot in a few.

On the interpreter the calibration matches the steady speed and the check
passes (29 ticks in a traced boot).

**Wider impact.** This is not specific to RAD. Every driver delay and every
delay-counted timeout in the kernel is about 5× short on jitv2. Most of our
devices answer at once, so it rarely matters. Suspect it when a jitv2-only
spurious timeout appears on a device that completes asynchronously (SCSI,
AIC-7880, ISP1040).

**Seen biting.** IMPACT's context switch waits for an idle board with up
to 6 x 15000 DMABUSY polls 10 us apart (about 0.9 s nominal, about 0.2 s
under jitv2). Behind long drawing it ran out, logged "mgras: hq idle wait
timeout" and "FlushHQ (rss_sh) failed for ctxsw", and the X server died.
Mitigated in the model: a STATUS or DMABUSY read of a busy board waits up
to 100 us for it (`Mgras::settle`), so polls count board work.

**Reproduce.** On an overlay boot, trace BAR0 0x30 reads in `Rad1::bar_read`
silently (store the first read, print both at the second). An `eprintln!` or
a gdb breakpoint on the first read stretches the delay and hides the bug.

**Fix options (none chosen).** Make the calibration see steady-state speed
(for instance, have jitv2 compile a page synchronously when it spins in a
tight self-loop at boot). Or accept the short delays and keep device models
tolerant of them. Do not slow the RAD UST down: the kernel hard-codes 960 ns.
