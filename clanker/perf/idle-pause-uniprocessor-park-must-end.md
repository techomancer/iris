# idle-pause: a uniprocessor park must end on its own, not only on a line

Status: rule (2026-11-19).

**Symptom.** Octane (single R10000), V12, `idle-pause` build: a minute after
start (the warm-up) the PROM menu and fx stopped responding. `idle status`
showed parks=6, slept=601 s, wake_irq=0, every wake `wake_other` (the
monitor stopping the CPU). Status 0x34005081: IE set, IM = IP4|IP6.

**Cause.** The PROM's menu loop (drawing the cursor into the VPro FIFO and
polling the console) is ~21 distinct kernel-mode PCs in one cluster with IE
and IM set, so the detector calls it idle. But the PROM enables no HEART
source, so no Cause.IP line ever rises. A uniprocessor `park` had no time
bound (the slice only re-checked `running`), so it slept for good. IE/IM
being set says a line *could* be taken, not that any source will send one.

**Fix.** `park` on a uniprocessor ends after `UP_PARK_MAX_NS` (100 ms) with
no line; that sets `quiet`, and later parks last one `SLICE_NS` (1 ms) until
an interrupt wake clears it. IRIX's clock tick always wakes it first, so a
settled IRIX guest is unchanged; a guest with no interrupt source runs
about a thousand short batches a second (still low host CPU). Counted in
`wake_slice`. Test: `a_park_with_no_interrupt_source_ends_and_then_slices`.

A poll loop the detector mistakes for idle still runs at one batch per
millisecond while it waits; if fx-style polled I/O turns out slow, back off
after a quiet park that is followed by leaving the cluster.

See also `idle-pause-must-not-park-with-interrupts-masked.md` (IM = 0) and
`idle-pause-masked-interrupt-must-wake.md`.
