# IRIX panics early when the O2's rendering engine is slower than its timeout

Status: fixed (2026-10-26). Also hit with the graphics console and less
than 1 GB.

**Symptom.** `headless = true` (no PS/2 keyboard, serial console) with the
IRIX 6.5.22 disk: the PROM menu comes up on serial, "1) Start System"
loads the kernel, and right after the banner:

    PANIC: TLBMISS: KERNEL FAULT
    EXC code:8, `Read TLB Miss '
    Bad addr: 0xff800000

The PC varies between runs (0x800d1430, 0x800f7318, 0x802cdf14); the bad
address does not. Same at 504ca31d (before the 64-bit bus), with and without
`lightning`, with 256 MB or 1 GB. The graphics console (keyboard attached)
boots the same disk to login. Not a regression: IRIX was never booted
headless; the bring-up log has only the PROM and NetBSD on serial.

**Reading the address.** 0xff800000 is not what the kernel dereferenced.
At 0x802cdf14 the instruction is `lw v1, 40(s3)` with `s3 = 0`: a NULL
struct pointer, VA 0x28. The user refill handler fetches that page's PTE
through Context's PTEBase, which IRIX points at its K2 page-table window at
0xff800000; nothing is mapped there that early, so the nested miss is the
panic and BadVAddr is the window. `badvaddr == 0xff800000` therefore means
"NULL (or other low user address) dereference in kernel mode".

**Not it.** CRIME's revision (now always 0xa1) and MACE's revision bit
(now always set) were tied to the keyboard; both fixed, neither changed the
panic.

**Cause.** With symbols (`nm` of the guest's `/unix`, kept as
`ignore/ip32/unix-6522.map`) the fault is `crmFifoWait+0xfc`, called from
`cn_init` → `gfx_earlyinit` → `crime_earlyinit` → `crimeInitHW`. That
call waits for the rendering engine's interface buffer to be empty
(`a1 = 0x7f`, so the threshold is 0), reading the status word at
0x1500_4000 (level in 24:18) with `us_delay(1)` between reads, about 4096
times. When it runs out it reports through a structure that does not exist
yet that early, and that is the NULL. Our engine had 8640 primitives queued
and finished them in host time, after the guest's 4 ms of guest time were
gone. Console type and memory size only change how much is queued then.

**Fix.** A status read on a busy engine first gives it up to 100 us of host
time (`RE_STATUS_POLL_WAIT`, `src/dev/o2gfx.rs`); see below for the loop. A guest polling status is
waiting for the engine anyway. Boots to login with 32 MB up to 1 GB and
headless.

**Again on a loaded host (2026-11-01).** The 100 us wait gave the kernel's
whole loop about 0.4 s of host time. With the host's load average at 120 to
300, graphics-console boots of a 400 MHz R12000 O2 hit the same panic (EPC
0x803050ac, BadVAddr 0xff800000, Cause TLBL) in five of eleven tries.
Holding the engine's thread for 2 to 3 s while the kernel polled brought it
on PRId 0x0e23 and 0x0e35 alike, so the processor revision plays no part. A
status read that follows another busy one, with no write between, now waits
up to 20 ms (`RE_STATUS_LOOP_WAIT`), ending as soon as the engine is idle.
The first read after a write keeps the 100 us bound so a driver checking for
room still overlaps the engine.

**Rule.** A guest loop that polls a device on its own clock (`us_delay`,
Count) can outrun a device modelled on a host thread. Make the poll itself
wait for the device, bounded, rather than lengthening anything guest-side.

**Gotcha found on the way.** With `jitv2` dispatch on, a `pc` breakpoint
inside compiled code may not fire; `j2 dispatch off` before setting it.
