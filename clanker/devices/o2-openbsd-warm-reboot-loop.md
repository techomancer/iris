# O2: OpenBSD 6.9 survives 89 warm reboots in a row (issue 308 not reproduced)

Status: open (2026-11-20). Issue techomancer/iris#308 reported OpenBSD 6.9
on the O2 coming up damaged after a warm reboot: segfaults and bus errors in
`rc`, sshd failing, `reorder_kernel` failing, "getty repeating too quickly".
Its lead: runs on a freshly copied flash passed, the failing run reused one.
None of this reproduces on current code or on the code from before 5e871a2b.

## What blocked the test first

On 2026-11-20 `main` could not boot OpenBSD on the O2 at all. Two
regressions, both fixed together with this note:

- `o2: keep openbsd's pci memory window off the no-ecc alias`: the 64-bit PCI memory window (0x2_0000_0000) folded onto RAM
  through the no-ECC alias once the alias was matched after the physical
  mask (5f3a9a55), so `ahc` saw "board is not responding" and no disk
  attached. See [o2-no-ecc-alias-is-memory.md](o2-no-ecc-alias-is-memory.md).
- `cpu: let a wait with ie clear complete on a requested interrupt`: on the R5000, OpenBSD's idle loop runs `wait` with IE clear and
  the CPU retried it for ever. See
  [../cpu/wait-is-not-a-nop.md](../cpu/wait-is-not-a-nop.md).

`tests/openbsd/boot-openbsd69-o2.toml` (`iris-ci boot-test`, see TESTING.md)
boots such a disk to its serial login and catches both. 2026-11-22, R5000,
default release features: `main` fails (`ahc0: board is not responding`,
`cmd fail`), the alias fix alone fails (console stops after `root on sd0a`),
both fixes reach `login:` in 165 s.

## The loop

Guest (a local workspace, not in the repository): OpenBSD 6.9
installed from `install69.iso` onto `obsd.raw`, 512 MB as `banks = [256,
256]`, PROM 4.18, NAT networking. `/etc/rc.local` (the workspace keeps it as
`rc.local.guest`), active while `/root/loop` holds a boot count:

1. increment the count, `sha256 -c` the 16 MB file written last boot, print
   `ISSUE308 boot N check OK|FAIL`;
2. write a new 16 MB file from `/dev/urandom`, `sha256` it, `sync`;
3. in the background: sleep 60, wait for `reorder_kernel` to finish, sleep
   20, `reboot`.

`watch-loop.sh N [FROM_LINE]` watches the console log and stops at boot N or
at the first `Segmentation fault`, `Bus error`, `repeating too quickly`,
`panic:`, `Stopped at`, `check FAIL`, `failed` or `core dumped`.

## Results

All release builds with `--features lightning,jitv2`.

| Build | CPU | Flash | Boots | Result |
|---|---|---|---|---|
| HEAD + both fixes | R5000 | as installed | 2-40 (39) | all OK |
| HEAD + both fixes | R10000 | reused after an iris restart | 41-60 (20) | all OK |
| 58fed776 (5e871a2b^) + both fixes | R10000 | reused | 61-90 (30) | all OK |

Every boot: checksum OK, sshd up, getty up, no fault. The only "failed" seen
was `reorder_kernel: failed` after I halted the guest while it was relinking
(relink.log showed `ld` still running), not a fault.

## Ruled out, or not supported

- Reusing the flash: the R10000 runs restarted iris on the saved
  `flash-install.bin` and stayed clean.
- Stale jitv2 pages after the PROM's bank sizing remaps memory on a warm
  reboot (5e871a2b, 2026-11-16): the build before it is clean too, so it is
  not shown to be the fix.
- Unflushed disk writes: 30 + 59 checksums of a file written just before
  `reboot` all matched.

The PROM's soft reboot does no hardware reset (the ARCS Reboot stub jumps
through `bfc00100` to the common reset path after writing CRIME 0x270), so
device state that survives a warm reboot is still a candidate, but nothing
in these runs exposed it.

## Not covered

- The interpreter without jitv2, and `lightning` off.
- Other memory layouts (one bank, 1 GB) and the R12000.
- The code from when the issue was filed (2026-10-31, a day after the O2
  profile landed): that tree cannot boot OpenBSD on the O2 because of the
  two regressions above, and later changes that could have fixed it include
  9350cf01 (shadow cache from reset), 260701f6 (AIC on its own thread),
  25e65c0e (flush disks at power-off) and 16ce0385 (keep RTC RAM).

If it comes back, ask the reporter for the build, the CPU, the memory layout
and whether jitv2 was on, and run the loop above with those.
