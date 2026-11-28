# End-to-end guests

Single-purpose bare-metal programs for behaviour a unit test cannot show:
something the running emulator does from the outside, such as a monitor
command, lands between two guest instructions and changes what the guest
sees. Each program is linked with the suite's `start.S` and `console.c`
only, and has a driver in `../run/` that boots it under a real `iris`,
drives the monitor, and turns the guest's result into an exit code.

Build them with `make -C cpu-tests e2e` (same compiler as the suite, see
`docs/toolchain.md`); each lands in `build/e2e/<name>.elf`. The drivers
need `python3` for `tools/mon.py` and a release `iris`.

## compare_after_debugger_read

A Compare write is judged against the guest's last Count read: a value
between that read and Count now is a deadline the guest missed, and IP7 is
raised at once. The program reads Count, spins on a flag without reading
Count again, and then writes Compare = that read + 0x100. Before releasing
it, the driver runs the monitor's `cop0` dump. The dump must not count as
the guest reading Count: if it does, the write looks like a deadline far in
the future and IP7 waits a whole Count wrap (issue 302).

```sh
make -C cpu-tests e2e
cpu-tests/run/compare-after-debugger-read.sh cop0   # dump, then release
cpu-tests/run/compare-after-debugger-read.sh none   # control: release only
```

Both exit 0 and print `PASS` when IP7 rises, 1 and `FAIL` when it does not.
`IRIS`, `MON_PORT` and `TIMEOUT` override the binary, the monitor port and
the time limit. The result is the guest's own Cause.IP7, not a timing.
