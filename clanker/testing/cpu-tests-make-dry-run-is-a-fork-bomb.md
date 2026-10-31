# Never run `make -n` in cpu-tests: it is a fork bomb

Status: rule, found 2026-10-30, the hard way.

`cpu-tests/Makefile` links `$(TARGET)` with a recipe ending in `@$(MAKE) check-size`, and
`check-size` depends on `$(TARGET)`. Make runs `$(MAKE)` lines even under `-n`, and a dry
run never produces the target, so each child relinks and recurses without end. `make -C
cpu-tests -n run` reached 11,280 processes, used all of a 125 GB host's memory, got an
unrelated VirtualBox VM killed by the OOM killer, and kept going after its parent shell
was killed, since nothing signals the children.

A real `make` terminates, because the target exists by the time the child looks. To see
whether the toolchain is present, read `toolchain.mk`. To stop a runaway, kill the exact
argv in a loop until none are left; `pkill -f` also matches your own shell.
