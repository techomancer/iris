# A backgrounded shutdown dies with its telnet session

Status: rule, 2026-11-23. Replaces "IRIX shutdown hangs on the GPU raster",
which was this mistake, not an emulator bug.

## Symptom

`tools/irix-sh.py run '/etc/shutdown -y -g0 -i0 &'` returns, telnet goes
away, but the guest never reaches the power-off dialog: the desktop stays up
(GL demos keep animating), `who -r` still says run level 2, and the
`shutdown` process is gone. It looked board-specific: on an idle desktop or
a slow raster the shutdown usually wins the race, with powerflip or ideas
hogging the guest CPU on the Newport GPU raster it lost every time.

## Cause

The `&` leaves `shutdown` in the telnet session, which irix-sh.py closes as
soon as the command line returns. The hangup kills `shutdown` before it
switches the run level whenever the guest is too busy to get there first.

## Rule

Run shutdown in the foreground (the session simply ends when the guest goes
down; see the `irix-guest-loop` skill), or detach it:
`nohup /etc/shutdown -y -g0 -i0 > /dev/null 2>&1 &`. With `nohup`, the GPU
raster with powerflip and ideas running reached the power-off dialog in
about two minutes.
