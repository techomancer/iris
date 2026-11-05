# A command line sent to the IRIX shell over telnet is cut at 256 characters

Status: rule, 2026-11-05.

The guest's tty is in canonical mode: a line longer than 255 characters is
truncated before the shell sees it. `tools/irix-sh.py run` sends each
command as one line, so a long `&&` chain loses its tail without an error,
with odd effects: `uuencode` reading the telnet session as its input and
hanging, a redirect to a cut-off file name, `results` renamed to `resul`.

Keep each command short, or push a script (`irix-sh.py push`) and run
that. `push` itself sends many short lines and is not affected.

Also: on some images xdm grabs the X server while its login box is up, so
X clients from a telnet shell (glxinfo, xdpyinfo, GL tests) hang until
someone logs in at the console. Log in with the monitor's `key type root`,
`key Return`. The VPro Octane images do not grab.
