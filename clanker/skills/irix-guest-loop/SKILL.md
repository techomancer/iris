---
name: irix-guest-loop
description: >-
  Drive a running IRIX guest in the IRIS emulator from the host: boot it, log
  in over telnet (tools/irix-sh.py), exchange files through the built-in NFS
  share, talk to the IRIS monitor (tools/mon.py or the iris MCP server), set
  up autologin for a desktop session, compile o32/n32/n64 code and shared
  libraries with native MIPSpro, inspect installed packages, and test new
  libraries against live IRIX binaries. Use for any task that has to run
  commands inside IRIX.
---

# IRIX guest loop

Use this skill whenever a task needs commands run inside a live IRIX guest:
building C/C++ or DSOs with MIPSpro, inspecting ELF files with native tools
(`elfdump`, `odump`, `nm`, `dwarfdump`), testing a rebuilt library against
real IRIX programs, or preparing software for packaging (see the
`irix-inst-packaging` skill for the packaging itself).

Everything here assumes IRIX 6.5 with MIPSpro 7.4.x installed. IRIX 5.3
works for the shell and file-exchange parts; it has no n32/n64 ABIs.

## 1. What you need

| Piece | Where |
|---|---|
| A bootable IRIX disk image | Install from original media: `clanker/irix/irix-install.md`. Never use a pre-built MAME CHD. |
| A config with telnet forwarded and NFS enabled | see below |
| Guest shell | `tools/irix-sh.py` (single-shot telnet sessions, exit status = the guest command's) |
| Monitor | `tools/mon.py "cmd" ...` or the iris MCP server (`clanker/iris_mcp.py`); command reference in `debug.md` |
| File exchange | built-in NFS: host `[nfs] shared_dir` appears as `/shared` in the guest |

Minimal additions to `iris.toml` (see `HELP.md` for the full file):

```toml
[nfs]
shared_dir = "./shared"

[[port_forward]]
proto = "tcp"
host_port = 2323
guest_port = 23
bind = "localhost"
```

`tools/irix-sh.py` connects to `localhost:2323` as `root` by default
(`--host`, `--port`, `--user` override). A fresh IRIX install has no root
password. If the guest has one, pass it with `--password` or the
`IRIS_ROOT_PASSWORD` environment variable. Never write a password into a
committed file, a skill or a note.

## 2. Boot and wait

1. Build and start iris yourself, in the background, logging to a file
   (`target/release/iris > iris-run.log 2>&1`). If the user already has a
   build they are using, do not overwrite it: build into your own target
   directory (`cargo build --release --target-dir target/agent ...`).
2. Release builds start the CPU by themselves. `developer` builds start it
   paused: send `start` to the monitor (`tools/mon.py start`).
3. `tools/irix-sh.py wait --timeout 900` returns once telnet answers with a
   login prompt (network and inetd are up).
4. `tools/irix-sh.py mount-shared` mounts the host share at `/shared` if the
   image does not mount it at boot.

Disk images are usually written directly (no overlay). **Never kill iris
while IRIX is running**: shut down from inside the guest (section 7) or use a
copy-on-write overlay (`STORAGE.md`). A killed guest corrupts XFS.

## 3. Running commands

```sh
tools/irix-sh.py run 'uname -a' 'cc -version'      # exit status = last command's
tools/irix-sh.py run -k 'cmd1' 'cmd2'              # keep going after a failure
tools/irix-sh.py push local.c /tmp/local.c          # text files only
tools/irix-sh.py pull /tmp/out.txt ./out.txt
```

Each session runs `/bin/sh` in `/tmp` with `DISPLAY=:0.0`. Exit status 124
means a timeout or a lost connection. Binary files go through `/shared`.

Guest shell gotchas:

- Stock IRIX has no bash, no `pkill`, and `chmod` wants `a+x`, not `+x`.
- root's *login* shell is csh. Commands that go through it (rsh, `su -`, an
  `.login`) must be csh-safe: no `2>&1`, no unset `$VAR`.
- `/shared` is mounted `noexec`: copy sources and binaries to `/tmp` before
  building or running them.
- `/tmp` is emptied at boot. Re-copy whatever you staged there.
- `showprods`, `versions` and `inst` page through `more`: pipe them to `cat`.

## 4. A desktop session (autologin)

X programs (GL tests, anything that opens a window) need a logged-in desktop.
Without one, xdm's login window owns the display. IRIX can log a user in
automatically:

```sh
tools/irix-sh.py run 'echo root > /etc/autologin' 'touch /etc/autologin.on'
```

- `/etc/autologin` holds the user name to log in.
- `/etc/autologin.on` is a **one-shot** flag: the login program deletes it
  when it logs the user in. Re-create it before every shutdown, or the next
  boot stops at the login window.
- The session appears about a minute after telnet comes up. Wait until
  `ps -ef | grep 4Dwm` shows the window manager (and `toolchest` for the
  full desktop), then `screen wait 2000 60000` on the monitor (returns once
  the screen has been still for 2 s) before taking screenshots.
- Without autologin you can log in on the screen itself: `ptr click X Y` on
  the user's icon in the xdm window, then `key Return` (see `screen`, `ptr`
  and `key` in debug.md; MCP `screenshot`, `click`, `press_key`).
- The screen saver blanks the screen after 10 minutes idle and every pixel
  probe then reads black. The desktop re-applies its saver settings after
  login, so run `/usr/bin/X11/xset s off` once the session is up and check
  that `xset q | grep timeout:` shows 0.

## 5. Building with MIPSpro

Copy sources from `/shared/...` into `/tmp/<work>` before compiling, so
build products and `so_locations`/`Makedepend` land on local XFS.

| ABI | Library directory | Compile | Shared library link |
|---|---|---|---|
| o32 | `/usr/lib` | `cc -o32 -mips2 -O2` | `cc -o32 -mips2 -shared -soname libfoo.so -set_version sgi1.0 -check_registry /usr/lib/so_locations -all ... -lc -no_unresolved` |
| n32 | `/usr/lib32` | `cc -n32 -mips3 -O2` | `cc -n32 -mips3 -shared -soname libfoo.so -set_version sgi1.0 -check_registry /usr/lib32/so_locations -all ... -lc -no_unresolved` |
| n64 | `/usr/lib64` | `cc -64 -mips3 -O2` (or `-mips4`) | `cc -64 -mips3 -shared -soname libfoo.so -set_version sgi1.0 -check_registry /usr/lib64/so_locations -all ... -lc -no_unresolved` |

- `-all` pulls every object of a static archive into the DSO.
- `-exports_file <file>` / `-hidden_symbol <sym>` control `.dynsym`. Omit
  them to match SGI's default (every non-static symbol exported).
- `-check_registry` assigns the DSO the quickstart base address registered
  for it. Never `-update_registry` the system `so_locations` from a scratch
  build; copy it to `/tmp` first.
- MIPSpro C is C89-ish: declarations at the top of a block, no
  `<stdint.h>`.

## 6. Inspecting what is installed

```sh
showfiles -- /usr/lib32/libz.so | cat     # which subsystem owns a file
versions -n | cat                         # installed products and versions
elfdump -h -L -Dc /usr/lib32/libz.so      # headers, liblist, dynamic tags
odump -D /usr/lib32/libz.so
grep -A 2 '^libz\.so' /usr/lib32/so_locations
```

### Testing a rebuilt library without installing it

Never overwrite `/usr/lib*/lib*.so` to test. Stage new libraries under
`/tmp/stage/{lib,lib32,lib64}` and point rld at them for one run:

| ABI | Variable |
|---|---|
| o32 | `LD_LIBRARY_PATH=/tmp/stage/lib` |
| n32 | `LD_LIBRARYN32_PATH=/tmp/stage/lib32` |
| n64 | `LD_LIBRARY64_PATH=/tmp/stage/lib64` |

`_RLD_ARGS="-v"` makes rld report which objects it loaded.

Before packaging a library:

1. Compare `.dynsym` (names and sizes) with the original, with `readelf -s
   -W` on the host or `elfdump -Dt` in the guest.
2. Build a test program against the **original** system header and run it
   against the **new** library.
3. Run the library's own test suite for every ABI, shared and static.
4. Run real IRIX programs that link the library with the staging variables
   set: no rld errors, same output.

## 7. Shutting down

```sh
tools/irix-sh.py run 'touch /etc/autologin.on; sync; shutdown -p -g0 -y'
```

Release builds exit when the guest powers off. `developer` builds stay up:
wait about a minute, check the log or `tools/mon.py status` for the
power-off, and only then stop the process. Kill iris by its exact PID
(`pgrep -f` also matches the shell running the command).
