#!/usr/bin/env python3
"""Drive a recording IMPACT session for the MGRAS replay goldens.

Start iris first with a recording from power-on, e.g.

  IRIS_MGRAS_REC=ignore/mgras-golden/desktop.rec \\
      target/release/iris --config ignore/ip28/iris-650.toml

then run this script. It starts the CPU (developer builds wait for `start`),
waits for the autologin desktop over telnet, and runs scripted scenes,
taking a recording checkpoint (`mgras rec mark`) and a screenshot after each:

  desktop   the session as autologin leaves it
  xterm     an xterm full of text (and xterm-closed once it exits)
  drag      a window dragged by its title bar (root's Console winterm)
  menu      the 4Dwm root menu (overlay planes) opened and dismissed

Screenshots land next to the recording as <rec>.<scene>.png. The recording
is stopped at the end (`mgras rec off`), which checkpoints the final state.
"""
import argparse
import os
import re
import socket
import subprocess
import sys
import time
import functools

print = functools.partial(print, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))


class Monitor:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=30)
        self.read_prompt()

    def read_prompt(self):
        data = b""
        while not data.endswith(b"> "):
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("monitor closed")
            data += chunk
        return data[:-2].decode("utf-8", "replace").strip()

    def cmd(self, text):
        self.sock.sendall(text.encode() + b"\n")
        return self.read_prompt()


def guest(*cmds, timeout=120):
    r = subprocess.run([sys.executable, os.path.join(HERE, "irix-sh.py"), "run", "-k", *cmds],
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout


def wait_desktop(limit):
    deadline = time.time() + limit
    while time.time() < deadline:
        try:
            out = guest("ps -ef | grep -c '[4]Dwm'", "ps -ef | grep -c '[t]oolchest'", timeout=60)
            counts = [int(x) for x in out.split() if x.isdigit()]
            if len(counts) == 2 and min(counts) > 0:
                return
        except (subprocess.TimeoutExpired, ValueError):
            pass
        time.sleep(10)
    raise TimeoutError("desktop did not come up")


class Mouse:
    """Closed-loop pointer steering from the VC3 cursor position, so X's
    pointer acceleration and the PS/2 axis sense don't matter."""

    def __init__(self, mon):
        self.mon = mon
        self.ysign = None

    def pos(self):
        m = re.search(r"cursor at \((-?\d+), (-?\d+)\)", self.mon.cmd("mgras"))
        return (int(m.group(1)), int(m.group(2))) if m else None

    def nudge(self, dx, dy, buttons):
        self.mon.cmd(f"ps2 mouse {dx} {dy} {buttons}")
        time.sleep(0.05)

    def calibrate(self):
        a = self.pos()
        self.nudge(0, 4, 0)
        time.sleep(0.3)
        b = self.pos()
        self.ysign = -1 if b[1] < a[1] else 1

    def goto(self, x, y, buttons=0, tol=2):
        for _ in range(400):
            cx, cy = self.pos()
            ex, ey = x - cx, y - cy
            if abs(ex) <= tol and abs(ey) <= tol:
                return
            # Small steps stay below X's acceleration threshold.
            step = lambda e: max(-4, min(4, e))
            self.nudge(step(ex), self.ysign * step(ey), buttons)
        raise RuntimeError(f"pointer did not reach ({x}, {y})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec", help="recording path iris was started with (for screenshot names)")
    ap.add_argument("--monitor-port", type=int, default=8888)
    ap.add_argument("--boot-timeout", type=float, default=1200)
    ap.add_argument("--no-start", action="store_true", help="CPU already running")
    a = ap.parse_args()

    mon = None
    for _ in range(60):
        try:
            mon = Monitor("127.0.0.1", a.monitor_port)
            break
        except OSError:
            time.sleep(1)
    if mon is None:
        sys.exit("no monitor")
    if not a.no_start:
        print(mon.cmd("start"))

    def checkpoint(scene):
        time.sleep(3)
        print(f"[{scene}]", mon.cmd("mgras rec mark"))
        print(mon.cmd(f"mgras shot {a.rec}.{scene}.png"))

    t0 = time.time()
    wait_desktop(a.boot_timeout)
    print(f"desktop up after {time.time() - t0:.0f}s")
    time.sleep(20)
    checkpoint("desktop")

    guest("xhost + >/dev/null 2>&1; true",
          "DISPLAY=:0 /usr/bin/X11/xterm -geometry 80x24+80+200 -e sh -c "
          "'ls -laR /usr/include | head -400; sleep 30' </dev/null >/dev/null 2>&1 &", "sleep 12")
    checkpoint("xterm")
    time.sleep(30)
    checkpoint("xterm-closed")

    mouse = Mouse(mon)
    mouse.calibrate()
    # Root's Console winterm sits bottom left; grab its title bar, drag it up.
    mouse.goto(300, 632)
    mouse.nudge(0, 0, 1)
    mouse.goto(500, 300, buttons=1)
    mouse.nudge(0, 0, 0)
    checkpoint("drag")

    mouse.goto(300, 700)
    mouse.nudge(0, 0, 2)
    time.sleep(2)
    print(mon.cmd(f"mgras shot {a.rec}.menu-open.png"))
    mouse.goto(150, 900, buttons=2)
    mouse.nudge(0, 0, 0)
    checkpoint("menu")

    print(mon.cmd("mgras rec off"))


if __name__ == "__main__":
    main()
