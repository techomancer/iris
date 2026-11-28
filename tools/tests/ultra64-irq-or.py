#!/usr/bin/env python3
"""ultra64-irq-or — the Ultra64 board's interrupt stays up while any source is pending.

    tools/tests/ultra64-irq-or.py IRIS ULTRA64_PEER [--port BASE]

Starts IRIS as a headless Indy at its built-in PROM with `[ultra64] enabled`,
no disks, and plays the N64 side with the `ultra64-peer` example
(`cargo build --example ultra64-peer`). The monitor makes the bus accesses
the IRIX driver would (reading GIO_INT_ACK, writing the RDB interrupt
clears) and `ioc status` shows whether GIO_EXP0 is up.

Two cases: an N64 GIO interrupt and an RDB write pending at once, acked RDB
first; and an RDB read and a GIO interrupt, acked GIO first. In each, the
line has to stay up after the first ack and fall after the second. Exits 0
when both pass.

BASE (default 8888) is the monitor port; serial A and B take BASE+1 and
BASE+2. The board's shared memory has a fixed name (iris_n64_bridge), so
only one IRIS with the board can run on a host at a time.
"""
import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mon import Monitor  # noqa: E402

GIO_INT_ACK = "0xffffffffbf400c00"
RDB_WRITE_INTR = "0xffffffffbf480008"
RDB_READ_INTR = "0xffffffffbf48000c"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("iris")
    ap.add_argument("peer")
    ap.add_argument("--port", type=int, default=8888)
    a = ap.parse_args()

    work = tempfile.mkdtemp(prefix="iris-u64-")
    cfg = os.path.join(work, "iris.toml")
    with open(cfg, "w") as f:
        f.write("headless = true\nno_audio = true\nscsi = {}\n\n[ultra64]\nenabled = true\n")
    log = open(os.path.join(work, "iris.log"), "w")
    iris = subprocess.Popen(
        [os.path.abspath(a.iris), "--config", cfg, "--headless",
         "--monitor-port", str(a.port),
         "--device", f"serial@serial2,backend=tcp:{a.port + 1}", "--device", f"serial@serial1,backend=tcp:{a.port + 2}"],
        cwd=work, stdout=log, stderr=subprocess.STDOUT)
    failures = []
    try:
        mon = connect(a.port, iris)
        wait_for_bridge(a.peer, iris)

        def peer(*msgs):
            subprocess.run([a.peer, *msgs], check=True, stdout=subprocess.DEVNULL)
            # The board's listener thread drains the ring on the signal.
            deadline = time.time() + 10
            while True:
                st = subprocess.run([a.peer, "--status"], check=True, capture_output=True, text=True).stdout
                n2h = st.split("n2h ")[1].strip().split("/")
                if n2h[0] == n2h[1]:
                    return
                if time.time() > deadline:
                    raise RuntimeError("the board never drained the n2h ring: " + st)
                time.sleep(0.05)

        def line():
            out = mon.cmd("ioc status")
            stat = next(l for l in out.splitlines() if "MAP stat=" in l)
            return "GIO_EXP0" in stat.split("mask0=")[0]

        def check(case, step, want):
            got = line()
            ok = got == want
            print(f"  {'ok  ' if ok else 'FAIL'} {step}: GIO_EXP0 {'up' if got else 'down'}"
                  f" (want {'up' if want else 'down'})")
            if not ok:
                failures.append(f"{case}: {step}")

        print(mon.cmd("stop"))

        case = "GIO interrupt + RDB write, RDB acked first"
        print(case)
        check(case, "idle", False)
        peer("gio", "2a", "rdb-write", "1234")
        check(case, "both pending", True)
        mon.cmd(f"mw {RDB_WRITE_INTR} 0 w")
        check(case, "after RDB write ack", True)
        out = mon.cmd(f"mem {GIO_INT_ACK} 1 w")
        print(f"       GIO_INT_ACK read: {out.split(':')[-1].strip()}")
        check(case, "after GIO_INT_ACK read", False)

        case = "RDB read + GIO interrupt, GIO acked first"
        print(case)
        peer("rdb-read", "gio", "01")
        check(case, "both pending", True)
        mon.cmd(f"mem {GIO_INT_ACK} 1 w")
        check(case, "after GIO_INT_ACK read", True)
        mon.cmd(f"mw {RDB_READ_INTR} 0 w")
        check(case, "after RDB read ack", False)
    finally:
        # Only the PROM runs and there are no disks, so SIGTERM (IRIS's own
        # orderly stop) is safe.
        iris.send_signal(signal.SIGTERM)
        try:
            iris.wait(timeout=30)
        except subprocess.TimeoutExpired:
            iris.kill()
        log.close()

    if failures:
        print(f"FAIL: {len(failures)} check(s): " + "; ".join(failures))
        print(f"iris log: {log.name}")
        return 1
    print("PASS")
    return 0


def connect(port, iris):
    deadline = time.time() + 120
    while True:
        if iris.poll() is not None:
            raise RuntimeError(f"iris exited with {iris.returncode}")
        try:
            return Monitor(port=port)
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.2)


def wait_for_bridge(peer, iris):
    deadline = time.time() + 60
    while subprocess.run([peer, "--status"], capture_output=True).returncode != 0:
        if iris.poll() is not None or time.time() > deadline:
            raise RuntimeError("the Ultra64 bridge never appeared (is [ultra64] enabled?)")
        time.sleep(0.2)


if __name__ == "__main__":
    sys.exit(main())
