#!/usr/bin/env python3
"""Race a monitor `j2 flush` against the compile workers' own flushes (#306).

Boots a guest in iris under `--ci`, with `IRIS_JIT_ARENA_FLUSH_MB` set low so
the compile workers fill their code arena and lead flushes of their own while
the guest boots. Meanwhile it asks the monitor, over and over, to `stop` the
CPU, `j2 flush` and `start` it again. A flush the monitor asks for runs on the
monitor's thread with the jitv2 lock held and joins the compile workers; a
worker leading a flush at that moment must not wait on that lock, or the
monitor command never returns.

    tools/jitv2-flush-race.py --iris target/release/iris --config ci.toml \\
        --seconds 600 --serial-log console.log

The config is an ordinary iris.toml for a single-CPU machine that boots on its
own. Give its disks `overlay = true`: under `--ci` the overlay lives in
`$TMPDIR` and the base image is never written. Give it a serial log to report
how far the guest got. Monitor, CI socket and any port forwards must not
collide with another iris.

Exit status: 0 when every command returned and the workers flushed at least
`--min-worker-flushes` times; 1 when a monitor command did not return within
`--cmd-timeout` (the deadlock); 2 when the run did not exercise the race
(too few worker flushes) or iris failed to start.

A hung iris cannot shut down; the harness then kills it. That is safe only
because `--ci` keeps the disk overlay in `$TMPDIR`: do not point this at a
config whose disks iris writes directly.
"""
import argparse
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

Monitor = import_module("mon").Monitor


class Log:
    """iris's stderr and stdout, copied to a file, with the flush lines counted."""

    def __init__(self, proc, path):
        self.worker = 0
        self.cpu = 0
        self.lines = 0
        self.out = open(path, "w")
        self.t = threading.Thread(target=self._read, args=(proc.stdout,), daemon=True)
        self.t.start()

    def _read(self, pipe):
        for raw in iter(pipe.readline, b""):
            line = raw.decode("latin-1", "replace")
            self.out.write(line)
            self.out.flush()
            self.lines += 1
            if "mega_flush (from jit thread)" in line:
                self.worker += 1
            elif "mega_flush (from cpu thread)" in line:
                self.cpu += 1


def ci_send(addr, cmd, args=None, timeout=60):
    """One request to iris's CI socket (`tcp:HOST:PORT` or a Unix path)."""
    if addr.startswith("tcp:"):
        host, port = addr[4:].rsplit(":", 1)
        s = socket.create_connection((host, int(port)), timeout=timeout)
    else:
        s = socket.socket(socket.AF_UNIX)
        s.settimeout(timeout)
        s.connect(addr)
    with s:
        s.sendall((json.dumps({"cmd": cmd, "args": args or {}}) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf or b"{}")


def connect_monitor(port, deadline, proc):
    while time.time() < deadline:
        if proc.poll() is not None:
            return None
        try:
            return Monitor(port=port, timeout=10)
        except OSError:
            time.sleep(0.5)
    return None


def last_serial_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            text = f.read().decode("latin-1", "replace")
    except OSError:
        return ""
    lines = [l.strip() for l in re.split(r"[\r\n]+", text) if l.strip()]
    return lines[-1] if lines else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iris", required=True, help="iris binary built with --features jitv2")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seconds", type=float, default=600, help="how long to race")
    ap.add_argument("--arena-mb", type=int, default=1, help="IRIS_JIT_ARENA_FLUSH_MB")
    ap.add_argument("--monitor-port", type=int, default=8888)
    ap.add_argument("--ci-socket", help="CI socket for the clean quit (default: the config's or iris's)")
    ap.add_argument("--cmd-timeout", type=float, default=60, help="a monitor command slower than this is a hang")
    ap.add_argument("--min-pause", type=float, default=0.05)
    ap.add_argument("--max-pause", type=float, default=1.0)
    ap.add_argument("--min-worker-flushes", type=int, default=5)
    ap.add_argument("--serial-log", help="the config's serial log, to report the guest's progress")
    ap.add_argument("--log", default="jitv2-flush-race.log", help="where iris's output goes")
    ap.add_argument("--stacks", help="on a hang, save every thread's stack here before killing iris")
    ap.add_argument("--seed", type=int, default=306)
    a = ap.parse_args()
    random.seed(a.seed)

    env = dict(os.environ, IRIS_JIT_ARENA_FLUSH_MB=str(a.arena_mb))
    argv = [a.iris, "--config", a.config, "--ci", "--headless", "--monitor-port", str(a.monitor_port)]
    if a.ci_socket:
        argv += ["--ci-socket", a.ci_socket]
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = Log(proc, a.log)
    mon = connect_monitor(a.monitor_port, time.time() + 120, proc)
    if mon is None:
        print(f"FAIL: no monitor on port {a.monitor_port} (iris exit {proc.poll()}); see {a.log}")
        if proc.poll() is None:
            proc.kill()
        return 2
    mon.s.settimeout(a.cmd_timeout)

    rounds, hung, error = 0, None, None
    start = time.time()
    try:
        mon.cmd("start")
        while time.time() - start < a.seconds:
            time.sleep(random.uniform(a.min_pause, a.max_pause))
            for c in ("stop", "j2 flush", "start"):
                t0 = time.time()
                try:
                    out = mon.cmd(c)
                except socket.timeout:
                    hung = (c, time.time() - t0)
                    break
                if c == "j2 flush" and "flush complete" not in out:
                    error = f"`j2 flush` said: {out!r}"
                    break
            if hung or error:
                break
            rounds += 1
            if rounds % 50 == 0:
                print(f"  {time.time() - start:6.0f}s  rounds {rounds}  worker flushes {log.worker}  cpu-side flushes {log.cpu}", flush=True)
    except (ConnectionError, OSError) as e:
        error = f"monitor connection lost: {e}"

    elapsed = time.time() - start
    print(f"rounds {rounds} in {elapsed:.0f}s; worker-led flushes {log.worker}; cpu-side flushes {log.cpu}")
    if a.serial_log:
        print(f"guest console, last line: {last_serial_line(a.serial_log)!r}")

    if hung:
        print(f"FAIL: `{hung[0]}` did not return within {hung[1]:.0f}s: the monitor's flush and a worker's flush wait for each other (#306)")
        if a.stacks:
            save_stacks(proc.pid, a.stacks)
        # A deadlocked iris cannot stop; its disk overlay is a --ci one in $TMPDIR.
        proc.send_signal(signal.SIGKILL)
        proc.wait()
        return 1

    status = 0
    if error:
        print(f"FAIL: {error}")
        status = 2
    elif log.worker < a.min_worker_flushes:
        print(f"INCONCLUSIVE: only {log.worker} worker-led flushes (want {a.min_worker_flushes}); lower --arena-mb or race longer")
        status = 2
    else:
        print("PASS: every flush returned while the compile workers flushed on their own")

    # Clean stop: the CI socket's quit stops the machine and exits iris.
    ci = a.ci_socket or ci_socket_of(a.config)
    try:
        if ci:
            ci_send(ci, "quit", {}, timeout=120)
        proc.wait(timeout=120)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"note: clean quit failed ({e}); stopping iris")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    return status


def save_stacks(pid, path):
    """Every thread's stack of the hung iris, with `sample` (macOS) or gdb."""
    import shutil
    if shutil.which("sample"):
        cmd = ["sample", str(pid), "2", "-file", path]
    elif shutil.which("gdb"):
        cmd = ["gdb", "-p", str(pid), "-batch", "-ex", "thread apply all bt"]
    else:
        print("note: no sample or gdb for the stacks")
        return
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120)
        if cmd[0] == "gdb":
            open(path, "wb").write(out.stdout)
        print(f"thread stacks: {path}")
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"note: stacks failed ({e})")


def ci_socket_of(config):
    try:
        text = open(config).read()
    except OSError:
        return None
    m = re.search(r'^\s*ci_socket\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else None


if __name__ == "__main__":
    sys.exit(main())
