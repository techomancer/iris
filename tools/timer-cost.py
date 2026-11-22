#!/usr/bin/env python3
"""Measure what a booted IRIX guest costs in timer wakes and host clock reads.

Boots a machine with `iris --ci`, waits for the console login prompt, and
reads the monitor's `timers` counters around two windows:

  idle     the guest sits at the console login prompt for --window seconds
  console  root logs in on the serial console and prints --lines lines to it

It then shuts the guest down from the console (`shutdown -p`) and waits for
iris to exit on its own. Rates come from the counters and the `host_ns` the
same command reports, not from timing the harness.

    tools/timer-cost.py --iris target/release/iris --config boot.toml \\
        --monitor-port 27840 --socket "$TMPDIR/iris-issue-784.sock"

Run it through the host's job gate if it has one; the guest lives for the
whole run. The config must use a copy-on-write overlay (or `--ci`, which
gives one) and a console on serial port A. Exit status: 0 when the guest
booted, shut down cleanly and stayed within every --max-* bound given;
1 otherwise. --json prints the numbers as one object.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mon import Monitor  # noqa: E402


def sample(mon: Monitor) -> dict[str, int]:
    out = mon.cmd("timers")
    words = out.split()
    if not words or words[0] != "timers:":
        raise RuntimeError(f"unexpected `timers` output: {out!r}")
    return {words[i]: int(words[i + 1]) for i in range(1, len(words) - 1, 2)}


def rates(a: dict[str, int], b: dict[str, int]) -> dict[str, float]:
    secs = (b["host_ns"] - a["host_ns"]) / 1e9
    out = {"seconds": round(secs, 3)}
    for k in ("wakes", "scans", "register_clock_reads"):
        out[k] = b[k] - a[k]
        out[k + "_per_s"] = round((b[k] - a[k]) / secs, 1)
    return out


class Ci:
    def __init__(self, exe: str, socket: str):
        self.base = [exe, "--socket", socket]

    def __call__(self, *args: str, timeout: float = 60) -> str:
        r = subprocess.run(self.base + list(args), capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"iris-ci {' '.join(args)}: rc {r.returncode}: {r.stderr.strip() or r.stdout.strip()}")
        return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iris", required=True, help="the iris binary")
    ap.add_argument("--iris-ci", help="the iris-ci binary (default: next to --iris)")
    ap.add_argument("--config", required=True, help="iris.toml; iris runs in its directory")
    ap.add_argument("--socket", required=True, help="the config's ci_socket")
    ap.add_argument("--monitor-port", type=int, required=True)
    ap.add_argument("--boot-timeout", type=float, default=900)
    ap.add_argument("--settle", type=float, default=30, help="seconds at the login prompt before measuring")
    ap.add_argument("--window", type=float, default=60, help="seconds of the idle window")
    ap.add_argument("--lines", type=int, default=500, help="lines printed in the console window")
    ap.add_argument("--log", help="iris's output (default: timer-cost-iris.log beside the config)")
    ap.add_argument("--max-idle-wakes-per-s", type=float)
    ap.add_argument("--max-idle-clock-reads-per-s", type=float)
    ap.add_argument("--max-console-clock-reads-per-line", type=float)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    cfg = Path(a.config).resolve()
    iris = str(Path(a.iris).resolve())
    ci = Ci(a.iris_ci or str(Path(iris).with_name("iris-ci")), a.socket)
    log_path = Path(a.log) if a.log else cfg.with_name("timer-cost-iris.log")
    result: dict = {"config": str(cfg)}
    problems: list[str] = []

    with open(log_path, "w") as log:
        proc = subprocess.Popen([iris, "--config", cfg.name, "--ci"], cwd=cfg.parent, stdout=log, stderr=subprocess.STDOUT)
    try:
        end = time.monotonic() + 120
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"iris exited with {proc.returncode} before its monitor was ready; see {log_path}")
            try:
                if "monitor  ready" in ci("status"):
                    break
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
            if time.monotonic() > end:
                raise RuntimeError("the monitor never became ready")
            time.sleep(1)

        ci("start")
        t0 = time.monotonic()
        ci("serial-wait", "login:", "--timeout", str(int(a.boot_timeout)), timeout=a.boot_timeout + 30)
        result["boot_to_login_s"] = round(time.monotonic() - t0, 1)
        mon = Monitor(port=a.monitor_port)

        time.sleep(a.settle)
        s0 = sample(mon)
        time.sleep(a.window)
        s1 = sample(mon)
        result["idle"] = rates(s0, s1)

        ci("login", "root", timeout=120)
        s2 = sample(mon)
        ci("run", f"awk 'BEGIN {{ for (i = 0; i < {a.lines}; i++) print \"timer-cost line\", i }}'",
           "--timeout", "600", timeout=630)
        s3 = sample(mon)
        console = rates(s2, s3)
        console["lines"] = a.lines
        console["register_clock_reads_per_line"] = round(console["register_clock_reads"] / a.lines, 2)
        result["console"] = console

        ci("serial-send", "sync; shutdown -p -g0 -y")
    except Exception as e:  # report, then never leave a running guest behind
        problems.append(str(e))
        if proc.poll() is None:
            print(f"timer-cost: {e}; iris is still running (pid {proc.pid}): shut the guest down, do not kill it",
                  file=sys.stderr)
            return 1

    try:
        rc = proc.wait(timeout=300)
    except subprocess.TimeoutExpired:
        print(f"timer-cost: iris did not exit within 300 s of shutdown (pid {proc.pid})", file=sys.stderr)
        return 1
    tail = log_path.read_text(errors="replace")
    result["iris_exit"] = rc
    result["clean_power_off"] = rc == 0 and "soft power-off" in tail
    if not result["clean_power_off"]:
        problems.append(f"no clean power-off (iris exit {rc}); see {log_path}")

    for key, section, field in (
        ("max_idle_wakes_per_s", "idle", "wakes_per_s"),
        ("max_idle_clock_reads_per_s", "idle", "register_clock_reads_per_s"),
        ("max_console_clock_reads_per_line", "console", "register_clock_reads_per_line"),
    ):
        bound = getattr(a, key)
        got = result.get(section, {}).get(field)
        if bound is not None and got is not None and got > bound:
            problems.append(f"{section} {field} {got} > {bound}")

    result["problems"] = problems
    if a.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"boot to login: {result.get('boot_to_login_s')} s")
        for name in ("idle", "console"):
            if name in result:
                r = result[name]
                print(f"{name:8} {r['seconds']:7.1f} s  wakes {r['wakes_per_s']:9.1f}/s  scans {r['scans_per_s']:9.1f}/s"
                      f"  register clock reads {r['register_clock_reads_per_s']:11.1f}/s")
        if "console" in result:
            print(f"console  {result['console']['register_clock_reads_per_line']} register clock reads per line")
        print(f"clean power-off: {result['clean_power_off']}")
        for p in problems:
            print(f"FAIL: {p}")
        print("PASS" if not problems else "FAIL")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
