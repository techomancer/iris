#!/usr/bin/env python3
"""Cold-start iris again and again and count how often the PROM reaches its menu.

Each round starts a fresh iris process with the given config, watches the
serial log the config names, and ends the round on the first of:

  ok       a --good pattern (default: the ARCS menu, "Option?")
  fail     a --bad pattern (default: the O2's serial loader "SL-9600" and a
           PROM exception "EXC")
  timeout  neither within --timeout seconds
  exited   iris stopped by itself

then stops iris with SIGTERM. Only run it on configs with no disk the guest
can write to (no [scsi.N], or overlay = true on each), since iris is stopped
while the PROM runs. The serial log and iris's output of every round that was
not ok are kept as fail-N.serial.log and fail-N.stderr.txt in --out; one line
per round goes to --out/results.txt.

Example, the O2 PROM with no disks, serial console on 23011:

  cat > o2.toml <<EOF
  nvram = "flash.bin"
  banks = [128, 0, 0, 0]
  headless = true
  no_audio = true
  monitor_port = 23010
  devices = [ { type = "serial", at = "serial1", backend = "tcp:23011" }, { type = "serial", at = "serial2", backend = "tcp:23012" } ]
  serial_log = "serial.log"
  serial_log_overwrite = true
  [machine]
  profile = "o2_ip32"
  [network]
  mode = "off"
  [scsi]
  EOF
  tools/prom-boot-loop.py --iris target/release/iris --config o2.toml --count 100

The run directory is the config's directory, so relative paths in it
(nvram, serial_log) land next to it.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time


def serial_log_path(config):
    """The serial_log value of a flat TOML file (top-level key only)."""
    with open(config) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line.startswith("["):
                break
            if line.startswith("serial_log") and "=" in line:
                key, val = (s.strip() for s in line.split("=", 1))
                if key == "serial_log":
                    return val.strip("\"'")
    sys.exit(f"{config}: no top-level serial_log key")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iris", required=True, help="iris binary")
    ap.add_argument("--config", required=True, help="iris config file")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=240.0, help="seconds per round")
    ap.add_argument("--good", action="append", help="pattern that ends a round as ok (repeatable)")
    ap.add_argument("--bad", action="append", help="pattern that ends a round as a failure (repeatable)")
    ap.add_argument("--out", help="directory for results (default: the config's directory)")
    ap.add_argument("--keep-going", action="store_true",
                    help="go on even if the first round times out (normally a sign the patterns are wrong)")
    args = ap.parse_args()
    good = args.good or ["Option?"]
    bad = args.bad or ["SL-9600", "EXC"]
    iris = os.path.abspath(args.iris)
    config = os.path.abspath(args.config)
    run_dir = os.path.dirname(config)
    out = os.path.abspath(args.out or run_dir)
    os.makedirs(out, exist_ok=True)
    log = os.path.join(run_dir, serial_log_path(config))
    results = open(os.path.join(out, "results.txt"), "a")
    counts = {}
    for i in range(args.count):
        if os.path.exists(log):
            os.remove(log)
        err_path = os.path.join(out, "stderr.txt")
        err = open(err_path, "w")
        t0 = time.time()
        p = subprocess.Popen([iris, "--config", config], cwd=run_dir, stdout=err, stderr=err)
        verdict, why = "timeout", ""
        while time.time() - t0 < args.timeout:
            time.sleep(0.5)
            if p.poll() is not None:
                verdict = "exited"
                break
            try:
                text = open(log, errors="replace").read()
            except FileNotFoundError:
                continue
            hit = next((b for b in bad if b in text), None)
            if hit:
                time.sleep(1)  # let the rest of the message reach the log
                verdict, why = "fail", hit
                break
            if any(g in text for g in good):
                verdict = "ok"
                break
        took = time.time() - t0
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        err.close()
        counts[verdict] = counts.get(verdict, 0) + 1
        results.write(f"{i} {verdict} {why} {took:.1f}s load={os.getloadavg()[0]:.1f}\n")
        results.flush()
        if verdict != "ok":
            if os.path.exists(log):
                shutil.move(log, os.path.join(out, f"fail-{i}.serial.log"))
            shutil.move(err_path, os.path.join(out, f"fail-{i}.stderr.txt"))
            if i == 0 and verdict in ("timeout", "exited") and not args.keep_going:
                print("the first round neither reached a --good nor a --bad pattern; stopping")
                break
        print(f"{i} {verdict} {why} {took:.1f}s", flush=True)
    print(counts)
    sys.exit(0 if set(counts) <= {"ok"} else 1)


if __name__ == "__main__":
    main()
