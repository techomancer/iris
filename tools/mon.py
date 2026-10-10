#!/usr/bin/env python3
"""Run monitor commands on a running iris (127.0.0.1:8888) and print the output.

    tools/mon.py "mgras status" "rss regs"

Each argument is one command; outputs are printed in order, separated by a
"== cmd" header when there is more than one.
"""
import re
import socket
import sys


class Monitor:
    def __init__(self, host="127.0.0.1", port=8888, timeout=60):
        self.s = socket.create_connection((host, port), timeout=timeout)
        self.buf = b""
        self._until_prompt()

    def _until_prompt(self):
        while not re.search(rb"(^|\n)> $", self.buf):
            chunk = self.s.recv(65536)
            if not chunk:
                raise ConnectionError("monitor closed")
            self.buf += chunk
        out, self.buf = self.buf, b""
        return out.decode("latin-1", "replace")

    def cmd(self, c):
        self.s.sendall((c + "\n").encode())
        out = self._until_prompt()
        return re.sub(r"\n?> $", "", out).replace("\r", "").strip()


def main():
    cmds = sys.argv[1:]
    if not cmds:
        print(__doc__.strip())
        return 2
    m = Monitor()
    for c in cmds:
        if len(cmds) > 1:
            print(f"== {c}")
        print(m.cmd(c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
