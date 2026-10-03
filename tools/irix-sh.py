#!/usr/bin/env python3
"""Single-shot shell access to an IRIX guest over telnet (host port forward).

Each invocation opens a telnet session to the guest (default localhost:2323,
the [[port_forward]] rule in iris.toml), logs in (root, no password by
default), runs the requested action, prints the result and exits with the
guest command's exit status. Designed for scripted/autonomous loops, not for
interactive use.

Actions:
  wait                      poll until a login prompt answers (boot / network up)
  run CMD [CMD ...]         run shell commands (sh), print output, exit = last rc
  push LOCAL REMOTE         copy a TEXT file into the guest (heredoc)
  pull REMOTE [LOCAL]       print / save a text file from the guest
  mount-shared              mkdir /shared and mount the host NFS share there

Every session exports DISPLAY=:0.0 and starts in /tmp. Command completion is
detected with a unique end marker, never with sleeps.

Examples:
  tools/irix-sh.py wait --timeout 600
  tools/irix-sh.py mount-shared
  tools/irix-sh.py run 'cp /shared/gltest/*.c /shared/gltest/irix.sh /tmp' \\
                       'cd /tmp && sh irix.sh glprim'
  tools/irix-sh.py run 'cd /tmp && ./glprim --hold 3000'
"""
import argparse
import os
import random
import re
import socket
import sys
import time

IAC, DONT, DO, WONT, WILL, SB, SE = 255, 254, 253, 252, 251, 250, 240
OPT_ECHO, OPT_SGA = 1, 3


class Telnet:
    def __init__(self, host, port, timeout):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.buf = b""
        self.log = os.environ.get("IRIX_SH_DEBUG")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def _negotiate(self, data):
        """Strip telnet commands from `data`, answering option requests:
        accept the server's ECHO/SGA, refuse everything else."""
        out = bytearray()
        i = 0
        while i < len(data):
            c = data[i]
            if c != IAC:
                out.append(c)
                i += 1
                continue
            if i + 1 >= len(data):
                break
            cmd = data[i + 1]
            if cmd == IAC:
                out.append(IAC)
                i += 2
            elif cmd in (DO, DONT, WILL, WONT) and i + 2 < len(data):
                opt = data[i + 2]
                if cmd == DO:
                    self.sock.sendall(bytes([IAC, WILL if opt == OPT_SGA else WONT, opt]))
                elif cmd == WILL:
                    self.sock.sendall(bytes([IAC, DO if opt in (OPT_ECHO, OPT_SGA) else DONT, opt]))
                i += 3
            elif cmd == SB:
                end = data.find(bytes([IAC, SE]), i)
                i = len(data) if end < 0 else end + 2
            else:
                i += 2
        return bytes(out)

    def read_until(self, pattern, timeout):
        """Read until regex `pattern` (bytes) matches; return text up to and
        including the match. Raises TimeoutError."""
        rx = re.compile(pattern)
        deadline = time.time() + timeout
        while True:
            m = rx.search(self.buf)
            if m:
                text = self.buf[: m.end()]
                self.buf = self.buf[m.end():]
                return text, m
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError(f"timed out waiting for {pattern!r}; last output:\n"
                                   + self.buf[-400:].decode("latin-1", "replace"))
            self.sock.settimeout(min(left, 1.0))
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError("connection closed; last output:\n"
                                      + self.buf[-400:].decode("latin-1", "replace"))
            chunk = self._negotiate(chunk)
            if self.log:
                sys.stderr.write(chunk.decode("latin-1", "replace"))
            self.buf += chunk

    def send(self, text):
        self.sock.sendall(text.replace("\n", "\r\n").encode("latin-1"))


class Session:
    def __init__(self, args):
        self.args = args
        self.t = Telnet(args.host, args.port, args.connect_timeout)

    def login(self):
        a = self.args
        self.t.read_until(rb"login: ?$", a.login_timeout)
        self.t.send(a.user + "\n")
        text, _ = self.t.read_until(rb"(Password: ?$|[#$] ?$)", a.login_timeout)
        if text.rstrip().endswith(b"Password:"):
            self.t.send((a.password or "") + "\n")
            self.t.read_until(rb"[#$] ?$", a.login_timeout)
        # Quiet, predictable shell: no echo of our input, and after the ready
        # marker no prompts at all (they would leak into command output).
        # Root's login shell may be csh (IRIX 6.5.0), so switch to sh first.
        self.t.send("exec /bin/sh\n")
        # Wait for the new shell's prompt: input sent before it starts can
        # be lost with the old one.
        self.t.read_until(rb"[#$] ?$", a.login_timeout)
        self.t.send("stty -echo; DISPLAY=:0.0; export DISPLAY; cd /tmp; "
                    "echo '__IRIX''SH_READY__'; PS1=''; PS2=''; export PS1 PS2\n")
        self.t.read_until(rb"__IRIXSH_READY__\r?\n", a.login_timeout)

    def run(self, cmd, timeout):
        """Run one shell command; return (output, rc). The markers are split
        in the command text ('__B''EGIN') so an echoed command line can never
        match them."""
        tag = "%08x" % random.getrandbits(32)
        begin, end = f"__IRIXSH_BEGIN_{tag}__", f"__IRIXSH_END_{tag}_"
        self.t.send(f"echo '{begin[:6]}''{begin[6:]}'; {cmd}\n"
                    f"echo '{end[:6]}''{end[6:]}'$?__\n")
        self.t.read_until(re.escape(begin.encode()) + rb"\r?\n", timeout)
        text, m = self.t.read_until(re.escape(end.encode()) + rb"(\d+)__", timeout)
        out = text[: m.start()].decode("latin-1", "replace").replace("\r\n", "\n").replace("\r", "")
        return out, int(m.group(1))

    def close(self):
        try:
            self.t.send("exit\n")
        except OSError:
            pass
        self.t.close()


def act_wait(args):
    deadline = time.time() + args.timeout
    while True:
        try:
            s = Session(args)
            s.t.read_until(rb"login: ?$", 30)
            s.t.close()
            print("login prompt is up")
            return 0
        except (OSError, TimeoutError, ConnectionError) as e:
            if time.time() > deadline:
                print(f"no login prompt after {args.timeout}s: {e}", file=sys.stderr)
                return 1
            time.sleep(5)


def with_session(args, fn):
    s = Session(args)
    try:
        s.login()
        return fn(s)
    finally:
        s.close()


def act_run(args):
    def go(s):
        rc = 0
        for c in args.cmds:
            out, rc = s.run(c, args.timeout)
            sys.stdout.write(out)
            if args.verbose:
                print(f"[rc={rc}] {c}", file=sys.stderr)
            if rc != 0 and not args.keep_going:
                break
        return rc
    return with_session(args, go)


def act_push(args):
    data = open(args.local, "r", encoding="latin-1").read()
    if not data.endswith("\n"):
        data += "\n"
    eof = "__IRIXSH_EOF__"
    if eof in data:
        print("file contains the heredoc terminator", file=sys.stderr)
        return 2

    def go(s):
        s.t.send(f"cat > '{args.remote}' <<'{eof}'\n{data}{eof}\n")
        out, rc = s.run(f"wc -c < '{args.remote}'", args.timeout)
        print(f"pushed {args.local} -> {args.remote} ({out.strip()} bytes)")
        return rc
    return with_session(args, go)


def act_pull(args):
    def go(s):
        out, rc = s.run(f"cat '{args.remote}'", args.timeout)
        if args.local:
            open(args.local, "w", encoding="latin-1").write(out)
            print(f"pulled {args.remote} -> {args.local} ({len(out)} bytes)")
        else:
            sys.stdout.write(out)
        return rc
    return with_session(args, go)


def act_mount_shared(args):
    def go(s):
        out, rc = s.run("mkdir -p /shared; (mount | grep -q ' /shared ') || "
                        f"mount {args.nfs_host}:/shared /shared; mount | grep ' /shared '",
                        args.timeout)
        sys.stdout.write(out)
        return rc
    return with_session(args, go)


def main():
    # Common options are accepted before or after the action.
    common = argparse.ArgumentParser(add_help=False)
    for name, kw in [("--host", dict(default="localhost")),
                     ("--port", dict(type=int, default=2323)),
                     ("--user", dict(default="root")),
                     ("--password", dict(default=None)),
                     ("--connect-timeout", dict(type=float, default=10)),
                     ("--login-timeout", dict(type=float, default=60)),
                     ("--timeout", dict(type=float, default=300, help="per-command timeout (wait: total)")),
                     ("--nfs-host", dict(default="192.168.0.1"))]:
        common.add_argument(name, **kw)
    sub_common = argparse.ArgumentParser(add_help=False)
    for a in common._actions:
        sub_common.add_argument(*a.option_strings, dest=a.dest, type=a.type,
                                default=argparse.SUPPRESS, help=a.help)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                parents=[common])
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("wait", parents=[sub_common])
    r = sub.add_parser("run", parents=[sub_common])
    r.add_argument("cmds", nargs="+")
    r.add_argument("-k", "--keep-going", action="store_true", help="continue after a failing command")
    r.add_argument("-v", "--verbose", action="store_true")
    ps = sub.add_parser("push", parents=[sub_common])
    ps.add_argument("local")
    ps.add_argument("remote")
    pl = sub.add_parser("pull", parents=[sub_common])
    pl.add_argument("remote")
    pl.add_argument("local", nargs="?")
    sub.add_parser("mount-shared", parents=[sub_common])
    args = p.parse_args()
    fn = {"wait": act_wait, "run": act_run, "push": act_push, "pull": act_pull,
          "mount-shared": act_mount_shared}[args.action]
    try:
        sys.exit(fn(args))
    except (TimeoutError, ConnectionError, OSError) as e:
        print(f"irix-sh: {e}", file=sys.stderr)
        sys.exit(124)


if __name__ == "__main__":
    main()
