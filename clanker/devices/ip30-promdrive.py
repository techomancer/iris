#!/usr/bin/env python3
"""promdrive.py PORT TOTAL_SECONDS CMD... - drive the Octane PROM on its
serial console (TCP PORT): wait for the maintenance menu, enter the
command monitor, run each CMD at its prompt, and echo everything the
PROM says to stdout. Stops after TOTAL_SECONDS."""
import socket
import sys
import time

port = int(sys.argv[1])
total = float(sys.argv[2])
cmds = list(sys.argv[3:])
t0 = time.time()

s = None
while time.time() - t0 < 30:
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=2)
        break
    except OSError:
        time.sleep(1)
if s is None:
    sys.exit("no console on port %d" % port)
s.settimeout(0.5)
buf = b""
state = "boot"
last_send = time.time()


def send(txt):
    global last_send
    s.sendall(txt.encode())
    last_send = time.time()
    sys.stdout.write("\n>>> sent %r\n" % txt)
    sys.stdout.flush()


while time.time() - t0 < total:
    try:
        d = s.recv(4096)
        if not d:
            break
        sys.stdout.write(d.decode("latin-1"))
        sys.stdout.flush()
        buf += d
        buf = buf[-4000:]
    except socket.timeout:
        pass
    tail = buf[-400:]
    if state == "boot" and (b"Option?" in tail or b"choice" in tail or b"Maintenance" in tail) and time.time() - last_send > 3:
        send("5\r")
        state = "monitor"
    elif state == "monitor" and b">> " in tail and time.time() - last_send > 2:
        if cmds:
            send(cmds.pop(0) + "\r")
            buf = b""
        else:
            state = "done"
    elif state == "boot" and time.time() - last_send > 60:
        # nothing recognisable: try the menu choice anyway
        send("\r")
s.close()
