# XDMCP reverse proxy (remote X login into the guest)

XDMCP (X Display Manager Control Protocol, UDP/177) lets a modern **X server**
display the IRIX login screen and desktop by asking the guest's `xdm` for a
session. Behind IRIS's software NAT the guest isn't directly addressable, so a
small application-layer gateway (ALG) makes it work — analogous to the FTP PASV
gateway. Both **IRIX 5.3 and 6.5** speak XDMCP 1.0; there's no version skew.

This applies to **NAT mode** (`[network] mode = "nat"`). In **PCAP bridged
mode** the guest is a real LAN host, so XDMCP works directly with no proxy.

## How it works

```
  X server (Xephyr/XQuartz)                IRIS host                 guest (IRIX xdm)
  ─────────────────────────                ─────────                 ────────────────
  XDMCP Query/Request  ──udp 177──▶  (redirect 177→11177)
                                      udp forward 11177 ──▶  gateway:11177 ──▶ :177
                                      ALG rewrites Request's
                                      connection-address → gateway
  X11 session  ◀──tcp 6000+N──  relay  ◀── gateway:(6000+N) ◀── xdm dials gateway:(6000+N)
```

1. A UDP port-forward `host:11177 → guest:177` carries the XDMCP control channel.
2. The ALG (`src/net/xdmcp.rs` + `net.rs`) rewrites the `Request` packet's
   *connection-addresses* to the NAT gateway and records `display → X-server
   address` (the datagram's source).
3. The guest's `xdm` opens the X11 session to `gateway:(6000+display)`. The NAT
   relays that to the real X server: to `127.0.0.1` for an X server on the IRIS
   host (the generic gateway→loopback path), or to the LAN address recorded in
   step 2 for an X server on another machine.

## Setup

### 1. IRIS: add the XDMCP forward

GUI → **Network** tab → **+ Add forward → "XDMCP (host 11177 to guest 177, UDP)"**.
It binds all interfaces so LAN X servers work. Or in `iris.toml`:

```toml
[[network.port_forward]]
proto = "udp"
host_port = 11177
guest_port = 177
bind = "any"
```

Port 11177 is the default (unprivileged so IRIS needs no elevation). Any
UDP forward to guest port 177 activates the ALG.

### 2. Guest: enable XDMCP in `xdm`

On the IRIX guest, make sure `xdm` accepts XDMCP queries: `/usr/lib/X11/xdm/Xaccess`
must allow the querying host (a bare `*` line allows direct queries), and
`xdm-config` must keep `DisplayManager.requestPort: 177` (not `0`). Restart `xdm`.

### 3. X-server host: redirect 177 → 11177

Stock X servers always send XDMCP to UDP **177** with no way to change the port,
so redirect 177→11177 on the **IRIS host** (one-time, needs admin once — keeps
IRIS itself unprivileged):

- **macOS / BSD (pf):** add to `/etc/pf.conf`, then `sudo pfctl -ef /etc/pf.conf`:
  ```
  rdr pass on lo0 inet proto udp from any to any port 177 -> 127.0.0.1 port 11177
  rdr pass on en0 inet proto udp from any to any port 177 -> 127.0.0.1 port 11177
  ```
- **Linux (iptables):**
  ```
  sudo iptables -t nat -A PREROUTING -p udp --dport 177 -j REDIRECT --to-ports 11177
  sudo iptables -t nat -A OUTPUT     -p udp --dport 177 -j REDIRECT --to-ports 11177
  ```

### 4. X server: it must listen on TCP

The X11 session connects over TCP, so the X server must accept TCP (many disable
it by default with `-nolisten tcp`). The easy test path is **Xephyr**, which
listens and queries in one command:

```
Xephyr :1 -query <iris-host> -ac -screen 1280x1024
```

(On macOS, XQuartz: Preferences → Security → "Allow connections from network
clients", then `Xephyr`/`Xnest` via XQuartz, or `X :1 -query <iris-host>`.)

The IRIX `xdm` greeter should appear in the Xephyr window; log in to get the
full desktop.

## Caveats

- **Auth:** only `MIT-MAGIC-COOKIE-1` / no-auth are supported. `XDM-AUTHORIZATION-1`
  cryptographically binds the addresses, so the ALG's rewrite would break it — it
  is detected and left unrewritten (the session won't establish). Configure `xdm`
  for magic-cookie or no auth.
- One active session per display number (the X11 port `6000+display`).
- IPv4 only.
