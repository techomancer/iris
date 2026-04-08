# IRIX 6.5 Networking Configuration

## Required files

| File | Contents | Example |
|------|----------|---------|
| /etc/sys_id | Hostname | `IRIS` |
| /etc/hosts | IP-to-hostname mapping | `192.168.0.2 IRIS` |
| /etc/config/ifconfig-ec0.options | IP + netmask (hex) | `192.168.0.2 netmask 0xffffff00` |
| /etc/config/static-route.options | Default gateway | `$ROUTE $QUIET add net default 192.168.0.1` |
| /etc/config/network | Enable networking | `on` |

## Common mistakes

- **Wrong filename:** Use `ifconfig-ec0.options`, NOT `ifconfig-1.options`.
  IRIX names config files after the interface device name.

- **Missing IP in options:** The IP address goes IN `ifconfig-ec0.options`
  along with the netmask. It's not just options — it's the full ifconfig args.

- **Wrong gateway file:** Use `/etc/config/static-route.options`, NOT
  `/etc/defaultrouter`. The format uses shell variables: `$ROUTE $QUIET add net default <ip>`.

- **Netmask format:** IRIX uses hex notation: `0xffffff00` for 255.255.255.0.

## NVRAM MAC address (one-time setup)

The Seeq Ethernet controller reads its MAC from NVRAM. A fresh install has
no MAC set, which prevents networking.

1. Boot to PROM monitor (press Escape during countdown)
2. `>> setenv -f eaddr 08:00:69:de:ad:01` (any SGI OUI `08:00:69` MAC)
3. From iris monitor (telnet 127.0.0.1 8888): `rtc save`

## iris emulator network configuration

The emulator provides a NAT gateway with built-in DHCP:
- Gateway: 192.168.0.1 (hardcoded in GatewayConfig)
- Guest: 192.168.0.2 (assigned via DHCP or static)
- Netmask: 255.255.255.0
- DNS: forwarded to host's resolver

Port forwarding configured in iris.toml:
```toml
[[port_forward]]
proto = "tcp"
host_port = 2323
guest_port = 23
bind = "localhost"
```

## Keyboard workaround

Alt-tabbing away from the Rex window corrupts IRIX X11 keyboard input
(terminal apps show escape codes). Once networking is up, use:
```bash
telnet 127.0.0.1 2323
```
This connects via the port forward to IRIX's telnet daemon with a clean
terminal — no keyboard corruption issues.
