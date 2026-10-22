# Inbound port-forwards to rsh/rlogin need a reserved source port

## Symptom

`rsh` into the guest through a `[[port_forward]]` to guest port 514 always
fails, while telnet through the same mechanism works fine. `rshd` closes the
connection without writing a single byte, so a client that reports the server's
own framing has nothing to show:

```
rsh: server closed connection without response
```

`/var/adm/SYSLOG` on the guest has the real reason:

```
rshd[123]: Connection from 192.168.0.1 on illegal port 49152
```

## Cause

`poll_tcp_fwd_listeners` synthesizes the SYN it injects into the guest, and it
allocated the source port from `fwd_ephemeral_next`, which starts at 49152. The
guest never sees the host client's real source port — only the one we make up.

BSD r-services authenticate with `.rhosts`/`hosts.equiv` trust, which is only
meaningful if the client proved it was root by binding a reserved port. So
`rshd` rejects anything outside 512..1023 *before reading the request*:

```c
if (fromp->sin_port >= IPPORT_RESERVED || fromp->sin_port < IPPORT_RESERVED/2)
        exit(1);        /* "Connection from %s on illegal port" */
```

A client that binds a reserved port on the host — as a correct rsh does — makes
no difference, because the NAT discards it. This is also why running the client
under `sudo` changes nothing, which makes the failure look unrelated to ports.

## Fix

Allocate the injected source port from a separate 512..1023 counter when the
forward targets 513/514. The chosen port becomes part of the
`tcp_fwd_pending` / `tcp_nat` / `tcp_tw` keys, so probe past any port still live
for that guest port rather than blindly cycling — a wrapped counter would
otherwise overwrite an in-flight or established entry and break that connection.
512 ports covers any realistic number of concurrent rsh/rlogin sessions; if the
range is somehow exhausted, drop the accept.

## Guest-side setup this still needs

The forward makes the connection appear to come from the gateway
(192.168.0.1), and reverse DNS for it fails (queries go to the upstream
resolver, which knows nothing about RFC1918 space). Trust must be granted to
the gateway:

- `/etc/inetd.conf`: `shell stream tcp nowait root /usr/etc/rshd rshd`
- `/etc/hosts`: `192.168.0.1 gateway`
- `~/.rhosts`, mode 600 — `/.rhosts` for root, since `hosts.equiv` never
  applies to root: `gateway <local-username>`

## The stderr channel

The rsh protocol's second connection is a reverse one: the client passes a port
number and `rshd` dials *back* to it from a `rresvport()`. That direction
already lands correctly — `nfs_remap_dst` maps guest→`192.168.0.1:N` onto host
`127.0.0.1:N` — but the NAT rewrites its source port to an OS-chosen ephemeral,
and `rcmd(3)` clients check that the reverse connection's source port is
reserved too. Clients that send `0` as the stderr port (`rcp` does, as do most
modern implementations) sidestep this entirely. Supporting the stderr channel
would require the outbound NAT connect to bind a reserved port, i.e. root on
the host.

## Verified

IRIX 5.3, NAT mode, forward `2514 → 514`:

```
$ rsh -p 2514 root@127.0.0.1 uname -a
IRIX IRIS 5.3 12200159 IP22 mips
```

Use `127.0.0.1`, not `localhost`: `bind = "localhost"` binds IPv4 only, so a
client that resolves `localhost` to `::1` gets ECONNREFUSED — a failure that
looks exactly like the forward not being there.
