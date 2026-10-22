# PROM `boot -f bootp()file`: what the ARCS PROM demands of BOOTP and TFTP

Source: `stand/arcs/lib/libsk/fs/bootp.c` and `net/udpip.c` (irixhistory).
Test: `test/tftp/` (guest userspace) and a live `boot -f bootp()/unix`.

## The BOOTP reply must name the file

`bootp.c` aborts with "File X not found on server" when `bp_file` in the
reply is empty, and it then TFTPs **exactly the name the server returned**,
not what the user typed. A gateway that answers BOOTP but leaves `file`
zeroed (fine for DHCP clients) makes every PROM network boot fail before a
single TFTP packet is sent. `handle_bootp` fills it (plus `sname`) when the
TFTP root has the file.

## Every reopen is a second BOOTP, and its reply must be unicast

The ELF loader seeks backwards. The bootp "filesystem" can't seek, so it
closes and reopens, which runs a whole new BOOTP + TFTP. On the first BOOTP
`ciaddr` is 0, so the PROM sets `IFF_PROMISC` and accepts anything. After
that it has an address, and `_ip_input` drops every packet whose `ip_dst`
is not exactly its own address, **broadcast included**. A broadcast reply
to the second BOOTP is silently lost:

```
Obtaining /unix from server iris
No server for /unix.
error reopening file in reverse seek
```

Reply to `ciaddr` if set, else `yiaddr` (bootpd's behaviour). DHCP replies
keep the broadcast IRIX's own client was already working with.

## Names and modes

The PROM always asks for `octet`, and passes absolute names through
(`bootp()/unix` → RRQ `/unix`). Command-line clients (`/usr/bsd/tftp`,
tftp-hpa) default to `netascii`. Serve absolute names relative to the root
(`tftpd -s`) and support both modes. Refusing either one looks like "TFTP
is broken" from the field.

## Getting to the Command Monitor

`reboot`, `init 0` + any key, and even a monitor `reset` all autobooted
here despite `nvram AutoLoad no`; that was not chased. Pressing Esc during
"Starting up the system" always works: send `ps2 esc` from the monitor
console every 2 s from launch, then `ps2 type 5` for the Command Monitor.
