# A receive channel that refuses while reporting itself active

**Fixed 2026-09-19** (`hpc3.rs`). Found with NetBSD/sgimips; IRIX never showed
it. Kept because the two wrong diagnoses on the way are more instructive than
the fix, which is four lines.

## Symptom

A bulk transfer into a NetBSD guest stopped after a few megabytes and the whole
interface went mute until reboot — DNS, ping, new connections, all dead. Stall
point random across runs of the same file (1.9, 3.2, 3.7, 5.0, 8.4, 15.8 MB), so
a race. Small transfers always worked, which is why nothing had noticed.

## The bug

The receive channel walks the driver's descriptor chain until it reaches the
end-of-chain guard, which the host still owns (`ROWN=0`) because the driver has
not reaped and re-armed it yet. We refused the frame — correct — and left the
channel `ACTIVE` — not correct, and it is the only thing the driver can see:

```c
/* if_sq.c, sq_rxintr() */
status = sq_hpc_read(sc, sc->hpc_regs->enetr_ctl);
if ((status & sc->hpc_regs->enetr_ctl_active) == 0) {
        ... re-arm the channel ...
}
```

That ACTIVE bit is our channel's `ctrl` bit. Claiming to be active while
refusing every frame tells the driver there is nothing to fix, so it does
nothing, so the host never hands a descriptor back, so we keep refusing.

The fix: clear ACTIVE when refusing on an `EOX` descriptor. The chain really is
exhausted there, stopping is what the hardware does rather than running off the
end, and it is what lets the driver's own restart path run.

Measured: 405 MB at 9.7 MiB/s on NetBSD 11.0, where nothing had ever got past
16 MB. IRIX regression check — 20 000 flood pings at 1400 bytes, 28 MB, 0.0%
loss — unaffected.

## The instrument that found it

Counters on the receive pump, printed by `seeq status`:

    rx_delivered=2625   (frozen)
    rx_refused=161167 → 195318   (~880/s, climbing)

A frame in hand, refused forever. Everything before this was guesswork; this
took one run. **Add the counter before theorising** — that is the lesson.

Then `pdma status` named the state exactly:

    [10] ENET RX : Active=true CBP=17ef6800 BC=e0000213 CTRL=200
                                   EOX=1 EOP=1 XIE=1 ROWN=0

`pdma chain <addr>` walks the ring showing ROWN per descriptor if more detail
is ever needed.

## Two diagnoses that were wrong, and why

**"CLRINT loses the RX interrupt."** `net.rs` carries a FIXME saying exactly
that, which is what made it convincing. Two fixes were built on it. The first —
stop `reset_interrupt` forging `OLD` and re-evaluate the line immediately —
broke networking outright: `dhcpcd` never completed a lease. The second re-armed
from the enet thread instead, survived DHCP, and did not fix the stall. Both
reverted. The FIXME may still describe a real bug; it was not this one.

**"The enet thread stops pumping."** Based on the guest reporting transmitted
pings while `net status icmp` showed no entries. **That evidence was bad**: ICMP
NAT entries expire after 30 s and the table was read after the pings finished.
An empty NAT table proves nothing unless it is read inside the entry's lifetime.

A third theory — that `enetr_ctl`'s ACTIVE bit and the refusal condition could
disagree — was half right and worth recording: they cannot disagree for the
`NOT_ACTIVE` refusal, because `is_active()` reads that same bit. They disagree
only for the **ROWN** refusal, which is a separate branch, and that is the bug.

## What the guest-side counters say at a stall

Useful for recognising this class of fault again, from inside a NetBSD guest:

| | |
|---|---|
| `netstat -i -d` | `Opkts` advancing, `Oerrs 0`, **`Ipkts` frozen** |
| `vmstat -i` | `sq0 intr` frozen or collapsed to a trickle |
| `arp -a` | **empty** — cannot resolve the gateway |
| `ping` | `sendto: Host is down` (an ARP failure, not a transmit failure) |

Receive is the broken direction; everything else is downstream of the guest not
getting an ARP reply. A guest that "cannot transmit" is usually a guest that
cannot receive.
