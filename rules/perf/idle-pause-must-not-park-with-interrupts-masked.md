# The idle-park heuristic must not park when Status.IM is zero

`IdleParkState::park` waits for `(ip & im) != 0`. `update` decided whether to
park from the Status IE bit and from whether an interrupt was already ready,
and never checked whether IM was *zero* — with every mask bit clear that wait
cannot be satisfied by any interrupt from any source, so the CPU thread parked
and stayed parked.

Early in boot IRIX runs with IE set and IM zero, so a `--features idle-pause`
build hung before the kernel banner: 0 of 4 boots reached a login prompt, 4 of
4 with the feature off, 4 of 4 with the guard.

Instrumenting the park loop at the point of the hang:

```
status=0x00000081 cause=0x0000000c pending=0x8000 ip&im=0x0
```

`status=0x81` is IE plus KX with every IM bit clear, and `pending=0x8000` is
IP7 — the timer interrupt had already been raised and was masked off.

IE says the guest would take an interrupt if one arrived; IM says whether one
can be delivered at all. Parking needs both.

Easy to miss because `update` already computes `im` for `interrupt_ready`, so
it reads as though the mask is accounted for.
