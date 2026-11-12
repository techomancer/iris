# The O2 PROM does not wait for its MTE clear: a status read must

Status: fixed (2026-11-12).

**Symptom.** On a busy host the O2 now and then sat at `SL-9600-8E>` on
serial with nothing before it, instead of reaching the PROM menu. A later
form of the same race prints nothing and parks the CPU at `0xbfc003a0` on a
reserved instruction.

**Cause.** PROM 4.18 clears 32 KB at RAM base with the MTE (`0xbfc061f0`),
reads RE status once without acting on it (`0xbfc06400`), stores a cookie at
`0xa000_1008` and copies its POST code to `0xa000_4000`. It checks the cookie
after POST (`0xbfc044f4`) and drops to the serial loader if it changed. The MTE
runs on the CRIME-RE host thread, and a status read waited at most 100 us.
When the host kept that thread off a processor longer, the clear ran late and
zeroed the cookie, or later still the POST code.

**Rule.** A status read answers `BUS_BUSY` until every MTE transfer queued
before it is done (`transfers_queued` against `transfers_done` in `CrimeRe`).
Primitives keep the bounded wait. MTE copies longer than 100 us lose their
overlap with the CPU. Software that reuses memory a transfer wrote without
reading the status word still races the engine. The same race exists for
primitives, whose frame buffers are main memory: a read-back with no idle wait
could see pixels not yet drawn. It is left uncovered on purpose, as no known
guest does that and holding primitives would cost drawing overlap.
