# A serial backend's refused byte must not have gone anywhere

Status: rule, 2026-11-13.

`SerialBackend::try_send_byte` returning `false` tells the 16550 to hold the
byte and offer it again (`ns16550.rs`, `poll_tx`). So a refusal must mean
nothing was sent to any host end, or the retry duplicates it.

- **Fan-out** (TCP + Unix socket, CI + socket): one end may take a byte the
  other refuses. `FanoutBackend` then returns `true` and owes the byte to the
  slow end (one byte each), refusing the next byte until every debt is paid.
- **TCP**: a telnet-escaped `0xff` is two bytes and a socket may take one.
  The rest waits in the client's `pending`; telnet replies queue behind it,
  never inside an escape.
- **Log tee**: logs on acceptance only, else every retry lands in the log.

Two consequences that are easy to miss:

- Held-back bytes (TCP `pending`, fan-out debts) are only written on the next
  send unless the receive poll also flushes them. `recv_byte` does, so the
  last character of a prompt is not stuck until the guest prints again.
- A byte the 16550 gives up on (after `XMIT_RETRY_TIMEOUT`, or at once while
  the backend counts as dead) goes to the plain `send_byte`, not nowhere. The
  sockets drop it there as they always did, but the log tee records it, so a
  stalled telnet client does not punch holes in `serial_log`, and the
  16550's output tail (`bytes_out`, `output_tail`) still has every byte.

The O2's MACE serial DMA (`mace_sdma.rs`) writes the UART without checking
`tx_ready`, so with a full socket it overruns the holding register: bytes are
lost there as before the hold existed. Gating it needs the TX walk to stop
mid-block.
