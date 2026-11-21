# A slow serial client never holds the guest

Status: rule, 2026-11-21.

A UART's transmit completes at once and THRE never reflects the host. On
hardware the wire never refuses, and the only back-pressure is the guest
driver's own flow control (IRIX's console runs `ixon`, so ^S from a client
pauses it).

- The log tee and the CI capture take every byte first, so they are always
  complete.
- Each TCP or Unix-socket client has a `ClientQueue`: what its socket has not
  taken, written out by `SerialBackend::flush` from the UART's existing poll
  (the 16550's `pump`, the Z85C30's receive thread, the L1's poll), one
  non-blocking write each, not a syscall per byte.
- Past `CLIENT_BACKLOG` (1 MiB) the oldest whole units go: a telnet escaped
  `0xff` or reply is never split, and a unit the socket took half of goes out
  whole first. The count shows in `serial status` (Indy, Indigo2) and
  `comN status` (O2), and a TCP client gets `[iris: N bytes dropped]` before
  the bytes that follow. A raw Unix-socket client gets no marker.
- One client never affects another: each has its own queue.

The hold-and-retry designs this replaced (a held byte in the 16550, then a
backend hold with a 2 s limit) made the guest wait on the slowest client and
still lost bytes when one stopped.
