# Getting serial output out of bare-metal guest code

Three things silently swallow SCC output, all hit while bringing up the CPU
test-suite support. In order of how long each one costs you:

1. **IOC serial 1 is SCC channel *B*, not A.** `IOC_SERIAL1_CMD`/`_DATA`
   (`0x1FBD9830`/`0x34`) map to `Z85c30::write` indices 0/1, and that function
   dispatches `0 => channel B, 2 => channel A`. So the first serial port in the
   IOC map is ttyd1 → **TCP 8881**. Channel A (ttyd2, TCP 8880) is at
   `0x1FBD9838`/`0x3C`. Writing to `0x…30` and listening on 8880 gets you
   nothing, and looks exactly like a broken device.

2. **Transmit is gated on WR5.TX_ENABLE.** An uninitialised channel queues the
   byte and the TX thread never latches it. Minimum init for output: write `5`
   to the command register (select WR5), then `0x68` (`TX_8BITS | TX_ENABLE`).
   Baud/BRG can stay unset — `tx_delay` then works out to 4 µs per character.

3. **The TCP serial backend holds exactly one client, and only notices a dead
   one on a failed write.** `TcpSocketBackend::send_byte` writes to
   `conn` if it is `Some`; `conn` is only replaced by an `accept()` inside
   `recv_byte`, which runs when `guard.is_none()`. A client that connected and
   went away leaves `conn` populated, so a *new* connection sits unaccepted in
   the backlog and every transmitted byte is written into the dead socket.
   Symptom: you connect, get no telnet handshake (`ff fb 01 …`), and see no
   output. Treat "no handshake within a second" as "reconnect", or use
   `--serial-log FILE`, which tees ttyd1 to a file and avoids the socket
   entirely — much the better option for a CI harness.
