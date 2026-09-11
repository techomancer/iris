# PS/2 mouse not detected under Linux 3.2 guests (and keyboard sometimes flaky too)

## Symptom

Under a Linux 3.2-based guest (not IRIX itself, but IRIX systems bundling a
Linux-derived PS/2 stack, or any Linux guest on this platform), the mouse is
never detected — `dmesg` shows `bad data from KBC - timeout` or the mouse
stays in `PSMOUSE_CMD_MODE`/`PSMOUSE_INITIALIZING` and never reaches
`PSMOUSE_ACTIVATED` even though well-formed 3-byte motion packets are being
sent. Keyboard input can also intermittently misbehave.

## Root causes (two independent bugs, both in `src/ps2.rs`)

**1. Command `0xD3` (`I8042_CMD_AUX_LOOP`) was unhandled.**

Linux's `i8042_check_mux()` (`drivers/input/serio/i8042.c`) probes for active
multiplexing at boot by issuing `0xD3` three times, each followed by a magic
byte (`0xF0`, `0x56`, `0xA4`), and expects each byte echoed back verbatim
through the AUX/mouse data path. The third byte matching (`0xA4 == 0xA4`) is
specifically what tells Linux "no mux here" and to proceed normally.

Our `write_command` fell through to the unsupported-command default for
`0xD3` — no response byte was ever queued. The probe's status-port poll loop
never saw OBF set, so it either timed out (burning wall-clock, and letting the
init sequence spill into the keyboard's turn) or left mux detection in an
undefined state. Fix: treat `0xD3` as "next data-port write is echoed back
verbatim, tagged as AUX/mouse source" (`CommandState::AuxLoop`). This makes
all three probe steps succeed exactly as the echo semantics require.

**2. Status port AUX bit (0x20) was only set for motion-packet bytes.**

`read_status()` only asserted `I8042_STR_AUXDATA` (bit 5 / `0x20`) when the
head of `rx_queue` was tagged `Ps2Source::Mouse`. But mouse *command
responses* (ACK/BAT/ID — the 0xFA/0xAA/id-byte replies to F2 GETID, F4
ENABLE, F6, FF RESET, etc.) are tagged `Ps2Source::MouseCmd`, a separate
variant that intentionally isn't counted in `mouse_queue_bytes` (see the
`Ps2Source` doc comments). Those bytes never set the AUX status bit, so from
the guest's perspective every mouse init ACK looked like it came from the
*keyboard* port. A driver validating the response source (or just confused by
the mismatch) can fail to complete the mouse init handshake, never reach
`PSMOUSE_ACTIVATED`, and then silently drop legitimately-tagged motion
packets that *do* set AUX correctly later.

Real 8042 hardware asserts AUXDATA based on which physical port a byte
arrived from, not on payload semantics — command responses count. Fix:
`read_status()` now checks `matches!(source, Ps2Source::Mouse |
Ps2Source::MouseCmd)`.

**3. Port-level disable/enable commands (`0xA7`/`0xA8`/`0xAD`/`0xAE`) were unhandled.**

Independent of `0xD3` muxing, Linux's `i8042_check_aux()` verifies the
controller genuinely supports gating the AUX port before trusting it:
`i8042_toggle_aux(false)` sends `0xA7` (disable AUX) then reads CTR via
`0x20` and requires bit 5 (`I8042_CTR_AUXDIS`) to read back as 1; it then
calls `i8042_toggle_aux(true)` (`0xA8`) and requires bit 5 to clear. If
either check fails, `i8042_check_aux()` aborts with `-ENODEV` and mouse
support is disabled outright for the rest of the boot — a much harsher
failure mode than a bad packet, because no further probing is attempted.
`0xAD`/`0xAE` are the keyboard-port equivalents, gating CTR bit 4
(`I8042_CTR_KBDDIS`).

None of the four were handled — they fell into the same unsupported-command
default as `0xD3` originally did, so the CTR byte's bits 4/5 never moved and
the probe failed immediately. Fix: `0xA7`/`0xA8` and `0xAD`/`0xAE` now set/clear
bits 5/4 of `state.config` (already the CTR byte returned by `0x20`), and
`push_mouse_input`/`push_kb` additionally check their port's disable bit
before queuing data — so a disabled port is honored end-to-end, not just
reported correctly to the boot-time probe.

## Where

- `write_command` / `write_data` — `src/ps2.rs` (0xD3 → `CommandState::AuxLoop`
  → next `write_data` byte echoed back tagged `MouseCmd`; 0xA7/0xA8 and
  0xAD/0xAE toggle CTR bits 5/4 in `state.config`)
- `read_status` — AUX bit now covers both `Mouse` and `MouseCmd` sources
- `push_mouse_input` / `push_kb` — also check `config & 0x20` / `config & 0x10`
  so a disabled port actually stops delivering data, not just reports the bit
- Regression tests: `ps2::tests::aux_loopback_probe_echoes_verbatim`,
  `aux_port_disable_enable_toggles_ctr_bit5`,
  `kbd_port_disable_enable_toggles_ctr_bit4`,
  `disabled_aux_port_blocks_mouse_packets`

## Note

`command_state` gained a 7th discriminant (`AuxLoop` = 6) in the save-state
integer encoding (`save_state`/`load_state`) — old snapshots without this
state decode fine (default `_ => Idle`), but don't reuse discriminant 6 for
anything else.
