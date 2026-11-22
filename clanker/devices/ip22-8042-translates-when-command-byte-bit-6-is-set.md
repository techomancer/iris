# The Indy/Indigo2 8042 translates to set 1 while command-byte bit 6 is set

Status: rule (2026-11-19, issue #765).

**Symptom.** NetBSD/sgimips on an Indy (Newport console, `pckbc_hpc` +
`pckbd`): cursor keys do nothing in sysinst, and pressing `a` (or any key)
makes the menu act as if Enter were held down for ever.

**Cause.** `Ps2Controller` stored the 8042 command byte but ignored bit 6
(XLATE). NetBSD's `pckbc_attach` writes `KC8_CPU` (0x04), then
`pckbd_set_xtscancode` sets `KC8_TRANS` (0x40), reads the byte back with
0x20, selects set 2 on the keyboard and sets `t_translating = 1`: from then
on `pckbd` decodes **set 1**. iris kept sending raw set 2, so `a` (`1C`)
read as set-1 Enter, its break `F0 1C` as "release 0x70, press Enter", and
`E0 75` (Up) as an unknown extended key. No break ever arrived, so wscons
autorepeated.

**Rule.** Only an 8042 translates: `Ps2Controller::new_8042` (the IP22/IP24
IOC, `src/dev/ioc.rs`) honours bit 6; `Ps2Controller::new` (MACE on the O2,
IOC3, IOC4, which reach each device on its own port and have no command
byte) never translates, though its power-on `config` still reads 0x47.
Translation runs over the keyboard's own bytes only, in `queue_kbd`: scan
codes, the Read ID reply (`AB 83` becomes `AB 41`) and the Get Scancode Set
reply (`2` becomes `41`). Controller replies (command byte, self-test) are not
translated. The translator swallows `F0` and sets bit 7 of the next byte.
The table is the standard 8042 one (as in QEMU's `ps2.c`). The unit test
`translating_8042_turns_set_2_into_set_1_for_every_key` checks it against
`map_keycode_set1` for every key.

**Observed (Indy, R4400, embedded PROM, NetBSD 11.1 INSTALL32_IP2x).**
The PROM writes command bytes 0x20, 0x00, 0x10 and selects set 3, so it never
translates; its menu and command monitor take keys as before. NetBSD leaves the
command byte at 0x47 with the keyboard in set 2, and after the fix reads `E0 50`
/ `E0 D0` for Down, `E0 48` / `E0 C8` for Up and `1E` / `9E` for `a`. To trace
this, use `log ps2 file <path>` then `log ps2 on`: `IRIS_DEBUG_LOG=ps2` logs
every status poll the PROM makes to stderr and to the monitor, and the monitor
stops answering.

**IRIX.** IRIX 6.5 on an Indigo2 IP28 (same IOC controller) leaves the command
byte at 0x03 with the keyboard in set 3, so it never translates either. With
the fix, clogin took typed letters, Left and Return (`gust`, Left, Left, `e`
logged in as `guest`). No Indy IRIX disk was at hand.

**E2E check.** `tools/tests/netbsd-indy-keyboard.py` boots the install CD to
sysinst and judges which screen rows each key changes, because NetBSD 11.1's
glyphs do not reach the Newport screen under `--ci` (only its text cursor
does) while 11.99.11's do. It fails on the pre-fix build (cursor
never moves, screen keeps changing after `a`) and passes with the fix, on 11.1
and on the 11.99.11 daily.
