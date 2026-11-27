# Testing MIDI in an IRIX guest: startmidi -p rs232 and a libmd program

Status: howto, 2026-11-26 (Tezro, IRIX 6.5).

- `startmidi` defaults to RS-422 with an external clock (a Macintosh MIDI
  box on an Indy/Indigo2 SCC). On an RS-232 port (IOC3, IOC4, O2) give
  `-p rs232`: `nohup /usr/sbin/startmidi -n synth -d /dev/ttyd3 -p rs232 &`.
  It stays running; killing it unlinks the port. `startmidi` with no
  arguments lists the interfaces (IRIX's own "Software Synth" is always
  there). The System Manager's serial device panel does the same.
- A test sender needs only libmd (`cc x.c -lmd`, header
  `<dmedia/midi.h>`): `mdInit()` (returns the interface count), `mdGetName(i)`,
  `mdOpenOutPort(name)`, `mdSetStampMode(p, MD_NOSTAMP)` to send at once, and
  `mdSend(p, &ev, 1)` with `ev.msg[0..2]` = status, data1, data2.
- Check the host end with the monitor's `serial`: the `midi_synth` line counts
  messages and notes. With `[audio] host = "null"`, `IRIS_MIDI_CAPTURE=x.wav`
  still records the synth, so a test needs no speakers.
- IRIX `grep` has no `\|`: use `egrep 'a|b'`.
