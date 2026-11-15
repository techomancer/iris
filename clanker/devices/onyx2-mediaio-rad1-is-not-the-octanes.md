# The Onyx2's MediaIO RAD1 is not the Octane's: revision 7.0, no 0x88/0x8c mute, counter from reset

**Status: rule (2026-11-12, `1877e10c`, `313022cc`, `ce3a53f0`).** Use
`Rad1::mediaio()` (`src/dev/rad1.rs`) for the Onyx2's MediaIO daughtercard,
never `Rad1::new()`. The two differ in three places; nothing else of the
MediaIO part's behaviour has been checked.

**Revision byte.** The Octane's onboard RAD1 reads 0xc0, which IRIX lists as
"RAD revision 12.0". Fitted to the Onyx2, the first audio transfer
(soundscheme at login, any `sfplay`) panicked the kernel. A real Onyx2's
`hinv -v` lists the MediaIO card as "RAD revision 7.0"; with 0x70
(`RAD1_REVISION_MEDIAIO`) the panic is gone. Why the 12.0 kernel path fails
against this model is not known: the byte is the only difference tried.

**0x88 and 0x8c are not a mute.** IRIX on the Onyx2 rewrites 0x8c about once
a second (every 1024 status completions), alternating 2 and 3. On the
MediaIO card that register is a serial data line and 0x88 its latch strobe.
Taking 3 as the Octane's DAC mute muted every other second: `sfplay` of a
6.8 s sample played as one-second bursts with silences between. The MediaIO
card stores both registers without acting on them.

**Counter and 0x84.** The MediaIO driver never writes the counter control
word at 0x40, but reads the counter at 0x30 twice 10 us apart at start-up and
reports "RAD UST not incrementing" unless it moved. The Octane's counter runs
only after its control writes, so on the Onyx2 it never moved. The MediaIO
counter runs from reset (and again after a card reset). The start-up's write
of 0 to 0x84 is stored and reads back, instead of tripping the
unsupported-access log.

**Tests:** `the_mediaio_card_presents_revision_7`,
`the_mediaio_card_plays_through_its_0x88_and_0x8c_writes` and
`the_mediaio_counter_runs_from_reset_and_through_a_reset` in
`src/dev/rad1/tests.rs`.
