# The 8042's translate bit (config bit 6) has to translate

Status: rule (2026-11-16).

**Symptom.** The NetBSD/sgimips installer on an Indy ignored the cursor keys
and a menu letter repeated forever. The PROM and IRIX were fine.

**Cause.** The controller's config byte powered up as 0x47, so the translate
bit read as set, but nothing translated: keyboard bytes stayed set 2. NetBSD
(`pckbc_xt_translation`) sets the bit, reads it back, takes that as "the 8042
turns set 2 into set 1" and decodes set 1. A set 2 break (F0 xx) is not a
set 1 break, so no key ever came up.

**Rule.** With bit 6 set, the keyboard's bytes leave the 8042's data port as
set 1 (`SET2_TO_SET1` in `read_data`: E0 passes, F0 becomes bit 7 on the next
byte). The config byte powers up as 0x07, translation off, because the Indy
PROM writes 0x20/0x00/0x10/0x00 and the guests that matter set the bit
themselves. The 8042's own replies (self-test result, config read-back) are
tagged `Ps2Source::Controller` and never translated. Only the Indy and
Indigo2 IOC reads the data port. The O2 (`mace_ps2`) and the Octane and
Origin (`ioc3_kbc`) take bytes with `pop_from`, one port per device, so the
bit cannot reach them. Translating where the keys are queued instead would
have reached them, and would have turned the keyboard of an old O2 snapshot
(config saved as 0x47) into set 1.

**Checked.** `translation_table_agrees_with_the_set_1_and_set_2_key_maps`
checks the table against the independent set 1 and set 2 key maps (it caught
F7 = 83 and Alt+SysRq = 84, the only set 2 codes above 7F), and the table
equals QEMU's. IRIX 5.3 and 6.5.22 on the Indy and 6.5.22 on the Indigo2 (IP22,
IP28) boot and type as on unmodified main (guest config 0x03, scan code set 3);
the O2, Octane and Fuel PROMs get identical keyboard bytes.
