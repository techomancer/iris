# The IP35 PROM checks its flash CPU record against the node board EEPROM's format-3 extended data

Status: confirmed 2026-11-07 on IP35 PROM 6.210 (Tezro, all three R16000
parts then modelled). The fix is in the tree as `CpuContext`.

**Symptom.** Every power-on printed `*** Module 001c01 CPU configuration
values mismatch` / "Please reflash the PROM image", even with a correct flash
record.

**Cause.** The PROM reads the node board's EEPROM through the L1 (component 0,
board area 2) and, when the area's format byte is 2 or more, one
extended-data record after the revision fields. The record's first byte
selects its layout: 0 group/capability/variety, 1 those plus the laser (MAC)
field, 2 nothing, 3 the CPU configuration. Format 3 is ten type/length
fields, in order: HubModeBits, CpuCfg0Reg..CpuCfg3Reg (four bytes each,
big-endian, the mode word per slot), HubSpeed and CpuSpeed (four bytes, MHz),
MinIrixVer, MinPromVer, MinL1Ver (text; only their presence is checked). The
PROM turns these into its configuration (the mode word from CpuCfg0Reg, the
clocks times 1e6) and compares them with the flash record. A format-0 area
has no values; the PROM then falls back to a table row chosen by the board
word, and a flash record that matches neither draws the mismatch.

**Fix.** `L1Config::cpu_context` / `CpuContext` (`src/dev/l1/model.rs`);
the Tezro fills it from its firmware record (`ip53_machine::cpu_context`).
Test: `the_node_board_record_carries_a_configured_cpu_configuration`
(`src/dev/l1/tests.rs`). The Fuel and the Origin 3200 set none and rely on
the table-row fallback, which their flash records match (see
`docs/fuel-ip34.md`).

**How it was found.** `strings` on the PROM payload (field names near
`HubModeBits:`), the payload disassembled with `llvm-objdump
--triple=mips64` after wrapping it as an ELF, and a constant-propagation scan
of lui/daddiu/dsll32 chains to find the code that builds each string address
(the strings are not referenced by a single immediate).
