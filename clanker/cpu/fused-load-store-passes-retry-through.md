# A fused load/store must pass EXEC_RETRY through, not raise it

Status: fixed 2026-10-28 (#270). Kind: rule.

**Symptom.** Indigo2 IP22 (R4400) with any IMPACT board, built with
`lightning` (or `opcodefusion` alone): the PROM panics during power-on
diagnostics with `Exception PC: 0x9fc36988, Exception RA: 0x9fc16fa0`,
bad address 0x10c. Default and `gpujit`-only builds boot. Newport, XZ and
Extreme boot under `lightning`. NVRAM contents do not matter.

**Cause.** `exec_fused_addr_load_store_tail` (ADDU/SUBU/ADDIU fused with
the following load or store) sent every status other than
`EXEC_COMPLETE` to `handle_exception`. IMPACT answers some reads with
`BUS_BUSY` (== `EXEC_RETRY`, 0x100), which has no exception bit, so
`handle_exception` took ExcCode from it: 0, an interrupt with no IP bits
set. The PROM's IMPACT setup (`addu v0,v0,at; lbu t3,0x444b(v0)` at
0x9fc3b918) hit it before the PROM had an exception frame, and the
handler then faulted on the null frame pointer at 0x9fc36988. That
second fault is the one the panic reports.

**Rule.** Any handler that finishes a raw read/write status must use
`finish_status`, which passes `EXEC_RETRY` and `EXEC_BREAKPOINT` through.
The fused tail has already moved PC to the load/store, so a retry runs
only the load/store again. jitv2's codegen learnt the same lesson
separately (see the comment near `emit_bail` in `jitv2/codegen.rs`).

**Finding it.** A breakpoint on the RAM exception vectors
(`0xffffffff80000180`; the PROM runs with BEV clear) stops on the
original exception, and `dt` shows the history. `ex` catches did not
stop it. Test: `test_addr_calc_load_store_fusion_retry`.
