# Linux's qla2xxx logs in to ISP2200 loop ports through request-ring mailbox entries

Status: rule (Linux 7.3.4 qla2xxx 12.00.00, ISP2200 on the Octane's PCI cage, 2026-11-15).

**Symptom.** qla2xxx loads, the loop comes up ("LOOP UP detected"), and no
disk ever appears. `scsi2 notes` shows `loop command 0x0074 not modelled`.

**Cause.** On the parts before the 24xx (not FWI2-capable) the driver logs in
to each loop port two ways: synchronously with mailbox command 0x0074 (Login
Loop Port, `qla2x00_login_local_device`), and from its async discovery state
machine with a *mailbox entry* in the request ring, entry type 0x39
(`MBX_IOCB_TYPE`, `struct mbx_entry` in `qla_def.h`), built by
`qla2x00_login_iocb` (m0 = 0x006f), `qla2x00_logout_iocb` (0x0071) and
`qla2x00_adisc_iocb` (0x0064 into `async_pd_dma`). The firmware answers in the
response ring with an entry of the same type: handle copied, `entry_status`
and `status` 0, the output mailboxes at the same offsets
(`qla2x00_mbx_iocb_entry`). Both paths were refused, so no port was ever
logged in. IRIX and the BSDs never send either, which is why IRIX 6.5.30
worked.

**Fix.** `Isp2200::mailbox_entry` serves the four commands Linux puts in an
entry and answers anything else there with 0x4001; 0x0074 answers COMPLETE
for a present port. Test: `isp2200::tests::loop_tests::port_login`.

Layout of `struct mbx_entry` (little-endian): type 0, count 1, handle 4,
loop ID 8 (the high byte on parts without extended IDs), status 10, m0 24,
m1 26, m2 28, m3 30, m6 32, m7 34, m9 36, m10 38.
