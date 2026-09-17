DONE net - ftp hangs at 48K, examine tcp window handling

rex
skip first/last for blocks?

DONE line stipple + lsadvlast (connected I_LINE continuation; rex3.rs pixel_count + setup-on-axis-mismatch)

DONE why is inactive terminal caret slanted and login picture frame too, bresenham innacuracy or something else?

logicop=src fastpath

DONE fract lines (draw_fline + fline_apply_fract)

DONE aa lines (draw_aline + endptfilter / aweight)

DONE dithering

HALFASSED cursor hotspot alignment

DONEish faster/better/lower latency gfifo

DONE octahedra not rendering the bottom of teh screen

cpu
DONE add callback on status writes so we can add side effects of user<>kernel 64<>32 switch and such

DONE nanotlb, cache last successfull address full translation for 3 access types.
reset on kernel<>user switch and tlb ops

DONE split translate into translate_32/64_kernel/user add function pointer

DONE new self-calibrating fasttick (since replaced: CP0 Count now runs at a fixed 33 MHz and IP7 comes from a host timer)

DONEish fpu
look at ide fpu test, convert to user space test, compile and run on irix, fix failures

watchhi/watchlo register support for debugging, translate variants that fire exception when hit


vino — SELECT_D1 per-channel routing + I2C repeated-start in vino.rs.
       Remaining: CDMC register fidelity beyond basic UYVY tweak;
       end-to-end visual verification under IRIX (vl_eoe / vino_eoe /
       indycam_eoe on 5.3 and 6.5) — see rules/irix/vino-verification-checklist.md

DONE scsi - eject, load cd

DONE scsi - C9 command

scsi - cdrom no sense in irix 5.3

DONE scsi - large requests fail

DONE ui - file selection for cd load (RCtrl+F12 in iris, Ctrl/Cmd+F12 and the SCSI menu in iris-gui)

DONE bus - writemask on write64 and write32 for more efficient uncached store left/right?

all - examine locks, possibly switch to spin, first check peripherials then lock in hpc and ioc


IP7 interrupt weirdness after reset/reboot

Xsgi crash in irix 5.3 R5K

DONE? early rex3 revision?



