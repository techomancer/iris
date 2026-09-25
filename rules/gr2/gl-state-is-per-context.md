# GR2 GL state goes through kernel memory on context switches

Symptom: a GL demo renders in one flat colour (amesh lines all purplish grey)
after another demo ran (blast). The trace shows the right per-vertex `cpack`
colours, but every `exec VERTEX` line has the same lit colour.

Cause: IRIS GL `winopen` does not turn lighting / blending / fog off; it relies
on a new GE context starting from the defaults. The kernel switches contexts
with `GE_HQMSAV` (0x1F0: id; DATA state, mode) and saves / restores the GE
state through its own memory (0x1E1 / 0x1E2 via HQ2_GEDMA), sized by what the
microcode reports in shram word 0x302. See HQ2.h "KERNEL TOKENS".

Emulator (src/dev/gr2/gl.rs, `gl_switch_context`, `gl_cx_*`): `GlState` is the
saved image, so it must stay plain data: `#[repr(C)]`, numbers and arrays
only, no pointers, no padding (compile-time asserts; `span: [f64; 6]` stays the
first field). New fields go anywhere after `span` and must be 4-byte types.
Growing the struct is fine (the kernel allocates what shram 0x302 says).

Replays and mid-session traces have no saved image for the running context:
a switch to it with state 1 keeps the live state (the lighting replay tests
depend on this).

Diagnosis tip: compare the colour sent (`0x6913` cpack, `0x182` colour) with
the `color (...)` on the `exec VERTEX` line. Different = lighting or fog is on.

GE_HQMSAV state 3 = detach (mode change or context exit, `Gr2DestroyDDRN`).
After it the GE has no owner: never report the detached id in shram 0x303
again, or the kernel dereferences its freed RRM node and panics ("KERNEL
FAULT", bad addr 0x4) at the next switch.
