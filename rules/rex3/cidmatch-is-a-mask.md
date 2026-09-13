# CIDMATCH selects permitted CIDs

REX3 CLIPMODE bits 12:9 are a four-bit permission mask, not an equality value.
Bit N permits a framebuffer write to a destination with two-bit CID N.
Mask 0 rejects all CIDs; mask 15 permits all. Popup bits are not CID bits.
Framebuffer reads bypass this check; screen-to-screen copies check the destination.
See `docs/rex3.pdf`, table 16 and section 3.3.

The StudioPaint artifact capture had CLIPMODE 0x00000403, selecting CID 1
with mask 0010. The pending destination rectangle had AUX bytes 0x11,
so its CID was 1. Comparing AUX's low nibble against 2 incorrectly rejected
those writes. This bug existed in both interpreter drawing and REX JIT;
the interpreter's screen-to-screen path also omitted destination CID checking.
Disabling REX JIT therefore did not eliminate the bug.

Regression tests `cid_write_masks_interpreter` and `cid_write_masks_jit`
cover all masks, CIDs, and popup values for blocks, integer lines, and copies.
The compiled test checks the dispatch counter to exclude interpreter fallback.
Graphics test initialization must use mask 15 when CID clipping is unwanted,
and explicit JIT compilation requests must use the same CLIPMODE key.

The register capture establishes a clipping defect, but confirmation that it
fully resolves StudioPaint's artifacts requires a guest reproduction with the
updated binary. Do not restart a running guest to validate without authorization.
