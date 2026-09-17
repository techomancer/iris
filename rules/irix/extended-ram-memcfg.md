# Extended RAM — MEMCFG synthesis for himem banks

IRIS can allocate banks 2–3 (himem at `0x20000000` / `0x28000000`) from
`banks` in config, but the embedded Indy PROM (`070-9101-011.bin`) often POSTs
only lomem (banks 0–1). Without valid MEMCFG halves for banks 2–3, IRIX stays
at 256 MB even when the GUI shows 384 or 512 MB.

## Fix (automatic)

After each MEMCFG0/1 write, if banks 0–1 are valid (VLD=1) and configured
himem banks are still invalid, `MemoryController::synthesize_himem_banks` patches
MEMCFG1 and fires the remap callback. See `src/mc.rs`.

## Verify

1. Set `banks = [128, 128, 64, 64]` (384 MB for IRIX 6.5), Stop → Start.
2. Monitor telnet `127.0.0.1:8888`: `mc regs` — banks 2–3 should show VLD=1.
3. In IRIX: `hinv -t memory` or System Manager → About This System.

## Layout reference

| banks | Guest (typical) |
|-------|-----------------|
| `[128,128,0,0]` | 256 MB (authentic Indy max) |
| `[128,128,64,64]` | 384 MB (IRIX 6.5) |
| `[128,128,128,128]` | 512 MB (IRIX 5.3 / emulator max) |

Guest RAM = sum of banks with MEMCFG VLD=1 after boot, not the config total alone.
