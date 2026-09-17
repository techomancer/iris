# Machine profile vs guest "IP22" in Software Manager

**Keywords:** machine,profile,ip22,ip24,indy,indigo2,guinness,software manager,hinv

## Naming (not a bug)

| Name | Where | Meaning |
|------|-------|---------|
| **IP24** | GUI Platform, TOML `[machine] profile = "indy_ip24"` | SGI Indy board (Guinness) product designation |
| **IP22** | IRIX Software Manager, `hinv` | Kernel platform family for IP22-class machines (Indy, Indigo2, …) |
| **indy** | GUI status bar | Config preset **filename**, not hardware type |

On a real Indy running IRIX 6.5, Software Manager and `hinv` still report **IP22** as the platform family (e.g. "IP22 Processor"). Selecting **SGI Indy (IP24)** in the GUI does not mean IRIX will say "IP24" in inventory.

## Config is enforced at `Machine::new`

`[machine] profile` drives the `guinness` flag passed to MC, IOC, and HPC3:

- `indy_ip24` → `guinness = true`, MC SYSID `0x00000013`
- `indigo2_ip22` → `guinness = false`, MC SYSID `0x00000010`

Both profiles are supported in the default build (GUI platform dropdown and CI use the same binary).

## Verify after changing profile

1. Monitor console (`127.0.0.1:8888`): `mc regs` → SYSID `00000013` on Indy IP24; `00000010` on Indigo2 IP22.
2. IRIX guest: `hinv | head -20` → IP22-family processor line + R4400; Indy-appropriate devices (single Newport, not dual-head Indigo2).
3. TOML with `profile = "indigo2_ip22"` → Start succeeds; hardware is fullhouse, not Guinness.

## Window title

`emulator_name()` may randomize the host window title (e.g. "Incredible Rust Indy Simulator"). That string is unrelated to guest inventory.
