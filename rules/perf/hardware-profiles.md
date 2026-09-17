# Hardware profiles vs MAME (Phase 3)

| Capability | MAME Indy | IRIS Phase 3 |
|------------|-----------|--------------|
| IRIX 6.5 desktop | Slow; DRC often flaky | Premiere stack (`lightning` + `rex-jit` + `idle-pause`; jitv2 optional) |
| Config GUI | None | Full `MachineConfig` + export TOML |
| CI automation | External scripts | `iris-ci` + TCP on Windows |
| Extended RAM | Limited | 384/512 MB GUI presets |
| Networking | Yes | NAT + pcap + port forward |
| Audio tuning | Basic | HAL2-Pump spin thread + underrun stats |
| Indigo2 single-head | Yes (slow) | `profile = indigo2_ip22` — default build, no cargo feature |
| Indigo2 dual-head | Yes (slow) | `profile = indigo2_ip22`, `graphics.heads = 2` |

Profiles in TOML:

```toml
[machine]
profile = "indy_ip24"   # default — enforced at Machine::new (guinness=true)
# profile = "indigo2_ip22"
```

`profile` is **not cosmetic**: it sets MC/IOC/HPC3 Guinness layout. IRIX still reports **IP22** as the platform family on Indy — see [`rules/gui/machine-profile-vs-guest-ip22.md`](../gui/machine-profile-vs-guest-ip22.md).

R4400 vs R5000 remains a **compile-time** Cargo feature — GUI shows rebuild command on Debug tab.

## RAM presets (stability)

| IRIX version | Recommended `banks` | Guest RAM |
|--------------|---------------------|-----------|
| 6.5 | `[128, 128, 64, 64]` | 384 MB |
| 6.5 / Indy authentic | `[128, 128, 0, 0]` | 256 MB |
| 5.3 | `[128, 128, 128, 128]` | 512 MB |

512 MB on IRIX 6.5 is not documented as supported — use 384 MB (`irix-install/iris-windows-384.toml`) if apps quit unexpectedly after a GUI RAM upgrade.

There is **no guest CPU MHz / overclock** config; status-bar MIPS is host emulation throughput only.
