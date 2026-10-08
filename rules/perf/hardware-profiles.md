# Hardware profiles vs MAME (Phase 3)

| Capability | MAME Indy | IRIS Phase 3 |
|------------|-----------|--------------|
| IRIX 6.5 desktop | Slow; DRC often flaky | Premiere stack (`lightning` + `rex-jit` + `idle-pause`; jitv2 optional) |
| Config GUI | None | Full `MachineConfig` + export TOML |
| CI automation | External scripts | `iris-ci` + TCP on Windows |
| Extended RAM | Limited | 384/512 MB GUI presets |
| Networking | Yes | NAT + pcap + port forward |
| Audio tuning | Basic | TimerManager frame pacing, persistent output/resampling, underrun stats |
| Indigo2 single-head | Yes (slow) | `profile = indigo2_ip22` — default build, no cargo feature |
| Indigo2 dual-head | Yes (slow) | `profile = indigo2_ip22`, `graphics.heads = 2` |

Profiles in TOML:

```toml
[machine]
profile = "indy_ip24"   # default — enforced at Machine::new (guinness=true)
# profile = "indigo2_ip22"
```

`profile` is **not cosmetic**: it sets MC/IOC/HPC3 Guinness layout. IRIX still reports **IP22** as the platform family on Indy — see [`rules/gui/machine-profile-vs-guest-ip22.md`](../gui/machine-profile-vs-guest-ip22.md).

R4400, R5000, and R10000 are runtime CPU settings; all models are built in.
Pair R10000 with `indigo2_ip28`, which has an embedded IP28 PROM. GR2 XZ/Extreme
and IMPACT boards are built in; choose them with `[graphics] board`.

## RAM presets (stability)

| IRIX version | Recommended `banks` | Guest RAM |
|--------------|---------------------|-----------|
| 6.5 | `[128, 128, 64, 64]` | 384 MB |
| 6.5 / Indy authentic | `[128, 128, 0, 0]` | 256 MB |
| 5.3 | `[128, 128, 128, 128]` | 512 MB |

On IP22/IP24, 512 MB across four banks on IRIX 6.5 is not documented as
supported — use 384 MB (`irix-install/iris-windows-384.toml`) if apps quit unexpectedly after a GUI RAM upgrade.

IP28 supports two 512 MB banks (1 GB); its larger MEMCFG granule is separate
from the IP22/IP24 four-bank limitation. See `rules/irix/ip28-512mb-banks.md`.

`[clock] fixed_mhz` sets CP0 Count MHz (default 33 on IP22/IP24, 97.5 on IP28).
IRIX reports twice that as CPU MHz; status-bar MIPS measures host throughput.
