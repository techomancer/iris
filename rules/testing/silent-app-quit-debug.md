# Silent IRIX app quit — debug capture checklist

**Keywords:** userspace, app quit, monitor, SYSLOG, 512 MB, IRIX 6.5
**Category:** testing

When IRIX apps close with no error dialog, collect evidence before guessing at fixes.

## Quick triage

1. **RAM layout** — IRIX 6.5: use `[128,128,64,64]` (384 MB), not 512 MB (`iris-windows-384.toml`). Verify `hinv -t memory` matches config after cold start.
2. **jitv2 A/B** — if built with `--features jitv2`, compare against a build without it on the same apps (the v1 JIT's `nojit` A/B toggle no longer exists — jitv2 is a compile-time-only feature with no runtime disable).
3. **Disk** — `cow status` in monitor; reset overlay if dirty sectors grew after panics.

## Capture bundle (send to developer)

| Artifact | How |
|----------|-----|
| TOML `banks` | `irix-install/iris-windows.toml` |
| Guest RAM | `hinv -t memory` in IRIX shell |
| Host log | `wsl\capture-app-crash.ps1` → `premiere-debug.log` |
| Monitor | telnet 8888: `stop` → `status` / `regs` / `bt` / `dt 80` |
| Guest | `ps -ef`, `tail -50 /var/adm/SYSLOG` after quit |
| jitv2 A/B | stable with jitv2 compiled out? yes/no |

## Monitor commands at quit moment

```text
stop
status
regs
bt
dt 80
exception all on
cow status
mc status
```

Developer build adds `debug on`, `log mips mask insn`, `dt file crash-trace.txt 1048576`.

See also `rules/jitv2/`, `rules/testing/disk-image-hygiene.md`.
