# Debug builds never start the guest CPU on their own

Status: rule (2026-11-22).

**Symptom.** In a debug `iris-gui` or `iris-appkit` (`cargo build -p ...`
without `--release`), Power On reports the machine started, but the status
says Powered off (or shows 0 MIPS) and the monitor's `status` says
`stopped pc=ffffffffbfc00000`.

**Cause.** `Machine::start` autostarts the CPU only under
`#[cfg(not(any(debug_assertions, feature = "developer")))]`: any build with
debug assertions starts paused, like a `developer` build, so a debugger or
the monitor can attach first. The front ends share the worker
(`model/handle.rs`), so every shell behaves the same.

**Rule.** Test a debug front end by sending `start` on the monitor
(`tools/mon.py start`) after Power On, or use a release build. Do not "fix"
the shells to start the CPU themselves; a release build starts it.
