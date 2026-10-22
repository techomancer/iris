# iris-ci is the canonical CI socket interface

**Keywords:** iris-ci,wrapper,ci,socket,bs512,csh,run,put,get,boot,login
**Category:** snapshot

# iris-ci — the right way to drive the CI socket

Built alongside `iris` from `src/iris_ci_main.rs`. Talks to `/tmp/iris.sock` with typed clap subcommands. Use this, not raw `nc` + JSON, for any new automation, runbook, or test scenario.

## Common workflows

```bash
iris-ci boot                  # PROM menu → IRIS console login (~40s on M2)
iris-ci login                 # send root + handle vt100 prompt + wait #
iris-ci run 'echo hello'      # send shell command, get stdout, exit on guest failure
iris-ci put localfile.tar     # copy host file into guest, no bs=512 math
iris-ci get /tmp/log --to .   # pull guest file out, no conv=sync math
iris-ci save base/desktop
iris-ci diff a b              # per-device + chunk + cow-sector deltas
iris-ci script tests/x.iris   # batch-run a sequence (one cmd per line, # comments)
iris-ci pull http://reg/foo bar
```

`iris-ci --help` for the full subcommand list, `iris-ci <cmd> --help` for any subcommand.

## Why not raw nc + JSON

Three real bugs that bit during dogfooding and that the wrapper handles for you:

### 1. csh redirect syntax

IRIX root login uses csh by default. `2>&1` is sh-only. Use `>& /dev/null` for combined stdout+stderr in csh.

### 2. csh echoes typed input verbatim

Any wait pattern that appears in your typed command will match the input echo BEFORE the command runs. So a marker like `IRIS-CI-RC=` matches both:
- the typed-input echo line (which contains literal `IRIS-CI-RC=$status`)
- the actual output line (which contains `IRIS-CI-RC=0`)

Wait for `\nIRIS-CI-RC=` (newline-prefixed) — only matches at the start of the OUTPUT line, never inside the typed-input echo line because the echo is on its own line with no leading `\n` immediately before the marker.

### 3. IRIX raw block-device gotchas

- Reads MUST use `bs=512` or any 512-multiple. `bs=64` returns `Read error: I/O error` with no SCSI-level diagnostic.
- Writes must be padded to `bs`. From a 28-byte input file, `dd bs=512 conv=sync,notrunc` zero-pads to 512. Without `conv=sync`, the partial-block write fails.
- After receiving via `dd … of=FILE bs=512 count=N`, the guest file is N×512 bytes — too long. Truncate with `dd if=/dev/null of=FILE bs=1 seek=ORIG count=0`.

`iris-ci put` and `iris-ci get` handle all three transparently. The user passes a host filename and a guest path; the wrapper computes counts, chooses csh-correct redirects, runs `wc -c` for size lookup, and truncates as needed.

## When to read the JSON directly

For automation that doesn't want to depend on `iris-ci` (e.g. a test harness in another language), the underlying socket protocol is newline-delimited JSON. Each request is one JSON object with `cmd` and `args`; each response is one JSON object with `ok` and `data` or `error`. See `src/ci.rs` for the dispatch table. Don't expect to do this comfortably from a shell script — that's why iris-ci exists.

## Implementation notes

- Single-request, single-response per invocation. Connect, write one line, read one line, shutdown the write side so the server's read loop exits cleanly.
- `cmd_run` waits for `\nIRIS-CI-RC=` then drains the trailing `<digits>\nIRIS N# ` to keep the next command's drain clean. Sleeps 150ms between the wait and the trailing read to let those bytes arrive.
- `extract_run_stdout` skips the first `\n` (end of typed echo line), strips the trailing `\nIRIS-CI-RC=` marker, normalises CRLF.
- `cmd_put` uses `dd if=/dev/null of=FILE bs=1 seek=N count=0` for truncation rather than perl; perl isn't reliably installed in IRIX 6.5.
- `cmd_get` uses `wc -c < FILE` for size lookup. Avoids parsing `ls -l` columns which vary across IRIX versions.

