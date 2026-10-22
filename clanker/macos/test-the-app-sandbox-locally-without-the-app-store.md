# Test the macOS App Sandbox locally — no App Store round-trip needed

The App Sandbox is enforced by the `com.apple.security.app-sandbox` entitlement
**at signing time**, not by App Store distribution. So you can run a genuinely
sandboxed `IRIS.app` locally and exercise the security-scoped bookmark /
folder-grant / CHD-fold flow without waiting on App Review.

## Recipe

```
./scripts/build-macos.sh appstore
open IRIS.app
```

The `appstore` variant (added 2026-06-19) does two things the plain build did not:
1. Compiles `--features appstore,iris/lightning`, so the **real** bookmark code
   (`#[cfg(feature = "appstore")]` in `macos_sandbox.rs`) and the
   `IRIS_CHD_DIFF_DIR` container redirect are active — not the off-sandbox stubs.
2. Signs with `installer/iris-gui-sandbox-local.entitlements` (app-sandbox =
   true), so the process actually runs sandboxed.

The earlier script signed with the MAS entitlements but built **without** the
feature, i.e. a sandboxed app with the bookmark logic compiled out — useless for
this test. That was the trap.

## Verify it's really sandboxed

```
codesign -d --entitlements - IRIS.app 2>/dev/null | grep app-sandbox
codesign --verify --verbose IRIS.app
ls ~/Library/Containers/io.github.<owner>.iris/    # created on first launch
strings -a IRIS.app/Contents/MacOS/iris-gui | grep IRIS_CHD_DIFF_DIR  # feature present
```

## Two gotchas that cost time

- **XML comments can't contain `--`.** codesign's entitlements parser
  (`AMFIUnserializeXML`) rejects a double hyphen inside a comment with
  "syntax error near line N". Writing `--features`/`--sign` in a comment broke
  it. The script now `plutil -lint`s the entitlements before signing.
- **A piped invocation hides the failure.** `build-macos.sh appstore 2>&1 | tail`
  reports the pipe's exit code (tail = 0), masking a codesign error. Run it
  without a pipe, or check for the "Signing bundle…" success line.

## Entitlements: local vs MAS

`iris-gui-sandbox-local.entitlements` omits the MAS-only
`com.apple.application-identifier` / `com.apple.developer.team-identifier` keys —
those need an embedded provisioning profile a real team identity that ad-hoc
(`codesign --sign -`) can't provide.

**Ad-hoc is enough for the within-session fold test** (grant a folder, boot,
quit → the `.diff.chd` folds away): the open-panel folder pick grants access for
the whole process lifetime, which is what the fold uses — it doesn't depend on a
bookmark resolving. **Cross-launch persistence** (relaunch, disk still
reachable) needs app-scoped bookmarks to resolve, which needs a stable identity:
`CODESIGN_IDENTITY="Developer ID Application: …" ./scripts/build-macos.sh appstore`.
