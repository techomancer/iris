# A Mac build number must beat every build ever uploaded; the version stays put

Status: rule (2026-11-28), `.github/workflows/appstore.yml`, `scripts/dist/version.sh`.

Symptom: every TestFlight build of IRIS sat in "Waiting for Review" before
testers got it, while the other app's builds went straight out.

Cause: the workflow stamped a new date (`20261128.0156`) into both
`CFBundleShortVersionString` and `CFBundleVersion`. TestFlight groups builds
by version, and Beta App Review applies to the first build of each version
sent to an external group, so every upload was a first build. A build whose
version is not the one being prepared in App Store Connect (1.13 at the time)
cannot be attached to it either.

Rule:
- The version (`CFBundleShortVersionString`) is the release, from
  `scripts/dist/appstore-version`, the same for every upload until the next
  release. Change it when the next release starts, and keep it equal to the
  version being prepared in App Store Connect. It must be above the last
  version Apple approved for the App Store.
- The build number (`CFBundleVersion`) is `20261129.<run number>`. On macOS,
  unlike iOS, it must be above every build ever uploaded for the app, whatever
  its version (TN2420); the dated builds reached `20261128.156`, so a bare
  run number would be refused. Apple compares it part by part and drops
  leading zeros (`.0156` is 156).
- Re-running a run that already uploaded reuses its number and is refused.
