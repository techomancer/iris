# Distribution builds

Every file IRIS publishes is made by a script in this directory. The GitHub
workflows install toolchains and call these scripts with explicit options, so
a local run with the same options produces the same files. Output goes to
`dist/` unless you pass `--out`.

| Script | Makes |
|---|---|
| `package-windows.sh` | Inno Setup installer, portable zip, CLI zip (run from Git Bash) |
| `package-macos.sh` | universal `IRIS.app` as dmg and zip, universal CLI tarball; signs and notarizes when credentials are set |
| `macos-bundle.sh` | just `IRIS.app`: builds iris-gui, adds Info.plist and icon, signs |
| `package-linux.sh` | deb, rpm, Arch pkg, optional GUI tarball, CLI tarball |
| `appimage.sh` | AppImage (in the Arch container CI uses) |
| `version.sh` | the version string: `nightly.YYYY.MM.DD`, `X.Y.Z`, or `dev.<commit>` |
| `release-notes.sh` | the release body with the downloads table, and the list of expected files |
| `publish-github.sh` | replaces a GitHub release with the files in a directory |

The packaging scripts take the same options:

```
--version V          default dev.<commit>
--target TRIPLE      default the host
--gui-features LIST  default iris/lightning,iris/jitv2,iris/gpujit,iris/idle-pause,hostgl
--cli-features LIST  default lightning,jitv2,gpujit,idle-pause,hostgl
--out DIR            default dist/
--skip-build         package what is already in target/
```

Packaged GUIs always add `bundled`, which hides the tools that only make sense
in a source checkout.

Examples:

```
scripts/dist/macos-bundle.sh                         # dist/IRIS.app, universal
scripts/dist/macos-bundle.sh --arch arm64 --features iris/lightning
scripts/dist/package-linux.sh --version 1.2.3
```

## Workflows

- `build.yml` builds everything for one commit and checks that every file the
  downloads table links to exists. Run it by hand to test a branch.
- `nightly.yml` runs daily at 05:00 UTC, builds `main` as
  `nightly.YYYY.MM.DD`, and replaces the single `nightly` pre-release. It skips
  the day when `main` has not moved.
- `release.yml` runs when a `vX.Y.Z` tag is pushed and publishes release
  `X.Y.Z`.
- `appstore.yml` (manual) builds the sandboxed, JIT-free App Store app,
  versioned as the newest `vX.Y.Z` tag on the commit it builds.

## macOS signing and notarization

`build.yml` signs and notarizes the macOS app and CLI when these repository
secrets exist, and falls back to ad-hoc signing with a warning when they do
not:

| Secret | What it is |
|---|---|
| `MACOS_CERTIFICATE` | base64 of a .p12 holding the **Developer ID Application** certificate and its private key |
| `MACOS_CERTIFICATE_PWD` | the .p12 password |
| `APP_STORE_CONNECT_KEY_ID`, `APP_STORE_CONNECT_ISSUER_ID`, `APP_STORE_CONNECT_PRIVATE_KEY` | App Store Connect API key (base64 of the .p8), used for notarization. The App Store workflow already uses these. |
| `MACOS_NOTARIZE_APPLE_ID`, `MACOS_NOTARIZE_PASSWORD`, `MACOS_TEAM_ID` | alternative to the API key: Apple ID, app-specific password, team ID |

Locally, `package-macos.sh` reads `MACOS_SIGN_IDENTITY` and either
`NOTARY_KEY_ID`/`NOTARY_ISSUER_ID`/`NOTARY_KEY_PATH` or
`NOTARY_APPLE_ID`/`NOTARY_PASSWORD`/`NOTARY_TEAM_ID`.

## Versions

The version goes into the About window, `iris-ci --version`, the macOS
Info.plist, the Windows installer and every file name. deb and rpm versions
need a leading digit, so a nightly is packaged as `0~nightly.YYYY.MM.DD`,
which sorts below any real release.
