// Bake the release version into APP_VERSION at compile time.
//
// In CI, RELEASE_VERSION is set to the date-stamped tag (e.g. "2025-06-09-02-00").
// Locally, it falls back to the Cargo.toml version with a "-dev" suffix so
// debug builds are distinguishable from releases.
//
// The rerun-if-env-changed directives are required: without them, Cargo's
// build-script caching would keep APP_VERSION frozen at whatever value was
// baked on the first compile, even after RELEASE_VERSION changes between
// the `cargo test` and `cargo build` steps in CI.
fn main() {
    println!("cargo:rerun-if-env-changed=RELEASE_VERSION");
    println!("cargo:rerun-if-env-changed=CARGO_PKG_VERSION");

    let version = std::env::var("RELEASE_VERSION")
        .unwrap_or_else(|_| std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".into()));

    let profile = std::env::var("PROFILE").unwrap_or_default();
    let full_version = if profile == "debug" && std::env::var("RELEASE_VERSION").is_err() {
        format!("{}-dev", version)
    } else {
        version
    };

    println!("cargo:rustc-env=APP_VERSION={}", full_version);

    // `native_mac`: the `macos-gui` front-end is compiled in. The feature is
    // a no-op off macOS, so gate on the target here once instead of repeating
    // `all(target_os = "macos", feature = "macos-gui")` at every use.
    println!("cargo::rustc-check-cfg=cfg(native_mac)");
    let macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    if macos && std::env::var_os("CARGO_FEATURE_MACOS_GUI").is_some() {
        println!("cargo:rustc-cfg=native_mac");
    }
}
