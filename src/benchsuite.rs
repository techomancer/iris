//! The benchmark suite's guest binary, carried inside the emulator.
//!
//! `bench/` builds a bare-metal MIPS image with a cross toolchain and drops it
//! in `bench/build/`. That is right for development and useless everywhere
//! else: a released application has no toolchain, and a sandboxed one has no
//! writable path to unpack an image to either. So a known-good build is checked
//! in at `bench/prebuilt/` and linked into the binary, and
//! `Machine::load_elf_bytes` loads it straight out of `.rodata` — no file, no
//! subprocess, no build step, identical on macOS, Windows and Linux.
//!
//! Precedent: the 512 KB PROM is already embedded (`crate::prombin`), though as
//! a generated Rust array rather than a real file. `include_bytes!` is smaller,
//! compiles faster, and leaves the artifact diffable as the binary it is.
//!
//! **The copy must not drift.** Accuracy is scored against golden checksums
//! compiled *into* this image, so a stale image against fresh goldens reports
//! failures that are not real. `.github/workflows/bench.yml` rebuilds it and
//! fails on any difference; `make -C bench prebuilt` refreshes it.

/// The guest image, ELF32 MSB, ~285 KB.
pub static SUITE_ELF: &[u8] = include_bytes!("../bench/prebuilt/irisbench.elf");

/// Short blake3 of the guest binary, in the form stored on every result and in
/// `data/bench_reference.json`.
///
/// Reference figures only mean something against the exact suite that produced
/// them — add or change a kernel and every stored number silently becomes a
/// comparison between two different workloads. Short because it is an identity
/// tag a human pastes into a JSON file, not a security digest.
pub fn suite_id() -> String {
    suite_id_of(SUITE_ELF)
}

pub fn suite_id_of(bytes: &[u8]) -> String {
    format!("blake3:{}", &blake3::hash(bytes).to_hex()[..16])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_image_is_a_loadable_elf() {
        let elf = crate::elf::parse(SUITE_ELF).expect("embedded suite must parse as ELF");
        assert!(!elf.segments.is_empty(), "no PT_LOAD segments");
        assert!(elf.entry != 0, "no entry point");
        // KSEG0. The suite links to a fixed address above the PROM's load area;
        // anything else means the link script changed underneath us.
        assert!(elf.entry >= 0xFFFF_FFFF_8800_0000, "entry {:#x} is not in KSEG0", elf.entry);
    }

    #[test]
    fn the_suite_id_is_stable_and_short() {
        let id = suite_id();
        assert!(id.starts_with("blake3:"));
        assert_eq!(id.len(), "blake3:".len() + 16);
        assert_eq!(id, suite_id(), "suite_id must not depend on anything but the bytes");
    }
}
