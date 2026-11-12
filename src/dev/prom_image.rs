//! SGI PROM image loading: version-six single-segment containers, the IRIX
//! installer's 256-byte header (the O2's `ip32prom.image`) and raw payloads.
//! A recognized but malformed container is rejected rather than retried as a
//! raw image.
//!
//! Ported from hw/mips/sgi/prom-image.c in jamesbraid/qemu-sgi-origami.

use std::io::{self, Read};

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn word(data: &[u8], off: usize) -> u64 {
    u64::from_be_bytes(data[off..off + 8].try_into().unwrap())
}
fn byte_sum(data: &[u8]) -> u64 {
    data.iter().map(|b| *b as u64).sum()
}

/// Read a PROM file and decode its payload, retaining the path in errors.
pub fn load(path: &std::path::Path, payload_limit: usize) -> io::Result<Vec<u8>> {
    let with_path = |error: io::Error| {
        io::Error::new(error.kind(), format!("PROM image {}: {error}", path.display()))
    };
    let file = std::fs::File::open(path).map_err(with_path)?;
    // Bound the encoded file by zlib's compression bound, the container
    // header, gzip framing and the accepted zero tail.
    let size = payload_limit as u64;
    let file_limit = size
        .saturating_add(size >> 12)
        .saturating_add(size >> 14)
        .saturating_add(size >> 25)
        .saturating_add(13 + 4096 + 12 + 7);
    if file.metadata().map_err(with_path)?.len() > file_limit {
        return Err(with_path(invalid(
            "encoded PROM image exceeds flash capacity",
        )));
    }
    let mut bytes = Vec::new();
    file.take(file_limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(with_path)?;
    if bytes.len() as u64 > file_limit {
        return Err(with_path(invalid(
            "encoded PROM image exceeds flash capacity",
        )));
    }
    decode(&bytes, payload_limit).map_err(with_path)
}

/// A PROM image and where it came from.
pub struct Loaded {
    pub bytes: Vec<u8>,
    /// The file it was read from; None for the PROM built into iris.
    pub source: Option<std::path::PathBuf>,
}

impl Loaded {
    /// The file's path, or "the built-in PROM", for messages.
    pub fn name(&self) -> String {
        self.source.as_ref().map_or_else(|| "the built-in PROM".into(), |p| p.display().to_string())
    }
}

/// `prom = "(embedded)"`: use the PROM built into iris, whatever files exist.
pub fn is_embedded_keyword(prom: &str) -> bool {
    prom == "(embedded)"
}

/// The PROM `profile` runs: the file `prom` names, which must load; or, when `prom` is
/// left at its default, `prom.bin` or the file name the machine's releases shipped
/// under if one is in the working directory; else the PROM built into iris.
pub fn resolve(profile: crate::config::MachineProfile, prom: &str, payload_limit: usize) -> Result<Loaded, String> {
    let implicit: Vec<&str> = std::iter::once(crate::config::DEFAULT_PROM).chain(profile.release_prom_file()).collect();
    resolve_with(profile.embedded_prom(), prom, &implicit, payload_limit)
}

fn resolve_with(
    built_in: crate::dev::embedded_prom::EmbeddedProm, prom: &str, implicit: &[&str], payload_limit: usize,
) -> Result<Loaded, String> {
    let from_file = |path: &str| {
        let loaded = load(std::path::Path::new(path), payload_limit).map(|bytes| Loaded { bytes, source: Some(path.into()) });
        if loaded.is_ok() {
            println!("Loaded PROM from {path}");
        }
        loaded
    };
    if prom != crate::config::DEFAULT_PROM && !is_embedded_keyword(prom) {
        return from_file(prom).map_err(|e| e.to_string());
    }
    if prom == crate::config::DEFAULT_PROM {
        for path in implicit {
            match from_file(path) {
                Ok(loaded) => return Ok(loaded),
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.to_string()),
            }
        }
    }
    let bytes = crate::dev::embedded_prom::get(built_in)?;
    println!("Using the built-in {built_in:?} PROM");
    Ok(Loaded { bytes, source: None })
}

/// The payload behind the installer's header: "PROM", then the payload's
/// length at +12 (big-endian), in 256 bytes. Recognized only when that length
/// matches the file, so a raw image that happens to start "PROM" stays raw.
fn installer_payload(file: &[u8]) -> Option<&[u8]> {
    let payload = file.get(0x100..)?;
    let said = u32::from_be_bytes(file[12..16].try_into().unwrap()) as usize;
    (file.starts_with(b"PROM") && said == payload.len()).then_some(payload)
}

/// Accept raw payload bytes, the installer's header form or the supported
/// stored/gzip container subset. Gzip output is bounded before decoding, and
/// exactly one member is required.
pub fn decode(file: &[u8], payload_limit: usize) -> io::Result<Vec<u8>> {
    if let Some(payload) = installer_payload(file) {
        if payload.len() > payload_limit {
            return Err(invalid("PROM image exceeds flash capacity"));
        }
        return Ok(payload.to_vec());
    }
    if file.len() < 0x48 || word(file, 0x40) != 0x4a464b535743534d {
        if file.len() > payload_limit {
            return Err(invalid("PROM image exceeds flash capacity"));
        }
        return Ok(file.to_vec());
    }
    if file.len() < 4096 {
        return Err(invalid("truncated SGI PROM container header"));
    }
    if word(file, 0x48) != 6 {
        return Err(invalid("unsupported SGI PROM container version"));
    }
    if word(file, 0x50) != file.len() as u64 {
        return Err(invalid("SGI PROM declared end does not match file size"));
    }
    if u32::from_be_bytes(file[0x7c..0x80].try_into().unwrap()) != 1 {
        return Err(invalid("unsupported SGI PROM segment count"));
    }
    if word(file, 0x98) != 4096 {
        return Err(invalid("unsupported SGI PROM payload offset"));
    }
    let flags = word(file, 0x90);
    if flags != 0x10 && flags != 0x13 {
        return Err(invalid("unsupported SGI PROM flags"));
    }
    let length = word(file, 0xb0);
    if length == 0 || length > payload_limit as u64 {
        return Err(invalid(
            "SGI PROM decoded length is zero or exceeds capacity",
        ));
    }
    let payload = if flags == 0x10 {
        if word(file, 0xb8) != 0 || word(file, 0xc8) != 0 || word(file, 0xd0) != length {
            return Err(invalid("unsupported SGI PROM uncompressed metadata"));
        }
        if length != (file.len() - 4096) as u64 {
            return Err(invalid("invalid SGI PROM stored length"));
        }
        file[4096..].to_vec()
    } else {
        let stored = word(file, 0xb8);
        if stored == 0 || stored > (file.len() - 4096) as u64 {
            return Err(invalid("invalid SGI PROM stored length"));
        }
        let end = 4096 + stored as usize;
        if file.len() - end > 7 || file[end..].iter().any(|b| *b != 0) {
            return Err(invalid("unsupported SGI PROM trailing bytes"));
        }
        if byte_sum(&file[4096..end]) != word(file, 0xc8) {
            return Err(invalid("SGI PROM stored checksum mismatch"));
        }
        let mut decoder = flate2::bufread::GzDecoder::new(&file[4096..end]);
        let mut decoded = Vec::new();
        (&mut decoder).take(length + 1).read_to_end(&mut decoded)?;
        if decoded.len() as u64 != length || !decoder.get_ref().is_empty() {
            return Err(invalid("invalid SGI PROM gzip segment or decoded length"));
        }
        decoded
    };
    if byte_sum(&payload) != word(file, 0xc0) {
        return Err(invalid("SGI PROM decoded checksum mismatch"));
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    // Cases follow tests/unit/test-sgi-prom-image.c in
    // jamesbraid/qemu-sgi-origami: wrapped, gzip, raw, malformed and files.
    use super::*;
    use std::io::Write;
    fn put(v: &mut [u8], off: usize, n: u64) {
        v[off..off + 8].copy_from_slice(&n.to_be_bytes());
    }
    fn container(payload: &[u8], gzip: bool) -> Vec<u8> {
        let stored = if gzip {
            let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            enc.write_all(payload).unwrap();
            enc.finish().unwrap()
        } else {
            payload.to_vec()
        };
        let mut v = vec![0; 4096];
        put(&mut v, 0x40, 0x4a464b535743534d);
        put(&mut v, 0x48, 6);
        put(&mut v, 0x50, (4096 + stored.len()) as u64);
        v[0x7c..0x80].copy_from_slice(&1u32.to_be_bytes());
        put(&mut v, 0x90, if gzip { 0x13 } else { 0x10 });
        put(&mut v, 0x98, 4096);
        put(&mut v, 0xb0, payload.len() as u64);
        put(&mut v, 0xc0, byte_sum(payload));
        if gzip {
            put(&mut v, 0xb8, stored.len() as u64);
            put(&mut v, 0xc8, byte_sum(&stored));
        } else {
            put(&mut v, 0xd0, payload.len() as u64);
        }
        v.extend(stored);
        v
    }
    /// Equivalent PROM payloads must boot from each supported image format.
    #[test]
    fn raw_stored_and_gzip_images_load_with_the_same_capacity_limit() {
        let path = crate::test_temp::TempPath::new("prom-load");
        for file in [vec![9, 8, 7, 6], container(&[9, 8, 7, 6], false), container(&[9, 8, 7, 6], true)] {
            std::fs::write(&path, file).unwrap();
            assert_eq!(load(&path, 4).unwrap(), [9, 8, 7, 6]);
            assert!(load(&path, 3).is_err(), "decoded payload exceeds capacity");
        }
        std::fs::write(&path, []).unwrap();
        assert!(load(&path, 4).unwrap().is_empty());
    }

    /// An invalid PROM must identify its input file instead of executing a
    /// malformed container.
    #[test]
    fn invalid_or_unavailable_images_report_their_path() {
        let path = crate::test_temp::TempPath::new("prom-invalid");
        let error = load(&path, 4).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(error.to_string().contains(&path.display().to_string()));
        let reject = |file: &[u8]| {
            std::fs::write(&path, file).unwrap();
            let error = load(&path, 4).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            assert!(error.to_string().contains(&path.display().to_string()));
        };
        reject(&vec![0; 5000]); // encoded file bound
        let original = container(&[4, 3, 2, 1], false);
        reject(&original[..100]); // recognized but truncated header
        for (off, n) in [(0x48, 7), (0x50, 4096), (0x90, 0x14), (0x98, 4095), (0xb0, 0), (0xd0, 3)] {
            let mut file = original.clone();
            put(&mut file, off, n);
            reject(&file);
        }
        for gzip in [false, true] {
            let mut file = container(&[4, 3, 2, 1], gzip);
            file[0xc7] ^= 1;
            reject(&file);
        }
        let gzip = container(&[4, 3, 2, 1], true);
        for tail in [7, 8] {
            let mut file = gzip.clone();
            file.extend(vec![0; tail]);
            let size = file.len();
            put(&mut file, 0x50, size as u64);
            if tail == 7 {
                std::fs::write(&path, &file).unwrap();
                assert_eq!(load(&path, 4).unwrap(), [4, 3, 2, 1]);
            } else {
                reject(&file);
            }
        }
    }

    /// The installer's form: "PROM", a word iris does not read (0x4400 in
    /// the O2's `ip32prom.image`), then the payload length, in 256 bytes.
    fn installer(payload: &[u8], said: u32) -> Vec<u8> {
        let mut v = vec![0; 0x100];
        v[..4].copy_from_slice(b"PROM");
        v[8..12].copy_from_slice(&0x4400u32.to_be_bytes());
        v[12..16].copy_from_slice(&said.to_be_bytes());
        v.extend(payload);
        v
    }

    /// The header is stripped only when its length matches the file; a raw
    /// image that starts "PROM" is still raw. The header is checked before
    /// the size, so an image whose header and payload fill the whole part
    /// still loads from the right place.
    #[test]
    fn installer_header_is_stripped_only_when_its_length_matches() {
        assert_eq!(decode(&installer(&[9, 8, 7, 6], 4), 4).unwrap(), [9, 8, 7, 6]);
        assert!(decode(&installer(&[9, 8, 7, 6], 4), 3).is_err(), "payload exceeds capacity");
        let full = installer(&[0x5a; 0x300], 0x300);
        assert_eq!(decode(&full, full.len()).unwrap(), [0x5a; 0x300]);
        let wrong = installer(&[9, 8, 7, 6], 5);
        assert_eq!(decode(&wrong, wrong.len()).unwrap(), wrong, "a mismatched length is raw");
    }

    /// Every file under `dir`, for the real-image test below.
    fn files_under(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for path in entries.filter_map(|e| e.ok().map(|e| e.path())) {
            if path.is_dir() {
                files_under(&path, out);
            } else {
                out.push(path);
            }
        }
        out.sort();
    }

    /// Real PROM files load to the flash contents each machine runs from:
    /// the Origin, Fuel and Tezro `.img` containers to the `.payload.bin`
    /// beside them, the O2's `ip32prom.image` to the bytes behind its
    /// header, and the Octane's raw `IP30prom.bin` to itself. The images
    /// are not in git; `IRIS_PROM_DIR` (default `ignore/proms`) is searched
    /// for them, whatever the layout, and the test skips without any.
    #[test]
    fn real_prom_files_load_to_their_payload() {
        let dir = std::env::var("IRIS_PROM_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| "ignore/proms".into());
        let mut files = Vec::new();
        files_under(&dir, &mut files);
        let mut checked = 0;
        for path in &files {
            let name = path.file_name().unwrap().to_string_lossy();
            let raw = || std::fs::read(path).unwrap();
            if let Some(stem) = name.strip_suffix(".img").filter(|s| s.ends_with("prom")) {
                // The Origin 200/2000 PROM is a 1 MiB part, the Fuel's and Tezro's 8 MiB.
                let limit = if stem.starts_with("ip27") { 1 << 20 } else { 8 << 20 };
                let Ok(payload) = std::fs::read(path.with_file_name(format!("{stem}.payload.bin"))) else { continue };
                assert_eq!(load(path, limit).unwrap(), payload, "{}", path.display());
            } else if name == "ip32prom.image" {
                let file = raw();
                let payload = load(path, 1 << 19).unwrap();
                assert_eq!(payload, &file[0x100..], "{}: the header is stripped", path.display());
                assert!(payload.len() < 1 << 19, "{}: shorter than the part", path.display());
            } else if name == "IP30prom.bin" {
                assert_eq!(load(path, 1 << 20).unwrap(), raw(), "{}", path.display());
            } else {
                continue;
            }
            checked += 1;
        }
        if checked == 0 {
            eprintln!("SKIPPED: no PROM images under {}", dir.display());
        } else {
            eprintln!("checked {checked} PROM images under {}", dir.display());
        }
    }

    /// A PROM file the user names must load, and wins over the built-in PROM; left at the
    /// default, the files in the working directory are tried and a missing one is no error.
    #[cfg(feature = "embedded-proms")]
    #[test]
    fn a_named_file_wins_and_must_load() {
        use crate::dev::embedded_prom::{self, EmbeddedProm::Ip32};
        let dir = crate::test_temp::TempPath::dir("prom-resolve");
        let (good, bad, missing) = (dir.join("good.bin"), dir.join("bad.bin"), dir.join("missing.bin"));
        std::fs::write(&good, [1, 2, 3, 4]).unwrap();
        std::fs::write(&bad, vec![0; 5000]).unwrap(); // over the limit
        let run = |prom: &std::path::Path, implicit: &[&std::path::Path]| {
            let implicit: Vec<&str> = implicit.iter().map(|p| p.to_str().unwrap()).collect();
            resolve_with(Ip32, prom.to_str().unwrap(), &implicit, 4096)
        };
        let default = std::path::Path::new(crate::config::DEFAULT_PROM);
        let built_in = embedded_prom::get(Ip32).unwrap();
        // A named file wins, and a named file that will not load is an error.
        assert_eq!(run(&good, &[&missing]).unwrap().bytes, [1, 2, 3, 4]);
        assert!(run(&missing, &[&good]).err().unwrap().contains("missing.bin"));
        assert!(run(&bad, &[&good]).is_err());
        // At the default, an implicit file is used when there, skipped when missing, and an
        // error when it is there but will not load.
        assert_eq!(run(default, &[&missing, &good]).unwrap().bytes, [1, 2, 3, 4]);
        let none = run(default, &[&missing]).unwrap();
        assert!(none.source.is_none() && none.bytes == built_in);
        assert!(run(default, &[&bad, &good]).is_err());
        // The keyword is the built-in PROM whatever files exist.
        let keyword = resolve_with(Ip32, "(embedded)", &[good.to_str().unwrap()], 4096).unwrap();
        assert!(keyword.source.is_none() && keyword.bytes == built_in);
    }

    /// Every profile starts from its built-in PROM when no file is named.
    #[cfg(feature = "embedded-proms")]
    #[test]
    fn every_machine_starts_without_a_file() {
        for p in crate::config::MachineProfile::EVERY {
            let loaded = resolve(p, "(embedded)", 8 << 20).unwrap();
            assert_eq!(loaded.bytes, crate::dev::embedded_prom::get(p.embedded_prom()).unwrap(), "{p:?}");
        }
    }

    /// Without the `embedded-proms` feature the built-in PROM is an error naming the feature.
    #[cfg(not(feature = "embedded-proms"))]
    #[test]
    fn without_the_feature_a_machine_needs_a_prom_file() {
        let e = resolve(crate::config::MachineProfile::OctaneIp30, "(embedded)", 8 << 20).err().unwrap();
        assert!(e.contains("embedded-proms") && e.contains("Ip30"), "{e}");
    }
}
