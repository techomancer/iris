//! SGI PROM image loading: version-six single-segment containers and raw
//! payloads. A recognized but malformed container is rejected rather than
//! retried as a raw image.
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

/// Accept raw payload bytes or the supported stored/gzip container subset.
/// Gzip output is bounded before decoding, and exactly one member is required.
pub fn decode(file: &[u8], payload_limit: usize) -> io::Result<Vec<u8>> {
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
        let path = std::env::temp_dir().join(format!("iris-prom-load-{}", std::process::id()));
        for file in [vec![9, 8, 7, 6], container(&[9, 8, 7, 6], false), container(&[9, 8, 7, 6], true)] {
            std::fs::write(&path, file).unwrap();
            assert_eq!(load(&path, 4).unwrap(), [9, 8, 7, 6]);
            assert!(load(&path, 3).is_err(), "decoded payload exceeds capacity");
        }
        std::fs::write(&path, []).unwrap();
        assert!(load(&path, 4).unwrap().is_empty());
        std::fs::remove_file(path).unwrap();
    }

    /// An invalid PROM must identify its input file instead of executing a
    /// malformed container.
    #[test]
    fn invalid_or_unavailable_images_report_their_path() {
        let path = std::env::temp_dir().join(format!("iris-prom-invalid-{}", std::process::id()));
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
        std::fs::remove_file(path).unwrap();
    }

}
