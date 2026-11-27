//! The O2 PROM's environment: an `env` segment of the flash at 0x4000, a `SHDR` header,
//! `name=value\0` entries ended by an empty string, and a last word that zeroes the body.

use std::ops::Range;

pub const ENV_OFFSET: usize = 0x4000;
const HEADER: usize = 0x40;
const MAGIC: &[u8; 4] = b"SHDR";
/// The length the PROM gives a segment it creates.
const CREATED_LEN: usize = 0x3f0;
const DEFAULTS: [&str; 10] = [
    "AutoLoad=Yes", "console=g", "diskless=0", "dbaud=9600", "volume=80", "sgilogo=y",
    "monitor=h", "TimeZone=PST8PDT", "netaddr=192.0.2.1", "crt_option=1",
];

fn words_sum(bytes: &[u8]) -> u32 {
    bytes.chunks_exact(4).fold(0u32, |s, w| s.wrapping_add(u32::from_be_bytes(w.try_into().unwrap())))
}

/// The segment at `at` if its header checks out: magic, a length that is a
/// multiple of 4 and holds the header, and header words summing to zero.
fn header_at(flash: &[u8], at: usize) -> Option<Range<usize>> {
    let head = flash.get(at..at + HEADER)?;
    let len = u32::from_be_bytes(head[0x0c..0x10].try_into().unwrap()) as usize;
    let ok = &head[8..12] == MAGIC && len % 4 == 0 && len > HEADER && words_sum(head) == 0;
    (ok && at + len <= flash.len()).then_some(at..at + len)
}

fn is_env(flash: &[u8], seg: &Range<usize>) -> bool {
    flash[seg.start + 0x10] == 3
        && &flash[seg.start + 0x14..seg.start + 0x18] == b"env\0"
        && words_sum(&flash[seg.start + HEADER..seg.end]) == 0
}

/// The PROM's scan: from the start in 256-byte steps, skipping past each valid
/// segment, the first `env` segment whose body checksum also holds.
pub fn find(flash: &[u8]) -> Option<Range<usize>> {
    let mut at = 0;
    while at < flash.len() {
        at = match header_at(flash, at) {
            Some(seg) if is_env(flash, &seg) => return Some(seg),
            Some(seg) => seg.end.next_multiple_of(0x100),
            None => at + 0x100,
        };
    }
    None
}

/// The entries of a segment's body, up to the empty string.
fn entries(body: &[u8]) -> impl Iterator<Item = &[u8]> {
    body.split(|&b| b == 0).take_while(|e| !e.is_empty())
}

pub fn console_is_serial(flash: &[u8]) -> bool {
    let Some(seg) = find(flash) else { return false };
    let body = &flash[seg.start + HEADER..seg.end - 4];
    entries(body).find_map(|e| e.strip_prefix(b"console=")).is_some_and(|v| v.first() == Some(&b'd'))
}

/// Write `entries` as the body of the segment `seg`, zero the tail and fix the last word.
fn write_body(flash: &mut [u8], seg: &Range<usize>, list: &[Vec<u8>]) -> Result<(), String> {
    let used: usize = list.iter().map(|e| e.len() + 1).sum::<usize>() + 1;
    let body = seg.start + HEADER..seg.end - 4;
    if used > body.len() {
        return Err(format!("the O2 env segment holds {} bytes; the entries need {used}", body.len()));
    }
    flash[body.clone()].fill(0);
    let mut at = body.start;
    for e in list {
        flash[at..at + e.len()].copy_from_slice(e);
        at += e.len() + 1;
    }
    let sum = words_sum(&flash[seg.start + HEADER..seg.end - 4]);
    flash[seg.end - 4..seg.end].copy_from_slice(&sum.wrapping_neg().to_be_bytes());
    Ok(())
}

/// A header for a segment of `len` bytes at the start of `head`, with its checksum.
fn write_header(head: &mut [u8], len: usize) {
    head[..HEADER].fill(0);
    head[8..12].copy_from_slice(MAGIC);
    head[0x0c..0x10].copy_from_slice(&(len as u32).to_be_bytes());
    head[0x10..0x12].copy_from_slice(&[3, 4]);
    head[0x14..0x17].copy_from_slice(b"env");
    head[0x34..0x38].copy_from_slice(b"v1.0");
    let sum = words_sum(&head[..HEADER]);
    head[0x3c..0x40].copy_from_slice(&sum.wrapping_neg().to_be_bytes());
}

/// The segment the PROM creates when it finds none: header, the default
/// entries, checksums.
fn create_default(flash: &mut [u8]) -> Result<Range<usize>, String> {
    let seg = ENV_OFFSET..ENV_OFFSET + CREATED_LEN;
    write_header(flash.get_mut(seg.clone()).ok_or("the flash is too small for the O2 env segment")?, CREATED_LEN);
    let list: Vec<Vec<u8>> = DEFAULTS.iter().map(|e| e.as_bytes().to_vec()).collect();
    write_body(flash, &seg, &list)?;
    Ok(seg)
}

/// Store `console=d` as `setenv console d` does: move it to the end, keep the
/// segment's length. With no segment, or an empty one, start from the PROM's defaults.
pub fn set_console_serial(flash: &mut [u8]) -> Result<(), String> {
    let seg = match find(flash).filter(|s| flash[s.start + HEADER] != 0) {
        Some(seg) => seg,
        None => create_default(flash)?,
    };
    let body = &flash[seg.start + HEADER..seg.end - 4];
    let mut list: Vec<Vec<u8>> = entries(body).filter(|e| !e.starts_with(b"console=")).map(<[u8]>::to_vec).collect();
    list.push(b"console=d".to_vec());
    write_body(flash, &seg, &list)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank() -> Vec<u8> {
        let mut flash = vec![0xffu8; 0x8000];
        create_default(&mut flash).unwrap();
        flash
    }

    #[test]
    fn default_segment_checksums() {
        let flash = blank();
        let seg = find(&flash).unwrap();
        assert_eq!(seg, ENV_OFFSET..ENV_OFFSET + 0x3f0);
        assert_eq!(&flash[ENV_OFFSET + 0x3c..ENV_OFFSET + 0x40], 0xce14138eu32.to_be_bytes());
        assert_eq!(&flash[seg.end - 4..seg.end], 0xc11fb2bau32.to_be_bytes());
        assert!(!console_is_serial(&flash));
    }

    #[test]
    fn setenv_console_d_moves_the_entry_to_the_end() {
        let before = blank();
        let mut flash = before.clone();
        set_console_serial(&mut flash).unwrap();
        let seg = find(&flash).unwrap();
        let want = DEFAULTS.iter().filter(|e| !e.starts_with("console=")).chain(&["console=d"]).fold(Vec::new(), |mut v, e| {
            v.extend_from_slice(e.as_bytes());
            v.push(0);
            v
        });
        assert_eq!(&flash[seg.start + HEADER..seg.start + HEADER + want.len()], &want[..]);
        assert_eq!(&flash[seg.end - 4..seg.end], 0x4d562687u32.to_be_bytes());
        assert_eq!(flash[..seg.start + HEADER], before[..seg.start + HEADER], "the header is unchanged");
        assert!(console_is_serial(&flash));
    }

    #[test]
    fn a_segment_with_a_stale_tail_still_validates_after_the_write() {
        let mut flash = blank();
        let seg = find(&flash).unwrap();
        flash[seg.end - 40..seg.end - 4].fill(0x5a);
        let sum = words_sum(&flash[seg.start + HEADER..seg.end - 4]);
        flash[seg.end - 4..seg.end].copy_from_slice(&sum.wrapping_neg().to_be_bytes());
        assert!(find(&flash).is_some());
        set_console_serial(&mut flash).unwrap();
        assert!(console_is_serial(&flash));
        assert_eq!(flash[seg.end - 40], 0, "the tail is zeroed");
    }

    #[test]
    fn no_segment_or_an_empty_one_starts_from_the_defaults() {
        let mut flash = vec![0xffu8; 0x8000];
        set_console_serial(&mut flash).unwrap();
        assert!(console_is_serial(&flash));
        // A factory-style empty segment, longer than the PROM creates.
        let mut flash = vec![0u8; 0x8000];
        write_header(&mut flash[ENV_OFFSET..], 0x400);
        assert_eq!(find(&flash), Some(ENV_OFFSET..ENV_OFFSET + 0x400));
        set_console_serial(&mut flash).unwrap();
        assert_eq!(find(&flash), Some(ENV_OFFSET..ENV_OFFSET + CREATED_LEN));
        assert!(console_is_serial(&flash));
    }
}
