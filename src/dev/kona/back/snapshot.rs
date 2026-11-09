//! The back end's bulk state in the chunk store, for a snapshot.
//!
//! Every allocated word array of the pixel store (the colour, depth, stencil and
//! the other words of a pixel's record, and the display-ID, overlay and popup
//! planes beside them) and the cursor's glyph memory are written as 64 KiB chunks
//! of big-endian words. A chunk of all zeros is not written: its place in the
//! list holds `ZERO`, so a mostly empty array costs a list of markers. An
//! allocated array that is all zero still gets its entry, so a restored store
//! allocates what the saved one had (the store's hash tells an unallocated array
//! from a zero one).
//!
//! Not saved here: the tile ring and the draw-state table (the board is
//! quiesced, so both are empty), the composed frame and the baked display tables
//! (rebuilt from the tables as loaded). The small state (the pipe's registers,
//! the display tables, the contexts) is the machine's `Saveable` part, which
//! this module does not cover. A snapshot loads only into the build that saved it.

use super::display::tables::Tables;
use super::store::texture::{Page, PAGE_WORDS, PLANES};
use super::store::{PixelStore, POSITIONS};
use crate::chunk_store::{ChunkHash, ChunkStore, CHUNK_SIZE};
use serde::{Deserialize, Serialize};
use std::io;

/// The hash a zero chunk's place holds; the store never holds a chunk by it.
pub const ZERO: ChunkHash = [0; 32];

/// One saved memory: its name and its chunks, in order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KonaChunk {
    pub name: String,
    pub hashes: Vec<ChunkHash>,
}

const WORDS: usize = CHUNK_SIZE / 4;

/// Bytes as chunks, with a zero chunk's place holding `ZERO`.
pub fn put_bytes(cs: &ChunkStore, bytes: &[u8]) -> io::Result<Vec<ChunkHash>> {
    bytes.chunks(CHUNK_SIZE).map(|c| if c.iter().all(|&b| b == 0) { Ok(ZERO) } else { cs.put(c) }).collect()
}

/// What `put_bytes` wrote, `len` bytes of it.
pub fn get_bytes(cs: &ChunkStore, hashes: &[ChunkHash], len: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(len);
    for (i, h) in hashes.iter().enumerate() {
        let want = (len.saturating_sub(i * CHUNK_SIZE)).min(CHUNK_SIZE);
        if *h == ZERO {
            out.resize(out.len() + want, 0);
        } else {
            let b = cs.get(h).map_err(|e| format!("chunk {i}: {e}"))?;
            if b.len() != want {
                return Err(format!("chunk {i} holds {} bytes, not {want}", b.len()));
            }
            out.extend_from_slice(&b);
        }
    }
    if out.len() != len {
        return Err(format!("{} bytes saved, {len} expected", out.len()));
    }
    Ok(out)
}

fn put(cs: &ChunkStore, words: &[u32]) -> io::Result<Vec<ChunkHash>> {
    let mut bytes = Vec::with_capacity(CHUNK_SIZE);
    words
        .chunks(WORDS)
        .map(|c| {
            if c.iter().all(|&w| w == 0) {
                return Ok(ZERO);
            }
            bytes.clear();
            c.iter().for_each(|w| bytes.extend_from_slice(&w.to_be_bytes()));
            cs.put(&bytes)
        })
        .collect()
}

fn get(cs: &ChunkStore, hashes: &[ChunkHash], len: usize) -> Result<Vec<u32>, String> {
    // A manifest is data: more or fewer chunks than the array needs is refused before any is read.
    if hashes.len() != len.div_ceil(WORDS) {
        return Err(format!("{} chunks for {len} words", hashes.len()));
    }
    let mut words = Vec::with_capacity(len);
    for (i, h) in hashes.iter().enumerate() {
        let want = (len - i * WORDS).min(WORDS);
        if *h == ZERO {
            words.resize(words.len() + want, 0);
            continue;
        }
        let bytes = cs.get(h).map_err(|e| format!("chunk {i}: {e}"))?;
        if bytes.len() != want * 4 {
            return Err(format!("chunk {i} holds {} bytes, not {}", bytes.len(), want * 4));
        }
        words.extend(bytes.chunks_exact(4).map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]])));
    }
    if words.len() != len {
        return Err(format!("{} words saved, {len} expected", words.len()));
    }
    Ok(words)
}

/// Save the store's allocated arrays and the cursor memory.
pub fn save(store: &PixelStore, tables: &Tables, cs: &ChunkStore) -> io::Result<Vec<KonaChunk>> {
    let mut out = Vec::new();
    for w in store.allocated() {
        let a = store.array(w).expect("allocated");
        // SAFETY: `i` is below the array's length.
        let words: Vec<u32> = (0..a.len()).map(|i| unsafe { a.load(i) }).collect();
        out.push(KonaChunk { name: format!("pixel-{w}"), hashes: put(cs, &words)? });
    }
    out.push(KonaChunk { name: "cursor-ram".into(), hashes: put(cs, tables.cursor_words())? });
    // Texture memory: each plane that holds pages, by its number and the length of its page table (both in the
    // name), one chunk to a page and a zero marker where the table has no page. A page is one chunk long.
    let tex = store.texture();
    for (plane, pages) in tex.used() {
        let hashes = (0..pages)
            .map(|pg| match tex.page(plane, pg) {
                Some(p) => Ok(put(cs, p)?.remove(0)),
                None => Ok(ZERO),
            })
            .collect::<io::Result<Vec<_>>>()?;
        out.push(KonaChunk { name: format!("texture-g{plane}-{pages}"), hashes });
    }
    Ok(out)
}

/// A snapshot's bulk state read into memory and checked, ready to apply.
pub struct Prepared {
    data: Vec<(usize, Vec<u32>)>,
    cursor: Vec<u32>,
    /// Texture memory's planes, by number.
    planes: Vec<(u16, Vec<Option<Box<Page>>>)>,
}

/// Read and check every chunk `save` wrote for a store and tables of the same configuration, and
/// change nothing.
pub fn prepare(store: &PixelStore, tables: &Tables, cs: &ChunkStore, list: &[KonaChunk]) -> Result<Prepared, String> {
    let mut arrays = Vec::new();
    let mut cursor = None;
    let mut texture = Vec::new();
    for c in list {
        if let Some((p, n)) = c.name.strip_prefix("texture-g").and_then(|r| r.split_once('-')).and_then(|(p, n)| Some((p.parse::<u16>().ok()?, n.parse::<usize>().ok()?))) {
            if p as usize >= PLANES || texture.iter().any(|&(q, _, _)| q == p) {
                return Err(format!("chunk {} names no texture plane or repeats one", c.name));
            }
            if n > store.texture().plane_pages() || c.hashes.len() != n {
                return Err(format!("{}: the saved plane is past a bank of one raster manager's memory", c.name));
            }
            texture.push((p, n, c));
            continue;
        }
        match c.name.strip_prefix("pixel-").map(str::parse::<usize>) {
            Some(Ok(w)) if w < POSITIONS => arrays.push((w, c)),
            Some(_) => return Err(format!("chunk {} names no word position", c.name)),
            None if c.name == "cursor-ram" => cursor = Some(c),
            None => return Err(format!("unknown chunk {}", c.name)),
        }
    }
    let entries = store.width() as usize * store.height() as usize;
    let mut data = Vec::new();
    for (w, c) in arrays {
        data.push((w, get(cs, &c.hashes, entries).map_err(|e| format!("{}: {e}", c.name))?));
    }
    let cursor = match cursor {
        Some(c) => get(cs, &c.hashes, tables.cursor_words().len()).map_err(|e| format!("cursor-ram: {e}"))?,
        None => vec![0; tables.cursor_words().len()],
    };
    let mut planes = Vec::new();
    let mut held = 0;
    for (p, n, c) in texture {
        let mut pages = Vec::with_capacity(n);
        for (i, h) in c.hashes.iter().enumerate() {
            if *h == ZERO {
                pages.push(None);
                continue;
            }
            let words = get(cs, std::slice::from_ref(h), PAGE_WORDS).map_err(|e| format!("{} page {i}: {e}", c.name))?;
            pages.push(Some(Box::<Page>::try_from(words.into_boxed_slice()).map_err(|_| "a page is not PAGE_WORDS long".to_string())?));
            held += 1;
        }
        planes.push((p, pages));
    }
    if held > store.texture().cap_pages() {
        return Err(format!("{held} texture pages saved, past the {} the product's memory holds", store.texture().cap_pages()));
    }
    Ok(Prepared { data, cursor, planes })
}

/// Put what `prepare` read into the store and tables: every array the snapshot names is allocated
/// and filled, and the others are zeroed, so the store reads as it did when saved.
pub fn apply(store: &PixelStore, tables: &mut Tables, p: Prepared) -> Result<(), String> {
    store.reset();
    for (plane, pages) in p.planes {
        if !store.texture().set_pages(plane, pages) {
            return Err(format!("texture plane {plane}: refused"));
        }
    }
    for (w, words) in p.data {
        store.ensure(w);
        let a = store.array(w).ok_or("no array after ensure")?;
        // SAFETY: the array holds `entries` words.
        words.iter().enumerate().for_each(|(i, &v)| unsafe { a.store(i, v) });
    }
    tables.load_cursor_words(&p.cursor);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::kona::back::store::{DID_PLANE, POPUP_PLANE};
    use crate::dev::kona::back::KonaBackend;
    use crate::dev::kona::KonaConfig;
    use crate::dev::kona::tests::ScratchDir;

    /// A chunk store's directory that goes with the test, however it ends.
    fn dir(tag: &str) -> ScratchDir {
        ScratchDir::new(&format!("snap-{tag}"))
    }

    fn board() -> KonaBackend {
        KonaBackend::new(KonaConfig { managed: (200, 100), ..KonaConfig::default() })
    }

    /// A board with a few arrays drawn on: colour words, the ID plane, and a zero array.
    fn drawn() -> KonaBackend {
        let b = board();
        for (w, v) in [(1usize, 0xff12_3456u32), (2, 0xff65_4321)] {
            b.store.write(7, 9, w, v);
            b.store.write(199, 99, w, v ^ 1);
        }
        b.store.write(3, 3, DID_PLANE, 13 << 2);
        b.store.ensure(POPUP_PLANE);
        b
    }

    /// Save, load into a fresh board, save again: the store reads the same, the
    /// arrays allocated are the same (an allocated zero array too), and the second save
    /// writes the same chunk lists.
    #[test]
    fn a_saved_store_restores_into_a_fresh_board_and_saves_the_same() {
        let d = dir("round");
        let cs = ChunkStore::new(d.path());
        let mut a = drawn();
        let first = a.save_chunks(&cs).unwrap();
        let mut b = board();
        b.load_chunks(&cs, &first).unwrap();
        assert_eq!(b.store.checksum(), a.store.checksum());
        assert_eq!(b.store.allocated(), a.store.allocated());
        assert_eq!(b.store.read(7, 9, 1), 0xff12_3456);
        assert_eq!(b.save_chunks(&cs).unwrap(), first, "save, load, save");
    }

    /// A chunk of zeros is not written, and its place holds the zero hash.
    #[test]
    fn zero_chunks_are_not_stored() {
        let d = dir("zero");
        let cs = ChunkStore::new(d.path());
        let mut a = drawn();
        let list = a.save_chunks(&cs).unwrap();
        let popup = list.iter().find(|c| c.name == format!("pixel-{POPUP_PLANE}")).unwrap();
        assert!(popup.hashes.iter().all(|h| *h == ZERO), "an all-zero array is all zero markers");
        assert!(!cs.has(&ZERO));
        let colour = list.iter().find(|c| c.name == "pixel-1").unwrap();
        assert!(colour.hashes.iter().any(|h| *h != ZERO) && colour.hashes.iter().all(|h| *h == ZERO || cs.has(h)));
    }

    /// A missing chunk is refused and the board is left as it was; so is a name no
    /// store position has.
    #[test]
    fn a_bad_snapshot_leaves_the_board_alone() {
        let d = dir("bad");
        let cs = ChunkStore::new(d.path());
        let mut a = drawn();
        let list = a.save_chunks(&cs).unwrap();
        let mut b = board();
        b.store.write(1, 1, 1, 77);
        let before = b.store.checksum();
        let gone = list.iter().find(|c| c.name == "pixel-1").unwrap().hashes.iter().find(|h| **h != ZERO).copied().unwrap();
        std::fs::remove_file(cs.path_for(&gone)).unwrap();
        assert!(b.load_chunks(&cs, &list).is_err());
        assert_eq!(b.store.checksum(), before);
        let odd = [KonaChunk { name: "pixel-9999".into(), hashes: vec![] }];
        assert!(b.load_chunks(&cs, &odd).unwrap_err().contains("no word position"));
        assert!(b.load_chunks(&cs, &[KonaChunk { name: "nonsense".into(), hashes: vec![] }]).is_err());
    }

    /// The cursor memory comes back with its registers: the glyph the host sees is the same.
    #[test]
    fn the_cursor_memory_is_restored() {
        let d = dir("cursor");
        let cs = ChunkStore::new(d.path());
        let mut a = board();
        {
            let mut t = a.tables.lock();
            let mut glyph = vec![0u32; 260];
            glyph[0] = 0x8000_0000;
            t.load_cursor(260, &glyph);
            t.indirect(0x101, 260);
            t.indirect(0x102, 34 | 33 << 13);
            t.indirect(0x103, 2);
        }
        let list = a.save_chunks(&cs).unwrap();
        let mut b = board();
        {
            // The registers are the small state, saved elsewhere; set as a restore would.
            let mut t = b.tables.lock();
            t.indirect(0x101, 260);
            t.indirect(0x102, 34 | 33 << 13);
            t.indirect(0x103, 2);
        }
        b.load_chunks(&cs, &list).unwrap();
        assert_eq!(b.tables.lock().snapshot().hw_cursor(), a.tables.lock().snapshot().hw_cursor());
        assert!(b.tables.lock().snapshot().hw_cursor().is_some());
    }

    #[test]
    fn texture_memory_is_restored_by_page() {
        let d = dir("texture");
        let cs = ChunkStore::new(d.path());
        let mut a = board();
        let t = a.store.texture();
        assert!(t.set(2, 0, 0xff01_0203) && t.set(2, 69_999, 0xff0a_0b0c) && t.set(5, 17, 7));
        let list = a.save_chunks(&cs).unwrap();
        let named = |n: &str| list.iter().find(|c| c.name == n).map(|c| c.hashes.iter().filter(|h| **h != ZERO).count());
        assert_eq!((named("texture-g2-5"), named("texture-g5-1")), (Some(2), Some(1)), "a chunk to a held page, a marker to the rest");
        let mut b = board();
        b.load_chunks(&cs, &list).unwrap();
        let t = b.store.texture();
        assert_eq!((t.used(), t.get(2, 0), t.get(2, 69_999), t.get(5, 17), t.resident_pages()), (vec![(2, 5), (5, 1)], 0xff01_0203, 0xff0a_0b0c, 7, 3));
        assert_eq!(b.save_chunks(&cs).unwrap(), list);
    }

    /// A plane that is not one, is repeated or is longer than a bank, a table of the wrong length, pages past the
    /// ceiling, and the older layouts' chunks, are refused.
    #[test]
    fn texture_planes_from_a_snapshot_are_checked() {
        let d = dir("texture-planes");
        let cs = ChunkStore::new(d.path());
        let mut a = board();
        assert!(a.store.texture().set(1, 0, 1));
        let list = a.save_chunks(&cs).unwrap();
        let h = put(&cs, &[0; PAGE_WORDS]).unwrap();
        let with = |name: &str, n: usize| -> Vec<KonaChunk> { list.iter().cloned().chain([KonaChunk { name: name.into(), hashes: h.iter().cycle().take(n).copied().collect() }]).collect() };
        let past = format!("texture-g3-{}", a.store.texture().plane_pages() + 1);
        for (name, n) in [("texture-g1-1", 1), (&format!("texture-g{PLANES}-1"), 1), ("texture-g3-2", 1), (past.as_str(), a.store.texture().plane_pages() + 1), ("texture-g9-999999999999", 1), ("texture-p3-8", 1), ("texture-8", 1)] {
            assert!(board().load_chunks(&cs, &with(name, n)).is_err(), "{name}");
        }
        assert!(board().load_chunks(&cs, &with("texture-g3-1", 1)).is_ok());
    }

    /// The manifest names every chunk the board saved, so the garbage collector keeps
    /// them, and not the zero marker.
    #[test]
    fn the_manifest_keeps_the_boards_chunks_alive() {
        use crate::snapshot::ChunksManifest;
        let d = dir("gc");
        let cs = ChunkStore::new(d.path());
        let mut a = drawn();
        let list = a.save_chunks(&cs).unwrap();
        let orphan = cs.put(b"nothing refers to this").unwrap();
        let m = ChunksManifest { kona_chunks: list.clone(), ..Default::default() };
        let live: std::collections::HashSet<_> = m.referenced_hashes().copied().collect();
        assert!(!live.contains(&ZERO) && live.len() >= 2);
        cs.gc(&live).unwrap();
        assert!(list.iter().flat_map(|c| &c.hashes).all(|h| *h == ZERO || cs.has(h)), "the board's chunks survive");
        assert!(!cs.has(&orphan), "an unreferenced one does not");
        // And the manifest round-trips through postcard as the registry reads it.
        let bytes = postcard::to_allocvec(&m).unwrap();
        assert_eq!(postcard::from_bytes::<ChunksManifest>(&bytes).unwrap().kona_chunks, list);
    }
}
