//! The SGI PROMs iris carries (the `embedded-proms` feature; without it [`get`] is an
//! error). For each one `iris-prom` stores the PROM a machine runs (the published file
//! unwrapped by `prom_image::decode`) as a deflate file of its own,
//! `data/proms/<board>.deflate`; `data/proms/proms.toml` lists where each came from. The
//! IO6 is the one file stored as published: it is the BaseIO flash's contents, and the
//! IP27 PROM unpacks the container from there itself.

/// A PROM, named by the board or card that runs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddedProm {
    /// Indy.
    Ip24,
    /// Indigo2.
    Ip22,
    /// Indigo2 IMPACT.
    Ip28,
    /// O2.
    Ip32,
    /// Octane.
    Ip30,
    /// BaseIO card (Origin 2000 and Onyx2 deskside).
    Io6,
    /// Origin 200, Origin 2000 and Onyx2 node board.
    Ip27,
    /// Fuel, Tezro, Origin 3200 and Onyx 3000.
    Ip35,
}

/// The PROM a machine runs, inflated from its embedded file.
#[cfg(feature = "embedded-proms")]
pub fn get(id: EmbeddedProm) -> Result<Vec<u8>, String> {
    use std::io::Read;

    let packed: &[u8] = match id {
        EmbeddedProm::Ip24 => include_bytes!("../../data/proms/ip24.deflate"),
        EmbeddedProm::Ip22 => include_bytes!("../../data/proms/ip22.deflate"),
        EmbeddedProm::Ip28 => include_bytes!("../../data/proms/ip28.deflate"),
        EmbeddedProm::Ip32 => include_bytes!("../../data/proms/ip32.deflate"),
        EmbeddedProm::Ip30 => include_bytes!("../../data/proms/ip30.deflate"),
        EmbeddedProm::Io6 => include_bytes!("../../data/proms/io6.deflate"),
        EmbeddedProm::Ip27 => include_bytes!("../../data/proms/ip27.deflate"),
        EmbeddedProm::Ip35 => include_bytes!("../../data/proms/ip35.deflate"),
    };
    let mut bytes = Vec::new();
    flate2::read::DeflateDecoder::new(packed)
        .read_to_end(&mut bytes)
        .unwrap_or_else(|e| panic!("embedded {id:?} PROM: {e}"));
    Ok(bytes)
}

/// The PROM a machine runs: not in this build.
#[cfg(not(feature = "embedded-proms"))]
pub fn get(id: EmbeddedProm) -> Result<Vec<u8>, String> {
    Err(format!("this build has no built-in {id:?} PROM (feature embedded-proms), so name one with `prom`"))
}

#[cfg(all(test, feature = "embedded-proms"))]
mod tests {
    use super::*;

    /// Each PROM inflates to what its machine runs: the size, and the blake3 of the payload
    /// (the published file unwrapped, except the IO6).
    #[test]
    fn each_prom_inflates_to_what_its_machine_runs() {
        let proms = [
            (EmbeddedProm::Ip24, 524_288, "08712626f4b43cc3b42c439f73f48c2ecc45888643f0f1879b2f0809989bb7c5"),
            (EmbeddedProm::Ip22, 524_288, "cb37a7122629b367e418eda87660a627a2e7d79a7cf28401bd1162b90328d57a"),
            (EmbeddedProm::Ip28, 524_288, "1f58759cfc5e81d2244537e84384537bee9cd5e8b423665372dbe544c6a44c22"),
            (EmbeddedProm::Ip32, 431_616, "8dd07f3c8b11d1e5e2f06efaae45611dc3339f62f39c9d4471335ee799c77161"),
            (EmbeddedProm::Ip30, 879_312, "c5e0f898812856438c5373b7c05505e852443f98dee983cebf3d7352f84a7d0b"),
            (EmbeddedProm::Io6, 365_328, "25e9a3cca96a8600a5721af6af68e9a22c83c5b59d0ee36dba4cea7fb6361d05"),
            (EmbeddedProm::Ip27, 908_752, "78e98e0b166aede4731a2cf02f89586b45909215f43cdc62469a55afabe662ea"),
            (EmbeddedProm::Ip35, 1_476_264, "3f07f9eac22def9c4b8ad946d92d3d1442acc9299f555875e36e26a807a07c88"),
        ];
        for (id, len, hash) in proms {
            let bytes = get(id).unwrap();
            assert_eq!(bytes.len(), len, "{id:?}");
            assert_eq!(blake3::hash(&bytes).to_hex().as_str(), hash, "{id:?}");
        }
    }

    /// Every catalog row's version is the one a banner inside its PROM states, so the
    /// versions rest on the PROMs' own bytes, not on a boot or a file name. Where a PROM
    /// holds several banners (the Octane's first stage and its main monitor, the IP27's
    /// own and the BaseIO's) the row's version need only be one of them.
    #[test]
    fn each_catalog_version_is_in_a_banner_of_its_prom() {
        #[derive(serde::Deserialize)]
        struct Catalog {
            prom: Vec<Row>,
        }
        #[derive(serde::Deserialize)]
        struct Row {
            board: String,
            version: String,
        }
        let catalog: Catalog = toml::from_str(include_str!("../../data/proms/proms.toml")).unwrap();
        assert_eq!(catalog.prom.len(), 8);
        for row in catalog.prom {
            let id = match row.board.as_str() {
                "ip24" => EmbeddedProm::Ip24,
                "ip22" => EmbeddedProm::Ip22,
                "ip28" => EmbeddedProm::Ip28,
                "ip32" => EmbeddedProm::Ip32,
                "ip30" => EmbeddedProm::Ip30,
                "io6" => EmbeddedProm::Io6,
                "ip27" => EmbeddedProm::Ip27,
                "ip35" => EmbeddedProm::Ip35,
                other => panic!("{other}: not an embedded board"),
            };
            let banners = crate::dev::prom_image::banners(&get(id).unwrap());
            // The version as a whole token: "4.1" must not match inside "4.17".
            let has = |text: &str| {
                text.match_indices(&row.version).any(|(at, v)| {
                    let before = text[..at].chars().next_back();
                    let after = text[at + v.len()..].chars().next();
                    !before.is_some_and(|c| c.is_ascii_alphanumeric() || c == '.')
                        && !after.is_some_and(|c| c.is_ascii_alphanumeric() || c == '.')
                })
            };
            assert!(banners.iter().any(|b| has(&b.text)), "{}: version {} is in none of {banners:?}", row.board, row.version);
        }
    }
}
