// System Snapshot — save and restore full machine state to/from a directory.
//
// Layout of saves/<name>/ (schema_version = 2):
//   snapshot.toml  — manifest (always TOML, human-readable)
//   cpu.bin        — CPU core (GPRs, CP0, FPU), TLB entries  (postcard BinValue)
//   mc.bin         — Memory Controller registers + GIO DMA state
//   ioc.bin        — IOC interrupt registers
//   hpc3.bin       — HPC3 state register, PBUS PIO, DMA channel registers
//   rex3.bin       — REX3 drawing registers, VC2, XMAP9, CMAP palette
//   {scc,pit,ps2,rtc,eeprom,scsi,seeq}.bin — peripheral device state
//   cow.toml       — COW overlay dirty sectors per SCSI device (stays TOML)
//   bank0.bin      — 128 MB RAM bank A (raw u8, big-endian word layout)
//   bank1.bin      — 128 MB RAM bank B
//   bank2.bin      — 128 MB RAM bank C
//   bank3.bin      — 128 MB RAM bank D
//
// schema_version = 1: same layout but device state is *.toml (hex strings).
// schema_version = 0 (no manifest): legacy, also *.toml.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use toml::Value;

/// On-disk schema version for the snapshot directory layout. Bumped when a
/// device's save_state format changes incompatibly. Old snapshots without a
/// manifest are treated as v0 (legacy, best-effort load).
///
/// v1 → v2: device state moved from *.toml (hex strings, ~80 ms cpu.toml
/// parse) to *.bin (postcard-encoded BinValue tree, sub-millisecond). Manifest
/// and cow.toml stay TOML.
///
/// v2 → v3: RAM banks and framebuffers moved from raw `bank{N}.bin`/`rex3_*.bin`
/// files to a content-addressable chunk store at `saves/.cas/`. Each snapshot
/// writes a tiny `chunks.bin` manifest of per-bank/per-framebuffer chunk
/// hashes. Two snapshots from the same parent share 95–99% of chunks, so a
/// fresh save-after-bundle-install costs only the bytes that changed.
///
/// v3 → v4: the L1D cache tag moved to the Figure 11-4 field positions. A v3
/// reader would take a v4 tag as clean and drop the line without writing it
/// back, so the bump is what makes it refuse the snapshot instead.
pub const SCHEMA_VERSION: u32 = 4;

const MANIFEST_FILE: &str = "snapshot.toml";

pub struct Snapshot {
    pub dir: PathBuf,
}

/// Identity of a configured SCSI disk image at capture time. Recorded so a
/// restore can refuse to paste captured RAM/COW state onto a different base
/// disk (which would silently corrupt the guest filesystem). Identity is the
/// configured path plus the host file size in bytes — cheap to compute and
/// catches the common wrong-disk / wrong-config / resized-image cases without
/// hashing multi-GB images.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskRef {
    pub id: u8,
    pub path: String,
    pub size_bytes: u64,
}

/// Build the list of cargo features enabled in this binary. Recorded in the
/// manifest and required to match on restore, since features such as `ci_clock`
/// (synthetic clock) and `jit` change CPU/timer semantics that the captured
/// state assumes. Sorted for stable comparison.
pub fn enabled_features() -> Vec<String> {
    let mut f: Vec<String> = Vec::new();
    macro_rules! push_if { ($name:literal) => { if cfg!(feature = $name) { f.push($name.to_string()); } } }
    push_if!("jit");
    push_if!("rex-jit");
    push_if!("lightning");
    push_if!("ci_clock");
    push_if!("tlbvmap");
    push_if!("chd");
    push_if!("camera");
    push_if!("developer");
    push_if!("developer_ip7");
    push_if!("tlbstats");
    push_if!("instr_stats");
    push_if!("debug_cache");
    f.sort();
    f
}

/// Top-level snapshot manifest. Lives at `saves/<name>/snapshot.toml`. Written
/// first on save and read first on load so the rest of the pipeline can fail
/// fast with a clear error before reading half a snapshot.
#[derive(Debug, Clone)]
pub struct Manifest {
    pub schema_version: u32,
    pub iris_git_rev: Option<String>,
    pub host_arch: String,
    pub created_at_unix: u64,
    pub parent: Option<String>,
    pub description: Option<String>,
    pub installed_bundles: Vec<String>,
    /// Cargo features enabled in the build that captured this snapshot.
    /// Empty for legacy manifests written before provenance was recorded.
    pub features: Vec<String>,
    /// Configured SCSI disks at capture time. Empty for legacy manifests.
    pub disks: Vec<DiskRef>,
    /// Configured nvram file path at capture time. None for legacy manifests.
    pub nvram: Option<String>,
}

impl Manifest {
    /// Build a manifest describing the current build/host, with no parent or
    /// description. Caller can mutate fields before writing.
    pub fn for_current_save() -> Self {
        let created_at_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            schema_version: SCHEMA_VERSION,
            iris_git_rev: option_env!("IRIS_GIT_REV").map(String::from),
            host_arch: std::env::consts::ARCH.to_string(),
            created_at_unix,
            parent: None,
            description: None,
            installed_bundles: Vec::new(),
            features: enabled_features(),
            // Filled in by the Machine, which is the only thing that knows the
            // configured disks and nvram path.
            disks: Vec::new(),
            nvram: None,
        }
    }

    pub fn to_toml(&self) -> Value {
        let mut tbl = toml::map::Map::new();
        tbl.insert("schema_version".into(), Value::Integer(self.schema_version as i64));
        if let Some(rev) = &self.iris_git_rev {
            tbl.insert("iris_git_rev".into(), Value::String(rev.clone()));
        }
        tbl.insert("host_arch".into(), Value::String(self.host_arch.clone()));
        tbl.insert("created_at_unix".into(), Value::Integer(self.created_at_unix as i64));
        if let Some(parent) = &self.parent {
            tbl.insert("parent".into(), Value::String(parent.clone()));
        }
        if let Some(d) = &self.description {
            tbl.insert("description".into(), Value::String(d.clone()));
        }
        let bundles: Vec<Value> = self.installed_bundles.iter()
            .map(|s| Value::String(s.clone())).collect();
        tbl.insert("installed_bundles".into(), Value::Array(bundles));

        // Provenance (additive; omitted when empty so legacy diffs stay clean).
        if !self.features.is_empty() {
            let feats: Vec<Value> = self.features.iter().map(|s| Value::String(s.clone())).collect();
            tbl.insert("features".into(), Value::Array(feats));
        }
        if !self.disks.is_empty() {
            let disks: Vec<Value> = self.disks.iter().map(|d| {
                let mut dt = toml::map::Map::new();
                dt.insert("id".into(), Value::Integer(d.id as i64));
                dt.insert("path".into(), Value::String(d.path.clone()));
                dt.insert("size_bytes".into(), Value::Integer(d.size_bytes as i64));
                Value::Table(dt)
            }).collect();
            tbl.insert("disks".into(), Value::Array(disks));
        }
        if let Some(nv) = &self.nvram {
            tbl.insert("nvram".into(), Value::String(nv.clone()));
        }
        Value::Table(tbl)
    }

    pub fn from_toml(v: &Value) -> Result<Self, String> {
        let tbl = v.as_table().ok_or("manifest: not a table")?;
        let schema_version = tbl.get("schema_version")
            .and_then(|x| x.as_integer())
            .ok_or("manifest: missing schema_version")? as u32;
        let host_arch = tbl.get("host_arch")
            .and_then(|x| x.as_str())
            .ok_or("manifest: missing host_arch")?
            .to_string();
        let created_at_unix = tbl.get("created_at_unix")
            .and_then(|x| x.as_integer())
            .map(|i| i as u64)
            .unwrap_or(0);
        let iris_git_rev = tbl.get("iris_git_rev").and_then(|x| x.as_str()).map(String::from);
        let parent = tbl.get("parent").and_then(|x| x.as_str()).map(String::from);
        let description = tbl.get("description").and_then(|x| x.as_str()).map(String::from);
        let installed_bundles = tbl.get("installed_bundles")
            .and_then(|x| x.as_array())
            .map(|arr| arr.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        // Provenance — absent in legacy (pre-provenance) manifests; default to
        // empty/None so those still load (validation treats empty as "unknown").
        let features = tbl.get("features")
            .and_then(|x| x.as_array())
            .map(|arr| arr.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let disks = tbl.get("disks")
            .and_then(|x| x.as_array())
            .map(|arr| arr.iter().filter_map(|x| {
                let t = x.as_table()?;
                Some(DiskRef {
                    id: t.get("id").and_then(|v| v.as_integer())? as u8,
                    path: t.get("path").and_then(|v| v.as_str())?.to_string(),
                    size_bytes: t.get("size_bytes").and_then(|v| v.as_integer())? as u64,
                })
            }).collect())
            .unwrap_or_default();
        let nvram = tbl.get("nvram").and_then(|x| x.as_str()).map(String::from);
        Ok(Self {
            schema_version,
            iris_git_rev,
            host_arch,
            created_at_unix,
            parent,
            description,
            installed_bundles,
            features,
            disks,
            nvram,
        })
    }
}

impl Snapshot {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    // ---- helpers ----

    pub fn write_toml(&self, name: &str, v: &Value) -> std::io::Result<()> {
        let path = self.dir.join(name);
        let s = toml::to_string_pretty(v)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut f = fs::File::create(&path)?;
        f.write_all(s.as_bytes())?;
        Ok(())
    }

    pub fn read_toml(&self, name: &str) -> std::io::Result<Value> {
        let path = self.dir.join(name);
        let mut f = fs::File::open(&path)?;
        let mut s = String::new();
        f.read_to_string(&mut s)?;
        toml::from_str::<Value>(&s)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    pub fn write_bin(&self, name: &str, data: &[u8]) -> std::io::Result<()> {
        let path = self.dir.join(name);
        fs::write(path, data)?;
        Ok(())
    }

    pub fn read_bin(&self, name: &str) -> std::io::Result<Vec<u8>> {
        let path = self.dir.join(name);
        fs::read(path)
    }

    /// Postcard-encode a `toml::Value` (via the tagged `BinValue` mirror) and
    /// write it as `<name>`. Sub-millisecond for typical device tables vs ~80
    /// ms TOML parse on cpu.toml.
    pub fn write_value_bin(&self, name: &str, v: &Value) -> std::io::Result<()> {
        let bv = BinValue::from_toml(v);
        let bytes = postcard::to_allocvec(&bv)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        self.write_bin(name, &bytes)
    }

    /// Inverse of `write_value_bin`. Returns the reconstructed `toml::Value`.
    pub fn read_value_bin(&self, name: &str) -> std::io::Result<Value> {
        let bytes = self.read_bin(name)?;
        let bv: BinValue = postcard::from_bytes(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(bv.into_toml())
    }

    /// Postcard-encode a `ChunksManifest` (v3+ snapshots).
    pub fn write_chunks_manifest(&self, m: &ChunksManifest) -> std::io::Result<()> {
        let bytes = postcard::to_allocvec(m)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        self.write_bin("chunks.bin", &bytes)
    }

    pub fn read_chunks_manifest(&self) -> std::io::Result<ChunksManifest> {
        let bytes = self.read_bin("chunks.bin")?;
        postcard::from_bytes(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Write a device save_state value, picking `<base>.bin` for v2+ and
    /// `<base>.toml` for legacy schemas. Centralizes the per-call branching
    /// in machine.rs.
    pub fn write_state(&self, base: &str, v: &Value, schema_version: u32) -> std::io::Result<()> {
        if schema_version >= 2 {
            self.write_value_bin(&format!("{}.bin", base), v)
        } else {
            self.write_toml(&format!("{}.toml", base), v)
        }
    }

    /// Read a device save_state value. For v2+ tries `<base>.bin` first and
    /// falls back to `<base>.toml` for snapshots half-migrated by external
    /// tooling. For legacy schemas reads `<base>.toml` directly.
    pub fn read_state(&self, base: &str, schema_version: u32) -> std::io::Result<Value> {
        if schema_version >= 2 {
            match self.read_value_bin(&format!("{}.bin", base)) {
                Ok(v) => Ok(v),
                Err(_) => self.read_toml(&format!("{}.toml", base)),
            }
        } else {
            self.read_toml(&format!("{}.toml", base))
        }
    }

    pub fn ensure_dir(&self) -> std::io::Result<()> {
        fs::create_dir_all(&self.dir)
    }

    /// Write the manifest to `snapshot.toml`. Always called first on save.
    pub fn write_manifest(&self, m: &Manifest) -> std::io::Result<()> {
        self.write_toml(MANIFEST_FILE, &m.to_toml())
    }

    /// Read the manifest. Returns `Ok(None)` if `snapshot.toml` is absent
    /// (legacy snapshots taken before this format was introduced).
    pub fn read_manifest(&self) -> Result<Option<Manifest>, String> {
        let path = self.dir.join(MANIFEST_FILE);
        if !path.exists() {
            return Ok(None);
        }
        let v = self.read_toml(MANIFEST_FILE).map_err(|e| e.to_string())?;
        Manifest::from_toml(&v).map(Some)
    }
}

// ---- ChunksManifest: per-bank / per-framebuffer chunk hash lists (v3+) ----

use crate::chunk_store::ChunkHash;

/// Per-snapshot pointer into the content-addressable chunk store. Every bank
/// and (optionally) each framebuffer is split into 64 KB chunks; this
/// manifest records the BLAKE3 hash of each chunk in order. Loading a
/// snapshot fetches the chunks and concatenates them back into the bank's
/// big-endian byte stream.
///
/// Stored as `chunks.bin` in the snapshot dir, postcard-encoded.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChunksManifest {
    /// One entry per RAM bank (0..3). Empty inner Vec means the bank wasn't
    /// captured (e.g. zero-sized in this configuration).
    pub bank_chunks: [Vec<ChunkHash>; 4],
    /// REX3 framebuffer chunks: (rgb, aux). `None` when running headless.
    pub framebuffer_chunks: Option<(Vec<ChunkHash>, Vec<ChunkHash>)>,
}

impl ChunksManifest {
    /// Iterate every chunk hash referenced by this manifest. Used by `gc`
    /// to build the live set across all kept snapshots.
    pub fn referenced_hashes(&self) -> impl Iterator<Item = &ChunkHash> {
        self.bank_chunks.iter().flatten().chain(
            self.framebuffer_chunks
                .iter()
                .flat_map(|(rgb, aux)| rgb.iter().chain(aux.iter())),
        )
    }
}

// ---- BinValue: tagged binary mirror of toml::Value ----
//
// Postcard is non-self-describing — it cannot deserialize directly into the
// untagged `toml::Value` enum (which relies on `deserialize_any`). BinValue
// carries an explicit variant tag so postcard can round-trip it. The
// conversion to/from `toml::Value` is a single tree walk and runs in low
// milliseconds even for the largest device tables.
//
// Datetime is rare in our save_state output — encode it as an ISO-8601 string
// and reparse on the way back. If parsing fails the value falls back to a
// plain `toml::Value::String` so a malformed datetime never panics a load.

/// Tagged binary mirror of `toml::Value`. Order-preserving for tables (matches
/// `toml::Value::Table` which uses an `IndexMap` under the hood).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BinValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<BinValue>),
    Table(Vec<(String, BinValue)>),
    Datetime(String),
}

impl BinValue {
    pub fn from_toml(v: &Value) -> Self {
        match v {
            Value::String(s) => BinValue::String(s.clone()),
            Value::Integer(i) => BinValue::Integer(*i),
            Value::Float(f) => BinValue::Float(*f),
            Value::Boolean(b) => BinValue::Boolean(*b),
            Value::Array(arr) => {
                BinValue::Array(arr.iter().map(BinValue::from_toml).collect())
            }
            Value::Table(tbl) => {
                let mut out = Vec::with_capacity(tbl.len());
                for (k, v) in tbl {
                    out.push((k.clone(), BinValue::from_toml(v)));
                }
                BinValue::Table(out)
            }
            Value::Datetime(dt) => BinValue::Datetime(dt.to_string()),
        }
    }

    pub fn into_toml(self) -> Value {
        match self {
            BinValue::String(s) => Value::String(s),
            BinValue::Integer(i) => Value::Integer(i),
            BinValue::Float(f) => Value::Float(f),
            BinValue::Boolean(b) => Value::Boolean(b),
            BinValue::Array(arr) => {
                Value::Array(arr.into_iter().map(BinValue::into_toml).collect())
            }
            BinValue::Table(entries) => {
                let mut tbl = toml::map::Map::new();
                for (k, v) in entries {
                    tbl.insert(k, v.into_toml());
                }
                Value::Table(tbl)
            }
            BinValue::Datetime(s) => match s.parse::<toml::value::Datetime>() {
                Ok(dt) => Value::Datetime(dt),
                Err(_) => Value::String(s),
            },
        }
    }
}

// ---- scalar hex helpers ----

/// Encode a u64 as a hex string Value (e.g. "0x000000001234abcd").
pub fn hex_u64(v: u64) -> Value { Value::String(format!("0x{:016x}", v)) }

/// Encode a u32 as a hex string Value (e.g. "0x1234abcd").
pub fn hex_u32(v: u32) -> Value { Value::String(format!("0x{:08x}", v)) }

/// Encode a u16 as a hex string Value (e.g. "0x1234").
pub fn hex_u16(v: u16) -> Value { Value::String(format!("0x{:04x}", v)) }

/// Encode a u8 as a hex string Value (e.g. "0x12").
pub fn hex_u8(v: u8)   -> Value { Value::String(format!("0x{:02x}", v)) }

// ---- TOML helpers ----

/// Build a TOML array of hex strings from a slice of u64.
pub fn u64_slice_to_toml(slice: &[u64]) -> Value {
    Value::Array(slice.iter().map(|&v| hex_u64(v)).collect())
}

/// Build a TOML array of hex strings from a slice of u32.
pub fn u32_slice_to_toml(slice: &[u32]) -> Value {
    Value::Array(slice.iter().map(|&v| hex_u32(v)).collect())
}

/// Build a TOML array of hex strings from a slice of u16.
pub fn u16_slice_to_toml(slice: &[u16]) -> Value {
    Value::Array(slice.iter().map(|&v| hex_u16(v)).collect())
}

/// Build a TOML array of hex strings from a slice of u8.
pub fn u8_slice_to_toml(slice: &[u8]) -> Value {
    Value::Array(slice.iter().map(|&v| hex_u8(v)).collect())
}

/// Parse a hex string or integer TOML value as u64.
pub fn toml_u64(v: &Value) -> Option<u64> {
    match v {
        Value::String(s) => u64::from_str_radix(s.trim_start_matches("0x"), 16).ok(),
        Value::Integer(i) => Some(*i as u64),
        _ => None,
    }
}

/// Parse a hex string or integer TOML value as u32.
pub fn toml_u32(v: &Value) -> Option<u32> {
    match v {
        Value::String(s) => u64::from_str_radix(s.trim_start_matches("0x"), 16).ok().map(|x| x as u32),
        Value::Integer(i) => Some(*i as u32),
        _ => None,
    }
}

/// Parse a hex string or integer TOML value as u16.
pub fn toml_u16(v: &Value) -> Option<u16> {
    match v {
        Value::String(s) => u64::from_str_radix(s.trim_start_matches("0x"), 16).ok().map(|x| x as u16),
        Value::Integer(i) => Some(*i as u16),
        _ => None,
    }
}

/// Parse a hex string or integer TOML value as u8.
pub fn toml_u8(v: &Value) -> Option<u8> {
    match v {
        Value::String(s) => u64::from_str_radix(s.trim_start_matches("0x"), 16).ok().map(|x| x as u8),
        Value::Integer(i) => Some(*i as u8),
        _ => None,
    }
}

/// Extract a bool from a TOML Value::Boolean.
pub fn toml_bool(v: &Value) -> Option<bool> {
    v.as_bool()
}

/// Load a u64 slice from a TOML array, filling as many entries as available.
pub fn load_u64_slice(v: &Value, dst: &mut [u64]) {
    if let Value::Array(arr) = v {
        for (i, item) in arr.iter().enumerate() {
            if i >= dst.len() { break; }
            if let Some(x) = toml_u64(item) { dst[i] = x; }
        }
    }
}

/// Load a u32 slice from a TOML array.
pub fn load_u32_slice(v: &Value, dst: &mut [u32]) {
    if let Value::Array(arr) = v {
        for (i, item) in arr.iter().enumerate() {
            if i >= dst.len() { break; }
            if let Some(x) = toml_u32(item) { dst[i] = x; }
        }
    }
}

/// Load a u16 slice from a TOML array.
pub fn load_u16_slice(v: &Value, dst: &mut [u16]) {
    if let Value::Array(arr) = v {
        for (i, item) in arr.iter().enumerate() {
            if i >= dst.len() { break; }
            if let Some(x) = toml_u16(item) { dst[i] = x; }
        }
    }
}

/// Load a u8 slice from a TOML array.
pub fn load_u8_slice(v: &Value, dst: &mut [u8]) {
    if let Value::Array(arr) = v {
        for (i, item) in arr.iter().enumerate() {
            if i >= dst.len() { break; }
            if let Some(x) = toml_u8(item) { dst[i] = x; }
        }
    }
}

/// Get a field from a TOML table by key.
pub fn get_field<'a>(table: &'a Value, key: &str) -> Option<&'a Value> {
    table.as_table()?.get(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_tmp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let p = std::env::temp_dir().join(format!("iris-snap-test-{}-{}", tag, nanos));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn manifest_round_trip_full() {
        let m = Manifest {
            schema_version: 1,
            iris_git_rev: Some("abc123".into()),
            host_arch: "aarch64".into(),
            created_at_unix: 1_700_000_000,
            parent: Some("base/desktop".into()),
            description: Some("post install".into()),
            installed_bundles: vec!["grep-2.5.4".into(), "sed-4.2.2".into()],
            features: vec!["jit".into(), "tlbvmap".into()],
            disks: vec![DiskRef { id: 1, path: "irix53.raw".into(), size_bytes: 4294967296 }],
            nvram: Some("nvram-irix53.bin".into()),
        };
        let v = m.to_toml();
        let m2 = Manifest::from_toml(&v).expect("parse");
        assert_eq!(m2.schema_version, m.schema_version);
        assert_eq!(m2.iris_git_rev, m.iris_git_rev);
        assert_eq!(m2.host_arch, m.host_arch);
        assert_eq!(m2.created_at_unix, m.created_at_unix);
        assert_eq!(m2.parent, m.parent);
        assert_eq!(m2.description, m.description);
        assert_eq!(m2.installed_bundles, m.installed_bundles);
        assert_eq!(m2.features, m.features);
        assert_eq!(m2.disks, m.disks);
        assert_eq!(m2.nvram, m.nvram);
    }

    #[test]
    fn manifest_round_trip_minimal() {
        let m = Manifest {
            schema_version: 1,
            iris_git_rev: None,
            host_arch: "x86_64".into(),
            created_at_unix: 0,
            parent: None,
            description: None,
            installed_bundles: vec![],
            features: vec![],
            disks: vec![],
            nvram: None,
        };
        let v = m.to_toml();
        let m2 = Manifest::from_toml(&v).expect("parse");
        assert!(m2.iris_git_rev.is_none());
        assert!(m2.parent.is_none());
        assert!(m2.description.is_none());
        assert!(m2.installed_bundles.is_empty());
        assert!(m2.features.is_empty());
        assert!(m2.disks.is_empty());
        assert!(m2.nvram.is_none());
    }

    #[test]
    fn manifest_rejects_missing_schema_version() {
        let mut tbl = toml::map::Map::new();
        tbl.insert("host_arch".into(), Value::String("aarch64".into()));
        let v = Value::Table(tbl);
        assert!(Manifest::from_toml(&v).is_err());
    }

    #[test]
    fn manifest_disk_round_trip() {
        let dir = unique_tmp_dir("manifest");
        let snap = Snapshot::new(&dir);
        let m = Manifest::for_current_save();
        snap.write_manifest(&m).expect("write");
        let loaded = snap.read_manifest().expect("read").expect("present");
        assert_eq!(loaded.schema_version, SCHEMA_VERSION);
        assert_eq!(loaded.host_arch, std::env::consts::ARCH);
        // cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_absent_returns_none() {
        let dir = unique_tmp_dir("missing");
        let snap = Snapshot::new(&dir);
        let loaded = snap.read_manifest().expect("read");
        assert!(loaded.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn for_current_save_uses_runtime_arch() {
        let m = Manifest::for_current_save();
        assert_eq!(m.schema_version, SCHEMA_VERSION);
        assert_eq!(m.host_arch, std::env::consts::ARCH);
        assert!(m.parent.is_none());
    }

    fn sample_value() -> Value {
        // Mirrors a slice of cpu.toml: top-level scalars + a sub-table with
        // mixed integer/string/array entries. Order matters for the table
        // round-trip assertion.
        let mut cp0 = toml::map::Map::new();
        cp0.insert("cp0_index".into(), Value::String("0x00000001".into()));
        cp0.insert("cp0_count".into(), Value::String("0x000000000badf00d".into()));
        cp0.insert("cp0_status".into(), Value::Integer(0x4040_0000));
        let mut tbl = toml::map::Map::new();
        tbl.insert("pc".into(), Value::String("0x9fc00000".into()));
        tbl.insert(
            "gpr".into(),
            Value::Array(vec![
                Value::String("0x0000000000000000".into()),
                Value::String("0x0000000000000001".into()),
                Value::String("0xffffffff80001234".into()),
            ]),
        );
        tbl.insert("cp0".into(), Value::Table(cp0));
        tbl.insert("running".into(), Value::Boolean(true));
        tbl.insert("ratio".into(), Value::Float(1.5));
        Value::Table(tbl)
    }

    #[test]
    fn binvalue_round_trip_matches_toml() {
        let v = sample_value();
        let bv = BinValue::from_toml(&v);
        let back = bv.into_toml();
        assert_eq!(back, v);
    }

    #[test]
    fn binvalue_postcard_round_trip() {
        let v = sample_value();
        let bv = BinValue::from_toml(&v);
        let bytes = postcard::to_allocvec(&bv).expect("encode");
        let bv2: BinValue = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(bv2.into_toml(), v);
    }

    #[test]
    fn write_state_v2_writes_bin_and_reads_back() {
        let dir = unique_tmp_dir("state-v2");
        let snap = Snapshot::new(&dir);
        let v = sample_value();
        snap.write_state("cpu", &v, 2).expect("write v2");
        assert!(dir.join("cpu.bin").exists(), "expected cpu.bin to be written");
        assert!(!dir.join("cpu.toml").exists(), "v2 must not write cpu.toml");
        let back = snap.read_state("cpu", 2).expect("read v2");
        assert_eq!(back, v);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_state_v1_writes_toml_and_reads_back() {
        let dir = unique_tmp_dir("state-v1");
        let snap = Snapshot::new(&dir);
        let v = sample_value();
        snap.write_state("cpu", &v, 1).expect("write v1");
        assert!(dir.join("cpu.toml").exists(), "expected cpu.toml to be written");
        assert!(!dir.join("cpu.bin").exists(), "v1 must not write cpu.bin");
        let back = snap.read_state("cpu", 1).expect("read v1");
        assert_eq!(back, v);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_state_v2_falls_back_to_toml_when_bin_missing() {
        // External tooling may legitimately produce a v2 manifest with .toml
        // device files (e.g. dump-and-edit workflow). Loader must be tolerant.
        let dir = unique_tmp_dir("state-fallback");
        let snap = Snapshot::new(&dir);
        let v = sample_value();
        snap.write_toml("cpu.toml", &v).expect("write toml");
        let back = snap.read_state("cpu", 2).expect("read with fallback");
        assert_eq!(back, v);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Hand-runnable bench: `cargo test --release --features lightning -- --ignored bench_cpu_toml_vs_bin --nocapture`.
    /// Reads saves/working/cpu.toml (3.6 MB legacy snapshot) and prints the
    /// parse-time delta between toml::from_str and postcard::from_bytes.
    #[test]
    #[ignore]
    fn bench_cpu_toml_vs_bin() {
        let path = "saves/working/cpu.toml";
        let s = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skipping: cannot read {}: {}", path, e);
                return;
            }
        };
        println!("cpu.toml: {} bytes", s.len());

        let runs = 5;
        let mut toml_total_us = 0u128;
        let mut toml_v: Option<Value> = None;
        for _ in 0..runs {
            let t = std::time::Instant::now();
            toml_v = Some(toml::from_str::<Value>(&s).unwrap());
            toml_total_us += t.elapsed().as_micros();
        }
        println!("toml::from_str avg over {} runs: {:.2} ms",
                 runs, toml_total_us as f64 / runs as f64 / 1000.0);

        let v = toml_v.take().unwrap();
        let bv = BinValue::from_toml(&v);
        let bytes = postcard::to_allocvec(&bv).unwrap();
        println!("postcard encoded: {} bytes (vs toml {} bytes, ratio {:.2}x)",
                 bytes.len(), s.len(), s.len() as f64 / bytes.len() as f64);

        let mut bin_total_us = 0u128;
        for _ in 0..runs {
            let t = std::time::Instant::now();
            let bv: BinValue = postcard::from_bytes(&bytes).unwrap();
            let _ = bv.into_toml();
            bin_total_us += t.elapsed().as_micros();
        }
        println!("postcard decode + into_toml avg over {} runs: {:.2} ms",
                 runs, bin_total_us as f64 / runs as f64 / 1000.0);
        println!("speedup: {:.1}x",
                 toml_total_us as f64 / bin_total_us as f64);
    }

    #[test]
    fn binvalue_payload_is_smaller_than_toml() {
        // Sanity check the size win on a representative-ish payload.
        let mut tbl = toml::map::Map::new();
        let big_arr: Vec<Value> = (0..1024)
            .map(|i| Value::String(format!("0x{:016x}", i as u64)))
            .collect();
        tbl.insert("gpr_big".into(), Value::Array(big_arr));
        let v = Value::Table(tbl);
        let toml_bytes = toml::to_string(&v).unwrap().into_bytes();
        let bv = BinValue::from_toml(&v);
        let bin_bytes = postcard::to_allocvec(&bv).unwrap();
        assert!(
            bin_bytes.len() < toml_bytes.len(),
            "bin {} bytes should be smaller than toml {} bytes",
            bin_bytes.len(),
            toml_bytes.len()
        );
    }
}
