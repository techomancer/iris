// System Snapshot — save and restore full machine state to/from a directory.
//
// Layout of saves/<name>/ for an Indy or Indigo2 (the current schema is
// `SCHEMA_VERSION`; the other machines save their own device files):
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

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use toml::Value;

use crate::chunk_store::{get_chunks_as_words, put_words_as_chunks, ChunkHash, ChunkStore};
use crate::machine_state::GuestState;
use crate::workstation::Workstation;

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
///
/// v4 → v5: each of the O2's MACE UARTs saves the 16550 core's registers,
/// receive FIFO and typed backlog where it saved eight registers and a queue.
/// A v4 reader would find none of it in a v5 snapshot and restore a console
/// with every UART register at zero under a running kernel; a v5 reader does
/// the same with a v4 O2 snapshot, so the O2 refuses those. Nothing else
/// changed shape: the serial DMA, the Indy's SCC and the rest are as before.
/// Octane snapshots start at v5, and an Octane reads only snapshots of the
/// schema it was built with, so the next bump refuses every older one.
///
/// v5 → v6: the PCI bus saves its own state (each function's header, the pin
/// and wire levels) where MACE and the Bridge each saved headers and pin
/// levels in their own layout. A v5 reader would load none of it and leave
/// every BAR unassigned under a running kernel, and a v6 reader does the
/// same with a v5 snapshot, so both machines refuse the other schema.
///
/// v6 → v7: `ChunksManifest` gains the KONA board's chunk lists (the pixel store's
/// arrays and the cursor memory). postcard is not self-describing, so a v6 reader
/// would take the new field's bytes for something else and a v7 reader would fail on
/// a v6 `chunks.bin`; the bump makes both refuse. Nothing else changed shape.
///
/// v7 → v8: `ChunksManifest` gains named IMPACT board state chunks. Metadata
/// remains in the device state file; large framebuffer, texture, frontend and
/// raster-engine payloads are content-addressed here.
///
/// v8 → v9: `ChunksManifest` gains named VPro/Odyssey board state and VRAM
/// chunks. The board's small metadata remains in its device state file.
///
/// v9 → v10: `ChunksManifest` gains named GR2 board state, VRAM, and Z-buffer
/// chunks. The board's small metadata remains in its device state file.
///
/// v10 → v11: disks, overlays and placed devices are named by slot path (`xio15/pci0/0/t1`) where they were numbered by
/// controller and target id, so each side refuses the other schema.
pub const SCHEMA_VERSION: u32 = 11;

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
    /// Where the disk is: its SCSI target or loop ID, or for an NVMe card the card.
    pub at: crate::slot_path::MachineSlotPath,
    pub path: String,
    pub size_bytes: u64,
}

impl DiskRef {
    /// The disks of `tables` as a manifest records them, by slot; CD-ROMs only when `cdroms` (the Indy and Indigo2 list them, the O2 and Octane do not).
    pub fn from_tables(tables: &[crate::scsi::TableRef], cdroms: bool) -> Vec<DiskRef> {
        tables.iter()
            .filter(|t| cdroms || !t.is_cdrom)
            .map(|t| DiskRef { at: t.at.clone(), path: t.path.clone(), size_bytes: t.size_bytes })
            .collect()
    }
}

/// Reject a restore whose CPU model differs from the running machine's.
/// `None` from the snapshot means a legacy manifest — the model was a build
/// flag then, so there is nothing to compare and the restore is allowed.
pub fn cpu_model_mismatch(snapshot: Option<&str>, current: &str) -> Option<String> {
    match snapshot {
        Some(sn) if sn != current => Some(format!(
            "CPU model differs: snapshot {} vs current {} — set [machine] cpu = \"{}\"",
            sn, current, sn.to_lowercase()
        )),
        _ => None,
    }
}

/// `Some(reason)` if a snapshot taken with cache model `snapshot` cannot load
/// into one running `current`. A full cache's state holds line data and dirty
/// lines a shadow cache has nowhere to put, and a shadow snapshot has none of
/// the tags a full cache would need, so the two never mix.
///
/// Manifests from before the choice existed carry no cache model; theirs was
/// the only one for the CPU: shadow on the R10000 (named "shadow"), full
/// otherwise.
pub fn cache_model_mismatch(snapshot: Option<&str>, snapshot_cpu: Option<&str>, current: &str) -> Option<String> {
    let snapshot = snapshot.unwrap_or(if snapshot_cpu == Some("shadow") { "shadow" } else { "full" });
    (snapshot != current).then(|| format!(
        "cache model differs: snapshot {snapshot} vs current {current} — set [machine] cache = \"{snapshot}\""))
}

/// Feature names older builds recorded that are no longer cargo features (the
/// code they gated is now always built). Ignored when comparing a snapshot's
/// recorded features against the running build.
pub const RETIRED_FEATURES: &[&str] = &["chd", "camera"];

/// Build the list of cargo features enabled in this binary. Recorded in the
/// manifest and required to match on restore, since features such as `ci_clock`
/// (synthetic clock) change CPU/timer semantics that the captured
/// state assumes. Sorted for stable comparison.
pub fn enabled_features() -> Vec<String> {
    let mut f: Vec<String> = Vec::new();
    macro_rules! push_if { ($name:literal) => { if cfg!(feature = $name) { f.push($name.to_string()); } } }
    push_if!("rex-jit");
    push_if!("lightning");
    push_if!("jitv2");
    push_if!("jitv2_opcodefusion");
    push_if!("opcodefusion");
    push_if!("idle-pause");
    push_if!("ci_clock");
    push_if!("tlbvmap");
    push_if!("developer");
    push_if!("tlbstats");
    push_if!("tlbcheck");
    push_if!("instr_stats");
    f.sort();
    f
}

/// Refuse a snapshot of any schema but the current one.
pub fn check_schema(name: &str, found: u32) -> Result<(), String> {
    if found != SCHEMA_VERSION {
        return Err(format!("snapshot {name} is schema {found}; this build reads schema {SCHEMA_VERSION}"));
    }
    Ok(())
}

/// Check that a snapshot fits this build, CPU and disks: the same build
/// features, CPU model and cache model, and every disk it recorded configured
/// at the same size. A moved disk or NVRAM file, or another iris revision, only
/// warns. `IRIS_SNAPSHOT_SKIP_CHECK=1` makes a mismatch a warning.
pub fn check_provenance(m: &Manifest, want: &Layout) -> Result<(), String> {
    let mut fatal = Vec::new();
    let features: Vec<String> = m.features.iter()
        .filter(|f| !RETIRED_FEATURES.contains(&f.as_str())).cloned().collect();
    if features != enabled_features() {
        fatal.push(format!("build features differ: snapshot [{}] vs current [{}]",
            m.features.join(","), enabled_features().join(",")));
    }
    if let Some(e) = cpu_model_mismatch(m.cpu_model.as_deref(), &want.cpu.0) {
        fatal.push(e);
    }
    if let Some(e) = cache_model_mismatch(m.cpu_cache.as_deref(), m.cpu_model.as_deref(), &want.cpu.1) {
        fatal.push(e);
    }
    for d in &m.disks {
        match want.disks.iter().find(|c| c.at == d.at) {
            None => fatal.push(format!("snapshot disk at {} ('{}') is not configured in this run", d.at, d.path)),
            Some(c) if c.size_bytes != d.size_bytes => fatal.push(format!(
                "disk at {} ('{}') size differs: snapshot {} bytes vs current {}", d.at, d.path, d.size_bytes, c.size_bytes)),
            Some(c) if c.path != d.path => eprintln!(
                "load_snapshot: disk at {} path differs: snapshot '{}' vs current '{}' (same size; continuing)", d.at, d.path, c.path),
            Some(_) => {}
        }
    }
    if m.nvram.is_some() && m.nvram != want.nvram {
        eprintln!("load_snapshot: nvram differs: snapshot {:?} vs current {:?} (continuing)", m.nvram, want.nvram);
    }
    if let (Some(rev), Some(mine)) = (&m.iris_git_rev, option_env!("IRIS_GIT_REV")) {
        if rev != mine {
            eprintln!("load_snapshot: snapshot was captured at iris {rev} but current build is {mine}");
        }
    }
    if !fatal.is_empty() {
        let msg = format!("snapshot provenance mismatch:\n  - {}", fatal.join("\n  - "));
        if std::env::var("IRIS_SNAPSHOT_SKIP_CHECK").is_ok_and(|v| !v.is_empty() && v != "0") {
            eprintln!("load_snapshot: {msg} [IRIS_SNAPSHOT_SKIP_CHECK set; continuing]");
        } else {
            return Err(format!("{msg}\n(set IRIS_SNAPSHOT_SKIP_CHECK=1 to override)"));
        }
    }
    Ok(())
}

/// Each disk's copy-on-write dirty sectors, by the disk's slot path.
pub type OverlaySets = Vec<(crate::slot_path::MachineSlotPath, Vec<u64>)>;

/// A SCSI bus that holds disks, at its slot, for the overlays a snapshot keeps.
pub struct DiskBus<'a> {
    pub at: crate::slot_path::MachineSlotPath,
    pub bus: &'a crate::scsi::ScsiBus,
    /// The slot of a disk on the bus: a target or a loop ID of it. None for a card with one disk
    /// and no target slot, such as an NVMe, where the card is the disk's slot.
    child: Option<fn(&crate::slot_path::MachineSlotPath, u8) -> crate::slot_path::MachineSlotPath>,
}

impl<'a> DiskBus<'a> {
    /// A controller's bus: its disks are its targets.
    pub fn targets(at: crate::slot_path::MachineSlotPath, bus: &'a crate::scsi::ScsiBus) -> Self {
        Self { at, bus, child: Some(crate::slot_path::MachineSlotPath::target) }
    }

    /// A Fibre Channel adapter's loop: its disks are loop IDs.
    pub fn loop_ids(at: crate::slot_path::MachineSlotPath, bus: &'a crate::scsi::ScsiBus) -> Self {
        Self { at, bus, child: Some(crate::slot_path::MachineSlotPath::loop_id) }
    }

    pub fn card(at: crate::slot_path::MachineSlotPath, bus: &'a crate::scsi::ScsiBus) -> Self {
        Self { at, bus, child: None }
    }

    /// The slot of the disk at `id` on this bus.
    pub fn disk(&self, id: usize) -> crate::slot_path::MachineSlotPath {
        self.child.map_or_else(|| self.at.clone(), |child| child(&self.at, id as u8))
    }

    fn id_of(&self, at: &crate::slot_path::MachineSlotPath) -> Option<usize> {
        let Some(child) = self.child else { return (*at == self.at).then_some(0) };
        let n = at.last_number()?;
        (child(&self.at, n) == *at).then_some(n as usize)
    }
}

/// The overlay buses of a machine: its SCSI controllers at `slots`, in the order its slot list
/// declares them, then its cards'.
pub fn disk_buses<'a>(
    slots: impl IntoIterator<Item = crate::slot_path::MachineSlotPath>,
    controllers: impl IntoIterator<Item = &'a crate::scsi::ScsiBus>,
    cards: Option<&'a crate::devices::AttachedCards>,
) -> Vec<DiskBus<'a>> {
    let mut v: Vec<_> = slots.into_iter().zip(controllers).map(|(at, bus)| DiskBus::targets(at, bus)).collect();
    v.extend(cards.into_iter().flat_map(|c| c.disk_buses()));
    v
}

/// Copy every disk's overlay into a directory named by its bus's slot path under the snapshot,
/// and write `cow.toml`: each disk's dirty sectors, keyed by the disk's slot path.
pub fn export_overlays(snap: &Snapshot, buses: &[DiskBus]) -> Result<(), String> {
    let mut tbl = toml::map::Map::new();
    for b in buses {
        let dir = snap.dir.join(b.at.to_string());
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let sets = b.bus.export_overlays(&dir).map_err(|e| format!("COW overlay export ({}): {e}", b.at))?;
        for (id, dirty) in sets {
            let arr = dirty.into_iter().map(|v| Value::Integer(v as i64)).collect();
            tbl.insert(b.disk(id).to_string(), Value::Array(arr));
        }
    }
    snap.write_toml("cow.toml", &Value::Table(tbl)).map_err(|e| e.to_string())
}

/// Where `export_overlays` put the overlay of the disk at `at`, relative to the snapshot: in its bus's directory,
/// named for its target or loop ID, or `scsi0.overlay` in the card's own directory for a card with one disk.
pub fn overlay_file(at: &crate::slot_path::MachineSlotPath) -> String {
    match at.target_id().or_else(|| at.loop_number()).zip(at.parent()) {
        Some((id, bus)) => format!("{bus}/scsi{id}.overlay"),
        None => format!("{at}/scsi0.overlay"),
    }
}

/// What a `cow.toml` lists. Keys that are not slot paths are ignored.
pub fn parse_cow_table(cow_toml: &Value) -> OverlaySets {
    let mut sets = Vec::new();
    if let Some(tbl) = cow_toml.as_table() {
        for (k, v) in tbl {
            let (Ok(at), Some(arr)) = (k.parse::<crate::slot_path::MachineSlotPath>(), v.as_array()) else { continue };
            sets.push((at, arr.iter().filter_map(|x| x.as_integer().map(|i| i as u64)).collect()));
        }
    }
    sets
}

/// The overlays `export_overlays` wrote under `dir`, put back on the buses that hold their
/// disks. A disk with no entry keeps its overlay, and an entry for a slot with no disk is ignored.
pub fn import_overlays(dir: &std::path::Path, buses: &[DiskBus], sets: &OverlaySets) -> Result<(), String> {
    for (at, dirty) in sets {
        for b in buses {
            let Some(id) = b.id_of(at) else { continue };
            b.bus.import_overlays(&dir.join(b.at.to_string()), &[(id, dirty.clone())])
                .map_err(|e| format!("COW overlay import ({at}): {e}"))?;
        }
    }
    Ok(())
}

/// Refuse a snapshot whose devices are not this machine's: one that moved (the error names the
/// `at` that pins it), and one added to the config since, whose bus the saved state has no slot for.
pub fn check_slots(m: &Manifest, current: &[(String, crate::slot_path::MachineSlotPath)]) -> Result<(), String> {
    let mut wrong = Vec::new();
    for (ty, at) in &m.devices {
        if !current.iter().any(|(t, a)| t == ty && a == at) {
            wrong.push(format!("snapshot has a {ty} at {at}, this machine has none there: set at = \"{at}\""));
        }
    }
    for (ty, at) in current {
        if !m.devices.iter().any(|(t, a)| t == ty && a == at) {
            wrong.push(format!("config has a {ty} at {at}, which the snapshot has none of: remove it, or save again"));
        }
    }
    if wrong.is_empty() { Ok(()) } else { Err(format!("snapshot devices differ:\n  - {}", wrong.join("\n  - "))) }
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
    /// Configured SCSI disks at capture time.
    pub disks: Vec<DiskRef>,
    /// The other placed devices at capture time: their type and resolved slot.
    pub devices: Vec<(String, crate::slot_path::MachineSlotPath)>,
    /// Configured nvram file path at capture time. None for legacy manifests.
    pub nvram: Option<String>,
    /// Emulated CPU at capture time. None for legacy manifests, and for those
    /// the check is skipped — the model was a build flag before it was config.
    pub cpu_model: Option<String>,
    /// Cache model at capture time, `CpuModel::CACHE_KIND`: "full" or
    /// "shadow". None for manifests from before it was a choice.
    pub cpu_cache: Option<String>,
    /// The machine, for one that is not the Indy and Indigo2 family: "o2_ip32"
    /// for the O2. None for the Indy and Indigo2 and every older manifest. A
    /// snapshot loads only into the machine it names.
    pub machine: Option<String>,
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
            devices: Vec::new(),
            nvram: None,
            cpu_model: None,
            cpu_cache: None,
            machine: None,
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
                dt.insert("at".into(), Value::String(d.at.to_string()));
                dt.insert("path".into(), Value::String(d.path.clone()));
                dt.insert("size_bytes".into(), Value::Integer(d.size_bytes as i64));
                Value::Table(dt)
            }).collect();
            tbl.insert("disks".into(), Value::Array(disks));
        }
        if !self.devices.is_empty() {
            let devices: Vec<Value> = self.devices.iter().map(|(ty, at)| {
                let mut dt = toml::map::Map::new();
                dt.insert("type".into(), Value::String(ty.clone()));
                dt.insert("at".into(), Value::String(at.to_string()));
                Value::Table(dt)
            }).collect();
            tbl.insert("devices".into(), Value::Array(devices));
        }
        if let Some(cm) = &self.cpu_model {
            tbl.insert("cpu_model".into(), Value::String(cm.clone()));
        }
        if let Some(cc) = &self.cpu_cache {
            tbl.insert("cpu_cache".into(), Value::String(cc.clone()));
        }
        if let Some(nv) = &self.nvram {
            tbl.insert("nvram".into(), Value::String(nv.clone()));
        }
        if let Some(m) = &self.machine {
            tbl.insert("machine".into(), Value::String(m.clone()));
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
                    at: t.get("at").and_then(|v| v.as_str())?.parse().ok()?,
                    path: t.get("path").and_then(|v| v.as_str())?.to_string(),
                    size_bytes: t.get("size_bytes").and_then(|v| v.as_integer())? as u64,
                })
            }).collect())
            .unwrap_or_default();
        let devices = tbl.get("devices")
            .and_then(|x| x.as_array())
            .map(|arr| arr.iter().filter_map(|x| {
                let t = x.as_table()?;
                Some((t.get("type")?.as_str()?.to_string(), t.get("at")?.as_str()?.parse().ok()?))
            }).collect())
            .unwrap_or_default();
        let nvram = tbl.get("nvram").and_then(|x| x.as_str()).map(String::from);
        let cpu_model = tbl.get("cpu_model").and_then(|x| x.as_str()).map(String::from);
        let cpu_cache = tbl.get("cpu_cache").and_then(|x| x.as_str()).map(String::from);
        let machine = tbl.get("machine").and_then(|x| x.as_str()).map(String::from);
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
            devices,
            nvram,
            cpu_model,
            cpu_cache,
            machine,
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
        ChunksManifest::from_bytes(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Write a device save_state value as `<base>.bin`.
    pub fn write_state(&self, base: &str, v: &Value) -> std::io::Result<()> {
        self.write_value_bin(&format!("{base}.bin"), v)
    }

    pub fn read_state(&self, base: &str) -> std::io::Result<Value> {
        self.read_value_bin(&format!("{base}.bin"))
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

// ---- One save, load and rollback for every machine, over the `Workstation` snapshot hooks ----

/// Present while a save is under way, so `read` refuses a directory a save died in.
/// The second name is an older build's.
const INCOMPLETE: [&str; 2] = ["snapshot_incomplete", "resident_snapshot_incomplete"];

/// A machine's state as a snapshot holds it.
#[derive(Default)]
pub struct Image<'a> {
    /// Device tables in load order, kept as `<name>.bin`.
    pub states: Vec<(String, Value)>,
    /// Raw files: the boot flash, a second head's planes.
    pub files: Vec<(String, Vec<u8>)>,
    /// RAM as words, by `bank_chunks` list. Parts of one list are joined.
    pub memory: Vec<(usize, std::borrow::Cow<'a, [u32]>)>,
    /// The first head's RGB and aux planes.
    pub framebuffer: Option<(Vec<u32>, Vec<u32>)>,
    pub boards: Vec<Board>,
}

impl Image<'_> {
    pub fn state(&self, name: &str) -> Result<&Value, String> {
        self.states.iter().find(|s| s.0 == name).map(|s| &s.1).ok_or_else(|| format!("snapshot has no state {name}"))
    }

    pub fn file(&self, name: &str) -> Result<&[u8], String> {
        self.files.iter().find(|f| f.0 == name).map(|f| &f.1[..]).ok_or_else(|| format!("snapshot has no file {name}"))
    }

    pub fn board(&self, state: &str) -> Option<&Board> {
        self.boards.iter().find(|b| b.state == state)
    }
}

/// A graphics board's state: its metadata, kept as `<state>.bin`, and its large named parts.
pub struct Board {
    pub state: String,
    pub list: BoardList,
    pub metadata: String,
    pub parts: Vec<(String, Vec<u32>)>,
}

impl Board {
    pub fn new(state: impl Into<String>, list: BoardList, (metadata, parts): (String, Vec<(String, Vec<u32>)>)) -> Self {
        Self { state: state.into(), list, metadata, parts }
    }
}

/// The list of `chunks.bin` that holds a kind of board's parts.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BoardList { Impact, Vpro, Gr2 }

impl BoardList {
    fn label(self) -> &'static str {
        match self { Self::Impact => "IMPACT", Self::Vpro => "VPro", Self::Gr2 => "GR2" }
    }

    fn of(self, c: &mut ChunksManifest) -> &mut Vec<NamedChunk> {
        match self { Self::Impact => &mut c.mgras_chunks, Self::Vpro => &mut c.odyssey_chunks, Self::Gr2 => &mut c.gr2_chunks }
    }
}

/// What a machine expects of a snapshot: its identity for the manifest and the
/// checks, and what the snapshot must hold, so `read` can refuse one before the machine is touched.
#[derive(Default)]
pub struct Layout {
    /// The manifest's `machine`; None for the Indy and Indigo2.
    pub machine: Option<&'static str>,
    pub disks: Vec<DiskRef>,
    pub slots: Vec<(String, crate::slot_path::MachineSlotPath)>,
    pub nvram: Option<String>,
    /// CPU model and cache kind.
    pub cpu: (String, String),
    pub states: Vec<String>,
    /// Raw files and the length each must have; 0 for any.
    pub files: Vec<(String, usize)>,
    /// Words in each `bank_chunks` list.
    pub memory: Vec<usize>,
    pub framebuffer: bool,
    /// A list holding several boards has each board's parts named `<state>/<part>`.
    pub boards: Vec<(String, BoardList)>,
}

/// What the last load left for `parent` and `rollback`.
#[derive(Default)]
pub struct Restored {
    name: Option<String>,
    kept: Option<Box<Kept>>,
}

/// A snapshot read and checked, ready to put into a machine.
pub struct Kept {
    pub image: Image<'static>,
    dir: PathBuf,
    /// Every disk's copy-on-write sets, by the disk's slot.
    overlays: OverlaySets,
}

/// Save to `saves/<name>/`, then run on if the machine was running.
pub fn save_in<W: Workstation + ?Sized>(m: &mut W, saves: &Path, name: &str) -> Result<(), String> {
    if let Some(why) = m.snapshot_refusal() {
        return Err(why);
    }
    let parent = m.snapshots().and_then(|r| r.name.clone());
    let (lifecycle, was_running) = (m.lifecycle(), m.cpu_is_running());
    m.stop();
    let saved = write(&*m, saves, name, parent);
    if lifecycle != GuestState::Idle {
        m.restart_peripherals();
    }
    if was_running {
        m.cpu_start();
    }
    m.set_lifecycle(lifecycle);
    saved?;
    println!("Snapshot saved to {}/{name}", saves.display());
    Ok(())
}

fn write<W: Workstation + ?Sized>(m: &W, saves: &Path, name: &str, parent: Option<String>) -> Result<(), String> {
    let io = |e: std::io::Error| e.to_string();
    let (layout, image) = (m.snapshot_layout(), m.capture()?);
    let snap = Snapshot::new(saves.join(name));
    snap.ensure_dir().map_err(io)?;
    let marker = snap.dir.join(INCOMPLETE[0]);
    fs::write(&marker, b"snapshot is incomplete\n").map_err(io)?;
    let mut manifest = Manifest::for_current_save();
    manifest.machine = layout.machine.map(Into::into);
    manifest.parent = parent;
    manifest.disks = layout.disks;
    manifest.devices = layout.slots;
    manifest.nvram = layout.nvram;
    (manifest.cpu_model, manifest.cpu_cache) = (Some(layout.cpu.0), Some(layout.cpu.1));
    snap.write_manifest(&manifest).map_err(io)?;
    for (name, state) in &image.states {
        snap.write_state(name, state).map_err(io)?;
    }
    for (name, bytes) in &image.files {
        snap.write_bin(name, bytes).map_err(io)?;
    }
    let store = ChunkStore::new(saves);
    let put = |words: &[u32], what: &str| put_words_as_chunks(&store, words).map_err(|e| format!("CAS {what} put: {e}"));
    let mut chunks = ChunksManifest::default();
    for (list, words) in &image.memory {
        chunks.bank_chunks[*list].extend(put(words, "memory")?);
    }
    if let Some((rgb, aux)) = &image.framebuffer {
        chunks.framebuffer_chunks = Some((put(rgb, "framebuffer rgb")?, put(aux, "framebuffer aux")?));
    }
    for b in &image.boards {
        snap.write_state(&b.state, &Value::String(b.metadata.clone())).map_err(io)?;
        let several = image.boards.iter().filter(|o| o.list == b.list).count() > 1;
        for (part, words) in &b.parts {
            let hashes = put(words, &format!("{} {part}", b.list.label()))?;
            let name = if several { format!("{}/{part}", b.state) } else { part.clone() };
            b.list.of(&mut chunks).push(NamedChunk { name, words: words.len(), hashes });
        }
    }
    snap.write_chunks_manifest(&chunks).map_err(io)?;
    m.with_disk_buses(&mut |buses| export_overlays(&snap, buses))?;
    fs::remove_file(marker).map_err(io)
}

/// Read `saves/<name>/` and check it against the machine, which is left as it was.
pub fn read<W: Workstation + ?Sized>(m: &W, saves: &Path, name: &str) -> Result<Kept, String> {
    if let Some(why) = m.snapshot_refusal() {
        return Err(why);
    }
    let want = m.snapshot_layout();
    let snap = Snapshot::new(saves.join(name));
    if INCOMPLETE.iter().any(|f| snap.dir.join(f).exists()) {
        return Err(format!("snapshot {name} is incomplete"));
    }
    let man = snap.read_manifest()?.ok_or_else(|| format!("{}/{name}: no snapshot.toml", saves.display()))?;
    if man.machine.as_deref() != want.machine {
        return Err(format!("snapshot {name} is of another machine ({})", man.machine.as_deref().unwrap_or("Indy or Indigo2")));
    }
    if man.host_arch != std::env::consts::ARCH {
        return Err(format!("snapshot host_arch '{}' does not match current host '{}'", man.host_arch, std::env::consts::ARCH));
    }
    check_schema(name, man.schema_version)?;
    check_provenance(&man, &want)?;
    check_slots(&man, &want.slots)?;
    let mut image = Image::default();
    for base in &want.states {
        image.states.push((base.clone(), snap.read_state(base).map_err(|e| format!("{base}: {e}"))?));
    }
    for (file, len) in &want.files {
        let bytes = snap.read_bin(file).map_err(|e| format!("{file}: {e}"))?;
        if *len != 0 && bytes.len() != *len {
            return Err(format!("saved {file} is {} bytes, this machine's holds {len}", bytes.len()));
        }
        image.files.push((file.clone(), bytes));
    }
    let store = ChunkStore::new(saves);
    let get = |hashes: &[ChunkHash], what: &str| get_chunks_as_words(&store, hashes).map_err(|e| format!("CAS {what} get: {e}"));
    let mut chunks = snap.read_chunks_manifest().map_err(|e| format!("chunks.bin: {e}"))?;
    for (list, expect) in want.memory.iter().enumerate() {
        let words = get(&chunks.bank_chunks[list], "memory")?;
        if words.len() != *expect {
            return Err(format!("snapshot memory is {} MiB, this machine has {} MiB", words.len() >> 18, expect >> 18));
        }
        image.memory.push((list, words.into()));
    }
    image.framebuffer = match (want.framebuffer, chunks.framebuffer_chunks.take()) {
        (true, Some((rgb, aux))) => Some((get(&rgb, "framebuffer rgb")?, get(&aux, "framebuffer aux")?)),
        (true, None) => return Err("snapshot has no framebuffer for the configured graphics board".into()),
        (false, _) => None,
    };
    for list in [BoardList::Impact, BoardList::Vpro, BoardList::Gr2] {
        let label = list.label();
        let saved = std::mem::take(list.of(&mut chunks));
        let wanted: Vec<_> = want.boards.iter().filter(|b| b.1 == list).map(|b| b.0.as_str()).collect();
        match (wanted.as_slice(), saved.is_empty()) {
            ([], true) => {}
            ([], false) => return Err(format!("snapshot contains {label} state but this machine has no {label} board")),
            (_, true) => return Err(format!("snapshot has no state for the configured {label} board")),
            (states, false) => {
                for &state in states {
                    let metadata = snap.read_state(state).map_err(|e| format!("{state}: {e}"))?;
                    let metadata = metadata.as_str().ok_or_else(|| format!("{label} snapshot metadata is not a string"))?.to_string();
                    let prefix = format!("{state}/");
                    let mut parts = Vec::with_capacity(saved.len());
                    for p in &saved {
                        let name = match states.len() {
                            1 => p.name.as_str(),
                            _ => match p.name.strip_prefix(&prefix) { Some(n) => n, None => continue },
                        };
                        let words = get(&p.hashes, &format!("{label} {name}"))?;
                        if words.len() != p.words {
                            return Err(format!("CAS {label} {name} has {} words, expected {}", words.len(), p.words));
                        }
                        parts.push((name.to_string(), words));
                    }
                    if parts.is_empty() {
                        return Err(format!("snapshot has no state for the {label} board {state}"));
                    }
                    image.boards.push(Board { state: state.to_string(), list, metadata, parts });
                }
            }
        }
    }
    let overlays = snap.read_toml("cow.toml").map(|t| parse_cow_table(&t)).map_err(|e| format!("cow.toml: {e}"))?;
    Ok(Kept { image, dir: snap.dir, overlays })
}

/// Put a read snapshot into the machine, with its disks' overlays. The machine runs on only if `run`.
/// A failure leaves it stopped, with its devices reset or part loaded, wanting a reset or another load.
pub fn restore<W: Workstation + ?Sized>(m: &mut W, kept: &Kept, run: bool) -> Result<(), String> {
    m.stop();
    m.apply_image(&kept.image)?;
    m.with_disk_buses(&mut |buses| import_overlays(&kept.dir, buses, &kept.overlays))?;
    m.restart_peripherals();
    if run {
        m.cpu_start();
    }
    Ok(())
}

/// Load `saves/<name>/` and run. The name is the parent of the next save and what `rollback`
/// returns to; `keep` also holds the image, so that rollback need not read it again.
fn load<W: Workstation + ?Sized>(m: &mut W, saves: &Path, name: &str, keep: bool) -> Result<(), String> {
    let kept = read(m, saves, name)?;
    restore(m, &kept, true)?;
    if let Some(r) = m.snapshots() {
        *r = Restored { name: Some(name.into()), kept: keep.then(|| Box::new(kept)) };
    }
    println!("Snapshot loaded from {}/{name}", saves.display());
    Ok(())
}

pub fn load_in<W: Workstation + ?Sized>(m: &mut W, saves: &Path, name: &str) -> Result<(), String> {
    load(m, saves, name, false)
}

/// The CI socket's `restore`: load, and keep the image and name for `rollback`.
/// What the guest printed before is dropped, so the next command reads no stale output.
pub fn ci_restore<W: Workstation + ?Sized>(m: &mut W, saves: &Path, name: &str) -> Result<(), String> {
    if let Some(ci) = m.get_ci_serial() {
        ci.reset();
    }
    load(m, saves, name, true)
}

/// The CI socket's `rollback`: back to the last `ci_restore`'s image. Its disks' overlays are
/// imported again from the snapshot's directory, which the guest's writes since have not touched.
pub fn ci_rollback<W: Workstation + ?Sized>(m: &mut W, saves: &Path) -> Result<(), String> {
    if let Some(ci) = m.get_ci_serial() {
        ci.reset();
    }
    let r = m.snapshots().ok_or(crate::workstation::NOT_HERE)?;
    let Some(name) = r.name.clone() else {
        return Err("no previous restore to roll back to".into());
    };
    let Some(kept) = r.kept.take() else {
        return ci_restore(m, saves, &name);
    };
    let result = restore(m, &kept, true).map_err(|e| format!("rollback to {name}: {e}"));
    if let Some(r) = m.snapshots() {
        r.kept = Some(kept);
    }
    result
}

// ---- ChunksManifest: per-bank / per-framebuffer chunk hash lists (v3+) ----

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
    /// The KONA pipe's pixel store and cursor memory; empty when the machine has none.
    pub kona_chunks: Vec<crate::dev::kona::KonaChunk>,
    /// Named IMPACT architectural state; empty when the machine has no IMPACT board.
    pub mgras_chunks: Vec<NamedChunk>,
    /// Named VPro architectural state and VRAM; empty when the machine has no VPro board.
    pub odyssey_chunks: Vec<NamedChunk>,
    /// Named GR2 architectural state, VRAM, and Z-buffer; empty without GR2.
    pub gr2_chunks: Vec<NamedChunk>,
}

/// One named part of a graphics board's state: its length in words and the chunks that hold it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedChunk {
    pub name: String,
    pub words: usize,
    pub hashes: Vec<ChunkHash>,
}

impl ChunksManifest {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    /// Iterate every chunk hash referenced by this manifest. Used by `gc`
    /// to build the live set across all kept snapshots.
    pub fn referenced_hashes(&self) -> impl Iterator<Item = &ChunkHash> {
        self.bank_chunks
            .iter()
            .flatten()
            .chain(
                self.framebuffer_chunks
                    .iter()
                    .flat_map(|(rgb, aux)| rgb.iter().chain(aux.iter())),
            )
            // A zero chunk's place holds the zero hash, which the store never holds.
            .chain(
                self.kona_chunks
                    .iter()
                    .flat_map(|c| c.hashes.iter())
                    .filter(|h| **h != [0u8; 32]),
            )
            .chain(self.mgras_chunks.iter().flat_map(|c| c.hashes.iter()))
            .chain(self.odyssey_chunks.iter().flat_map(|c| c.hashes.iter()))
            .chain(self.gr2_chunks.iter().flat_map(|c| c.hashes.iter()))
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

// ---- required fields ----
//
// For state with no sensible default: a snapshot missing one of these is
// refused with the field's name, not loaded with a zero in its place.

pub fn field<'a>(table: &'a Value, key: &str) -> Result<&'a Value, String> {
    get_field(table, key).ok_or_else(|| format!("missing {key}"))
}

pub fn field_u64(table: &Value, key: &str) -> Result<u64, String> {
    toml_u64(field(table, key)?).ok_or_else(|| format!("{key}: not a number"))
}

/// A count stored as a number, refused above `limit`: the size of whatever
/// array it indexes.
pub fn usize_field(v: &Value, key: &str, limit: usize) -> Result<usize, String> {
    let n = field_u64(v, key)? as usize;
    if n > limit {
        return Err(format!("{key}: {n} is more than {limit}"));
    }
    Ok(n)
}

pub fn field_u32(table: &Value, key: &str) -> Result<u32, String> {
    toml_u32(field(table, key)?).ok_or_else(|| format!("{key}: not a 32-bit number"))
}

pub fn field_u16(table: &Value, key: &str) -> Result<u16, String> {
    toml_u16(field(table, key)?).ok_or_else(|| format!("{key}: not a 16-bit number"))
}

pub fn field_u8(table: &Value, key: &str) -> Result<u8, String> {
    toml_u8(field(table, key)?).ok_or_else(|| format!("{key}: not a byte"))
}

pub fn field_bool(table: &Value, key: &str) -> Result<bool, String> {
    toml_bool(field(table, key)?).ok_or_else(|| format!("{key}: not a boolean"))
}

/// A list of exactly `N` numbers, each converted by `get`.
fn field_list<T: Copy + Default, const N: usize>(
    table: &Value, key: &str, get: fn(&Value) -> Option<T>,
) -> Result<[T; N], String> {
    let items = field(table, key)?.as_array().ok_or_else(|| format!("{key}: not a list"))?;
    if items.len() != N {
        return Err(format!("{key}: {} entries, expected {N}", items.len()));
    }
    let mut out = [T::default(); N];
    for (o, item) in out.iter_mut().zip(items) {
        *o = get(item).ok_or_else(|| format!("{key}: an entry is out of range"))?;
    }
    Ok(out)
}

pub fn field_u64s<const N: usize>(table: &Value, key: &str) -> Result<[u64; N], String> {
    field_list(table, key, toml_u64)
}

pub fn field_u32s<const N: usize>(table: &Value, key: &str) -> Result<[u32; N], String> {
    field_list(table, key, toml_u32)
}

pub fn field_u8s<const N: usize>(table: &Value, key: &str) -> Result<[u8; N], String> {
    field_list(table, key, toml_u8)
}

/// A byte list of any length.
pub fn field_bytes(table: &Value, key: &str) -> Result<Vec<u8>, String> {
    field(table, key)?.as_array().ok_or_else(|| format!("{key}: not a list"))?
        .iter().map(|v| toml_u8(v).ok_or_else(|| format!("{key}: an entry is not a byte"))).collect()
}

/// A list of 64-bit numbers of any length.
pub fn field_u64_vec(table: &Value, key: &str) -> Result<Vec<u64>, String> {
    field(table, key)?.as_array().ok_or_else(|| format!("{key}: not a list"))?
        .iter().map(|v| toml_u64(v).ok_or_else(|| format!("{key}: an entry is not a number"))).collect()
}

/// A table of state, as save_state functions build them.
#[derive(Default)]
pub struct StateTable(toml::map::Map<String, Value>);

impl StateTable {
    pub fn new() -> Self { Self::default() }
    pub fn u64(&mut self, key: &str, v: u64) -> &mut Self { self.0.insert(key.into(), hex_u64(v)); self }
    pub fn u32(&mut self, key: &str, v: u32) -> &mut Self { self.0.insert(key.into(), hex_u32(v)); self }
    pub fn u16(&mut self, key: &str, v: u16) -> &mut Self { self.0.insert(key.into(), hex_u16(v)); self }
    pub fn u8(&mut self, key: &str, v: u8) -> &mut Self { self.0.insert(key.into(), hex_u8(v)); self }
    pub fn bool(&mut self, key: &str, v: bool) -> &mut Self { self.0.insert(key.into(), Value::Boolean(v)); self }
    pub fn u64s(&mut self, key: &str, v: &[u64]) -> &mut Self { self.0.insert(key.into(), u64_slice_to_toml(v)); self }
    pub fn u32s(&mut self, key: &str, v: &[u32]) -> &mut Self { self.0.insert(key.into(), u32_slice_to_toml(v)); self }
    pub fn u16s(&mut self, key: &str, v: &[u16]) -> &mut Self { self.0.insert(key.into(), u16_slice_to_toml(v)); self }
    pub fn u8s(&mut self, key: &str, v: &[u8]) -> &mut Self { self.0.insert(key.into(), u8_slice_to_toml(v)); self }
    pub fn value(&mut self, key: &str, v: Value) -> &mut Self { self.0.insert(key.into(), v); self }
    pub fn build(&mut self) -> Value { Value::Table(std::mem::take(&mut self.0)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the current schema loads; an older or newer one is refused with the schema this build reads.
    #[test]
    fn check_schema_names_the_schema_it_reads() {
        let e = check_schema("old", SCHEMA_VERSION - 1).unwrap_err();
        assert!(e.contains(&format!("schema {}", SCHEMA_VERSION - 1)) && e.contains(&format!("reads schema {SCHEMA_VERSION}")), "{e}");
        assert!(check_schema("newer", SCHEMA_VERSION + 1).is_err());
        assert!(check_schema("current", SCHEMA_VERSION).is_ok());
    }

    /// The current layout round-trips, and a cut-short file is refused.
    #[test]
    fn a_chunks_manifest_of_the_current_layout_round_trips_and_garbage_is_refused() {
        let mut m = ChunksManifest::default();
        m.bank_chunks[2].push([7; 32]);
        m.kona_chunks.push(crate::dev::kona::KonaChunk {
            name: "state-9".into(),
            hashes: vec![[9; 32]],
        });
        m.mgras_chunks.push(NamedChunk {
            name: "front".into(),
            words: 17,
            hashes: vec![[10; 32]],
        });
        m.odyssey_chunks.push(NamedChunk {
            name: "vram".into(),
            words: 23,
            hashes: vec![[11; 32]],
        });
        m.gr2_chunks.push(NamedChunk {
            name: "zbuf".into(),
            words: 29,
            hashes: vec![[12; 32]],
        });
        let bytes = postcard::to_allocvec(&m).unwrap();
        let back = ChunksManifest::from_bytes(&bytes).unwrap();
        assert_eq!(back.bank_chunks[2], m.bank_chunks[2]);
        assert_eq!(back.kona_chunks, m.kona_chunks);
        assert_eq!(back.mgras_chunks, m.mgras_chunks);
        assert_eq!(back.odyssey_chunks, m.odyssey_chunks);
        assert_eq!(back.gr2_chunks, m.gr2_chunks);
        assert!(back.referenced_hashes().any(|hash| *hash == [12; 32]));
        assert!(
            ChunksManifest::from_bytes(&bytes[..bytes.len() - 1]).is_err(),
            "cut short"
        );
        assert!(ChunksManifest::from_bytes(&[]).is_err());
    }

    /// Overlays are kept per bus under its slot and listed by each disk's slot, so the same target on two
    /// controllers, an NVMe card and a loop disk name four different files, and they come back by slot.
    #[test]
    fn overlays_round_trip_by_slot_across_buses() {
        use crate::scsi::{ScsiBus, ScsiIdentities, ScsiIdentity};
        let id = ScsiIdentity::new(b"TEST    ", b"SNAPSHOT        ", b"1.0 ");
        let bus = || ScsiBus::new(16, ScsiIdentities { disk: id, cdrom: id });
        let (a, b, card, fc) = (bus(), bus(), bus(), bus());
        let dir = unique_tmp_dir("overlays");
        let image = dir.join("d.raw");
        fs::write(&image, vec![0u8; 4096]).unwrap();
        let disk = crate::config::ScsiDeviceConfig { path: image.to_string_lossy().into_owned(), overlay: true, ..Default::default() };
        for (bus, at) in [(&a, 3), (&b, 3), (&card, 0), (&fc, 9)] {
            crate::scsi::attach_table(bus, at, "x", &disk, None, None).unwrap();
        }
        let loc = |s: &str| s.parse::<crate::slot_path::MachineSlotPath>().unwrap();
        let buses = [
            DiskBus::targets(loc("hpc3/c0"), &a),
            DiskBus::targets(loc("hpc3/c1"), &b),
            DiskBus::card(loc("xio13/pci0/1"), &card),
            DiskBus::loop_ids(loc("xio13/pci0/2"), &fc),
        ];
        let snap = Snapshot::new(dir.join("saves/s"));
        snap.ensure_dir().unwrap();
        export_overlays(&snap, &buses).unwrap();
        let mut sets = parse_cow_table(&snap.read_toml("cow.toml").unwrap());
        sets.sort();
        let mut named: Vec<String> = sets.iter().map(|s| s.0.to_string()).collect();
        named.sort();
        assert_eq!(named, ["hpc3/c0/t3", "hpc3/c1/t3", "xio13/pci0/1", "xio13/pci0/2/l9"]);
        for f in ["hpc3/c0/scsi3.overlay", "hpc3/c1/scsi3.overlay", "xio13/pci0/1/scsi0.overlay", "xio13/pci0/2/scsi9.overlay"] {
            assert!(snap.dir.join(f).exists(), "{f}");
        }
        import_overlays(&snap.dir, &buses, &sets).expect("each comes back from its own slot");
        let _ = fs::remove_dir_all(&dir);
    }

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
            disks: vec![DiskRef { at: "hpc3/c0/t1".parse().unwrap(), path: "irix53.raw".into(), size_bytes: 4294967296 }],
            devices: vec![("virtio_serial".into(), "gio1".parse().unwrap())],
            nvram: Some("nvram-irix53.bin".into()),
            cpu_model: Some("R5000".into()),
            cpu_cache: Some("shadow".into()),
            machine: Some("o2_ip32".into()),
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
        assert_eq!(m2.devices, m.devices);
        assert_eq!(m2.nvram, m.nvram);
        assert_eq!(m2.cpu_model, m.cpu_model);
        assert_eq!(m2.machine, m.machine);
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
            devices: vec![],
            nvram: None,
            cpu_model: None,
            cpu_cache: None,
            machine: None,
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
        assert!(m2.cpu_model.is_none(), "legacy manifest has no cpu_model");
        assert!(m2.machine.is_none(), "an Indy/Indigo2 manifest names no machine");
    }

    #[test]
    fn a_snapshot_only_restores_onto_the_cpu_it_was_taken_on() {
        // same model: fine
        assert!(cpu_model_mismatch(Some("R4400"), "R4400").is_none());
        assert!(cpu_model_mismatch(Some("R5000"), "R5000").is_none());
        // legacy manifest: nothing recorded, so nothing to refuse
        assert!(cpu_model_mismatch(None, "R5000").is_none());
        // crossed models: refused, and the message says how to fix it
        let e = cpu_model_mismatch(Some("R4400"), "R5000").expect("must refuse");
        assert!(e.contains("R4400") && e.contains("R5000"), "{}", e);
        assert!(e.contains(r#"cpu = "r4400""#), "must name the config fix: {}", e);
        assert!(cpu_model_mismatch(Some("R5000"), "R4400").is_some());
    }

    #[test]
    fn cache_model_mismatch_is_fatal_and_legacy_means_the_old_default() {
        assert!(cache_model_mismatch(Some("full"), Some("R4400"), "full").is_none());
        assert!(cache_model_mismatch(Some("shadow"), Some("R5000"), "shadow").is_none());
        assert!(cache_model_mismatch(Some("full"), Some("R4400"), "shadow").is_some());
        assert!(cache_model_mismatch(Some("shadow"), Some("R4400"), "full").is_some());
        // Before the choice: R4400/R5000 ran the full cache, the R10000 the shadow one.
        assert!(cache_model_mismatch(None, Some("R4400"), "full").is_none());
        assert!(cache_model_mismatch(None, Some("R4400"), "shadow").is_some());
        assert!(cache_model_mismatch(None, Some("shadow"), "shadow").is_none());
        assert!(cache_model_mismatch(None, None, "full").is_none());
    }

    /// The CPU model must survive a manifest round trip, and a legacy manifest
    /// (written before the model was config) must still parse as "unknown" so
    /// old snapshots keep loading instead of being refused outright.
    #[test]
    fn cpu_model_round_trips_and_legacy_stays_unknown() {
        let mut m = Manifest::for_current_save();
        m.cpu_model = Some("R5000".into());
        let back = Manifest::from_toml(&m.to_toml()).expect("round trip");
        assert_eq!(back.cpu_model.as_deref(), Some("R5000"));

        // a manifest with no cpu_model key at all
        let mut t = m.to_toml();
        t.as_table_mut().unwrap().remove("cpu_model");
        let legacy = Manifest::from_toml(&t).expect("legacy parses");
        assert!(legacy.cpu_model.is_none(), "absent cpu_model must read as unknown, not error");

        // and the execution-affecting flags are now recorded
        let f = enabled_features();
        for name in ["jitv2", "opcodefusion", "idle-pause"] {
            assert_eq!(f.contains(&name.to_string()), cfg!(feature = "jitv2") && name == "jitv2"
                || cfg!(feature = "opcodefusion") && name == "opcodefusion"
                || cfg!(feature = "idle-pause") && name == "idle-pause",
                "{} recorded iff enabled", name);
        }
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
    fn write_state_writes_bin_and_reads_back() {
        let dir = unique_tmp_dir("state-bin");
        let snap = Snapshot::new(&dir);
        let v = sample_value();
        snap.write_state("cpu", &v).expect("write");
        assert!(dir.join("cpu.bin").exists(), "expected cpu.bin to be written");
        assert!(!dir.join("cpu.toml").exists(), "no cpu.toml");
        assert_eq!(snap.read_state("cpu").expect("read"), v);
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

    #[test]
    fn required_fields_name_what_is_missing_or_short() {
        let v = StateTable::new().u64("a", 7).u32s("b", &[1, 2]).bool("c", true).build();
        assert_eq!(field_u64(&v, "a"), Ok(7));
        assert_eq!(field_u32s::<2>(&v, "b"), Ok([1, 2]));
        assert_eq!(field_bool(&v, "c"), Ok(true));
        assert_eq!(field_u64(&v, "z"), Err("missing z".into()));
        assert_eq!(field_u32s::<3>(&v, "b"), Err("b: 2 entries, expected 3".into()));
        assert!(field_bool(&v, "a").is_err());
    }
}
