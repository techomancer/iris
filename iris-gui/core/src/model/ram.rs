//! RAM bank helpers shared by the Memory tab, menus, and status readouts.

/// Quick preset totals (MB) for the Memory menu and new-machine dialog.
const RAM_PRESETS: &[u32] = &[32, 64, 96, 128, 192, 256, 384, 512];

const IP28_RAM_PRESETS: &[u32] = &[32, 64, 96, 128, 192, 256, 384, 512, 768, 1024];

/// An O2's kits are 32/64/128/256 MB per DIMM pair, four pairs at most, so the
/// reachable totals are the same sums the IP28 makes — but built out of those
/// kits rather than out of MC banks.
const O2_RAM_PRESETS: &[u32] = &[32, 64, 96, 128, 192, 256, 384, 512, 768, 1024];

const OCTANE_RAM_PRESETS: &[u32] = &[64, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192];

pub fn ram_presets(profile: iris::config::MachineProfile) -> &'static [u32] {
    use iris::config::MachineProfile as P;
    match profile {
        P::OctaneIp30 => OCTANE_RAM_PRESETS,
        P::Origin200Ip29 | P::Origin2000Ip27 | P::Onyx2Ip27 => &[128, 512, 1024, 2048],
        P::FuelIp34 => &[512],
        P::TezroIp53 => &[4096, 1024, 2048, 8192],
        P::Origin3200Ip35 | P::Onyx3000Ip35 => &[512, 1024, 2048, 4096],
        P::Indigo2Ip28 => IP28_RAM_PRESETS,
        P::O2Ip32 => O2_RAM_PRESETS,
        _ => RAM_PRESETS,
    }
}

/// Tooltip for a bank size `profile` cannot take.
pub fn bank_size_tip(profile: iris::config::MachineProfile) -> String {
    use iris::config::MachineProfile as P;
    match profile {
        P::OctaneIp30 => t!("ram-bank-octane-sizes"),
        P::O2Ip32 => t!("ram-bank-o2-sizes"),
        _ => t!("config-ram-bank-ip28-only"),
    }
}

pub fn active_banks(banks: &[u32]) -> usize {
    banks.iter().filter(|&&s| s > 0).count()
}

pub fn ram_summary(banks: &[u32]) -> String {
    let total: u32 = banks.iter().sum();
    let n = active_banks(banks);
    t!("ram-summary", total = total, banks = n)
}

/// Every total `cfg`'s machine can hold, smallest first, each with the bank
/// layout that makes it: the fewest banks, the largest first, among the
/// layouts `MachineConfig::validate` accepts for this machine and processor.
/// The Machine page's memory slider offers exactly these.
pub fn layouts(cfg: &iris::config::MachineConfig) -> Vec<(u32, Vec<u32>)> {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Mutex;
    static CACHE: Mutex<Option<HashMap<(String, String), Vec<(u32, Vec<u32>)>>>> = Mutex::new(None);
    let p = cfg.machine.profile;
    let key = (p.name().to_string(), cfg.cpu().name());
    if let Some(v) = CACHE.lock().ok().and_then(|c| c.as_ref().and_then(|m| m.get(&key).cloned())) {
        return v;
    }
    let n = p.memory_banks();
    let mut sizes: Vec<u32> = p.bank_sizes().to_vec();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    // Non-increasing layouts of `n` banks: the largest bank first, empty ones last.
    fn walk(sizes: &[u32], n: usize, at: usize, cur: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
        if cur.len() == n {
            out.push(cur.clone());
            return;
        }
        for (i, s) in sizes.iter().enumerate().skip(at) {
            cur.push(*s);
            walk(sizes, n, i, cur, out);
            cur.pop();
        }
    }
    let mut candidates = Vec::new();
    walk(&sizes, n, 0, &mut Vec::new(), &mut candidates);
    let mut best: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for banks in candidates {
        let total: u32 = banks.iter().sum();
        if total == 0 {
            continue;
        }
        let mut t = cfg.clone();
        t.banks.iter_mut().for_each(|b| *b = 0);
        for (i, b) in banks.iter().enumerate() {
            if let Some(slot) = t.banks.get_mut(i) {
                *slot = *b;
            }
        }
        t.banks_set = true;
        if t.validate().is_err() {
            continue;
        }
        let used = |v: &Vec<u32>| v.iter().filter(|b| **b > 0).count();
        match best.get(&total) {
            Some(b) if used(b) <= used(&banks) => {}
            _ => {
                best.insert(total, banks);
            }
        }
    }
    let v: Vec<(u32, Vec<u32>)> = best.into_iter().collect();
    // Only a real answer is kept: an empty one can come from a config that
    // fails validate() for some other reason (a half-typed setting), and
    // would hide the memory slider for every machine of this type.
    if !v.is_empty() {
        if let Ok(mut c) = CACHE.lock() {
            c.get_or_insert_with(HashMap::new).insert(key, v.clone());
        }
    }
    v
}

/// The layout for the total nearest `mb` that the machine can hold.
pub fn layout_near(cfg: &iris::config::MachineConfig, mb: u32) -> Option<(u32, Vec<u32>)> {
    layouts(cfg).into_iter().min_by_key(|(t, _)| (i64::from(*t) - i64::from(mb)).abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use iris::config::{MachineConfig, MachineProfile};

    #[test]
    fn every_machine_has_layouts_and_each_one_validates() {
        for p in MachineProfile::ALL {
            let mut cfg = MachineConfig::default();
            let from = cfg.machine.profile;
            cfg.machine.profile = p;
            crate::model::hardware::switch_profile(&mut cfg, from);
            let all = layouts(&cfg);
            assert!(!all.is_empty(), "{p:?}");
            for (total, banks) in &all {
                assert_eq!(banks.iter().sum::<u32>(), *total);
                let mut t = cfg.clone();
                t.banks.iter_mut().for_each(|b| *b = 0);
                t.banks[..banks.len()].copy_from_slice(banks);
                t.banks_set = true;
                t.validate().unwrap_or_else(|e| panic!("{p:?} {banks:?}: {e}"));
            }
        }
    }

    #[test]
    fn a_total_between_two_takes_the_nearer() {
        let cfg = MachineConfig::default();
        let (total, banks) = layout_near(&cfg, 200).unwrap();
        assert!(layouts(&cfg).iter().any(|(t, _)| *t == total));
        assert_eq!(banks.iter().sum::<u32>(), total);
    }
}
