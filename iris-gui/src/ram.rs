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
