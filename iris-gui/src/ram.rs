//! RAM bank helpers shared by the Memory tab, menus, and status readouts.

/// Quick preset totals (MB) for the Memory menu and new-machine dialog.
const RAM_PRESETS: &[u32] = &[32, 64, 96, 128, 192, 256, 384, 512];

const IP28_RAM_PRESETS: &[u32] = &[32, 64, 96, 128, 192, 256, 384, 512, 768, 1024];

pub fn ram_presets(ip28: bool) -> &'static [u32] {
    if ip28 { IP28_RAM_PRESETS } else { RAM_PRESETS }
}

pub fn active_banks(banks: &[u32; 4]) -> usize {
    banks.iter().filter(|&&s| s > 0).count()
}

pub fn ram_summary(banks: &[u32; 4]) -> String {
    let total: u32 = banks.iter().sum();
    let n = active_banks(banks);
    format!("{total} MB ({n} bank{})", if n == 1 { "" } else { "s" })
}
