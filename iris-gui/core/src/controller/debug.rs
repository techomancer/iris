//! Diagnostics: which features this build has (D13: shown to the user, in
//! About and Help > Diagnostics), and where the machine's debug ports listen.

use iris::config::MachineConfig;

/// One build feature and its state, as the user reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Feature {
    pub name: &'static str,
    pub state: String,
}

/// The build features worth telling a user about. The JITs are compiled in
/// but the App Store build turns them off at run time (IRIS_NO_JIT), so this
/// reports what actually runs.
pub fn build_features() -> Vec<Feature> {
    use iris::build_features as bf;
    let jit_off = std::env::var_os("IRIS_NO_JIT").is_some();
    let jit = |on: bool| {
        if !on {
            t!("menu-help-feature-off")
        } else if jit_off {
            t!("menu-help-feature-off-sandbox")
        } else {
            t!("menu-help-feature-on")
        }
    };
    vec![
        Feature { name: "rex-jit", state: jit(bf::REX_JIT) },
        Feature { name: "gr2-jit", state: jit(bf::GR2_JIT) },
        Feature { name: "gr4-jit", state: jit(bf::GR4_JIT) },
        Feature { name: "jitv2", state: jit(bf::JITV2) },
        Feature {
            name: "lightning",
            state: if bf::LIGHTNING { t!("menu-help-feature-on-no-debug") } else { t!("menu-help-feature-off") },
        },
        Feature { name: "pcap", state: if bf::PCAP { t!("menu-help-feature-on") } else { t!("menu-help-feature-off") } },
        Feature {
            name: "hostcall",
            state: if bf::HOSTCALL { t!("menu-help-feature-on") } else { t!("menu-help-feature-off") },
        },
    ]
}

/// The build features as text lines, for About, Diagnostics and a bug report.
pub fn diagnostics_text() -> String {
    build_features().iter().map(|f| t!("menu-help-feature", name = f.name, state = f.state.as_str())).collect::<Vec<_>>().join("\n")
}

/// The monitor console's TCP port for `cfg`.
pub fn monitor_port(cfg: &MachineConfig) -> u16 {
    cfg.monitor_port.unwrap_or(iris::monitor::DEFAULT_PORT)
}

/// Whether this build has the debug paths (the monitor's breakpoints, GDB):
/// a lightning build compiles them out.
pub fn debug_available() -> bool {
    !iris::build_features::LIGHTNING
}
