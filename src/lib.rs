#![allow(dead_code, unused_variables, unused_imports)]

#[cfg(all(feature = "lightning", feature = "developer"))]
compile_error!(
    "features `lightning` and `developer` are mutually exclusive: lightning strips \
     debuggability for speed (fixed jitv2 dispatch, no jitcheck/inline-compile \
     switching, etc.) while developer exists specifically to add it back — pick one"
);

/// Compile-time feature flags exposed for tooling (e.g. iris-gui) so it can
/// surface "CHD support required" / "camera support required" hints without
/// duplicating the cargo feature set.
pub mod build_features {
    pub const CHD:       bool = cfg!(feature = "chd");
    pub const CAMERA:    bool = cfg!(feature = "camera");
    pub const PCAP:      bool = cfg!(feature = "pcap");
    pub const JITV2:     bool = cfg!(feature = "jitv2");
    pub const REX_JIT:   bool = cfg!(feature = "rex-jit");
    /// N64 development board (Ultra64 GIO card) + POSIX shm bridge to an
    /// external gopher64. The GUI gates the "Enable dev board" toggle on this.
    pub const ULTRA64:   bool = cfg!(feature = "ultra64");
    /// DaynaPort SCSI/Link target — a SCSI-attached Ethernet adapter that can
    /// be configured on any SCSI id. The GUI gates the "DaynaPort" device kind
    /// on this.
    pub const DAYNAPORT: bool = cfg!(feature = "daynaport");
    /// Lightning build strips breakpoint checks and the traceback buffer
    /// from the MIPS executor hot path. Interactive debugging (GDB stub,
    /// monitor breakpoints) is non-functional in this build.
    pub const LIGHTNING: bool = cfg!(feature = "lightning");
    pub const IDLE_PAUSE: bool = cfg!(feature = "idle-pause");
    /// The emulated CPU, fixed at build time (the cache model differs deeply
    /// between the R4400 and R5000, so it's a compile-time choice, not a runtime
    /// setting). `r5k` selects the R5000; `r5ksc`/`r5ksc_triton` add a secondary
    /// cache. The GUI surfaces this read-only on the Memory tab.
    pub const CPU: &str = if cfg!(feature = "r5ksc_triton") {
        "MIPS R5000 (Triton on-die L2)"
    } else if cfg!(all(feature = "r5k", feature = "r5ksc")) {
        "MIPS R5000 (external L2)"
    } else if cfg!(feature = "r5k") {
        "MIPS R5000"
    } else {
        "MIPS R4400"
    };
}

pub mod config;
pub mod traits;
pub mod trace;
#[macro_use]
pub mod devlog;
pub mod timer;
pub mod prom;
pub mod prombin;
pub mod prombini2;
pub mod mips_isa;
pub mod mips_dis;
pub mod mips_core;
pub mod mips_tlb;
pub mod mips_cache_v2;
pub mod mips_exec;
pub mod mips_exec_test;
pub mod mips_instr_stats;
pub mod mem;
pub mod mc;
pub mod machine;
pub mod eeprom_93c56;
pub mod platform;
pub mod hpc3;
pub mod ioc;
pub mod physical;
pub mod ds1x86;
pub mod z85c30;
pub mod telnet;
pub mod monitor;
pub mod locks;
pub mod pit8254;
pub mod net;
pub mod nfsudp;
pub mod tftp;
pub mod testdev;
pub mod xdmcp;
#[cfg(feature = "pcap")]
pub mod net_pcap;
pub mod seeq8003;
#[cfg(feature = "daynaport")]
pub mod daynaport;
pub mod cow_disk;
#[cfg(feature = "chd")]
pub mod chd_disk;
pub mod scsi;
pub mod wd33c93a;
pub mod hal2;
pub mod ps2;
pub mod ui;
pub mod rex3;
pub mod rex3_simd;
pub mod compositor;
pub mod gl_compositor;
pub mod headless_gl;
pub mod debug_overlay;
pub mod vc2;
pub mod vc2_timings;
pub mod xmap9;
pub mod cmap;
pub mod bt445;
pub mod disp;
pub mod exp;
pub mod gdb_stub;
pub mod snapshot;
pub mod sgi_vh;
pub mod elf;
pub mod chunk_store;
pub mod validate;
pub mod registry;
pub mod thread_affinity;
pub mod perf_monitor;
pub mod ci;
pub mod hptimer;
pub mod hptimer_tests;
#[cfg(feature = "idle-pause")]
pub mod idle_park;
pub mod vga_font;
pub mod cdmc;
#[cfg(feature = "camera")]
pub mod camera;
pub mod saa7191;
pub mod video_source;
pub mod vino;
pub mod xz;
pub mod mgras;
pub mod ultra_proto;
pub mod ultra64;
#[cfg(feature = "rex-jit")]
pub mod rex3_jit;
#[cfg(feature = "jitv2")]
pub mod jitv2;
pub mod jit_feedback;

#[cfg(test)]
mod platform_profile_tests;