#![allow(dead_code, unused_variables, unused_imports)]

#[cfg(all(feature = "lightning", feature = "developer"))]
compile_error!(
    "features `lightning` and `developer` are mutually exclusive: lightning strips \
     debuggability for speed (fixed jitv2 dispatch, no jitcheck/inline-compile \
     switching, etc.) while developer exists specifically to add it back — pick one"
);

#[cfg(all(feature = "instr_stats", feature = "jitv2"))]
compile_error!(
    "features `instr_stats` and `jitv2` together are meaningless: instr_stats' \
     execution counters are interpreter-path only (see its own Cargo.toml doc \
     comment) and a jitv2 build routes the vast majority of instructions \
     through compiled code, bypassing them entirely — the resulting counts \
     would silently undercount almost everything rather than erroring. Build \
     instr_stats without jitv2 to get real interpreter-only counts."
);

#[cfg(feature = "r5ksc_triton")]
compile_error!(
    "`r5ksc_triton` models the O2's on-die R5000 L2, not a machine IRIS \
     emulates (IRIS targets Indy/Indigo2; an Indy R5000 board has the \
     external R4600SC-style secondary cache instead — that's `r5ksc`, which \
     is ALSO currently broken and separately refused below). It's also \
     unfinished on its own terms: `cargo test --features r5k,r5ksc_triton` \
     fails mips_cache_v2::tests::cache_op_index_inv_l1i. Refusing to build \
     rather than silently shipping a broken cache model for a machine this \
     emulator doesn't target. See src/mips_cache_v2.rs and \
     rules/testing/r5k-l1i-cache-bugs.md."
);

#[cfg(feature = "r5ksc")]
compile_error!(
    "`r5ksc` (external R4600SC-style secondary cache — what a real Indy \
     R5000 board has) does not currently work: `cargo test --features \
     r5k,r5ksc` fails mips_cache_v2's L1I tests. Plain `r5k` with no \
     secondary-cache feature at all also fails l1i_fetch_stress/ \
     l1i_l1d_coherence — there is currently no working R5000 \
     secondary-cache configuration. Refusing to build rather than silently \
     shipping a broken cache model. See src/mips_cache_v2.rs and \
     rules/testing/r5k-l1i-cache-bugs.md; `r5k` alone still builds (R5000 \
     CPU/FPU semantics are unaffected — only the L1I model under a \
     secondary-cache config is implicated) and is what \
     cpu-tests/run/matrix.sh's R5000 cell uses until this is fixed."
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
    // There is deliberately no `CPU` constant here any more. The emulated CPU
    // stopped being a build-time property in 96e5ddd: both cache models are
    // monomorphised into every binary and `Machine::new` picks between them
    // from `cfg.machine.cpu`. A constant derived from cargo features could only
    // report how the binary was compiled, which is no longer the same question
    // as which CPU is running — and it was being displayed to users as though
    // it were. Ask the config (`MachineConfig::machine.cpu`), or the guest,
    // which reads PRId.

    /// Every compile-time flag this binary was built with, in a fixed order.
    ///
    /// CPU model and execution engine come first: these change what the guest
    /// sees, not just how fast it sees it, and a benchmark result is meaningless
    /// without them. `r5k` in particular was missing from this list long enough
    /// that an R5000 build could report "build features: tlbvmap" and be taken
    /// for an R4400 — cpu-tests/run/matrix.sh has a whole guard against exactly
    /// that confusion.
    ///
    /// In the library rather than in `main.rs` because a saved benchmark result
    /// records this list, and the in-process runner has no startup banner to
    /// read it back out of.
    pub fn enabled() -> Vec<&'static str> {
        const FEATURES: &[(&str, bool)] = &[
            ("r5k", cfg!(feature = "r5k")),
            ("r5ksc", cfg!(feature = "r5ksc")),
            ("r5ksc_triton", cfg!(feature = "r5ksc_triton")),
            ("mips4", cfg!(feature = "mips4")),
            ("jitv2", cfg!(feature = "jitv2")),
            ("jitv2_opcodefusion", cfg!(feature = "jitv2_opcodefusion")),
            ("opcodefusion", cfg!(feature = "opcodefusion")),
            ("idle-pause", cfg!(feature = "idle-pause")),
            ("rex-jit", cfg!(feature = "rex-jit")),
            ("lightning", cfg!(feature = "lightning")),
            ("ppmem", cfg!(feature = "ppmem")),
            ("tlbvmap", cfg!(feature = "tlbvmap")),
            ("tlbstats", cfg!(feature = "tlbstats")),
            ("tlbcheck", cfg!(feature = "tlbcheck")),
            ("instr_stats", cfg!(feature = "instr_stats")),
            ("chd", cfg!(feature = "chd")),
            ("camera", cfg!(feature = "camera")),
            ("pcap", cfg!(feature = "pcap")),
            ("ci_clock", cfg!(feature = "ci_clock")),
            ("developer", cfg!(feature = "developer")),
            ("developer_ip7", cfg!(feature = "developer_ip7")),
            ("debug_cache", cfg!(feature = "debug_cache")),
        ];
        FEATURES.iter().filter(|(_, e)| *e).map(|(n, _)| *n).collect()
    }

    /// `enabled()` as the emulator prints it at startup.
    pub fn banner() -> String {
        let on = enabled();
        if on.is_empty() { "(none)".to_string() } else { on.join(" ") }
    }
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
#[cfg(feature = "ppmem")]
pub mod ppmem;
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
pub mod bench_report;
pub mod benchsuite;
pub mod bench_runner;
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
#[cfg(all(feature = "jitv2", not(feature = "j2wp")))]
pub mod jitv2_html_default;
#[cfg(all(feature = "jitv2", feature = "j2wp"))]
pub mod jitv2_html_j2wp;
pub mod jit_feedback;

#[cfg(test)]
mod platform_profile_tests;