//! Perfetto instrumentation macros and edge-tracker types (see
//! `src/perfetto/mod.rs`).
//!
//! Always compiled, unlike the `perfetto` module itself, so call sites build
//! with or without `--features perfetto`; without it every macro expands to
//! nothing (arguments are not evaluated) and the tracker types are
//! zero-sized.
//
// Each macro takes an `EventDef` place expression with a 'static lifetime
// (`events::VSYNC`, `events::IOC_LINES[3]`) and an optional argument
// (anything `as u64` accepts). The `{ #[cfg] {..} }` shape keeps them valid
// in both statement and expression position.

/// A level drawn as a slice (`pf_level!`). Holds the session generation in
/// which the open slice began; 0 = none.
#[derive(Default)]
pub struct PfLevel {
    #[cfg(feature = "perfetto")]
    pub(crate) open: std::sync::atomic::AtomicU32,
}

impl PfLevel {
    pub const fn new() -> Self {
        Self {
            #[cfg(feature = "perfetto")]
            open: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

/// Up to 32 levels at once, one slice track per bit (`pf_bits!`).
#[derive(Default)]
pub struct PfBits {
    #[cfg(feature = "perfetto")]
    pub(crate) bits: std::sync::atomic::AtomicU32,
    #[cfg(feature = "perfetto")]
    pub(crate) gen: std::sync::atomic::AtomicU32,
}

impl PfBits {
    pub const fn new() -> Self {
        Self {
            #[cfg(feature = "perfetto")]
            bits: std::sync::atomic::AtomicU32::new(0),
            #[cfg(feature = "perfetto")]
            gen: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

/// A counter emitted only when it changes (`pf_value!`), for values sampled
/// far more often than they move (a FIFO level read on every status poll).
#[derive(Default)]
pub struct PfValue {
    #[cfg(feature = "perfetto")]
    pub(crate) last: std::sync::atomic::AtomicI64,
    #[cfg(feature = "perfetto")]
    pub(crate) gen: std::sync::atomic::AtomicU32,
}

impl PfValue {
    pub const fn new() -> Self {
        Self {
            #[cfg(feature = "perfetto")]
            last: std::sync::atomic::AtomicI64::new(0),
            #[cfg(feature = "perfetto")]
            gen: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

/// Instant event.
#[macro_export]
macro_rules! pf_instant {
    ($def:expr) => { $crate::pf_instant!($def, 0u64) };
    ($def:expr, $arg:expr) => {{
        #[cfg(feature = "perfetto")]
        {
            if $crate::perfetto::on(&$def) {
                $crate::perfetto::emit(&$def, $crate::perfetto::Kind::Instant, ($arg) as u64);
            }
        }
    }};
}

/// Open a slice. Pair with `pf_end!` on the same track (for a `Global`
/// track, possibly from another thread).
#[macro_export]
macro_rules! pf_begin {
    ($def:expr) => { $crate::pf_begin!($def, 0u64) };
    ($def:expr, $arg:expr) => {{
        #[cfg(feature = "perfetto")]
        {
            if $crate::perfetto::on(&$def) {
                $crate::perfetto::emit(&$def, $crate::perfetto::Kind::Begin, ($arg) as u64);
            }
        }
    }};
}

/// Close the innermost open slice on `$def`'s track.
#[macro_export]
macro_rules! pf_end {
    ($def:expr) => {{
        #[cfg(feature = "perfetto")]
        {
            if $crate::perfetto::on(&$def) {
                $crate::perfetto::emit(&$def, $crate::perfetto::Kind::End, 0);
            }
        }
    }};
}

/// Set the counter track named after `$def` to `$value` (signed).
#[macro_export]
macro_rules! pf_counter {
    ($def:expr, $value:expr) => {{
        #[cfg(feature = "perfetto")]
        {
            if $crate::perfetto::on(&$def) {
                $crate::perfetto::emit(&$def, $crate::perfetto::Kind::Counter, ($value) as i64 as u64);
            }
        }
    }};
}

/// Slice covering the rest of the enclosing block. Statement position only.
#[macro_export]
macro_rules! pf_scope {
    ($def:expr) => { $crate::pf_scope!($def, 0u64) };
    ($def:expr, $arg:expr) => {
        #[cfg(feature = "perfetto")]
        let _pf_scope = $crate::perfetto::Scope::begin(&$def, ($arg) as u64);
    };
}

/// Drive a `PfLevel`: `pf_level!(tracker, DEF, is_high [, arg])`. The begin
/// carries `arg`.
#[macro_export]
macro_rules! pf_level {
    ($tracker:expr, $def:expr, $level:expr) => { $crate::pf_level!($tracker, $def, $level, 0u64) };
    ($tracker:expr, $def:expr, $level:expr, $arg:expr) => {{
        #[cfg(feature = "perfetto")]
        { $tracker.set(&$def, $level, ($arg) as u64); }
    }};
}

/// Drive a `PfBits`: `pf_bits!(tracker, DEFS, bits)` where `DEFS` is a
/// `[EventDef; 32]` with one entry per bit.
#[macro_export]
macro_rules! pf_bits {
    ($tracker:expr, $defs:expr, $bits:expr) => {{
        #[cfg(feature = "perfetto")]
        { $tracker.update(&$defs, ($bits) as u32); }
    }};
}

/// Drive a `PfValue`: `pf_value!(tracker, DEF, value)`.
#[macro_export]
macro_rules! pf_value {
    ($tracker:expr, $def:expr, $value:expr) => {{
        #[cfg(feature = "perfetto")]
        { $tracker.set(&$def, ($value) as i64); }
    }};
}
