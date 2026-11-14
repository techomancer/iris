//! `regs!`, the form generated register definitions take (the tests show
//! one). The vocabulary is QEMU's registerfields.h and RegisterAccessInfo.
//! Each register line becomes a module of consts:
//!
//! - `OFFSET` (from the unit's `BASE`), `WIDTH` in bits, `RESET` when the
//!   map knows it, and the `RO`, `W1C`, `COR` and `RSVD` masks (0 when the
//!   map lists none). Writable bits are `!(RO | RSVD)`: there is no mask
//!   for them.
//! - an array adds `COUNT`, `STRIDE`, `END`, `at(i)` and `index(off)`.
//! - each field is a module with `SHIFT`, `LENGTH` and `MASK` (in place),
//!   and, when the map lists values for it, a `u64` const per value and a
//!   `Value` enum.
//!
//! `unit NAME base N { }` becomes a module with `BASE`; `const NAME = V;`
//! a `u64` const. Everything is const: no traits and no crates.

macro_rules! regs {
    // A file of units, each with its own base.
    (
        $( unit $unit:ident base $ubase:literal { $($body:tt)* } )+
    ) => {
        $(
            #[allow(non_snake_case, dead_code)]
            pub mod $unit {
                regs! { base $ubase; $($body)* }
            }
        )+
    };
    // A flat file: one base, constants, then registers. A constant and a
    // register name both start with an ident, so the constants are taken one
    // by one and the registers in one repetition. That recurses once per
    // constant: see the recursion limit in lib.rs.
    ( base $base:literal; $($rest:tt)* ) => {
        pub const BASE: u64 = $base;
        regs! { @consts $($rest)* }
    };
    ( @consts const $cname:ident = $cval:literal; $($rest:tt)* ) => {
        pub const $cname: u64 = $cval;
        regs! { @consts $($rest)* }
    };
    ( @consts $($rest:tt)* ) => { regs! { @registers $($rest)* } };
    (
        @registers
        $(
            $name:ident $off:literal width $width:literal
            $(reset $reset:literal)? $(ro $ro:literal)? $(w1c $w1c:literal)?
            $(cor $cor:literal)? $(rsvd $rsvd:literal)?
            $(count $count:literal stride $stride:literal end $end:literal)?
            $({ $( $field:ident : $lo:literal .. $hi:literal $({ $($vname:ident = $vval:literal),+ $(,)? })? ),* $(,)? })?;
        )*
    ) => {
        $(
            #[allow(non_snake_case, dead_code)]
            pub mod $name {
                pub const OFFSET: u64 = $off;
                pub const WIDTH: u32 = $width;
                $(pub const RESET: u64 = $reset;)?
                pub const RO: u64 = 0 $(| $ro)?;
                pub const W1C: u64 = 0 $(| $w1c)?;
                pub const COR: u64 = 0 $(| $cor)?;
                pub const RSVD: u64 = 0 $(| $rsvd)?;
                $(
                    pub const COUNT: usize = $count;
                    pub const STRIDE: u64 = $stride;
                    pub const END: u64 = $end;
                    const _: () = assert!(OFFSET + COUNT as u64 * STRIDE == END);

                    /// Offset of element `i`.
                    pub const fn at(i: usize) -> u64 {
                        assert!(i < COUNT);
                        OFFSET + i as u64 * STRIDE
                    }

                    /// The element whose bytes hold `off`; `None` off the
                    /// array or between its elements.
                    pub const fn index(off: u64) -> Option<usize> {
                        if off < OFFSET || off >= END {
                            return None;
                        }
                        let d = off - OFFSET;
                        if d % STRIDE >= WIDTH as u64 / 8 {
                            return None;
                        }
                        Some((d / STRIDE) as usize)
                    }
                )?
                $($(
                    #[allow(non_snake_case, dead_code)]
                    pub mod $field {
                        pub const SHIFT: u32 = $lo;
                        pub const LENGTH: u32 = $hi - $lo + 1;
                        pub const MASK: u64 = (u64::MAX >> (64 - LENGTH)) << SHIFT;
                        $(
                            #[repr(u8)]
                            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
                            #[allow(non_camel_case_types)]
                            pub enum Value { $($vname = $vval),+ }
                            $(pub const $vname: u64 = $vval;)+

                            impl Value {
                                /// `None` means not a value this map lists.
                                pub const fn from_bits(v: u64) -> Option<Self> {
                                    match (v & MASK) >> SHIFT {
                                        $($vval => Some(Self::$vname),)+
                                        _ => None,
                                    }
                                }

                                /// The field value, unshifted.
                                pub const fn bits(self) -> u64 {
                                    self as u64
                                }
                            }
                        )?
                    }
                )*)?
            }
        )*
    };
}

/// The union of the in-place masks of the named fields of a register:
/// `field_masks!(reg::LINK_CONTROL; MAX_BURST, NULL_TIMEOUT)`.
macro_rules! field_masks {
    ($reg:path ; $($field:ident),+ $(,)?) => {{
        use $reg as r;
        0 $(| r::$field::MASK)+
    }};
}

#[cfg(test)]
mod tests {
    mod flat {
        regs! {
            base 0x400000;
            const KEY = 0x53474972756c6573;
            WIDGET_ID 0x0000 width 64 reset 0x1c110049 { REVISION: 28..31, PART: 12..27, MANUFACTURER: 1..11 };
            PROTECTION_OVERRIDE 0x0108 width 64;
            SCRATCH 0x0150 width 64 reset 0x0 count 2 stride 0x8 end 0x160;
            CRB_C 0x0410 width 64 reset 0x0 ro 0xf0 w1c 0x100 cor 0x6 rsvd 0x1000 count 15 stride 0x20 end 0x5f0;
            LINK_CONTROL 0x0128 width 64 reset 0x3fc03ff040a {
                MAX_BURST: 32..41,
                LINK_STATUS: 12..13 { LINK_DOWN = 0, LINK_WORKING = 2 },
                WHOLE: 0..63,
            };
            // A field named like a register constant shares no namespace with it.
            CLASH 0x0200 width 64 reset 0x5 { RESET: 0..3, COUNT: 4..7 };
        }
    }
    mod units {
        regs! {
            unit ii base 0x400000 { WIDGET_ID 0x0000 width 64 reset 0x1c110049; }
            unit pi base 0x0 { const SHIFT = 33; CPU 0x20 width 32; }
        }
    }
    use flat::*;

    #[test]
    fn a_register_expands_to_its_offset_width_reset_and_masks() {
        assert_eq!(BASE, 0x40_0000);
        assert_eq!(KEY, 0x5347_4972_756c_6573);
        assert_eq!((WIDGET_ID::OFFSET, WIDGET_ID::WIDTH, WIDGET_ID::RESET), (0, 64, 0x1c11_0049));
        assert_eq!((WIDGET_ID::RO, WIDGET_ID::W1C, WIDGET_ID::COR, WIDGET_ID::RSVD), (0, 0, 0, 0));
        assert_eq!(PROTECTION_OVERRIDE::OFFSET, 0x108);
        assert_eq!((CRB_C::RO, CRB_C::W1C, CRB_C::COR, CRB_C::RSVD), (0xf0, 0x100, 0x6, 0x1000));
    }

    #[test]
    fn a_field_has_its_shift_length_and_in_place_mask() {
        assert_eq!((WIDGET_ID::REVISION::SHIFT, WIDGET_ID::REVISION::LENGTH), (28, 4));
        assert_eq!(WIDGET_ID::REVISION::MASK, 0xf000_0000);
        assert_eq!((WIDGET_ID::PART::MASK, WIDGET_ID::MANUFACTURER::MASK), (0x0fff_f000, 0xffe));
        assert_eq!(LINK_CONTROL::MAX_BURST::MASK, 0x3ff << 32);
        assert_eq!(LINK_CONTROL::WHOLE::MASK, u64::MAX);
        assert_eq!((CLASH::RESET, CLASH::RESET::MASK, CLASH::COUNT::MASK), (5, 0xf, 0xf0));
    }

    #[test]
    fn listed_field_values_convert_both_ways() {
        use LINK_CONTROL::LINK_STATUS::Value;
        assert_eq!(Value::from_bits(0x2000 | 0x40a), Some(Value::LINK_WORKING));
        assert_eq!(Value::from_bits(0x400), Some(Value::LINK_DOWN));
        assert_eq!(Value::from_bits(0x1000), None);
        assert_eq!((Value::LINK_WORKING.bits(), Value::LINK_DOWN.bits()), (2, 0));
        assert_eq!((LINK_CONTROL::LINK_STATUS::LINK_WORKING, LINK_CONTROL::LINK_STATUS::LINK_DOWN), (2, 0));
    }

    #[test]
    fn an_array_gives_its_extent_element_offsets_and_decode() {
        assert_eq!((SCRATCH::COUNT, SCRATCH::STRIDE, SCRATCH::END), (2, 8, 0x160));
        assert_eq!((SCRATCH::at(0), SCRATCH::at(1)), (0x150, 0x158));
        assert_eq!((SCRATCH::index(0x150), SCRATCH::index(0x15c), SCRATCH::index(0x160)), (Some(0), Some(1), None));
        assert_eq!(SCRATCH::index(0x14f), None);
        // 8-byte elements 0x20 apart: the gap between them decodes to nothing.
        assert_eq!((CRB_C::index(0x430), CRB_C::index(0x438), CRB_C::index(0x5f0)), (Some(1), None, None));
    }

    #[test]
    fn field_masks_unions_the_named_fields() {
        assert_eq!(field_masks!(WIDGET_ID; REVISION, MANUFACTURER), 0xf000_0ffe);
    }

    #[test]
    fn units_become_modules_with_their_own_base() {
        assert_eq!((units::ii::BASE, units::ii::WIDGET_ID::OFFSET), (0x40_0000, 0));
        assert_eq!((units::pi::BASE, units::pi::SHIFT, units::pi::CPU::WIDTH), (0, 33, 32));
    }
}
