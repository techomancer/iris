//! The generated Hub offsets and the `regs!` extras the model leans on,
//! checked against the manual's values.

use super::*;

#[test]
fn offsets_and_values_match_the_manual() {
    assert_eq!(PI_CPU_NUM::OFFSET, 0x20);
    assert_eq!(MD_MEMORY_CONFIG::OFFSET, 0x200018);
    assert_eq!(NI_STATUS_REV_ID::OFFSET, 0x600000);
    assert_eq!(IIO_WID::OFFSET, 0x400000);
    assert_eq!(IIO_ICRB_C::at(7), 0x4004f0);
    assert_eq!(IIO_ICRB_C::index(0x4004f0), Some(7));
    assert_eq!(IIO_ICRB_C::index(0x4004f8), None);
    use IIO_ILCSR::LNK_STAT::Value;
    assert_eq!(Value::from_bits(0x2000), Some(Value::WORKING));
    assert_eq!(Value::from_bits(0x1000), None);
    assert_eq!(Value::WORKING.bits(), 2);
}
