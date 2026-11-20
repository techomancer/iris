//! The generated Bedrock offsets and the `regs!` extras the model leans on,
//! checked against the manual's values.

use super::*;

#[test]
fn offsets_and_arrays_match_the_manual() {
    assert_eq!(pi::CPU_NUM::OFFSET, 0x20);
    assert_eq!(md::MEMORY_CONFIG::OFFSET, 0x8);
    assert_eq!(ni::PORT_RESET::OFFSET, 0x8);
    assert_eq!(io::LINK_CONTROL::OFFSET, 0x128);
    assert_eq!(lb::SOFT_RESET::OFFSET, 0x4018);
    assert_eq!(xb::XB_FIRST_ERROR::OFFSET, 0x90);
    assert_eq!(io::SCRATCH::at(1), 0x158);
    assert_eq!(io::SCRATCH::index(0x158), Some(1));
    assert_eq!(io::SCRATCH::index(0x160), None);
}
