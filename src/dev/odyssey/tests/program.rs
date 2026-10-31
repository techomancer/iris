//! Recognising the geometry program the GL library sets.

use super::*;

/// Set the three entries and the list at index 0 the way the library's
/// program words do, and let the board recognise the program.
fn set_program(s: &mut Board, list: &[u32], k: u32, entries: [u32; 3]) {
    s.gl_const[..list.len()].copy_from_slice(list);
    s.gl_entry[ODSY_ENTRY_29] = entries[0];
    s.gl_entry[ODSY_ENTRY_2A] = entries[1];
    s.gl_entry[ODSY_ENTRY_2C] = entries[2];
    s.gl_entry[ODSY_ENTRY_33] = k;
    s.gl_program();
}

/// The older GL libraries number every routine from 891 on 99 lower, their
/// program image having no spot routine. A one-light record and an
/// assembled two-light list in that numbering light as their current forms
/// do, and the current numbering still wins where it means something.
///
/// QEMU: command-fifo-gl-program-older-numbering.
#[test]
fn the_older_libraries_numbering_lights_their_programs() {
    let (mut s, _) = board("v12");

    // Record 3 of the current numbering is 0x5cb 0x5d8 0x5e5; the older
    // image's entries are 99 lower.
    set_program(&mut s, &[0x41e, 0x203, 0x467, 0x462], 0, [0x568, 0x575, 0x582]);
    assert!(s.gl_lit, "an older-numbered resident lit record is recognised");

    // An assembled list: the head 0x290 is below the spot routine, so it is
    // the same in both numberings; the routines past it move up by 99.
    let list = [0x290, 0x2b6, 0x2cf, 0x335, 0x2de, 0x300, 0x392, 0x437, 0x203, 0x25e, 0x462];
    set_program(&mut s, &list, 1, [0x290, 0x462, 0x462]);
    assert!(s.gl_lit, "an older-numbered assembled list is recognised");
    assert!(!s.gl_local_viewer);

    // The current numbering of the same record is recognised as before.
    set_program(&mut s, &[], 0, [0x5cb, 0x5d8, 0x5e5]);
    assert!(s.gl_lit);
}

/// The model cannot tell which library runs, so a program that means
/// something in the current numbering is never remapped, and an entry past
/// the older image's 2,266 records is no entry of it.
#[test]
fn the_older_numbering_is_tried_only_where_the_current_one_fails() {
    let (mut s, _) = board("v12");

    // 0x568 0x575 0x582 is not a current record, and 0x9f0 is past the
    // older image's end, so nothing recognises these.
    set_program(&mut s, &[], 0, [0x568, 0x575, 0x9f0]);
    assert!(!s.gl_lit);

    // The unlit record's entries sit below the spot routine in both numberings.
    set_program(&mut s, &[ODSY_UCODE_END], 0, [ODSY_UCODE_UNLIT, ODSY_UCODE_UNLIT_2A, ODSY_UCODE_UNLIT_2C]);
    assert!(!s.gl_lit);
}
