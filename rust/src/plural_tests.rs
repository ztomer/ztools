use super::{counted, noun, noun_as};

#[test]
fn one_is_singular_and_every_other_count_is_plural() {
    assert_eq!(counted(1_usize, "file"), "1 file");
    assert_eq!(counted(0_usize, "file"), "0 files");
    assert_eq!(counted(2_u32, "file"), "2 files");
    assert_eq!(counted(11_u64, "task"), "11 tasks");
}

#[test]
fn the_noun_alone_follows_the_same_rule() {
    assert_eq!(noun(1_usize, "row"), "row");
    assert_eq!(noun(3_usize, "row"), "rows");
}

#[test]
fn an_irregular_noun_names_its_plural() {
    assert_eq!(noun_as(1_usize, "entry", "entries"), "entry");
    assert_eq!(noun_as(2_usize, "entry", "entries"), "entries");
}
