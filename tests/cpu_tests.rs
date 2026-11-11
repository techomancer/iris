//! cpu-tests (`cpu-tests/`) under the emulator this build produced: one case
//! per CPU, for the engine the build was compiled with.
//!
//!     cargo test --release --test cpu_tests -- --ignored
//!     cargo test --release --features jitv2 --test cpu_tests -- --ignored
//!
//! The guest binary needs the pinned MIPS cross compiler
//! (`cpu-tests/docs/toolchain.md`) and is not checked in. The cases read it
//! from `IRIS_CPUTEST_ELF`, else `cpu-tests/build/cputest.elf`. They are
//! `#[ignore]`d, so a plain `cargo test` reports them as ignored rather than
//! building nothing and calling it a pass; asked for explicitly with no guest
//! binary, they fail and say how to build one.
//!
//! A cell passes when exactly the tests listed for it in
//! `cpu-tests/expected-failures.txt` fail and every planned test reported;
//! the failure names each test that broke the rule. The R10000, R12000 and
//! R14000 run on the same Indy-shaped machine as the others, which iris
//! allows any CPU under the test device; no real Indy carried one, so their
//! expectations come from the manuals (`cpu-tests/docs/r10000.md`).

mod common;

use common::cputest::*;
use common::*;

fn cell(cpu: &str) {
    require_optimised_build();
    let cell = format!("{}-{}", cpu, engine());
    let elf = guest_elf_path();
    assert!(
        elf.is_file(),
        "no cpu-tests guest at {}: build it with `make -C cpu-tests` in the toolchain \
         container (cpu-tests/docs/toolchain.md) or point IRIS_CPUTEST_ELF at one",
        elf.display()
    );
    let run = run_guest(cpu, &elf).unwrap_or_else(|e| panic!("{cell}: {e}"));
    let errs = run.errors(&cell);
    let reported = run.tap.results.len();
    let failed = run.tap.results.iter().filter(|(_, ok)| !ok).count();
    if let Some(path) = junit_path(&format!("cpu-tests-{cell}")) {
        iris::junit::write(junit_suite(&cell, &run.tap, &errs), &path).expect("write the JUnit report");
    }
    if !errs.is_empty() {
        panic!(
            "{}\n{reported} reported, {failed} failed (log: {})\n{}\n--- last 20 lines ---\n{}",
            headline(&cell, &errs),
            run.log.display(),
            errs.iter().map(|e| format!("check: {e}")).collect::<Vec<_>>().join("\n"),
            tail(&run.stdout, 20)
        );
    }
    println!("check: {cell} ok ({reported} reported, {failed} expected failures)");
}

#[test]
#[ignore = "bare-metal suite: needs an optimised build and the cpu-tests guest; cargo test --release --test cpu_tests -- --ignored"]
fn r4400() {
    cell("r4400");
}

#[test]
#[ignore = "bare-metal suite: needs an optimised build and the cpu-tests guest; cargo test --release --test cpu_tests -- --ignored"]
fn r5000() {
    cell("r5000");
}

#[test]
#[ignore = "bare-metal suite: needs an optimised build and the cpu-tests guest; cargo test --release --test cpu_tests -- --ignored"]
fn r10000() {
    cell("r10000");
}

#[test]
#[ignore = "bare-metal suite: needs an optimised build and the cpu-tests guest; cargo test --release --test cpu_tests -- --ignored"]
fn r12000() {
    cell("r12000");
}

#[test]
#[ignore = "bare-metal suite: needs an optimised build and the cpu-tests guest; cargo test --release --test cpu_tests -- --ignored"]
fn r14000() {
    cell("r14000");
}

mod parser {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn tap_names_end_at_the_first_space_and_skips_pass() {
        let t = parse_tap("junk\r\n1..3\r\nok 1 - a/b\r\nnot ok 2 - c/d\r\nok 3 - e/f # SKIP\r\n");
        assert_eq!(t.plan, Some(3));
        assert_eq!(
            t.results,
            vec![("a/b".into(), true), ("c/d".into(), false), ("e/f".into(), true)]
        );
        assert_eq!(t.skipped, vec![false, false, true]);
    }

    const DONE: &str = "IRIS-CPUTEST-DONE rc=1\n";

    fn want() -> BTreeSet<String> {
        ["c/d".to_string()].into()
    }

    #[test]
    fn only_listed_failures_in_a_complete_run_pass() {
        let t = parse_tap(&format!("1..2\nok 1 - a/b\nnot ok 2 - c/d\n{DONE}"));
        assert!(t.done);
        assert!(judge("x", &t, &want()).is_empty());
        let t = parse_tap(&format!("1..2\nnot ok 1 - a/b\nok 2 - c/d\n{DONE}"));
        assert_eq!(judge("x", &t, &want()).len(), 2, "a new failure and a new pass");
    }

    #[test]
    fn a_run_with_no_plan_fails() {
        let t = parse_tap(&format!("ok 1 - a/b\nnot ok 2 - c/d\n{DONE}"));
        assert_eq!(judge("x", &t, &want()), vec!["no plan line: the guest never reached its first test"]);
    }

    #[test]
    fn a_run_short_of_its_plan_fails() {
        let t = parse_tap(&format!("1..3\nok 1 - a/b\nnot ok 2 - c/d\n{DONE}"));
        assert_eq!(judge("x", &t, &want()), vec!["2 of 3 tests reported: the run stopped early"]);
    }

    #[test]
    fn a_run_past_its_plan_fails() {
        let t = parse_tap(&format!("1..1\nok 1 - a/b\nnot ok 2 - c/d\n{DONE}"));
        assert_eq!(judge("x", &t, &want()), vec!["2 tests reported against a plan of 1"]);
    }

    #[test]
    fn a_run_with_no_done_line_fails() {
        // Every planned result, then the emulator died before the summary.
        let t = parse_tap("1..2\nok 1 - a/b\nnot ok 2 - c/d\n");
        assert_eq!(
            judge("x", &t, &want()),
            vec!["no IRIS-CPUTEST-DONE line: the guest never finished its run"]
        );
    }

    #[test]
    fn a_repeated_result_line_fails() {
        let t = parse_tap(&format!("1..2\nok 1 - a/b\nok 1 - a/b\n{DONE}"));
        let errs = judge("x", &t, &want());
        assert!(
            errs.iter().any(|e| *e == "result 2 is numbered 1: a result line is repeated or missing"),
            "{errs:?}"
        );
    }

    #[test]
    fn junit_has_a_case_per_guest_test() {
        use iris::junit::TestResult;
        // a/b fails unexpectedly, c/d is an expected failure, e/f is skipped,
        // and the run never printed its done line.
        let t = parse_tap("1..3\nnot ok 1 - a/b\nnot ok 2 - c/d\nok 3 - e/f # SKIP\n");
        let errs = judge("r4400-interp", &t, &want());
        assert_eq!(headline("r4400-interp", &errs), "r4400-interp: 1 failing: a/b");
        let suite = junit_suite("r4400-interp", &t, &errs);
        assert_eq!((suite.tests(), suite.failures(), suite.skipped()), (4, 2, 2));
        let cases: Vec<(&str, String)> = suite
            .testcases
            .iter()
            .map(|c| {
                assert_eq!(c.classname.as_deref(), Some("cpu-tests.r4400.interp"));
                let m = match &c.result {
                    TestResult::Failure { message, .. } => format!("fail: {message}"),
                    TestResult::SkippedWithCause { message, .. } => format!("skip: {message}"),
                    r => format!("{r:?}"),
                };
                (c.name.as_str(), m)
            })
            .collect();
        assert_eq!(
            cases,
            [
                ("a/b", "fail: failed and is not an expected failure on r4400-interp".into()),
                ("c/d", "skip: expected failure (expected-failures.txt)".into()),
                ("e/f", "skip: skipped on this CPU".into()),
                ("run", "fail: no IRIS-CPUTEST-DONE line: the guest never finished its run".into()),
            ]
        );
    }
}
