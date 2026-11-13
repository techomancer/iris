//! JUnit reports for CI test reporters, from the host tools that run guests
//! (`iris-bench run --junit`, the bare-metal suite tests): one case per guest
//! test, so a report names the guest test that failed rather than only the
//! command that ran it. A thin layer over the junit-report crate.

use std::path::Path;

pub use junit_report::{Duration, TestCase, TestResult, TestSuite};

/// The class a cell's cases are filed under: `<suite>.<cpu>.<engine>` for
/// the cell label `<cpu>-<engine>`.
pub fn classname(suite: &str, cell: &str) -> String {
    format!("{suite}.{}", cell.replace('-', "."))
}

/// One suite named `class`, with every case filed under that class.
pub fn suite(class: &str, cases: impl IntoIterator<Item = TestCase>) -> TestSuite {
    let mut suite = TestSuite::new(class);
    for mut case in cases {
        case.set_classname(class);
        suite.add_testcase(case);
    }
    suite
}

/// Write `suite` as a JUnit report at `path`, creating its directory.
pub fn write(suite: TestSuite, path: &Path) -> Result<(), String> {
    let err = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| err(&e))?;
    }
    let mut report = junit_report::Report::new();
    report.add_testsuite(suite);
    let file = std::fs::File::create(path).map_err(|e| err(&e))?;
    report.write_xml(file).map_err(|e| err(&e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_well_formed_escaped_cases_under_one_class() {
        let class = classname("bench", "r4400-interp");
        assert_eq!(class, "bench.r4400.interp");
        let s = suite(&class, [
            TestCase::success("int/alu", Duration::ZERO),
            TestCase::failure("a<b>&\"c'", Duration::ZERO, "checksum", "0x1 != golden 0x2"),
        ]);
        let dir = std::env::temp_dir().join(format!("iris-junit-test-{}", std::process::id()));
        let path = dir.join("r.xml");
        write(s, &path).unwrap();
        let xml = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(xml.contains("tests=\"2\" errors=\"0\" failures=\"1\""), "{xml}");
        assert!(xml.contains("name=\"a&lt;b&gt;&amp;&quot;c&apos;\""), "{xml}");
        assert!(xml.contains("message=\"0x1 != golden 0x2\""), "{xml}");
        assert_eq!(xml.matches("classname=\"bench.r4400.interp\"").count(), 2, "{xml}");
    }
}
