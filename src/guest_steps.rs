//! IRIX guest steps, as data: commands to run, PROM settings to seed, and
//! boots to check.
//!
//! A file holds any mix of `[[step]]`, `[[nvram]]` and `[[boot]]` tables, and
//! [`load`] reads them all. A consumer takes the tables it needs and ignores
//! the rest: `iris-bench irix` reads steps, `iris-ci boot-test` reads the
//! other two.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Directories an IRIX 6.5 install actually puts programs in. `which` is a csh
/// script there and `command -v` is not in its Bourne shell, so a program is
/// probed by testing for it directly.
pub const IRIX_PATH: &str =
    "/bin:/usr/bin:/usr/sbin:/usr/bsd:/usr/etc:/usr/bin/X11:/usr/local/bin:/sbin:/usr/gfx";

#[derive(Debug, Default, Deserialize)]
pub struct StepFile {
    #[serde(default)]
    pub step: Vec<Step>,
    #[serde(default)]
    pub nvram: Vec<Nvram>,
    #[serde(default)]
    pub boot: Vec<Boot>,
}

#[derive(Debug, Deserialize)]
pub struct Step {
    pub name: String,
    pub unit: String,
    /// Work units the command performs. 0 means "not a fixed quantity" (a
    /// `find` over /usr, an x11perf run) — the duration is still comparable
    /// between runs on the same disk image, the rate is not.
    #[serde(default)]
    pub work: u64,
    /// Untimed preparation.
    #[serde(default)]
    pub setup: Option<String>,
    pub cmd: String,
    /// Program that must exist, or the step is skipped rather than failed.
    #[serde(default)]
    pub requires: Option<String>,
    /// Run it, but do not record a row (cleanup).
    #[serde(default)]
    pub skip_timing: bool,
    /// Seconds the setup and the command may each take. A step that overruns
    /// is FAILED. Defaults to the `--timeout` of the run.
    #[serde(default)]
    pub timeout: Option<u64>,
}

/// A PROM variable seeded before each boot, with `console` added per boot.
/// A blank NVRAM has no boot path, so without these the PROM cannot load the
/// disk.
#[derive(Debug, Deserialize)]
pub struct Nvram {
    pub name: String,
    pub value: String,
    /// Whether the booted guest lists this variable too. The guest's own
    /// `nvram` command shows only the ones its kernel reads.
    #[serde(default)]
    pub guest: bool,
}

/// One boot of the installed disk, with the check that says it worked.
#[derive(Debug, Deserialize)]
pub struct Boot {
    /// The JUnit case for the check below.
    pub name: String,
    /// The PROM `console` value for this boot: `d` for the serial line, `g`
    /// for the graphics board.
    pub console: String,
    /// The JUnit case for asking iris to quit afterwards.
    pub quit: String,
    /// Text the serial console must print.
    #[serde(default)]
    pub serial: Option<String>,
    /// A screenshot the screen must match, relative to this file.
    #[serde(default)]
    pub screen: Option<PathBuf>,
}

/// Wrap a guest command so it runs under a Bourne shell with a sane PATH,
/// whatever the login shell is. Consequence, documented in the shipped step files: a step
/// command may not contain a single quote.
pub fn wrap(cmd: &str) -> String {
    format!("sh -c 'PATH={}; export PATH; {}'", IRIX_PATH, cmd)
}

pub fn load(path: &Path) -> Result<StepFile, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    parse(&text, &path.display().to_string())
}

/// A step file's text, with `origin` naming it in an error.
pub fn parse(text: &str, origin: &str) -> Result<StepFile, String> {
    toml::from_str(text).map_err(|e| format!("{origin}: {e}"))
}

/// The `[[step]]` tables of each file, in order.
pub fn load_steps(paths: &[PathBuf]) -> Result<Vec<Step>, String> {
    let mut steps = Vec::new();
    for p in paths {
        steps.extend(load(p)?.step);
    }
    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_may_hold_any_mix_of_tables() {
        let f: StepFile = toml::from_str(
            "[[nvram]]\nname = \"AutoLoad\"\nvalue = \"Yes\"\n\
             [[boot]]\nname = \"b\"\nconsole = \"d\"\nquit = \"q\"\nserial = \"login:\"\n",
        ).unwrap();
        assert!(f.step.is_empty());
        assert_eq!((f.nvram[0].guest, f.boot[0].serial.as_deref()), (false, Some("login:")));
        assert!(toml::from_str::<StepFile>("").unwrap().boot.is_empty());
    }

    #[test]
    fn the_benchmark_files_have_no_boots() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/irix");
        for f in ["steps.toml", "benchies.toml"] {
            let file = load(&dir.join(f)).unwrap();
            assert!(!file.step.is_empty() && file.boot.is_empty() && file.nvram.is_empty(), "{f}");
        }
    }
}
