//! Where a device goes: a path of tagged steps from the machine root, such as `hpc3/c0/t1`, `xio15/pci0/0/t1` or `gio1`. A path
//! depends only on the machine and its config, never on attach order, and parses without the machine.

use std::fmt;
use std::str::FromStr;

/// The tags, in the order sibling hops sort. A tag that takes no number is a
/// whole token.
const TAGS: &[(&str, bool)] = &[
    ("node", true), // the XIO link of node N, where that node has its own
    ("xio", true),  // port N of the crossbar
    ("pci", true),  // PCI bus N of the parent
    ("", true),     // device N on the parent PCI bus
    ("f", true),    // PCI function N, reserved
    ("hpc3", false),
    ("gfx", false),
    ("gio", true), // GIO expansion slot N
    ("c", true),   // channel N of a controller with more than one
    ("t", true),   // SCSI target N
    ("loop", true), // Fibre Channel loop ID N
    ("usb", true), // USB port N
    ("serial", true), // serial port N, as numbered on the case
];

const BARE: u8 = 3;

fn tag(name: &str) -> Option<u8> {
    // `exp` is the older spelling of `gio`, read but never written.
    let name = if name == "exp" { "gio" } else { name };
    TAGS.iter().position(|t| t.0 == name).map(|i| i as u8)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct PathStep {
    tag: u8,
    n: u8,
}

impl PathStep {
    fn of(name: &str, n: u8) -> PathStep {
        PathStep { tag: tag(name).expect("a tag in TAGS"), n }
    }
    fn node(n: u8) -> PathStep { PathStep::of("node", n) }
    fn xio(n: u8) -> PathStep { PathStep::of("xio", n) }
    fn pci(n: u8) -> PathStep { PathStep::of("pci", n) }
    fn dev(n: u8) -> PathStep { PathStep { tag: BARE, n } }
    fn channel(n: u8) -> PathStep { PathStep::of("c", n) }
    fn target(n: u8) -> PathStep { PathStep::of("t", n) }
    fn loop_id(n: u8) -> PathStep { PathStep::of("loop", n) }
    fn hpc3() -> PathStep { PathStep::of("hpc3", 0) }
    fn gio(n: u8) -> PathStep { PathStep::of("gio", n) }
    fn serial(n: u8) -> PathStep { PathStep::of("serial", n) }
    fn number(self) -> u8 { self.n }
    fn is_target(self) -> bool { self.same_tag(PathStep::target(0)) }
    fn is_loop(self) -> bool { self.same_tag(PathStep::loop_id(0)) }
    fn same_tag(self, other: PathStep) -> bool { self.tag == other.tag }
}

impl fmt::Display for PathStep {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match TAGS[self.tag as usize] {
            (name, true) => write!(f, "{name}{}", self.n),
            (name, false) => f.write_str(name),
        }
    }
}

impl FromStr for PathStep {
    type Err = String;
    fn from_str(s: &str) -> Result<PathStep, String> {
        let bad = || format!("`{s}` is not a slot path step");
        // Whole tokens first, so the digit in `hpc3` is not read as a number.
        if let Some(i) = TAGS.iter().position(|t| !t.1 && t.0 == s) {
            return Ok(PathStep { tag: i as u8, n: 0 });
        }
        let (name, digits) = s.split_at(s.find(|c: char| c.is_ascii_digit()).ok_or_else(bad)?);
        let tag = tag(name).filter(|&t| TAGS[t as usize].1).ok_or_else(bad)?;
        if !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(bad());
        }
        let n = digits.parse().map_err(|_| format!("`{s}`: the number does not fit 0-255"))?;
        Ok(PathStep { tag, n })
    }
}

/// A path of hops from the machine root. The empty path is the root.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct MachineSlotPath(Vec<PathStep>);

impl MachineSlotPath {
    pub fn root() -> MachineSlotPath { MachineSlotPath(Vec::new()) }
    fn join(&self, h: PathStep) -> MachineSlotPath {
        let mut v = self.0.clone();
        v.push(h);
        MachineSlotPath(v)
    }
    pub fn pci(&self, n: u8) -> MachineSlotPath { self.join(PathStep::pci(n)) }
    pub fn dev(&self, n: u8) -> MachineSlotPath { self.join(PathStep::dev(n)) }
    pub fn channel(&self, n: u8) -> MachineSlotPath { self.join(PathStep::channel(n)) }
    pub fn target(&self, n: u8) -> MachineSlotPath { self.join(PathStep::target(n)) }
    pub fn loop_id(&self, n: u8) -> MachineSlotPath { self.join(PathStep::loop_id(n)) }
    pub fn gio(&self, n: u8) -> MachineSlotPath { self.join(PathStep::gio(n)) }
    pub fn hpc3(&self) -> MachineSlotPath { self.join(PathStep::hpc3()) }
    pub fn parent(&self) -> Option<MachineSlotPath> {
        self.0.split_last().map(|(_, rest)| MachineSlotPath(rest.to_vec()))
    }
    /// The number of the last step.
    pub fn last_number(&self) -> Option<u8> { self.0.last().map(|h| h.number()) }
    /// The id of a SCSI target path, the number of a Fibre Channel loop ID path, the number of a GIO slot path.
    pub fn target_id(&self) -> Option<u8> { self.last_if(PathStep::is_target) }
    pub fn loop_number(&self) -> Option<u8> { self.last_if(PathStep::is_loop) }
    pub fn gio_slot(&self) -> Option<u8> { self.last_if(|h| h.same_tag(PathStep::gio(0))) }
    /// N of a serial port path `serialN`, the number printed on the case.
    pub fn serial_port(&self) -> Option<u8> { self.last_if(|h| h.same_tag(PathStep::serial(0))) }
    fn last_if(&self, f: impl Fn(PathStep) -> bool) -> Option<u8> {
        self.0.last().copied().filter(|&h| f(h)).map(PathStep::number)
    }
    /// Whether `self` is `bus` or lies below it.
    pub fn starts_with(&self, bus: &MachineSlotPath) -> bool { self.0.starts_with(&bus.0) }
    pub fn depth(&self) -> usize { self.0.len() }
    /// Whether `s` is one step of a path, which a name or alias must not be.
    pub fn is_step(s: &str) -> bool { s.parse::<PathStep>().is_ok() }
    /// The node and crossbar port of a slot `[nodeN/]xioM...`; node 0 when it names none.
    pub fn xio_port(&self) -> Option<(u8, u8)> {
        let (node, rest) = match self.0.split_first()? {
            (h, rest) if h.same_tag(PathStep::node(0)) => (h.number(), rest),
            _ => (0, &self.0[..]),
        };
        rest.first().filter(|h| h.same_tag(PathStep::xio(0))).map(|h| (node, h.number()))
    }
    /// `self` followed by the steps of `rest`.
    pub fn extend(&self, rest: &MachineSlotPath) -> MachineSlotPath {
        MachineSlotPath([self.0.as_slice(), rest.0.as_slice()].concat())
    }
}

impl fmt::Display for MachineSlotPath {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, h) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str("/")?;
            }
            write!(f, "{h}")?;
        }
        Ok(())
    }
}

impl FromStr for MachineSlotPath {
    type Err = String;
    fn from_str(s: &str) -> Result<MachineSlotPath, String> {
        if s.is_empty() {
            return Err("a slot path is empty".into());
        }
        s.split('/').map(str::parse).collect::<Result<_, _>>().map(MachineSlotPath)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(s: &str) -> MachineSlotPath { s.parse().unwrap() }

    #[test]
    fn every_grammar_example_round_trips() {
        for s in [
            "hpc3/c0/t1", "hpc3/c1/t4", "gio0", "gfx", "pci0/1/t2", "pci0/3", "xio15/pci0/1/t2", "xio11",
            "xio10/pci0/1/pci0/0", "node1/xio11", "xio15/pci0/1/c1/t2", "xio15/pci0/5/usb0", "xio15/pci0/1/loop3",
            "xio15/pci1/1", "xio15/pci0/0/f1",
        ] {
            assert_eq!(loc(s).to_string(), s);
        }
    }

    #[test]
    fn exp_parses_as_gio_and_prints_gio() {
        assert_eq!(loc("exp1"), loc("gio1"));
        assert_eq!(loc("exp0").to_string(), "gio0");
    }

    #[test]
    fn siblings_order_by_number_not_text() {
        assert!(loc("pci0/2") < loc("pci0/10"));
        assert!(loc("xio9") < loc("xio15"));
        assert!(loc("hpc3/c0/t7") < loc("hpc3/c0/t15"));
    }

    #[test]
    fn bare_and_tagged_hops_differ() {
        assert_eq!(loc("pci0/3"), MachineSlotPath::root().pci(0).dev(3));
        assert_ne!(loc("pci0/3"), loc("pci0/t3"));
    }

    #[test]
    fn bad_strings_are_errors() {
        for s in ["", "/pci0", "pci", "pci0/", "t", "zz1", "pci256", "hpc30", "gfx1", "pci0//1", "t-1", "pci0x", "scsi0", "xio15/pci0/1/l3"] {
            assert!(s.parse::<MachineSlotPath>().is_err(), "{s:?} parsed");
        }
    }

    #[test]
    fn a_crossbar_port_is_read_with_or_without_its_node() {
        assert_eq!(loc("xio11/pci0/1").xio_port(), Some((0, 11)));
        assert_eq!(loc("node1/xio8").xio_port(), Some((1, 8)));
        assert_eq!(loc("pci0/3").xio_port(), None);
        assert_eq!(loc("node1/pci0").xio_port(), None);
    }

    #[test]
    fn join_parent_extend() {
        let l = MachineSlotPath::root().hpc3().channel(0).target(1);
        assert_eq!(l, loc("hpc3/c0/t1"));
        assert_eq!(l.parent(), Some(loc("hpc3/c0")));
        assert_eq!(MachineSlotPath::root().parent(), None);
        assert_eq!((l.target_id(), l.loop_number()), (Some(1), None));
        assert_eq!(loc("xio15/pci0").extend(&loc("1/t2")), loc("xio15/pci0/1/t2"));
    }
}
