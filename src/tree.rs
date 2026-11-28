//! The machine's slots as a tree, what is in each and where it came from: the monitor's `devices` and `--list-slots`.

use std::collections::HashMap;

use anstyle::{AnsiColor, Effects, Style};
use termtree::{GlyphPalette, Tree};

use crate::config::MachineConfig;
use crate::devices::DeviceModel;
use crate::slot_path::MachineSlotPath;
use crate::topology::{DevicePlacement, MachineSlotKind};

/// How the tree is drawn: box characters or ASCII branches, with ANSI colour or without.
#[derive(Clone, Copy, Debug)]
pub struct Look {
    pub unicode: bool,
    pub color: bool,
}

impl Look {
    /// Plain ASCII, for a socket or a file.
    pub const PLAIN: Look = Look { unicode: false, color: false };

    /// For stdout: box characters on a terminal that shows them (`supports_unicode::on` says yes to a pipe, so ask only a
    /// terminal), and colour always, which `anstream` strips off a pipe.
    pub fn stdout() -> Look {
        use std::io::IsTerminal;
        Look { unicode: std::io::stdout().is_terminal() && supports_unicode::supports_unicode(), color: true }
    }

    fn palette(self) -> GlyphPalette {
        match self.unicode {
            true => GlyphPalette::new(),
            false => GlyphPalette { middle_item: "|", last_item: "`", item_indent: "-- ", middle_skip: "|", last_skip: " ", skip_indent: "   " },
        }
    }

    /// `text` in `style`, then padded to `width` outside the colour.
    fn cell(self, text: &str, width: usize, style: Style) -> String {
        let pad = " ".repeat(width.saturating_sub(text.chars().count()));
        match self.color && !text.is_empty() {
            true => format!("{style}{text}{style:#}{pad}"),
            false => format!("{text}{pad}"),
        }
    }
}

/// One line of the tree: the slot's name, what is in it, who put it there and a note.
struct Row {
    parent: usize,
    path: MachineSlotPath,
    label: String,
    has: String,
    origin: &'static str,
    note: String,
    /// A run of free targets or loop IDs on one line: the first step's name and the last id.
    run: Option<(String, u8)>,
}

/// The tree of `cfg`'s machine: every slot under the slot it hangs from, with what is in it. `live` adds what a running
/// machine's serial ports are connected to.
pub fn render(cfg: &MachineConfig, look: Look, live: bool) -> Result<String, String> {
    let p = DevicePlacement::new(cfg)?;
    let at: HashMap<_, _> = p.placed().map(|(d, at)| (at, d)).collect();
    let board_implied = cfg.graphics.board.is_none() && !cfg.devices.iter().any(|d| matches!(d.device, DeviceModel::Graphics));
    let mut rows = vec![Row { parent: 0, path: MachineSlotPath::root(), label: p.machine().to_string(), has: String::new(), origin: "", note: String::new(), run: None }];
    for s in &p.slots {
        let parent = rows.iter().rposition(|r| s.path.starts_with(&r.path)).unwrap_or(0);
        let full = s.path.to_string();
        let name = match rows[parent].path.to_string() {
            up if up.is_empty() => full.as_str(),
            up => full.strip_prefix(&format!("{up}/")).unwrap_or(&full),
        };
        let named = at.get(&s.path).and_then(|d| d.name.as_deref());
        let names: Vec<&str> = s.aliases.iter().map(String::as_str).chain(named).collect();
        let label = if names.is_empty() { name.to_string() } else { format!("{name} ({})", names.join(", ")) };
        let host = s.path.serial_port().filter(|_| live).and_then(|n| crate::dev::serial_map::host_end(cfg.machine.profile, n));
        let (has, origin, note) = match (&s.kind, at.get(&s.path)) {
            (MachineSlotKind::Fixed(part) | MachineSlotKind::ScsiBus { part, .. } | MachineSlotKind::PciBus(part), _) => (part.to_string(), "built-in", String::new()),
            (_, Some(d)) => {
                let has = match d.device {
                    DeviceModel::Graphics => format!("graphics {}", cfg.graphics_board().as_str()),
                    _ => d.device.summary(),
                };
                let origin = if d.default || (board_implied && matches!(d.device, DeviceModel::Graphics)) { "default" } else { "config" };
                (has, origin, host.unwrap_or_default())
            }
            (_, None) if host.is_some() => ("serial".to_string(), "default", host.unwrap_or_default()),
            (kind, None) => ("-".to_string(), "", format!("takes: {}", kind.label())),
        };
        let free_target = has == "-" && names.is_empty() && s.path.target_id().or(s.path.loop_number()).is_some();
        if let (true, Some(last)) = (free_target, rows.last_mut()) {
            if let (Some((first, end)), true) = (&mut last.run, last.parent == parent) {
                *end = s.path.last_number().unwrap_or(0);
                last.label = format!("{first}-{end}");
                continue;
            }
        }
        let run = free_target.then(|| (name.to_string(), s.path.last_number().unwrap_or(0)));
        rows.push(Row { parent, path: s.path.clone(), label, has, origin, note, run });
    }
    Ok(draw(&rows, look))
}

/// Columns aligned across the whole tree: a node at depth `d` is indented `4 * d` characters by its branch.
fn draw(rows: &[Row], look: Look) -> String {
    let depth: Vec<usize> = (0..rows.len()).map(|i| std::iter::successors(Some(i), |&j| (j != 0).then(|| rows[j].parent)).count() - 1).collect();
    let width = |f: &dyn Fn(usize) -> usize| (1..rows.len()).map(f).max().unwrap_or(0);
    let label = width(&|i| 4 * depth[i] + rows[i].label.chars().count());
    let has = width(&|i| rows[i].has.chars().count());
    let origin = width(&|i| rows[i].origin.len());
    let text = |i: usize| match i {
        0 => rows[0].label.clone(),
        _ => {
            let r = &rows[i];
            let ty = Style::new().fg_color(Some(AnsiColor::Cyan.into()));
            let origin_style = match r.origin {
                "default" => Style::new().effects(Effects::DIMMED),
                "config" => Style::new().fg_color(Some(AnsiColor::Yellow.into())).effects(Effects::BOLD),
                _ => Style::new(),
            };
            let line = format!(
                "{}  {}  {}  {}",
                look.cell(&r.label, label - 4 * depth[i], Style::new()),
                look.cell(&r.has, has, if r.has == "-" { Style::new() } else { ty }),
                look.cell(r.origin, origin, origin_style),
                r.note
            );
            line.trim_end().to_string()
        }
    };
    fn grow(rows: &[Row], i: usize, text: &dyn Fn(usize) -> String) -> Tree<String> {
        Tree::new(text(i)).with_leaves((i + 1..rows.len()).filter(|&c| rows[c].parent == i).map(|c| grow(rows, c, text)))
    }
    let mut tree = grow(rows, 0, &text);
    tree.set_glyphs(look.palette());
    tree.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn indy(devices: &str) -> MachineConfig {
        MachineConfig::from_toml_str(&format!("[machine]\nprofile = \"indy_ip24\"\n{devices}")).unwrap().resolved().unwrap()
    }

    /// The same tree in both glyph sets, and in colour once the colour is stripped: runs of free targets share a line, and each row says where it came from.
    #[test]
    fn the_tree_draws_in_both_glyph_sets_and_colours_its_columns() {
        let cfg = indy("[[devices]]\ntype = \"hdd\"\nat = \"scsi0/t2\"\npath = \"irix.img\"\n[[devices]]\ntype = \"cdrom\"\nat = \"scsi0/t4\"\n");
        let ascii = render(&cfg, Look::PLAIN, false).unwrap();
        let rows: Vec<String> = ascii.lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect();
        assert_eq!(rows, [
            "SGI Indy (IP24)",
            "|-- hpc3 hpc3 built-in",
            "| `-- c0 (scsi0) wd33c93a built-in",
            "| |-- t1 - takes: scsi devices",
            "| |-- t2 hdd irix.img config",
            "| |-- t3 - takes: scsi devices",
            "| |-- t4 cdrom config",
            "| `-- t5-7 - takes: scsi devices",
            "|-- gfx graphics newport default",
            "|-- gio1 - takes: virtio devices",
            "|-- gio0 - takes: gio cards, virtio devices",
            "|-- serial1 - takes: serial devices",
            "`-- serial2 - takes: serial devices",
        ]);
        let unicode = render(&cfg, Look { unicode: true, color: false }, false).unwrap();
        assert_eq!(unicode.replace("├── ", "|-- ").replace("└── ", "`-- ").replace("│   ", "|   "), ascii);
        let colour = render(&cfg, Look { unicode: false, color: true }, false).unwrap();
        assert!(colour.contains('\x1b'), "{colour}");
        assert_eq!(anstream::adapter::strip_str(&colour).to_string(), ascii);
        // A board the user chose is theirs, not the machine's default.
        let xz = render(&indy("[graphics]\nboard = \"xz\"\n"), Look::PLAIN, false).unwrap();
        assert!(xz.lines().any(|l| l.contains("graphics xz") && l.ends_with("config")), "{xz}");
    }

    /// A running machine adds what its serial ports are connected to; a device there says which one.
    #[test]
    fn a_running_machines_serial_ports_show_their_host_end() {
        let cfg = indy("");
        let _kept = crate::dev::serial_map::mapped("SCC channel B", Some(1), true, &cfg, crate::dev::serial_host::null_serial_backend());
        let live = render(&cfg, Look::PLAIN, true).unwrap();
        assert!(live.lines().any(|l| l.contains("serial1") && l.contains("serial") && l.contains("default") && l.ends_with("SCC channel B (console): nothing")), "{live}");
        assert!(render(&cfg, Look::PLAIN, false).unwrap().contains("takes: serial devices"));
    }
}
