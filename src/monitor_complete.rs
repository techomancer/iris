//! Tab completion for the monitor's command line.
//!
//! The monitor has no table of subcommands: each device registers a command
//! name and a one-line usage string (`Device::register_commands`), and
//! `help` prints exactly that. Completion reads the same pairs, so a command
//! a device adds or renames completes without a second list to update. A
//! subcommand is a lowercase word that the usage string writes right after
//! the command name (`audio status | audio codec` offers `status` and
//! `codec`; `ptr down|up` offers both). Arguments past the subcommand, and
//! anything a usage string spells in angle brackets, are not completed.
//!
//! The function is pure (line and cursor in, candidates out) and runs only
//! when Tab is pressed at the terminal console.

use crate::monitor::Builtin;

/// The command prefix that addresses a CPU: `cpu regs`, `cpu 1 regs`.
const CPU_PREFIX: &str = "cpu";

/// Lowercase words that follow `name` in `help`, in order of appearance.
fn subcommands(name: &str, help: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (at, _) in help.match_indices(name) {
        let before_ok = help[..at].chars().next_back().map_or(true, |c| !c.is_alphanumeric() && c != '_' && c != '-');
        let Some(rest) = help[at + name.len()..].strip_prefix(' ') else { continue };
        if !before_ok {
            continue;
        }
        let token = rest.split_whitespace().next().unwrap_or("");
        for word in token.split(['|', '/', ',']) {
            let word_ok = word.len() > 1 && word.starts_with(|c: char| c.is_ascii_lowercase())
                && word.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
            if word_ok && !out.iter().any(|w| w == word) {
                out.push(word.to_string());
            }
        }
    }
    out
}

/// The monitor's reply to `help` split into its (command, usage) pairs and
/// whatever else was in the text. `help` writes `{name:12} - {usage}` per
/// command, so a line counts only if it has exactly that shape with a
/// lowercase-word name; an indented line continues the usage above it. Other
/// lines (log output that shares the socket) come back in the second part,
/// unchanged, for the caller to show.
pub fn parse_help(reply: &str) -> (Vec<(String, String)>, String) {
    let mut found: Vec<(String, String)> = Vec::new();
    let mut other = String::new();
    for raw in reply.split_inclusive('\n') {
        let line = raw.trim_end_matches(['\r', '\n']);
        if line.starts_with(char::is_whitespace) {
            if let Some((_, usage)) = found.last_mut() {
                usage.push('\n');
                usage.push_str(line);
                continue;
            }
        } else if let Some(pair) = help_line(line) {
            found.push(pair);
            continue;
        }
        other.push_str(raw);
    }
    (found, other)
}

/// `line` as `help` prints a command, or none.
fn help_line(line: &str) -> Option<(String, String)> {
    let name = line.split(' ').next()?;
    let is_name = name.starts_with(|c: char| c.is_ascii_lowercase())
        && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
    let usage = line.strip_prefix(&format!("{name:12} - "))?;
    is_name.then(|| (name.to_string(), usage.to_string()))
}

/// What can replace the word ending at `pos`: the byte offset the word starts
/// at and the candidates, sorted and without repeats. `commands` is what
/// `Monitor::commands` returns. No candidates means nothing to offer.
pub fn complete(line: &str, pos: usize, commands: &[(String, String)]) -> (usize, Vec<String>) {
    let head = &line[..pos];
    let start = head.rfind(char::is_whitespace).map_or(0, |i| i + head[i..].chars().next().map_or(1, char::len_utf8));
    let prefix = &head[start..];
    let mut words: Vec<&str> = head[..start].split_whitespace().collect();

    // `cpu regs` and `cpu 1 regs` are `regs` aimed at a CPU.
    if words.first() == Some(&CPU_PREFIX) && commands.iter().any(|(c, _)| c == CPU_PREFIX) {
        let numbered = words.get(1).is_some_and(|w| w.chars().all(|c| c.is_ascii_digit()));
        words.drain(..if numbered { 2 } else { 1 });
    }

    let mut found: Vec<String> = match words.as_slice() {
        [] => commands
            .iter()
            .map(|(c, _)| c.clone())
            .chain(Builtin::NAMES.iter().map(|(c, _)| c.to_string()))
            .collect(),
        [cmd] => commands.iter().filter(|(c, _)| c == cmd).flat_map(|(c, h)| subcommands(c, h)).collect(),
        _ => Vec::new(),
    };
    found.retain(|c| c.starts_with(prefix));
    found.sort();
    found.dedup();
    (start, found)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmds() -> Vec<(String, String)> {
        [
            ("audio", "MACE audio + AD1843: audio status | audio codec"),
            ("screen", "The guest's screen: screen info | screen cursor | screen shot <file.png> [x y w h] | screen wait [ms]"),
            ("ptr", "Pointer: ptr pos | ptr down|up [button] | ptr click <x> <y>"),
            ("bp", "Breakpoint commands: bp add <addr> [type], bp list, bp enable/disable <id>"),
            ("cpu", "CPU commands: start, stop, run"),
            ("regs", "Dump registers"),
            ("run", "Run instructions"),
            ("log", "[DEV] log <module|all> <on|off>  |  log <module> mask <cat|hex>  |  log status"),
        ]
        .iter()
        .map(|(c, h)| (c.to_string(), h.to_string()))
        .collect()
    }

    fn at_end(line: &str) -> (usize, Vec<String>) {
        complete(line, line.len(), &cmds())
    }

    #[test]
    fn an_empty_line_offers_every_command_and_the_built_ins() {
        let (start, all) = at_end("");
        assert_eq!(start, 0);
        for name in ["audio", "regs", "help", "quit", "exit"] {
            assert!(all.iter().any(|c| c == name), "{name} missing from {all:?}");
        }
        assert!(all.windows(2).all(|w| w[0] < w[1]), "sorted, no repeats");
    }

    #[test]
    fn a_first_word_completes_by_prefix() {
        assert_eq!(at_end("re"), (0, vec!["regs".to_string()]));
        assert_eq!(at_end("ru"), (0, vec!["run".to_string()]));
        assert_eq!(at_end("zz").1, Vec::<String>::new());
        // Leading blanks move where the word starts.
        assert_eq!(at_end("  au"), (2, vec!["audio".to_string()]));
    }

    #[test]
    fn a_second_word_completes_from_the_usage_string() {
        assert_eq!(at_end("audio "), (6, vec!["codec".to_string(), "status".to_string()]));
        assert_eq!(at_end("audio st"), (6, vec!["status".to_string()]));
        assert_eq!(at_end("screen s").1, vec!["shot".to_string()]);
        assert_eq!(at_end("ptr ").1, vec!["click", "down", "pos", "up"]);
        assert_eq!(at_end("bp ").1, vec!["add", "disable", "enable", "list"]);
    }

    #[test]
    fn angle_bracket_arguments_are_not_offered() {
        assert_eq!(at_end("log ").1, vec!["status".to_string()]);
        assert_eq!(at_end("screen shot ").1, Vec::<String>::new());
    }

    #[test]
    fn later_words_and_unknown_commands_offer_nothing() {
        assert!(at_end("audio status ").1.is_empty());
        assert!(at_end("nope ").1.is_empty());
        assert!(at_end("regs ").1.is_empty());
    }

    #[test]
    fn the_cpu_prefix_completes_the_command_behind_it() {
        assert_eq!(at_end("cpu re"), (4, vec!["regs".to_string()]));
        assert_eq!(at_end("cpu 1 re"), (6, vec!["regs".to_string()]));
        assert_eq!(at_end("cpu 1 audio ").1, vec!["codec", "status"]);
    }

    #[test]
    fn only_the_text_before_the_cursor_counts() {
        let line = "audio status";
        assert_eq!(complete(line, 8, &cmds()), (6, vec!["status".to_string()]));
        assert_eq!(complete(line, 2, &cmds()), (0, vec!["audio".to_string()]));
    }

    #[test]
    fn the_help_reply_gives_back_what_was_registered() {
        let reply = format!(
            "{:12} - MACE audio: audio status | audio codec\r\n\
             {:12} - log status\r\n              modules: net mc\r\n\
             HPC3: seeq - not a command\r\n\
             nope - one space is not the help shape\r\n\
             {:12} - usage\r\n",
            "audio", "log", "a-name-longer-than-twelve"
        );
        let (found, other) = parse_help(&reply);
        assert_eq!(
            found,
            vec![
                ("audio".to_string(), "MACE audio: audio status | audio codec".to_string()),
                ("log".to_string(), "log status\n              modules: net mc".to_string()),
                ("a-name-longer-than-twelve".to_string(), "usage".to_string()),
            ]
        );
        assert_eq!(other, "HPC3: seeq - not a command\r\nnope - one space is not the help shape\r\n");
        assert_eq!(parse_help(""), (vec![], String::new()));
    }

    #[test]
    fn a_command_name_inside_a_longer_word_is_not_a_subcommand() {
        assert!(subcommands("log", "dialog status | blog post").is_empty());
        assert_eq!(subcommands("log", "log status | log file <path>"), vec!["status", "file"]);
    }
}
