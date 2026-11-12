//! GUI text catalogs.
//!
//! Every user-visible string lives in a Fluent catalog under
//! `iris-gui/locales/<language>/*.ftl`, looked up by message id with `t!`.
//! `build.rs` embeds every folder it finds there, so adding a language is adding
//! a folder with translated copies of the English files (see
//! `locales/README.md`). Lookups fall back from the chosen language to its base
//! language (`pt-BR` to `pt`) and then to English, so a partial translation
//! still shows every string.
//!
//! A message whose value is empty (`some-subtitle =` on a line of its own) is
//! an empty string. [`subtitle`] and [`ResponseExt::tip`] draw nothing for an
//! empty string, which lets a catalog turn an optional caption or tooltip on or
//! off without touching the code.

use fluent_bundle::concurrent::FluentBundle;
use fluent_bundle::{FluentArgs, FluentResource, FluentValue};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::sync::{Arc, OnceLock};
use unic_langid::LanguageIdentifier;

include!(concat!(env!("OUT_DIR"), "/locales.rs"));

/// The language every other catalog falls back to. Its catalog is the
/// reference: it must define every id the code uses.
pub const FALLBACK: &str = "en";

/// Look up a GUI string by message id, with optional Fluent arguments:
/// `t!("menu-file")`, `t!("status-mips", mips = mips_text)`.
///
/// Arguments accept strings and numbers. Pass a number (not pre-formatted
/// text) wherever a translation may need to pick a plural form.
macro_rules! t {
    ($id:literal) => {
        $crate::i18n::text($id, None)
    };
    ($id:literal, $($key:ident = $value:expr),+ $(,)?) => {{
        let mut args = $crate::i18n::Args::new();
        $( args.set(stringify!($key), $crate::i18n::Arg::into_value($value)); )+
        $crate::i18n::text($id, Some(&args))
    }};
}

pub type Args<'a> = FluentArgs<'a>;

/// A value usable as a `t!` argument.
pub trait Arg<'a> {
    fn into_value(self) -> FluentValue<'a>;
}

macro_rules! arg_via_from {
    ($($ty:ty),*) => {$(
        impl<'a> Arg<'a> for $ty {
            fn into_value(self) -> FluentValue<'a> { FluentValue::from(self) }
        }
    )*};
}
arg_via_from!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

impl<'a> Arg<'a> for &'a str {
    fn into_value(self) -> FluentValue<'a> { FluentValue::from(self) }
}
impl<'a> Arg<'a> for String {
    fn into_value(self) -> FluentValue<'a> { FluentValue::from(self) }
}
impl<'a> Arg<'a> for &'a String {
    fn into_value(self) -> FluentValue<'a> { FluentValue::from(self.as_str()) }
}
impl<'a> Arg<'a> for std::borrow::Cow<'a, str> {
    fn into_value(self) -> FluentValue<'a> { FluentValue::from(self) }
}
impl<'a> Arg<'a> for std::path::Display<'a> {
    fn into_value(self) -> FluentValue<'a> { FluentValue::from(self.to_string()) }
}

type Bundle = FluentBundle<FluentResource>;

/// The active lookup chain: the chosen language first, English last.
struct Catalog {
    chain: Vec<Bundle>,
}

static ACTIVE: RwLock<Option<Arc<Catalog>>> = RwLock::new(None);
/// Ids already reported missing, so a missing string logs once, not per frame.
static MISSING: OnceLock<parking_lot::Mutex<HashSet<String>>> = OnceLock::new();

/// Turn `id =` with no value into `id = {""}`. Fluent requires every message to
/// have a value; this lets a catalog write an intentionally empty string the
/// obvious way. A following indented line is a multi-line value, not empty.
fn allow_empty_messages(source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let mut out = String::with_capacity(source.len() + 64);
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_end();
        let is_bare_id = trimmed.ends_with('=')
            && !line.starts_with([' ', '\t', '#', '-', '.'])
            && trimmed[..trimmed.len() - 1]
                .trim_end()
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        let continues = lines[i + 1..]
            .iter()
            .find(|l| !l.trim().is_empty())
            .is_some_and(|l| l.starts_with([' ', '\t']));
        out.push_str(trimmed);
        if is_bare_id && !continues {
            out.push_str(" {\"\"}");
        }
        out.push('\n');
    }
    out
}

/// Parse one embedded `.ftl` file. A syntax error is a bug in the catalog,
/// caught by the tests; at run time the valid entries are still used.
fn parse(lang: &str, file: &str, source: &str) -> FluentResource {
    match FluentResource::try_new(allow_empty_messages(source)) {
        Ok(res) => res,
        Err((res, errors)) => {
            log::error!("locales/{lang}/{file}: {errors:?}");
            res
        }
    }
}

fn files_for(lang: &str) -> Option<&'static [(&'static str, &'static str)]> {
    LOCALES.iter().find(|(code, _)| *code == lang).map(|(_, files)| *files)
}

fn bundle(lang: &str) -> Option<Bundle> {
    let files = files_for(lang)?;
    let id: LanguageIdentifier = lang.parse().ok()?;
    let mut bundle = Bundle::new_concurrent(vec![id]);
    // Bidi isolation marks around arguments render as boxes in egui's fonts.
    bundle.set_use_isolating(false);
    for (file, source) in files {
        if let Err(errors) = bundle.add_resource(parse(lang, file, source)) {
            log::error!("locales/{lang}/{file}: {errors:?}");
        }
    }
    Some(bundle)
}

/// The installed language that best serves `requested` (`pt-BR`, `pt_BR.UTF-8`,
/// `de`): an exact match, then the same base language.
fn resolve(requested: &str) -> Option<&'static str> {
    let tag = requested.split(['.', '@']).next()?.replace('_', "-");
    let wanted: LanguageIdentifier = tag.parse().ok()?;
    let codes = LOCALES.iter().map(|(code, _)| *code);
    codes.clone().find(|c| c.eq_ignore_ascii_case(&tag)).or_else(|| {
        codes.clone().find(|c| {
            c.parse::<LanguageIdentifier>().is_ok_and(|id| id.language == wanted.language)
        })
    })
}

/// The language the system asks for, if a catalog for it is installed.
fn system_language() -> Option<&'static str> {
    sys_locale::get_locales().find_map(|l| resolve(&l))
}

/// Select the GUI language. An empty `preference` follows the system. Takes
/// effect on the next frame, since every string is looked up as it's drawn.
pub fn set_language(preference: &str) {
    let lang = if preference.is_empty() { system_language() } else { resolve(preference) }
        .unwrap_or(FALLBACK);
    let mut codes = vec![lang];
    if let Ok(id) = lang.parse::<LanguageIdentifier>() {
        let base = id.language.as_str();
        if base != lang {
            if let Some(base) = resolve(base) { codes.push(base); }
        }
    }
    codes.push(FALLBACK);
    codes.dedup();
    let chain = codes.into_iter().filter_map(bundle).collect();
    *ACTIVE.write() = Some(Arc::new(Catalog { chain }));
}

/// Installed languages as `(code, name)`, where the name is the catalog's own
/// `language-name` (so each language is listed in itself).
pub fn languages() -> Vec<(&'static str, String)> {
    LOCALES
        .iter()
        .map(|(code, _)| {
            let name = bundle(code)
                .and_then(|b| {
                    let pattern = b.get_message("language-name")?.value()?;
                    Some(b.format_pattern(pattern, None, &mut vec![]).into_owned())
                })
                .unwrap_or_else(|| code.to_string());
            (*code, name)
        })
        .collect()
}

fn catalog() -> Arc<Catalog> {
    if let Some(c) = ACTIVE.read().as_ref() {
        return c.clone();
    }
    set_language("");
    ACTIVE.read().clone().expect("catalog set")
}

/// Look up `id`, formatting it with `args`. Use `t!` rather than calling this.
pub fn text(id: &str, args: Option<&Args>) -> String {
    let catalog = catalog();
    for bundle in &catalog.chain {
        let Some(pattern) = bundle.get_message(id).and_then(|m| m.value()) else { continue };
        let mut errors = vec![];
        let s = bundle.format_pattern(pattern, args, &mut errors);
        if !errors.is_empty() {
            log::warn!("GUI string {id}: {errors:?}");
        }
        return s.into_owned();
    }
    let missing = MISSING.get_or_init(Default::default);
    if missing.lock().insert(id.to_string()) {
        log::warn!("GUI string {id} is not in any catalog");
    }
    id.to_string()
}

/// A secondary caption under a control. Draws nothing when the catalog leaves
/// the string empty.
pub fn subtitle(ui: &mut egui::Ui, text: String) {
    if !text.is_empty() {
        ui.label(egui::RichText::new(text).weak().small());
    }
}

pub trait ResponseExt {
    /// `on_hover_text`, skipped when the catalog leaves the string empty.
    fn tip(self, text: String) -> Self;
}

impl ResponseExt for egui::Response {
    fn tip(self, text: String) -> Self {
        if text.is_empty() { self } else { self.on_hover_text(text) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_syntax::ast;
    use std::collections::{BTreeMap, BTreeSet};

    /// Message id -> the variables its value references, for one language.
    fn messages(lang: &str) -> BTreeMap<String, BTreeSet<String>> {
        let mut out = BTreeMap::new();
        for (file, source) in files_for(lang).unwrap() {
            let res = match FluentResource::try_new(allow_empty_messages(source)) {
                Ok(res) => res,
                Err((_, errors)) => panic!("locales/{lang}/{file}: {errors:?}"),
            };
            for entry in res.entries() {
                if let ast::Entry::Message(m) = entry {
                    let mut vars = BTreeSet::new();
                    if let Some(p) = &m.value { pattern_vars(p, &mut vars); }
                    let id = m.id.name.to_string();
                    assert!(out.insert(id.clone(), vars).is_none(), "{lang}: {id} defined twice");
                }
            }
        }
        out
    }

    fn pattern_vars(p: &ast::Pattern<&str>, vars: &mut BTreeSet<String>) {
        for el in &p.elements {
            if let ast::PatternElement::Placeable { expression } = el { expr_vars(expression, vars); }
        }
    }

    fn expr_vars(e: &ast::Expression<&str>, vars: &mut BTreeSet<String>) {
        match e {
            ast::Expression::Select { selector, variants } => {
                inline_vars(selector, vars);
                for v in variants { pattern_vars(&v.value, vars); }
            }
            ast::Expression::Inline(i) => inline_vars(i, vars),
        }
    }

    fn inline_vars(i: &ast::InlineExpression<&str>, vars: &mut BTreeSet<String>) {
        match i {
            ast::InlineExpression::VariableReference { id } => { vars.insert(id.name.to_string()); }
            ast::InlineExpression::Placeable { expression } => expr_vars(expression, vars),
            ast::InlineExpression::FunctionReference { arguments, .. } => {
                for a in &arguments.positional { inline_vars(a, vars); }
            }
            _ => {}
        }
    }

    /// Every `t!("id", k = v, ..)` in the source: (file, id, argument names).
    fn uses() -> Vec<(String, String, BTreeSet<String>)> {
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for e in std::fs::read_dir(dir).unwrap().flatten() {
                let p = e.path();
                // i18n.rs itself only has examples in its docs.
                if p.is_dir() { walk(&p, out) }
                else if p.extension().is_some_and(|x| x == "rs") && !p.ends_with("i18n.rs") { out.push(p) }
            }
        }
        let mut files = vec![];
        walk(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut files);
        let mut out = vec![];
        for f in files {
            let src = std::fs::read_to_string(&f).unwrap();
            let name = f.file_name().unwrap().to_string_lossy().into_owned();
            let mut rest = src.as_str();
            while let Some(at) = rest.find("t!(") {
                // Skip identifiers that merely end in `t` (`print!`, `assert!`).
                let prev = rest[..at].chars().next_back();
                rest = &rest[at + 3..];
                if prev.is_some_and(|c| c.is_alphanumeric() || c == '_') { continue; }
                // The id may start on the next line.
                let Some(after) = rest.trim_start().strip_prefix('"') else { continue };
                rest = after;
                let end = rest.find('"').unwrap();
                let id = rest[..end].to_string();
                // Argument names up to the matching close paren.
                let mut depth = 1;
                let mut close = end;
                for (i, c) in rest[end..].char_indices() {
                    match c {
                        '(' | '[' | '{' => depth += 1,
                        ')' | ']' | '}' => { depth -= 1; if depth == 0 { close = end + i; break; } }
                        _ => {}
                    }
                }
                let body = &rest[end + 1..close];
                let mut args = BTreeSet::new();
                let mut d = 0;
                let mut seg_start = 0;
                let b = body.as_bytes();
                for i in 0..=b.len() {
                    let c = b.get(i).copied().unwrap_or(b',');
                    match c {
                        b'(' | b'[' | b'{' => d += 1,
                        b')' | b']' | b'}' => d -= 1,
                        b',' if d == 0 => {
                            let seg = body[seg_start..i].trim();
                            if let Some((k, _)) = seg.split_once('=') {
                                let k = k.trim();
                                if !k.is_empty() && k.chars().all(|c| c.is_alphanumeric() || c == '_') {
                                    args.insert(k.to_string());
                                }
                            }
                            seg_start = i + 1;
                        }
                        _ => {}
                    }
                }
                out.push((name.clone(), id, args));
            }
        }
        out
    }

    #[test]
    fn english_defines_every_id_with_the_arguments_the_code_passes() {
        let en = messages(FALLBACK);
        let mut problems = vec![];
        for (file, id, args) in uses() {
            match en.get(&id) {
                None => problems.push(format!("{file}: {id} is not in locales/{FALLBACK}")),
                Some(vars) if !vars.is_subset(&args) => problems.push(format!(
                    "{file}: {id} uses {vars:?} but the code passes {args:?}"
                )),
                _ => {}
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    #[test]
    fn every_english_id_is_used() {
        let used: BTreeSet<String> = uses().into_iter().map(|(_, id, _)| id).collect();
        let unused: Vec<_> = messages(FALLBACK)
            .into_keys()
            .filter(|id| id != "language-name" && !used.contains(id))
            .collect();
        assert!(unused.is_empty(), "unused ids in locales/{FALLBACK}: {unused:?}");
    }

    #[test]
    fn translations_match_english() {
        let en = messages(FALLBACK);
        let mut problems = vec![];
        for (lang, _) in LOCALES.iter().filter(|(l, _)| *l != FALLBACK) {
            for (id, vars) in messages(lang) {
                match en.get(&id) {
                    None => problems.push(format!("{lang}: {id} is not an English id")),
                    Some(en_vars) if !vars.is_subset(en_vars) => {
                        problems.push(format!("{lang}: {id} uses {vars:?}, English passes {en_vars:?}"))
                    }
                    _ => {}
                }
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    #[test]
    fn empty_messages_hide_and_multiline_values_survive() {
        let src = "a =\nb = x\nc =\n    long\n    value\nd =\n";
        let res = FluentResource::try_new(allow_empty_messages(src)).expect("parses");
        let mut bundle = Bundle::new_concurrent(vec!["en".parse().unwrap()]);
        bundle.set_use_isolating(false);
        bundle.add_resource(res).unwrap();
        let get = |id| {
            let p = bundle.get_message(id).unwrap().value().unwrap();
            bundle.format_pattern(p, None, &mut vec![]).into_owned()
        };
        assert_eq!(get("a"), "");
        assert_eq!(get("b"), "x");
        assert_eq!(get("c"), "long\nvalue");
        assert_eq!(get("d"), "");
    }

    #[test]
    fn regional_requests_resolve_to_the_base_language() {
        assert_eq!(resolve("en_GB.UTF-8"), Some("en"));
        assert_eq!(resolve("en-US"), Some("en"));
        assert_eq!(resolve("xx"), None);
    }

    #[test]
    fn arguments_are_substituted_without_isolation_marks() {
        set_language("en");
        assert_eq!(text("language-name", None), "English");
    }
}
