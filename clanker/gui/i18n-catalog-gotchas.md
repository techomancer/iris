# GUI text catalogs (i18n) gotchas

- All GUI text is Fluent (`iris-gui/locales/<lang>/*.ftl`, `src/i18n.rs`).
  `build.rs` embeds every locale folder; nothing registers a language.
- Fluent rejects a message with no value. `src/i18n.rs` rewrites a bare
  `id =` (next non-blank line not indented) to `id = {""}` before parsing, so
  an empty value is how a catalog hides a `subtitle()` or `.tip()`. Don't
  "fix" those lines to `{""}` by hand, and don't remove the rewrite.
- `set_use_isolating(false)` is deliberate: Fluent's default wraps arguments
  in U+2068/U+2069, which egui's default fonts draw as boxes
  (`egui-default-font-glyph-coverage-avoid-tofu.md`).
- Ids are unique across all files of one language; a duplicate makes
  `add_resource` drop the second definition. Each file owns its prefix.
- The i18n tests scan `src/**/*.rs` for literal `t!("id", k = v)` calls. An id
  built at run time (`t!(some_var)`) doesn't compile — `t!` takes a literal on
  purpose so the scan sees every id. For an enum, `match` to literal ids.
