# iris-gui text catalogs

Every string the GUI shows lives here, in [Fluent](https://projectfluent.org/)
files, one folder per language. The code refers to strings by id:

```rust
ui.heading(t!("config-processor"));
ui.label(t!("status-mips", mips = mips_text));
subtitle(ui, t!("config-cpu-subtitle"));            // hidden when empty
ui.button(t!("common-browse")).tip(t!("config-prom-browse-tip"));
```

`t!` is defined in `core/src/model/i18n.rs`; the egui helpers `subtitle` and `ResponseExt::tip` are in `src/i18n.rs`.

## Editing English text

Change the value in `en/*.ftl`. The code doesn't need to change.

Optional captions and tooltips are called `*-subtitle` and `*-tip`. Leave the
value empty to hide one; the code still asks for it, so filling it in later
makes it appear:

```ftl
config-cpu-subtitle =
config-cpu-tip = Chosen when the machine starts.
```

Long values continue on indented lines:

```ftl
help-network-body =
    The guest reaches the host network through NAT.
    Ports 2323 and 2121 are forwarded to telnet and FTP.
```

Arguments are written `{ $name }`. Numbers can select a plural form:

```ftl
disks-missing = { $count ->
    [one] One disk image is missing.
   *[other] { $count } disk images are missing.
}
```

## Adding a language

1. Copy `en/` to a folder named for the language tag, such as `de/` or `pt-BR/`.
2. Translate the values, including `language-name` (the language's name in
   itself, as listed in View > Language).
3. Build. `build.rs` picks up the new folder; there's nothing to register.

A missing id falls back to the base language (`pt-BR` to `pt`) and then to
English, so a translation can be finished gradually. An empty value in a
translation is an intentional blank; delete the line to fall back to English.

`cargo test -p iris-gui-core i18n` checks that every id the code uses exists in
English with the arguments the code passes, that every English id is used, and
that every translation's ids and arguments match English.

## Writing style

- Sentence case for everything: "Save state", not "Save State".
- Plain words. Say what a control does, not how it's implemented. Register
  names, config keys and environment variables belong in docs, not labels.
- Labels are short noun phrases; buttons are verbs ("Create disk", "Start").
- An ellipsis (…) on a button or menu item means it opens a dialog or picker.
- No trailing period on labels, buttons or menu items. Full sentences in
  messages and tooltips end with one.
- Errors say what failed and what to do next, in one or two sentences.
- Don't repeat in a tooltip what the label already says. Most controls need no
  tooltip at all.
- Use "IRIX", "PROM", "NVRAM", "SCSI", "CHD" as written; "Indy", "Indigo2".

## Ids

Lowercase, hyphen-separated, prefixed by area: `menu-`, `config-`, `status-`,
`dialog-`, `bench-`, and so on. Each file covers one area of the GUI, and an id
must be unique across all files of a language. Shared words (OK, Cancel,
Browse…) are `common-*` in `common.ftl`.
