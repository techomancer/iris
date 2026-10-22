# egui default fonts: which glyphs render, which are tofu

iris-gui uses egui's built-in fonts (no custom font is loaded). Labels/buttons
render in the **Proportional** family, whose fallback chain (epaint 0.29
`fonts.rs`) is:

```
Ubuntu-Light  →  NotoEmoji-Regular  →  emoji-icon-font
```

Note **Hack is NOT in the Proportional chain** (it's Monospace-only). So a glyph
that exists only in Hack still renders as tofu (□) in a normal label.

## Two traps

**(1) Hack-only glyphs.** Hack covers many symbols (`●` U+25CF, `→` U+2192) but
it's Monospace-only, so those are **tofu in any normal label**. This is the one
that bit us repeatedly — `●` *looks* fine in a monospace context but renders as a
colored □ in the NET light / machine-switch marker.

**(2) dingbats vs emoji codepoints.** NotoEmoji ships the **emoji-presentation**
codepoints but not the adjacent plain **dingbats**.

Verified against the actual 0.29 cmaps (Proportional chain only):

| Want | Tofu — DO NOT USE | Renders — use |
|------|-------------------|---------------|
| filled dot | `●` U+25CF | **`•` U+2022** (size it up), or paint a circle |
| right arrow / breadcrumb | `→` U+2192 | **`»` U+00BB**, or ASCII `->` |
| check | `✓` U+2713 | `✅` U+2705 |
| cross / close | `✗` U+2717, `✕` U+2715 | `×` U+00D7, `❌` U+274C |
| power | `⏻` U+23FB | (none — drop it) |
| folder tabs | `🗂` U+1F5C2 | `📁` U+1F4C1, `📂` U+1F4C2 |

**Confirmed RENDER** (via Ubuntu-Light/NotoEmoji/emoji-icon): `•` `»` `×` `▶`
`⚠` `↩` `⬇` `■` `ℹ` `🌐` `💾` `📁` `📂` `📷` `−` `≈`.
**Confirmed TOFU:** `●` `→` `✓` `✗` `✕` `🗂` `⏻`.

For a colored **status dot**, the NET light and granted-folder indicator use
`RichText::new("\u{2022}").size(..).color(..)` (a sized, colored bullet). For an
inline "active" marker in a list, prefer `ui.selectable_label(active, name)` —
no glyph at all. When in doubt, paint the shape (`ui.painter().circle_filled`).

## How to check a glyph before using it

Dump the cmaps of the four files under `epaint_default_fonts-0.29.1/fonts/` and
test the codepoint against the **Proportional** chain (Ubuntu-Light + NotoEmoji +
emoji-icon-font) — **NOT** the union with Hack. That single mistake (including
Hack) is what made me wrongly bless `●` and `→`; excluding Hack, a hand-rolled
cmap parser (handle subtable formats 4, 6, 12) matches reality exactly. fontTools
is cleaner but isn't installable in this env (PEP 668 managed environment).

The real fix for an arbitrary glyph would be to load a font that covers it via
`egui::FontDefinitions`, but that adds binary weight; for a handful of icons,
staying inside the built-in coverage is simpler.
