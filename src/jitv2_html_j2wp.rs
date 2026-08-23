//! `j2 html`: complete self-contained implementation for the `j2wp`
//! whole-page design — data structs (`jitv2_html`), the HTML/JS renderer
//! (`render_jitv2_html`), and the page-snapshot collector
//! (`write_jitv2_html`) that populates a `jitv2_html::Snapshot` from every
//! claimed `PhysicalCodePage`'s `requested`/`compiled`/`denied` bitmaps and
//! single per-page compiled function. See `jitv2_html_default.rs` for the
//! default design's complete, separate counterpart — the two don't share a
//! collection body (or even the same `Snapshot` shape): §13's one page, one
//! function, many entry points model runs exactly one `walk_multi_entry`
//! per page (over every currently-published offset) instead of one walk
//! per entry the pre-§13 design used.

use std::io::Write;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::Serialize;

use crate::jitv2::Jitv2;
use crate::physical::HIMEM_BASE;
use crate::traits::BusDevice;

pub mod jitv2_html {
    use serde::Serialize;

    /// One analyzer-visited word in the page's single §13 multi-entry walk
    /// (`Analyzer::walk_multi_entry` over every requested-or-compiled
    /// offset) — one row in the page's single instruction-listing column.
    #[derive(Serialize)]
    pub struct WordRow {
        /// Word offset within the page (0..1024).
        pub word: u32,
        pub raw: u32,
        /// Full disassembly text (`mips_dis::disassemble`) — one column now
        /// (not one-per-entry), so there's room to show the real
        /// instruction instead of just a color swatch.
        pub dis: String,
        /// `analyzer::classify()`'s variant name (Sequential/Branch/Jump/
        /// RegJump/Excluded/RegionBoundary) — used directly as a CSS class
        /// for per-instruction-type coloring.
        pub kind: &'static str,
        pub is_entry_point: bool,
        pub is_fallback: bool,
        pub is_branch_target: bool,
        pub is_slot_only: bool,
    }

    /// One claimed `PhysicalCodePage` — §13: one compiled function per page,
    /// so page-level stats and a single analyzer walk (not one per entry).
    #[derive(Serialize)]
    pub struct PageDump {
        pub pfn: u32,
        pub phys_addr: u32,
        /// "low" (0x08000000 RAM window), "high" (0x20000000 RAM window), or
        /// "prom" (0x1FC00000, 1MB boot ROM) — purely a label for the
        /// region-overview grid; jitv2 itself tracks pages by bare pfn,
        /// agnostic to which bus window/device they fall in.
        pub window: &'static str,
        pub gen: u64,
        pub func: String,
        pub stale: bool,
        pub fr1: bool,
        pub entry_count: u32,
        pub denylisted_count: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instr_count: Option<u32>,
        /// Host (compiled machine code) size in bytes.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub code_size: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub call_count: Option<u64>,
        /// How many times `publish()` has committed a fresh compile for this
        /// page (`PhysicalCodePage::compile_count`) — recompile churn.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub compile_count: Option<u32>,
        /// Guest code size in bytes: 4 * (distinct visited words in
        /// `rows`), i.e. how many bytes of MIPS this page's one compiled
        /// function actually covers — the denominator for host
        /// bytes/guest instruction.
        pub guest_code_size: u32,
        /// The page's single reachability walk (union of every requested or
        /// compiled entry offset), ascending word order — one row per
        /// visited instruction, rendered as a single column.
        pub rows: Vec<WordRow>,
    }

    #[derive(Serialize)]
    pub struct Snapshot {
        pub pages: Vec<PageDump>,
    }
}

/// Render a [`jitv2_html::Snapshot`] as a complete, self-contained HTML
/// document: the snapshot embedded as a JSON `<script>` blob, with vanilla
/// JS/CSS doing all layout/interactivity client-side (no network requests —
/// works fine opened directly from `file://`).
///
/// Two-level structure per the design this command was built for: a region
/// overview (one CSS-grid cell per claimed page, grouped by which 256MB
/// physical window it falls in) that's clickable through to a per-page
/// detail overlay (all 1024 words, one coverage column per entry, colored by
/// the analyzer's instruction classification).
#[cfg(feature = "jitv2")]
pub fn render_jitv2_html(snapshot: &jitv2_html::Snapshot) -> String {
    let data_json = serde_json::to_string(snapshot).expect("Snapshot serialization is infallible (no maps with non-string keys, no floats)");
    format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>jitv2 dump</title>
<style>
  :root {{
    color-scheme: dark;
    --bg: #14161a;
    --panel: #1c1f26;
    --border: #2c3038;
    --text: #d8dce2;
    --dim: #7a8290;
    --row-even: #1a1d23;
    --row-odd: #17191e;
    --row-entry-even: #223047;
    --row-entry-odd: #1d2a3d;
    --row-hover: #2a3f5c;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font: 13px/1.4 ui-monospace, "SF Mono", Consolas, monospace;
    margin: 0;
    padding: 16px;
  }}
  h1 {{ font-size: 16px; margin: 0 0 4px; }}
  h2 {{ font-size: 13px; margin: 16px 0 8px; color: var(--dim); text-transform: uppercase; letter-spacing: 0.05em; }}
  .summary {{ color: var(--dim); margin-bottom: 16px; }}
  .region {{
    display: grid;
    grid-template-columns: repeat(128, 1fr);
    gap: 1px;
    margin-bottom: 8px;
  }}
  .cell {{
    aspect-ratio: 1;
    background: #23262d;
    border-radius: 1px;
  }}
  .cell.claimed {{
    cursor: pointer;
  }}
  .cell.claimed:hover {{ outline: 1px solid #fff; outline-offset: -1px; }}
  .region-label {{ color: var(--dim); font-size: 11px; margin-bottom: 4px; }}
  .heat-scale {{
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--dim);
    font-size: 11px;
    margin: 4px 0 16px;
  }}
  .heat-scale .bar {{
    width: 160px;
    height: 10px;
    border-radius: 2px;
    background: linear-gradient(to right, hsl(210 70% 32%), hsl(140 65% 40%), hsl(50 90% 50%), hsl(0 85% 55%));
  }}

  #overlay {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.6);
    z-index: 10;
  }}
  #overlay.open {{ display: flex; align-items: center; justify-content: center; }}
  #overlay-panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    width: 98vw;
    height: 94vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  #overlay-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }}
  #overlay-close {{
    cursor: pointer;
    background: none;
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 4px;
    padding: 4px 10px;
    font: inherit;
  }}
  #overlay-body {{ overflow: auto; padding: 8px 16px 16px; }}
  .stats {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px 20px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 14px;
    margin: 4px 0 12px;
    font-size: 12px;
  }}
  .stats .stat {{ white-space: nowrap; }}
  .stats .stat .k {{ color: var(--dim); margin-right: 4px; }}
  .stats .stat.warn .v {{ color: #e0c060; }}
  table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
  colgroup col.word {{ width: 76px; }}
  colgroup col.raw {{ width: 96px; }}
  colgroup col.flags {{ width: 150px; }}
  colgroup col.dis {{ width: auto; }}
  th, td {{
    padding: 1px 6px;
    text-align: left;
    white-space: nowrap;
    font-size: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  thead th {{
    position: sticky;
    top: 0;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    color: var(--dim);
    z-index: 2;
  }}
  tbody tr:nth-child(even) {{ background: var(--row-even); }}
  tbody tr:nth-child(odd) {{ background: var(--row-odd); }}
  tbody tr.entry-point:nth-child(even) {{ background: var(--row-entry-even); }}
  tbody tr.entry-point:nth-child(odd) {{ background: var(--row-entry-odd); }}
  tbody tr:hover {{ background: var(--row-hover) !important; }}
  td.offset {{ color: var(--dim); }}
  td.raw {{ color: var(--dim); }}
  td.flags {{ color: #6fd06f; }}
  td.dis {{ border-left: 3px solid transparent; padding-left: 8px; }}
  tr.kind-Sequential td.dis {{ border-left-color: #3a5c96; }}
  tr.kind-Branch td.dis, tr.kind-Jump td.dis {{ border-left-color: #d19f3c; }}
  tr.kind-RegJump td.dis {{ border-left-color: #c05f5f; }}
  tr.kind-Excluded td.dis {{ border-left-color: #82509a; }}
  tr.kind-RegionBoundary td.dis {{ border-left-color: #4c4c4c; }}
  tr.fallback td.dis {{ outline: 1px solid #e0c060; outline-offset: -2px; }}
  tr.entry-point td.offset::before {{ content: "\25b8 "; color: #6fd06f; }}
  .legend {{ display: flex; gap: 12px; color: var(--dim); font-size: 11px; margin: 4px 0 12px; flex-wrap: wrap; }}
  .legend span.swatch {{ display: inline-block; width: 10px; height: 3px; margin-right: 4px; vertical-align: 2px; }}
</style>
</head>
<body>
<h1>jitv2 dump</h1>
<div class="summary" id="summary"></div>

<h2>0x08000000 window (low RAM, 256MB)</h2>
<div class="region-label">128 pages/row &middot; 4KB/page &middot; click a page to open it</div>
<div class="region" id="region-low"></div>

<h2>0x20000000 window (high RAM, 256MB)</h2>
<div class="region-label">128 pages/row &middot; 4KB/page &middot; click a page to open it</div>
<div class="region" id="region-high"></div>

<h2>0x1FC00000 window (PROM, 1MB)</h2>
<div class="region-label">128 pages/row &middot; 4KB/page &middot; click a page to open it</div>
<div class="region" id="region-prom"></div>

<div class="heat-scale"><span>entry density:</span><span class="bar"></span><span>0 &rarr; most-populated page</span></div>

<div id="overlay">
  <div id="overlay-panel">
    <div id="overlay-header">
      <div id="overlay-title"></div>
      <button id="overlay-close">close</button>
    </div>
    <div id="overlay-body"></div>
  </div>
</div>

<script id="data" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const LOMEM_BASE = 0x08000000, HIMEM_BASE = 0x20000000, PROM_BASE = 0x1FC00000, PROM_SIZE = 1024*1024, PAGE_SIZE = 4096;
const RAM_PAGES_PER_WINDOW = (0x08000000 / PAGE_SIZE); // 256MB / 4KB
const PROM_PAGES = PROM_SIZE / PAGE_SIZE;
const COLS = 128;

function fmtHex(n, digits) {{ return '0x' + n.toString(16).padStart(digits, '0'); }}
function fmtBytes(n) {{
  if (n >= 1024*1024) return (n/(1024*1024)).toFixed(2) + 'MB';
  if (n >= 1024) return (n/1024).toFixed(2) + 'KB';
  return n + 'B';
}}

const byPfn = new Map(DATA.pages.map(p => [p.pfn, p]));
const low = DATA.pages.filter(p => p.window === 'low');
const high = DATA.pages.filter(p => p.window === 'high');
const prom = DATA.pages.filter(p => p.window === 'prom');

let totalEntries = 0, totalCodeSize = 0, totalCompiles = 0;
for (const p of DATA.pages) {{
  totalEntries += p.entry_count;
  totalCodeSize += (p.code_size || 0);
  totalCompiles += (p.compile_count || 0);
}}
document.getElementById('summary').textContent =
  `${{DATA.pages.length}} claimed pages (${{low.length}} low / ${{high.length}} high / ${{prom.length}} prom) ` +
  `· ${{totalEntries}} published entries · ${{fmtBytes(totalCodeSize)}} compiled code · ${{totalCompiles}} total compiles`;

// Entry-density heat color: hottest page (most published entries) anchors
// the top of the scale so the map stays useful regardless of absolute
// density — a 40-entry page should read as "hot" on a run where nothing
// else broke 50, not stay a dim blue forever waiting for a 1024/1024 page.
const maxDensity = Math.max(1, ...DATA.pages.map(p => p.entry_count));
function densityColor(count) {{
  const t = Math.min(1, count / maxDensity);
  // blue (cold) -> green -> yellow -> red (hot), matching .heat-scale .bar
  const hue = 210 - t * 210; // 210deg -> 0deg
  const light = 32 + t * 23;
  return `hsl(${{hue}} 70% ${{light}}%)`;
}}

function buildRegion(containerId, pages, base, pageCount) {{
  const el = document.getElementById(containerId);
  const rows = Math.ceil(pageCount / COLS);
  const claimed = new Map(pages.map(p => [Math.floor((p.phys_addr - base) / PAGE_SIZE), p]));
  const frag = document.createDocumentFragment();
  // Sparse: only render rows that actually contain a claimed page — a full
  // grid of every possible page in a 256MB window would be enormous DOM for
  // what's almost always a mostly-empty window.
  const usedRows = new Set([...claimed.keys()].map(i => Math.floor(i / COLS)));
  const sortedRows = [...usedRows].sort((a,b) => a-b);
  for (const row of sortedRows) {{
    for (let col = 0; col < COLS; col++) {{
      const idx = row * COLS + col;
      const page = claimed.get(idx);
      const cell = document.createElement('div');
      cell.className = 'cell' + (page ? ' claimed' : '');
      if (page) {{
        cell.style.background = densityColor(page.entry_count);
        cell.title = `pfn=${{fmtHex(page.pfn,8)}} phys=${{fmtHex(page.phys_addr,8)}} entries=${{page.entry_count}} size=${{fmtBytes(page.code_size||0)}} compiles=${{page.compile_count||0}}`;
        cell.addEventListener('click', () => openPage(page.pfn));
      }}
      frag.appendChild(cell);
    }}
  }}
  el.appendChild(frag);
  if (rows > sortedRows.length) {{
    const note = document.createElement('div');
    note.className = 'region-label';
    note.textContent = `(showing ${{sortedRows.length}} of ${{rows}} possible rows — empty rows omitted)`;
    el.after(note);
  }}
}}
buildRegion('region-low', low, LOMEM_BASE, RAM_PAGES_PER_WINDOW);
buildRegion('region-high', high, HIMEM_BASE, RAM_PAGES_PER_WINDOW);
buildRegion('region-prom', prom, PROM_BASE, PROM_PAGES);

const KIND_LEGEND = ['Sequential','Branch','Jump','RegJump','Excluded','RegionBoundary'];

function openPage(pfn) {{
  const page = byPfn.get(pfn);
  if (!page) return;
  document.getElementById('overlay-title').textContent =
    `pfn=${{fmtHex(page.pfn,8)}} phys=${{fmtHex(page.phys_addr,8)}} (${{page.window}}) gen=${{page.gen}} func=${{page.func}}${{page.stale ? ' STALE' : ''}}`;

  const hostBytes = page.code_size;
  const guestInstrs = page.rows.length;
  const bytesPerInstr = (hostBytes != null && guestInstrs > 0) ? (hostBytes / guestInstrs).toFixed(1) : null;

  const stats = document.createElement('div');
  stats.className = 'stats';
  const stat = (k, v, warn) => `<span class="stat${{warn ? ' warn' : ''}}"><span class="k">${{k}}</span><span class="v">${{v}}</span></span>`;
  stats.innerHTML =
    stat('entries', page.entry_count) +
    stat('denylisted', page.denylisted_count, page.denylisted_count > 0) +
    stat('compiles', page.compile_count != null ? page.compile_count : '?', (page.compile_count||0) > 10) +
    stat('calls', page.call_count != null ? page.call_count.toLocaleString() : '?') +
    stat('fr1', page.fr1) +
    stat('guest code', fmtBytes(page.guest_code_size) + ` (${{guestInstrs}} instrs)`) +
    stat('host code', hostBytes != null ? fmtBytes(hostBytes) : '?') +
    stat('host B/guest instr', bytesPerInstr != null ? bytesPerInstr : '?');

  const KIND_COLOR = {{
    Sequential: '#3a5c96', Branch: '#d19f3c', Jump: '#d19f3c',
    RegJump: '#c05f5f', Excluded: '#82509a', RegionBoundary: '#4c4c4c',
  }};
  const legend = document.createElement('div');
  legend.className = 'legend';
  legend.innerHTML = KIND_LEGEND.map(k => `<span><span class="swatch" style="background:${{KIND_COLOR[k]}}"></span>${{k}}</span>`).join('');

  const table = document.createElement('table');
  const colgroup = '<colgroup><col class="word"><col class="raw"><col class="flags"><col class="dis"></colgroup>';
  const head = '<thead><tr><th>offset</th><th>raw</th><th>flags</th><th>disassembly</th></tr></thead>';

  let body = '<tbody>';
  for (const r of page.rows) {{
    const flags = [];
    if (r.is_entry_point) flags.push('ENTRY');
    if (r.is_fallback) flags.push('fallback');
    if (r.is_branch_target) flags.push('branch-target');
    if (r.is_slot_only) flags.push('slot-only');
    const rowCls = [
      `kind-${{r.kind}}`,
      r.is_entry_point ? 'entry-point' : '',
      r.is_fallback ? 'fallback' : '',
    ].filter(Boolean).join(' ');
    body += `<tr class="${{rowCls}}" title="${{r.kind}}"><td class="offset">${{fmtHex(r.word*4,4)}}</td>` +
      `<td class="raw">${{r.raw.toString(16).padStart(8,'0')}}</td>` +
      `<td class="flags">${{flags.join(' ')}}</td>` +
      `<td class="dis">${{escapeHtml(r.dis)}}</td></tr>`;
  }}
  body += '</tbody>';
  table.innerHTML = colgroup + head + body;

  const bodyEl = document.getElementById('overlay-body');
  bodyEl.innerHTML = '';
  bodyEl.appendChild(stats);
  bodyEl.appendChild(legend);
  bodyEl.appendChild(table);
  document.getElementById('overlay').classList.add('open');
}}

function escapeHtml(s) {{
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}}

document.getElementById('overlay-close').addEventListener('click', () => {{
  document.getElementById('overlay').classList.remove('open');
}});
document.getElementById('overlay').addEventListener('click', (e) => {{
  if (e.target.id === 'overlay') document.getElementById('overlay').classList.remove('open');
}});
document.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape') document.getElementById('overlay').classList.remove('open');
}});
</script>
</body>
</html>
"##)
}


/// Collect every claimed page's single compiled-region state, render it,
/// and write it to `path` — the whole body of `j2 html`'s `"html"` match
/// arm, pulled out so the two designs' versions can live side by side
/// instead of being spliced together inline. `analyzer` mirrors the
/// caller's own `exec.jitv2_inline_analyzer` (a fresh `Analyzer` is fine
/// here too — this only uses its `walk_multi_entry` scratch buffer, never
/// reads back anything left over from a previous dispatch).
pub fn write_jitv2_html(
    jitv2: &Arc<Mutex<Jitv2>>,
    bus: &Arc<dyn BusDevice>,
    analyzer: &mut crate::jitv2::analyzer::Analyzer,
    path: &str,
    writer: &mut dyn Write,
) -> Result<(), String> {
    let jit = jitv2.lock();
    let mut pages: Vec<jitv2_html::PageDump> = Vec::new();
    // Snapshot every claimed page. §13: one compiled function per page, not
    // one per entry — so this runs exactly one `walk_multi_entry` per page
    // (over every currently-published offset, the same union `comp.rs`
    // itself compiles from), not one walk per entry the pre-§13 version
    // did. O(pages * ENTRIES_PER_PAGE) like the existing full-pool scans
    // (Jitv2::code_bytes_used doc comment) — fine for an on-demand dump,
    // not a hot path.
    for page in jit.claimed_pages() {
        let phys_addr = page.pfn * crate::jitv2::PAGE_SIZE;
        // PROM_BASE/PROM_SIZE mirror the private constants in
        // `prom.rs`/`physical.rs` (0x1FC00000, 1MB) — jitv2 can and does
        // compile boot-time PROM code (physical alias of
        // 0xFFFFFFFF_BFC00000), a third window distinct from both RAM
        // banks, easy to forget when eyeballing only the two documented
        // 256MB RAM windows.
        const PROM_BASE: u32 = 0x1FC00000;
        const PROM_END: u32 = 0x1FD00000;
        let window = if (PROM_BASE..PROM_END).contains(&phys_addr) {
            "prom"
        } else if phys_addr >= HIMEM_BASE {
            "high"
        } else {
            "low"
        };
        let page_gen = page.current_gen();

        let mut words = [0u32; crate::jitv2::ENTRIES_PER_PAGE];
        for (i, w) in words.iter_mut().enumerate() {
            *w = bus.read32(phys_addr + (i as u32) * 4).data;
        }

        // §13: one function/gen/instr_count/code_size/call_count/
        // compile_count for the whole page now.
        let func = page.func();
        let entry_gen = page.entry_gen();
        let stale = entry_gen != page_gen;
        #[cfg(feature = "developer")]
        let (instr_count, code_size, call_count, compile_count) = (
            Some(page.instr_count.load(Ordering::Relaxed)),
            Some(page.code_size.load(Ordering::Relaxed)),
            Some(page.call_count.load(Ordering::Relaxed)),
            Some(page.compile_count.load(Ordering::Relaxed)),
        );
        #[cfg(not(feature = "developer"))]
        let (instr_count, code_size, call_count, compile_count): (Option<u32>, Option<u32>, Option<u64>, Option<u32>) = (None, None, None, None);

        let mut entry_count = 0u32;
        let mut denylisted_count = 0u32;
        let mut entry_words: Vec<u16> = Vec::new();
        for off in 0..crate::jitv2::ENTRIES_PER_PAGE {
            if page.is_denylisted(off) { denylisted_count += 1; }
            if !page.is_published(off) { continue; }
            entry_count += 1;
            entry_words.push(off as u16);
        }

        let mut rows: Vec<jitv2_html::WordRow> = Vec::new();
        if !entry_words.is_empty() {
            let walked = analyzer.walk_multi_entry(&words, &entry_words, phys_addr, usize::MAX);
            rows = crate::jitv2::analyzer::instrs_linear(walked)
                .map(|instr| {
                    let kind = match crate::jitv2::analyzer::classify(instr.raw, instr.word, phys_addr) {
                        crate::jitv2::analyzer::Classify::Sequential => "Sequential",
                        crate::jitv2::analyzer::Classify::Branch { .. } => "Branch",
                        crate::jitv2::analyzer::Classify::Jump { .. } => "Jump",
                        crate::jitv2::analyzer::Classify::RegJump => "RegJump",
                        crate::jitv2::analyzer::Classify::Excluded => "Excluded",
                        crate::jitv2::analyzer::Classify::RegionBoundary => "RegionBoundary",
                    };
                    let paddr = phys_addr + (instr.word as u32) * 4;
                    jitv2_html::WordRow {
                        word: instr.word as u32,
                        raw: instr.raw,
                        dis: crate::mips_dis::disassemble(instr.raw, paddr as u64, None),
                        kind,
                        is_entry_point: instr.is_entry_point,
                        is_fallback: instr.is_fallback,
                        is_branch_target: instr.is_branch_target,
                        is_slot_only: instr.is_slot_only,
                    }
                })
                .collect();
        }
        // Guest bytes actually covered by this page's one compiled
        // function: 4 bytes/word for every distinct visited word
        // (instrs_linear already dedupes to one CompiledInstr per word
        // regardless of how many entries' walks reached it).
        let guest_code_size = (rows.len() as u32) * 4;

        pages.push(jitv2_html::PageDump {
            pfn: page.pfn,
            phys_addr,
            window,
            gen: page_gen,
            func: format!("{:#014x}", func as usize),
            stale,
            fr1: page.is_fr1(),
            entry_count,
            denylisted_count,
            instr_count,
            code_size,
            call_count,
            compile_count,
            guest_code_size,
            rows,
        });
    }
    drop(jit);

    let snapshot = jitv2_html::Snapshot { pages };
    let html = render_jitv2_html(&snapshot);
    match std::fs::File::create(path) {
        Ok(f) => {
            let mut bw = std::io::BufWriter::new(f);
            bw.write_all(html.as_bytes()).map_err(|e| format!("Cannot write {}: {}", path, e))?;
            std::io::Write::flush(&mut bw).map_err(|e| format!("Cannot write {}: {}", path, e))?;
            writeln!(writer, "jitv2: wrote {} claimed page(s) -> {}", snapshot.pages.len(), path).unwrap();
            Ok(())
        }
        Err(e) => Err(format!("Cannot open {}: {}", path, e)),
    }
}
