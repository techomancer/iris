//! `j2 html`: complete self-contained implementation for the default
//! (`not(feature = "j2wp")`) one-function-per-entry-point design — data
//! structs (`jitv2_html`), the HTML/JS renderer (`render_jitv2_html`), and
//! the page-snapshot collector (`write_jitv2_html`) that populates a
//! `jitv2_html::Snapshot` from every claimed `PhysicalCodePage`'s `entries`
//! table. See `jitv2_html_j2wp.rs` for the whole-page design's complete,
//! separate counterpart — the two don't share a collection body (or even
//! the same `Snapshot` shape) because they read a compiled entry's
//! func/gen/instr_count from structurally different places (`entries[off]`
//! vs the `requested`/`compiled`/`denied` bitmaps + single per-page `func`).

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

    /// One analyzer-classified, visited word within an entry's reachable
    /// region — `Analyzer::walk_bounded`'s `instrs_linear()` output, reduced
    /// to what the page renders per coverage cell.
    #[derive(Serialize)]
    pub struct WordCoverage {
        /// Word offset within the page (0..1024).
        pub word: u32,
        /// `analyzer::classify()`'s variant name (Sequential/Branch/Jump/
        /// RegJump/Excluded/RegionBoundary) — used directly as a CSS class
        /// for per-instruction-type coloring.
        pub kind: &'static str,
        pub is_fallback: bool,
        pub is_branch_target: bool,
    }

    /// One published `JitEntry` within a page, plus its analyzer coverage.
    #[derive(Serialize)]
    pub struct EntryDump {
        /// Word offset within the page this entry was compiled from.
        pub offset: u32,
        pub func: String,
        pub gen: u64,
        pub stale: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instr_count: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub code_size: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub call_count: Option<u64>,
        /// Reachable words from this entry's own analyzer walk, ascending
        /// offset order — one rendered coverage column per entry.
        pub coverage: Vec<WordCoverage>,
    }

    /// One claimed `PhysicalCodePage`.
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
        pub published_count: u32,
        pub denylisted_count: u32,
        pub total_code_size: u64,
        pub entries: Vec<EntryDump>,
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
    --col-hover: rgba(111, 162, 255, 0.10);
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
  table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
  colgroup col.fixed {{ width: 76px; }}
  colgroup col.cov {{ width: 15px; }}
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
  th.cov {{
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    text-align: right;
    padding: 4px 0;
    font-size: 10px;
    cursor: default;
  }}
  tbody tr:nth-child(even) {{ background: var(--row-even); }}
  tbody tr:nth-child(odd) {{ background: var(--row-odd); }}
  tbody tr.has-entry:nth-child(even) {{ background: var(--row-entry-even); }}
  tbody tr.has-entry:nth-child(odd) {{ background: var(--row-entry-odd); }}
  tbody tr:hover {{ background: var(--row-hover) !important; }}
  td.offset {{ color: var(--dim); }}
  td.flags-published {{ color: #6fd06f; }}
  td.flags-denylisted {{ color: #e06a6a; }}
  td.cov {{
    width: 15px;
    padding: 0;
    text-align: center;
    position: relative;
    border-left: 1px solid rgba(255,255,255,0.04);
  }}
  td.cov.col-hover, th.cov.col-hover {{ background: var(--col-hover); }}
  .cov-Sequential {{ background: #2d4a7a; }}
  .cov-Branch {{ background: #b8862f; }}
  .cov-Jump {{ background: #b8862f; }}
  .cov-RegJump {{ background: #a04f4f; }}
  .cov-Excluded {{ background: #6a3f7a; }}
  .cov-RegionBoundary {{ background: #3a3a3a; }}
  td.cov.col-hover.cov-Sequential {{ background: #3a5c96; }}
  td.cov.col-hover.cov-Branch, td.cov.col-hover.cov-Jump {{ background: #d19f3c; }}
  td.cov.col-hover.cov-RegJump {{ background: #c05f5f; }}
  td.cov.col-hover.cov-Excluded {{ background: #82509a; }}
  td.cov.col-hover.cov-RegionBoundary {{ background: #4c4c4c; }}
  .cov.fallback {{ outline: 1px solid #e0c060; outline-offset: -1px; }}
  .legend {{ display: flex; gap: 12px; color: var(--dim); font-size: 11px; margin: 4px 0 12px; flex-wrap: wrap; }}
  .legend span.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }}
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

let totalEntries = 0, totalCodeSize = 0;
for (const p of DATA.pages) {{ totalEntries += p.published_count; totalCodeSize += p.total_code_size; }}
document.getElementById('summary').textContent =
  `${{DATA.pages.length}} claimed pages (${{low.length}} low / ${{high.length}} high / ${{prom.length}} prom) ` +
  `· ${{totalEntries}} published entries · ${{fmtBytes(totalCodeSize)}} compiled code`;

// Entry-density heat color: hottest page (most published entries) anchors
// the top of the scale so the map stays useful regardless of absolute
// density — a 40-entry page should read as "hot" on a run where nothing
// else broke 50, not stay a dim blue forever waiting for a 1024/1024 page.
const maxDensity = Math.max(1, ...DATA.pages.map(p => p.published_count));
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
        cell.style.background = densityColor(page.published_count);
        cell.title = `pfn=${{fmtHex(page.pfn,8)}} phys=${{fmtHex(page.phys_addr,8)}} entries=${{page.published_count}} size=${{fmtBytes(page.total_code_size)}}`;
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
    `pfn=${{fmtHex(page.pfn,8)}} phys=${{fmtHex(page.phys_addr,8)}} (${{page.window}}) gen=${{page.gen}} ` +
    `entries=${{page.published_count}} denylisted=${{page.denylisted_count}} code=${{fmtBytes(page.total_code_size)}}`;

  const legend = document.createElement('div');
  legend.className = 'legend';
  legend.innerHTML = KIND_LEGEND.map(k => `<span><span class="swatch cov-${{k}}"></span>${{k}}</span>`).join('');

  // One coverage-word map per entry, keyed by word offset, for O(1) lookup
  // while building each of the 1024 rows below.
  const covByEntry = page.entries.map(e => {{
    const m = new Map();
    for (const w of e.coverage) m.set(w.word, w);
    return m;
  }});

  const table = document.createElement('table');
  let colgroup = '<colgroup><col class="fixed"><col class="fixed"><col class="fixed"><col class="fixed">';
  for (let i = 0; i < page.entries.length; i++) {{ colgroup += '<col class="cov">'; }}
  colgroup += '</colgroup>';

  // Column headers: each entry's page offset drawn as 3 vertical hex
  // digits (000..FFF), matching the row offsets' own hex format instead of
  // an arbitrary e0/e1/eN index — denser and directly comparable to the
  // offset column at a glance.
  let head = '<thead><tr><th>offset</th><th>flags</th><th>func</th><th>size</th>';
  page.entries.forEach((e, i) => {{
    const hex3 = e.offset.toString(16).padStart(3, '0');
    head += `<th class="cov" data-col="${{i}}" title="entry @ ${{fmtHex(e.offset*4,4)}} (${{e.func}})">${{hex3}}</th>`;
  }});
  head += '</tr></thead>';

  const entryAtOffset = new Map(page.entries.map((e,i) => [e.offset, i]));
  let body = '<tbody>';
  for (let word = 0; word < 1024; word++) {{
    const isEntry = entryAtOffset.has(word);
    const entryIdx = entryAtOffset.get(word);
    const entry = isEntry ? page.entries[entryIdx] : null;
    let flagsCls = '', flagsText = '';
    if (entry) {{ flagsCls = 'flags-published'; flagsText = 'PUBLISHED' + (entry.stale ? ' STALE' : ''); }}
    body += `<tr class="${{entry ? 'has-entry' : ''}}" data-row="${{word}}"><td class="offset">${{fmtHex(word*4,4)}}</td>` +
      `<td class="${{flagsCls}}">${{flagsText}}</td>` +
      `<td>${{entry ? entry.func : ''}}</td>` +
      `<td>${{entry && entry.code_size != null ? entry.code_size : ''}}</td>`;
    for (let i = 0; i < covByEntry.length; i++) {{
      const w = covByEntry[i].get(word);
      if (w) {{
        const fb = w.is_fallback ? ' fallback' : '';
        body += `<td class="cov cov-${{w.kind}}${{fb}}" data-col="${{i}}" title="${{w.kind}}${{w.is_fallback ? ' (fallback)' : ''}}"></td>`;
      }} else {{
        body += `<td class="cov" data-col="${{i}}"></td>`;
      }}
    }}
    body += '</tr>';
  }}
  body += '</tbody>';
  table.innerHTML = colgroup + head + body;

  // Current-row/current-column highlight: track the hovered cell's column
  // index and toggle it across every row's same-index <td>, alongside the
  // plain CSS :hover for the row itself — cheap enough (1024 rows) to do on
  // every mousemove without debouncing.
  let hoveredCol = null;
  const setColHover = (col) => {{
    if (hoveredCol === col) return;
    if (hoveredCol !== null) {{
      table.querySelectorAll(`[data-col="${{hoveredCol}}"]`).forEach(el => el.classList.remove('col-hover'));
    }}
    hoveredCol = col;
    if (hoveredCol !== null) {{
      table.querySelectorAll(`[data-col="${{hoveredCol}}"]`).forEach(el => el.classList.add('col-hover'));
    }}
  }};
  table.addEventListener('mouseover', (e) => {{
    const cell = e.target.closest('td.cov, th.cov');
    setColHover(cell ? cell.dataset.col : null);
  }});
  table.addEventListener('mouseleave', () => setColHover(null));

  const bodyEl = document.getElementById('overlay-body');
  bodyEl.innerHTML = '';
  bodyEl.appendChild(legend);
  bodyEl.appendChild(table);
  document.getElementById('overlay').classList.add('open');
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



/// Collect every claimed page's compiled-entry state, render it, and write
/// it to `path` — the whole body of `j2 html`'s `"html"` match arm, pulled
/// out so the two designs' versions can live side by side instead of being
/// spliced together inline. `analyzer` and `bus` mirror the caller's own
/// `exec.jitv2_inline_analyzer`/`exec.sysad` (a fresh `Analyzer` is fine
/// here too — this only uses its `walk_bounded` scratch buffer, never reads
/// back anything left over from a previous dispatch).
pub fn write_jitv2_html(
    jitv2: &Arc<Mutex<Jitv2>>,
    bus: &Arc<dyn BusDevice>,
    analyzer: &mut crate::jitv2::analyzer::Analyzer,
    path: &str,
    writer: &mut dyn Write,
) -> Result<(), String> {
    let jit = jitv2.lock();
    let mut pages: Vec<jitv2_html::PageDump> = Vec::new();
    // Snapshot every claimed page, running the same analyzer walk `j2
    // analyze` uses per published entry. O(pages * ENTRIES_PER_PAGE) like
    // the existing full-pool scans (Jitv2::code_bytes_used doc comment) —
    // fine for an on-demand dump, not a hot path.
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

        let mut entries: Vec<jitv2_html::EntryDump> = Vec::new();
        let mut published_count = 0u32;
        let mut denylisted_count = 0u32;
        let mut total_code_size = 0u64;
        for off in 0..crate::jitv2::ENTRIES_PER_PAGE {
            if page.is_denylisted(off) { denylisted_count += 1; }
            if !page.is_published(off) { continue; }
            published_count += 1;
            let entry = &page.entries[off];
            let entry_gen = entry.gen.load(Ordering::Relaxed);
            let stale = entry_gen != page_gen;

            #[cfg(feature = "developer")]
            let (instr_count, code_size, call_count) = (
                Some(entry.instr_count as u32),
                Some(entry.code_size),
                Some(entry.call_count.load(Ordering::Relaxed)),
            );
            #[cfg(not(feature = "developer"))]
            let (instr_count, code_size, call_count): (Option<u32>, Option<u32>, Option<u64>) = (None, None, None);
            if let Some(sz) = code_size { total_code_size += sz as u64; }

            let (walked, _non_empty) = analyzer.walk_bounded(&words, off as u16, phys_addr, usize::MAX);
            let coverage: Vec<jitv2_html::WordCoverage> = crate::jitv2::analyzer::instrs_linear(walked)
                .map(|instr| {
                    let kind = match crate::jitv2::analyzer::classify(instr.raw, instr.word, phys_addr) {
                        crate::jitv2::analyzer::Classify::Sequential => "Sequential",
                        crate::jitv2::analyzer::Classify::Branch { .. } => "Branch",
                        crate::jitv2::analyzer::Classify::Jump { .. } => "Jump",
                        crate::jitv2::analyzer::Classify::RegJump => "RegJump",
                        crate::jitv2::analyzer::Classify::Excluded => "Excluded",
                        crate::jitv2::analyzer::Classify::RegionBoundary => "RegionBoundary",
                    };
                    jitv2_html::WordCoverage {
                        word: instr.word as u32,
                        kind,
                        is_fallback: instr.is_fallback,
                        is_branch_target: instr.is_branch_target,
                    }
                })
                .collect();

            entries.push(jitv2_html::EntryDump {
                offset: off as u32,
                func: format!("{:#014x}", entry.func as usize),
                gen: entry_gen,
                stale,
                instr_count,
                code_size,
                call_count,
                coverage,
            });
        }

        pages.push(jitv2_html::PageDump {
            pfn: page.pfn,
            phys_addr,
            window,
            gen: page_gen,
            published_count,
            denylisted_count,
            total_code_size,
            entries,
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
