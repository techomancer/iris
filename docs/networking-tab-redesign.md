# Networking tab redesign — design & plan

Status (September 2026): **Phases 0, 1 and 2 shipped** (subnet logic, the
Networking tab, and the FTP ALG). NFS moved in-process instead
(`docs/nfsudp-plan.md`). **Phase 3 (in-app file bridge) was never started**, and
the suppaftp fork prompt it depended on (`docs/suppaftp-emu-fork-prompt.md`) was
removed from the repo. Original status: Phase 0 + 1 done; Phases 2 (FTP ALG) and
3 (in-app file bridge) pending. Captures the design agreed for the iris-gui
**Networking** config tab and the two new backend features it pulls in. Mirrors
the `docs/cow-chd-sync-plan.md` convention so the work survives across sessions.

## Goal

A regular user with **no networking knowledge** opens the Networking tab, accepts
the defaults, and it *just works*. Power users can still build odd subnets, port
forwards, and file-sharing setups. Everything must stay inside the Mac App Store
sandbox (no new entitlements; no reliance on spawning external binaries).

## Background — how the backend constrains the UI

- `nat_subnet` is stored as a **single CIDR string** (e.g. `192.168.0.0/24`).
  The two new UI controls (network + mask) just recompose into that one string.
- `iris::config::parse_nat_subnet` requires the **network address** (host bits
  zero), rejects prefix `> /30`, and always derives **gateway = network + 1**,
  **Indy `ec0` = network + 2**. So the host octet typed by the user is never the
  actual host — it must be `.0`, and both endpoints fall out of the subnet.
- `net.rs` already runs a real userspace TCP state machine *toward the guest*
  (`NatTcpEntry`: `server_seq` / `client_seq` / `client_win` / `retransmit`), and
  the port-forward path (`TcpFwdPending`) bridges a peer to the guest by injecting
  segments. Today that peer is always a host `TcpStream`. The **in-app file
  bridge** swaps that peer for an in-process protocol engine.
- NFS works by spawning an **external `unfsd`** (`src/main.rs`), which a sandboxed
  MAS app generally can't exec — so NFS likely doesn't function in the App Store
  build (open item below).

## Decisions

### A. Private-network controls (replace the single CIDR text field)
- **Network address** dropdown: *first-free 192.168.x* (default) / first-free
  172.16.x / first-free 10.x / **Custom…** (free-typed, snap-to-boundary + warn).
- **Subnet mask** dropdown — prefixes **8, 12, 16, 22, 24, 25, 26** (default
  **/24**) / **Custom…** (type bits → show mask). `/8 /12 /16` are the native
  sizes of the three RFC1918 blocks. Custom still reaches `/30`.
- **Live derived line**: gateway (net+1), Indy `ec0` (net+2), usable host range,
  broadcast, and conflict ✓/⚠.
- **`if-addrs`** dependency powers *first-free* selection and *overlap* warnings
  by reading the host's own interface addresses (no entitlement; does not trigger
  the macOS 15 Local Network prompt — that's for talking *to* LAN peers).
- **Sanity tiers / override dialog**:
  - **Hard** (engine can't represent): prefix `0` or `> /30`, malformed → blocked.
  - **Off-boundary** (e.g. `192.168.40.0/16`): snap to the real network
    (`192.168.0.0/16`) + a small grey note. Widening the mask is the common cause.
  - **Soft** (parses but unwise): not RFC1918, or overlaps a host network →
    confirmation dialog *"This networking configuration does not appear to be
    valid, please double-check…"* → **[Override Sanity Checks]** / **[Cancel]**
    (Cancel reverts to the previous good value). Big masks (`/8 /12`) overlap
    Docker/VPN ranges far more often, so this earns its keep.
  - Defaults are RFC1918 + /24 + conflict-checked, so a normal user **never**
    sees the dialog.
- The **troubleshooting** (`netfix`) dialog surfaces the Indy's *required* address
  (`ExpectedNet`); the config tab shows the host/gateway side.

### B. Port forwards
- `+ Add forward ▼` menu: **Telnet** (2323→23) / **FTP** (2121→21) / **Custom…**.
- Pre-filled rows use **unprivileged host ports** (>1024) — required to stay in
  the sandbox without root.

### C. FTP ALG (in `net.rs`) — *kept*
- Watches a forwarded FTP control stream, rewrites `PASV`/`PORT`, and opens the
  data port dynamically so the FTP **port-forward** works for a user's *own
  external* FTP client. Uses the existing `network.server` entitlement.

### D. In-app NAT-side file bridge — *the big one*
- The app is already a host on the virtual network (`gateway_ip`). An in-app
  client originates from a NAT address straight to the guest's daemon over the
  emulated SEEQ8003 — **no NAT traversal, no host sockets, no new entitlement**,
  and it reads/writes only user-selected local files (already entitled). This is
  the cleanest App Store file-sharing story and fills the NFS gap there.
- **Abstract the TCP-peer seam**: generalize the forward-path peer from a host
  `TcpStream` to an in-process stream trait, reusing the `NatTcpEntry` core.
- Ship **FTP-passive client first** (app opens both control + data connections to
  guest `ftpd` — no reverse channel), with **rcp/rsh behind the same seam later**
  (adds a guest-dials-back stderr channel + `.rhosts` trust).
- **UI**: folder picker (Browse + MRU), file list with push/pull, credentials for
  FTP.
- **Guest auto-provision** via the existing `netfix` serial path: enable the
  daemon / confirm trust or account, so the user only clicks Browse. The
  troubleshooting dialog and the bridge become one "set up sharing" flow.

### E. NFS
- Add an explanatory blurb + a **live-generated** mount command (gateway + folder
  fill in automatically).
- Keep NFS for the **notarized DMG** build; the App Store build relies on the
  in-app bridge. Final gating pending the unfsd-in-sandbox investigation.

## Entitlements / App Store
- **No new entitlements.** Port-forwards are already justified under
  `com.apple.security.network.server`; the FTP ALG reuses it. The in-app bridge
  adds **zero** socket surface. `if-addrs`/`getifaddrs` needs none.
- Unprivileged host ports keep forwards legal in the sandbox.

## Build order

| Phase | What | Risk | Status |
|---|---|---|---|
| **0** | Pure subnet/conflict logic (`iris-gui/src/netplan.rs`) + `if-addrs`: parse/compose CIDR, mask presets, first-free, conflict, sanity tiers, snap, derived addrs — 15 unit tests. No UI. | low | **done** |
| **1** | Networking tab UI on Phase 0: network preset combo, mask combo (8/12/16/22/24/25/26 + custom bits), CIDR field, derived line + conflict ✓/⚠, override modal (Cancel / Use suggested / Override), Add-forward menu (Telnet/FTP/Custom), NFS blurb + live mount cmd, troubleshooting-window Indy address. Network edits now mark cfg dirty. | low | **done** |
| **2** | FTP ALG in `net.rs`: rewrite passive-mode `227` replies on an inbound FTP control forward, bind a localhost data listener, register a transient (FIFO-bounded) data forward. Pure parse/rewrite unit-tested. | med | **done** |
| **3** | In-app file bridge — see decision below. | high | pending |

### Phase 3 plan (decided 2026-06-18)
Architecture **B** (pure NAT-side client, no host sockets). FTP reuses suppaftp's
protocol code via a **transport-generic fork** rather than a hand-rolled client —
spec in `docs/suppaftp-emu-fork-prompt.md`, since removed (make suppaftp's sync client generic
over an `FtpConnector` trait; default `TcpConnector` preserves behavior; passive
first; TLS only on the TCP path). With a virtual-net connector **no ALG/PASV
rewrite is needed** for the bridge (we reach the guest's PASV address directly on
the virtual net; the Phase 2 ALG stays only for the external-client forward).

Two tracks:
- **suppaftp-emu fork** — separate crate/repo, driven by the prompt (user kicks
  this off). IRIS links it by path/git.
- **IRIS foundation** (resumes once the crate API exists): (1) `NatEngine`
  in-process TCP-peer seam — generalize `TcpFwdPending`/`NatTcpEntry`'s
  `stream: TcpStream` to `{ HostSocket(TcpStream), InProc(..) }`; (2)
  `VirtualTcpStream: Read+Write` + a `VirtualConnector` impl of the crate's
  `FtpConnector`; (3) **dual-pane** UI (rfd Browse + MRU local, FTP `LIST`/`CWD`
  for the Indy); (4) rcp (hand-rolled — no crate exists); (5) guest daemon
  auto-provision via the `netfix` serial path.

Status: paused after delivering the fork prompt; IRIS side not started.

### Phase 2 implementation notes / limits
- Works because the NAT *relays application bytes* between an OS host socket and
  the userspace guest-side TCP, so the length-changing `227` rewrite needs no
  seq/ack surgery (the host stack re-sequences). `client_seq` still advances by
  the original payload length.
- Inbound FTP control connection identified by `server_ip == gateway && client_port == 21`.
- **Handles classic passive mode only** (`PASV` → `227`). Not handled yet:
  active mode (`PORT`), extended passive (`EPSV`/`229`), or a `227` split across
  TCP segments (no control-stream reassembly — fine for IRIX ftpd, which sends it
  in one segment).
- Data forwards are transient, FIFO-capped at 16, and truncated on reset /
  live-subnet-apply. **Unverified on a real boot** — needs a manual FTP transfer.

### Phase 1 implementation notes
- `cfg.nat_subnet` stays the raw-CIDR source of truth; preset/mask controls
  rewrite it (snapped via `to_cidr`), the CIDR field stays partial-typing-safe.
- Config-editor tab edits did **not** previously mark the config dirty (only the
  Memory/SCSI quick-menus did). The Networking tab now reports `changed` up via
  `TabOutcome` so its edits autosave; other tabs are unchanged (pre-existing gap
  left for a separate fix).
- The override modal fires only on a *committed* soft-invalid subnet (preset/mask
  pick or CIDR field losing focus), never on live keystrokes.

Phase 1 depends on 0. Phases 2 and 3 both touch `net.rs`; sequence them. Phase 3
is the largest.

## Open items
- **unfsd-in-sandbox**: confirm whether `unfsd` is bundled/signed and actually
  launches under the App Store sandbox. Drives whether the NFS panel is compiled
  out under `feature = "appstore"`.

## Verbiage (approved)

**NAT intro:** "IRIS gives the Indy its own private NAT network — the same trick
your home router uses. The Indy reaches the internet through IRIS, but nothing on
your real network can see it. Pick a subnet that doesn't overlap a network your
computer already uses (Wi-Fi, Ethernet, VPN, Docker…) — if it does, IRIS flags it
below, since an overlap can cut the Indy off from the internet."

**Port forwards helper:** "A port forward maps a port on your computer to a port
on the Indy, so host tools can reach guest services (log in, copy files…).
Inbound only, and it works once the guest is up on the NAT subnet. None exist by
default."

**NFS blurb:** "The Indy speaks NFS natively — the easiest, batteries-included way
to move files between your computer and the emulated machine. IRIS runs the NFS
server for you, backed by the folder you pick below; there's nothing to install
and no NFS know-how required. Pick a folder, boot the Indy, then mount it:
`mkdir /shared` / `mount <gateway>:<shared-dir> /shared`. Your files appear at
`/shared` on the Indy. The host address and path above update automatically to
match your subnet and folder."

**Override dialog:** "This networking configuration does not appear to be valid,
please double-check…" — buttons **[Override Sanity Checks]** / **[Cancel]**.
