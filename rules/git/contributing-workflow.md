# Git workflow for Cursor agents and contributors

This file lives in the repo so **Cursor (and other AI-assisted) sessions have a
stable reference** for how we use git on IRIS. Agents do not retain memory across
chats; pointing `CLAUDE.md` here keeps the same rules in every session.

**Goal:** every change is **tracked in git** — sync `main`, branch, commit, push,
open a PR — instead of leaving large untracked diffs or mega-commits that are hard
to review. One logical change at a time.

Philip (Chronic Engineering / `chronic8000`) contributes to the canonical repo
at [techomancer/iris](https://github.com/techomancer/iris).

Workspace path: `c:\Users\chron\CURSOR-PROJECTS\iris-main`

**Agents: read this before any commit, push, or PR.**
---

## Remotes

| Remote | URL | Purpose |
|--------|-----|---------|
| `origin` | `https://github.com/techomancer/iris.git` | Canonical upstream — **fetch/pull only** |
| `upstream` | same as `origin` | Alias for fetch reference |
| `fork` | `https://github.com/chronic8000/iris.git` | **Push branches here** (direct push to `origin` often returns 403) |

Local `main` tracks `origin/main` and must stay aligned with techomancer before
starting new work.

---

## Per-change workflow

Do this **after every logical change** (one feature, one fix, one doc update — never
a mega-commit).

### 1. Sync before you branch

```powershell
cd c:\Users\chron\CURSOR-PROJECTS\iris-main
git fetch origin
git checkout main
git pull origin main
```

`main` is our equivalent of Dani's `upstream-main`: always the latest merged
techomancer code.

### 2. Cut a focused branch

Branch names: `docs/…`, `fix/…`, `feat/…`, `gui/…` — short and descriptive.

```powershell
git checkout -b fix/short-description
```

**Never** commit directly on `main`.

### 3. Make one focused change

- Minimize diff scope; match existing code style.
- Do **not** commit unrelated edits, CRLF-only churn, or local dev artifacts.

### 4. Review before commit

```powershell
git status --short
git diff
```

**Never commit:**

- `jit-profile.bin`, `rex-jit-profile.bin`, `*.bak`
- `iris-gui-stderr.txt`, `iris-gui-stdout.txt`
- `irix-install/*.overlay`, guest NVRAM/disk images unless explicitly requested
- `.pr-body.md` or other throwaway PR draft files

### 5. Commit (PowerShell — no bash heredocs)

Use `-m` twice for title + body:

```powershell
git add path/to/changed/files
git commit -m "area: short imperative summary" -m "Optional longer explanation of why."
```

Commit message style (match upstream):

- Prefix: `docs:`, `fix:`, `gui:`, `rex3:`, `ci:`, etc.
- One sentence summary in present tense; body explains **why**, not just what.

**Only create commits when the user asked for the change to be committed/PR'd**, or
when they explicitly say to push after each change.

### 6. Push to fork (not origin)

```powershell
git push -u fork fix/short-description
```

If push to `origin` fails with 403, that is expected — use `fork`.

### 7. Open PR to techomancer/iris

Write the body to a temp file (avoids PowerShell quoting pain), then:

```powershell
# create .pr-body-temp.md with Summary + Test plan sections
gh pr create --repo techomancer/iris `
  --head chronic8000:fix/short-description `
  --base main `
  --title "area: same as commit subject" `
  --body-file .pr-body-temp.md
```

Delete the temp body file after PR creation.

PR template:

```markdown
## Summary
- Bullet(s): what changed and why

## Test plan
- [ ] What was verified (build, manual test, docs-only skim, etc.)
```

Return the PR URL to the user.

### 8. After merge — clean up local

```powershell
git checkout main
git pull origin main
git branch -d fix/short-description
```

Optionally sync the fork's `main` later (not required for every PR):

```powershell
git push fork main
```

---

## Coordination

- Announce in Discord (#emulation / SGUG) when starting non-trivial work so we
  don't collide with Dani (`danifunker`) or Dominik (`DominBear`).
- Prefer **one PR per feature**; if a PR grows large, split before pushing.
- Run `cargo build` / relevant smoke tests on the fork when touching emulator
  code; docs-only changes can skip full IRIX boot.

---

## Quick reference (copy-paste block)

```powershell
cd c:\Users\chron\CURSOR-PROJECTS\iris-main
git fetch origin
git checkout main
git pull origin main
git checkout -b fix/my-change
# ... edit files ...
git status --short
git add <files>
git commit -m "fix: describe change" -m "Why this was needed."
git push -u fork fix/my-change
gh pr create --repo techomancer/iris --head chronic8000:fix/my-change --base main --title "fix: describe change" --body-file .pr-body-temp.md
```

---

## Example (completed)

PR #53 — README fork section cleanup:

- Branch: `docs/readme-unified-platform-section`
- Commit: `docs: replace stale fork README section with unified platform overview`
- PR: https://github.com/techomancer/iris/pull/53

---

## What we do NOT do

- No `git push --force` to `main` on techomancer or fork without explicit user request.
- No `git config` changes.
- No amending commits unless user explicitly asks and commit was not pushed.
- No pushing secrets, disk images with user data, or build profiles.
- No single giant PRs spanning unrelated areas (HAL2 + REX3 + GUI + docs in one go).
