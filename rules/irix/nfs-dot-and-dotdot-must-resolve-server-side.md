# NFS LOOKUP `..` must resolve on the server — the client does not synthesize it

## Symptom

Absolute paths on an in-core NFS share work, relative ones don't. On the guest:

```
# cd /mnt/share/sub
# cd ..
..: No such file or directory
# ls ../other
../other not found
```

`ls -a` also never shows `.` or `..`. Once it fails, it keeps failing for the
life of the mount even after the server is fixed — the client caches the
negative lookup in its DNLC, so **testing a fix needs a fresh mount**.

## Cause

`src/nfsudp.rs` funnelled every LOOKUP name through `valid_component()`, which
rejects `.`, `..`, empty, and anything with a separator. That is right for
CREATE/MKDIR/REMOVE/RENAME, but wrong for LOOKUP:

- SVR4-derived clients (IRIX's is one) short-circuit `.` in `nfslookup()` but
  **send `..` over the wire** on every DNLC miss — `lookuppn()` only intercepts
  `..` at a *mount root* (to cross back into the covering filesystem) and at the
  process root. Everywhere else the server answers it.
- So `lookup(dirid, "..")` returned `None` → `NFS3ERR_NOENT`/`NFSERR_NOENT`, and
  `chdir("..")`, `../x`, and getcwd-style walks all got ENOENT.

`NfsBacking::readdir()` also skipped `.`/`..` with a comment saying the wire
layer would synthesize them — it never did, in either the v2 or v3 encoder.

## Fix

No tree structure is needed: `id_to_path` already holds the full root-relative
path for every fileid, so the parent is one `PathBuf::pop()` away.

- `parent_id()` pops one component and interns the result; `pop()` returning
  `false` means we're at the root, so `..` there yields `ROOT_ID`. That keeps
  containment — `..` can never leave the export.
- `lookup()` handles `.`/`..` **before** `valid_component()`, gated on
  `is_dir(dirid)`. `valid_component()` is unchanged, so the mutating procedures
  still refuse both names.
- `readdir()` emits `.` and `..` first. Both wire encoders name-sort before
  paging and `.` < `..` < everything else in byte order, so the index-based
  READDIR cookie stays stable.

The `..` entry must carry the **same** fileid `lookup` interns for that path —
interning is keyed on the relative path, so it does. getcwd() walks up by
matching a child's fileid against the parent's directory entries; mismatched
ids there produce a wrong `pwd` rather than an error.

## Watch out

- Directory `nlink` is still hardcoded to 2 (`attr_from`). Now that `.`/`..` are
  real entries, a directory with subdirectories should report `2 + subdirs`.
  Tools using the link-count leaf optimization (`fts`, some `find`s) treat
  `nlink == 2` as "no subdirectories" and stop descending. Computing it properly
  costs a `read_dir` per GETATTR, which is why it was left alone.
- `..` off the *mount root* never reaches us — the client's VFS crosses back to
  the covered vnode itself. Returning `ROOT_ID` is the safe answer regardless.
- Regression test: `nfsudp::tests::dot_and_dotdot_navigate_without_escaping`.
