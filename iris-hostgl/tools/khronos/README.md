# The Khronos OpenGL API Registry, pinned

`gl.xml` is the OpenGL API Registry from
<https://github.com/KhronosGroup/OpenGL-Registry>, unmodified:

| | |
|---|---|
| file | `xml/gl.xml` |
| commit | `1cdd228e34966dd6b95bd203e9f84faba0f371a1` (2026-08-28) |
| SHA-256 | `b9ca2cfa5c676e901c20d34af3407f1687cde0f1336a5ff7a8974d04c7494ad3` |
| licence | Apache-2.0 (the file's own header: Copyright The Khronos Group Inc., `SPDX-License-Identifier: Apache-2.0`); the licence text is `LICENSE-Apache-2.0.txt` |

It is kept here so that regenerating the host GL protocol needs no network and
behaves the same on every platform. Only `../glapi_from_registry.py` and
`../gl_avail.py` read it, and only when the protocol is regenerated; building
IRIS does not. `gl_avail.py` writes `src/avail.rs`, which GL version or
extension provides each entry point, for backends whose library answers every
name (Linux: glvnd and Mesa); rerun it whenever `calls.rs` changes.

To move to a newer registry: replace `gl.xml`, update the commit and checksum
above, run `glapi_from_registry.py`, `glshim.py` and `gl_avail.py`, and check that
`iris-hostgl/src/calls.rs` and the guest's encoders change only where you
expect -- the registry has renamed parameters and retyped them before.
