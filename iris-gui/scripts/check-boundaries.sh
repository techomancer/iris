#!/usr/bin/env bash
# Layer rules for the IRIS front ends (clanker/plans/iris-newgui-plan.md, "Layers").
#
# Enforced (exit 1):
#   - iris-gui/core names no UI toolkit (egui, eframe, rfd, objc2, gtk, glib,
#     Win32 UI, and winit beyond its plain `keyboard` key codes).
#   - iris-gui/core/src/controller spawns no raw threads: slow work is a
#     model::job::Job.
#   - shells listed in ENFORCED reach the emulator and the model only through
#     iris_gui_core::controller, and spawn no raw threads.
# Reported (the count only goes down; a shell that reaches 0 moves to ENFORCED):
#   - shells naming the emulator (`iris::`) or the model
#     (`iris_gui_core::model`, `core::model`) directly, and raw thread spawns.
#
# The egui app (iris-gui/src) is on its way out and is not counted: it may
# keep reaching past the controllers until it is removed. `-v` still lists it.
#
# Usage: iris-gui/scripts/check-boundaries.sh [-v]   (-v lists the files behind each count)
set -euo pipefail
cd "$(dirname "$0")/.."

VERBOSE=0
[ "${1:-}" = "-v" ] && VERBOSE=1

SHELLS=()
for d in frontends/*/src; do [ -d "$d" ] && SHELLS+=("$d"); done
# Not counted (see above).
LEGACY=(src)
# Shells that must already be clean.
ENFORCED=()
for d in frontends/*/src; do [ -d "$d" ] && ENFORCED+=("$d"); done

TOOLKITS='egui|eframe|rfd|objc2_app_kit|objc2_ui_kit|gtk4?|glib|libadwaita|windows::Win32::UI'
# The Objective-C runtime and Foundation are platform services, not a UI
# toolkit: the model may use them (sandbox bookmarks, the container home),
# controllers may not.
OBJC='objc2|objc2_foundation'
MODEL_REF='\biris::|\biris_gui_core::model\b|\bcore::model\b'
SPAWN='thread::spawn|thread::Builder'

fail=0
error() {
  if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::error::$*"; else echo "error: $*"; fi
  fail=1
}
note() {
  if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::warning::$*"; else echo "$*"; fi
}

# --- enforced: core is toolkit-free, controllers are thread-free ---------------
if hits=$(grep -rnE "\b($TOOLKITS)::|extern crate ($TOOLKITS)\b|use ($TOOLKITS)\b" core/src); then
  error "iris-gui/core uses toolkit code:"; echo "$hits"
fi
if hits=$(grep -rnE "\b($OBJC)::" core/src/controller); then
  error "iris-gui/core/src/controller uses the Objective-C runtime:"; echo "$hits"
fi
if hits=$(grep -rnE "\bwinit::" core/src | grep -vE "winit::keyboard::" ); then
  error "iris-gui/core uses winit beyond its key codes:"; echo "$hits"
fi
if hits=$(grep -rnE "$SPAWN" core/src/controller); then
  error "iris-gui/core/src/controller spawns threads; use model::job::Job:"; echo "$hits"
fi

# --- shells -------------------------------------------------------------------
refs_in() { grep -rnoE "$MODEL_REF" "$@" 2>/dev/null || true; }
spawns_in() { grep -rnoE "$SPAWN" "$@" 2>/dev/null || true; }

if [ "${#ENFORCED[@]}" -gt 0 ]; then
  hits=$(refs_in "${ENFORCED[@]}")
  if [ -n "$hits" ]; then
    error "an enforced shell names the emulator or the model; re-export it from iris_gui_core::controller:"
    echo "$hits"
  fi
  hits=$(spawns_in "${ENFORCED[@]}")
  if [ -n "$hits" ]; then
    error "an enforced shell spawns raw threads:"; echo "$hits"
  fi
fi

refs=""
spawns=""
if [ "${#SHELLS[@]}" -gt 0 ]; then
  refs=$(refs_in "${SHELLS[@]}")
  spawns=$(spawns_in "${SHELLS[@]}")
fi
n_refs=$(printf '%s' "$refs" | grep -c . || true)
n_spawn=$(printf '%s' "$spawns" | grep -c . || true)
total=$((n_refs + n_spawn))
note "shells name the emulator or the model $n_refs times and spawn raw threads $n_spawn times"
if [ "$VERBOSE" = 1 ]; then
  echo "--- emulator/model references per file"
  printf '%s\n' "$refs" | grep . | cut -d: -f1 | sort | uniq -c | sort -rn || true
  echo "--- raw thread spawns per file"
  printf '%s\n' "$spawns" | grep . | cut -d: -f1 | sort | uniq -c | sort -rn || true
fi

if [ "$VERBOSE" = 1 ]; then
  legacy=$(refs_in "${LEGACY[@]}"; spawns_in "${LEGACY[@]}")
  echo "--- not counted: the egui app (${LEGACY[*]}) reaches past the controllers $(printf '%s' "$legacy" | grep -c . || true) times"
fi

echo "boundaries: $total"
exit "$fail"
