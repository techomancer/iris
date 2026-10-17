#!/usr/bin/env bash
# Publish the files in a directory as a GitHub release, replacing any release
# that already has the tag.
#
#   scripts/dist/publish-github.sh --tag TAG --sha SHA --title TITLE
#        --notes FILE --dir DIR [--prerelease] [--move-tag] [--previous REF]
#
#   --move-tag    point TAG at SHA first, creating it if needed (the rolling
#                 "nightly" tag). Without it TAG must already exist at SHA.
#   --previous    append the commits since REF (a tag or SHA) to the notes.
#                 For a vX.Y.Z tag GitHub's generated notes are appended instead.
#
# Needs gh, authenticated (GH_TOKEN) with contents: write on GITHUB_REPOSITORY.
#
# Order matters so the old release stays up until the new one is complete:
# everything is uploaded to a new draft first, and only then is the old release
# deleted and the draft published under the tag. A failed run leaves at most a
# stale draft, which the next run deletes.
source "$(dirname "$0")/lib.sh"

TAG=""; SHA=""; TITLE=""; NOTES=""; DIR=""; PRERELEASE=false; MOVE_TAG=0; PREVIOUS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --tag)        TAG="$2"; shift 2 ;;
        --sha)        SHA="$2"; shift 2 ;;
        --title)      TITLE="$2"; shift 2 ;;
        --notes)      NOTES="$2"; shift 2 ;;
        --dir)        DIR="$2"; shift 2 ;;
        --prerelease) PRERELEASE=true; shift ;;
        --move-tag)   MOVE_TAG=1; shift ;;
        --previous)   PREVIOUS="$2"; shift 2 ;;
        -h|--help)    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done
for v in TAG SHA TITLE NOTES DIR; do [ -n "${!v}" ] || die "--$(echo "$v" | tr 'A-Z' 'a-z') is required"; done
need gh jq
R="${GITHUB_REPOSITORY:?set GITHUB_REPOSITORY=owner/name}"
STAGING="${TAG}-staging"

retry() {
    local n
    for n in 1 2 3 4; do
        "$@" && return 0
        [ "$n" = 4 ] && return 1
        warn "attempt $n failed: $*; retrying"
        sleep $((n * 5))
    done
}

# ── Clear out drafts a failed run left behind ────────────────────────────────
gh api --paginate "repos/$R/releases?per_page=100" \
    --jq ".[] | select(.draft and .name == \"$STAGING\") | .id" |
while read -r id; do
    log "deleting stale draft $id"
    gh api -X DELETE "repos/$R/releases/$id"
done

# ── Notes ────────────────────────────────────────────────────────────────────
BODY=$(cat "$NOTES")
OLD_SHA=""
[ -z "$PREVIOUS" ] || OLD_SHA=$(gh api "repos/$R/commits/$PREVIOUS" --jq .sha 2>/dev/null || true)
if [ -n "$PREVIOUS" ]; then
    if [ -n "$OLD_SHA" ] && [ "$OLD_SHA" != "$SHA" ]; then
        CHANGES=$(gh api "repos/$R/compare/$OLD_SHA...$SHA" \
            --jq '.commits | reverse | .[:50][] | "- \(.commit.message | split("\n")[0]) (\(.sha[:8]))"' || true)
        if [ -n "$CHANGES" ]; then
            BODY="$BODY

## Changes since the previous build

$CHANGES

[Full comparison](https://github.com/$R/compare/${OLD_SHA:0:12}...${SHA:0:12})"
        fi
    fi
elif [ "$MOVE_TAG" = 0 ]; then
    GENERATED=$(gh api -X POST "repos/$R/releases/generate-notes" -f tag_name="$TAG" --jq .body || true)
    [ -n "$GENERATED" ] && BODY="$BODY

$GENERATED"
fi

# ── Draft with every asset ───────────────────────────────────────────────────
DRAFT=$(jq -n --arg name "$STAGING" --arg tag "$STAGING" --arg sha "$SHA" --arg body "$BODY" \
    '{tag_name: $tag, name: $name, target_commitish: $sha, body: $body, draft: true}' \
    | gh api -X POST "repos/$R/releases" --input - --jq .id)
log "created draft $DRAFT"

upload() {  # <file>
    local name; name=$(basename "$1")
    # A failed upload can leave a half-made asset that blocks the retry.
    gh api "repos/$R/releases/$DRAFT/assets" --jq ".[] | select(.name == \"$name\") | .id" \
        | while read -r id; do gh api -X DELETE "repos/$R/releases/assets/$id"; done
    gh api -X POST -H "Content-Type: application/octet-stream" \
        "https://uploads.github.com/repos/$R/releases/$DRAFT/assets?name=$name" \
        --input "$1" --jq .name >/dev/null
}
count=0
for f in "$DIR"/*; do
    [ -f "$f" ] || continue
    log "uploading $(basename "$f")"
    retry upload "$f" || die "could not upload $(basename "$f")"
    count=$((count + 1))
done
[ "$count" -gt 0 ] || die "nothing to upload in $DIR"

# ── Swap ─────────────────────────────────────────────────────────────────────
OLD=$(gh api "repos/$R/releases/tags/$TAG" --jq .id 2>/dev/null || true)
if [ -n "$OLD" ]; then
    log "deleting the previous $TAG release ($OLD)"
    gh api -X DELETE "repos/$R/releases/$OLD"
fi

if [ "$MOVE_TAG" = 1 ]; then
    if gh api "repos/$R/git/ref/tags/$TAG" >/dev/null 2>&1; then
        gh api -X PATCH "repos/$R/git/refs/tags/$TAG" -f sha="$SHA" -F force=true >/dev/null
    else
        gh api -X POST "repos/$R/git/refs" -f ref="refs/tags/$TAG" -f sha="$SHA" >/dev/null
    fi
    log "moved $TAG to $SHA"
else
    AT=$(gh api "repos/$R/commits/$TAG" --jq .sha)
    [ "$AT" = "$SHA" ] || die "tag $TAG is at $AT, not the built commit $SHA"
fi

LATEST=true
[ "$PRERELEASE" = true ] && LATEST=false
URL=$(jq -n --arg tag "$TAG" --arg name "$TITLE" --arg sha "$SHA" \
        --argjson pre "$PRERELEASE" --arg latest "$LATEST" \
    '{tag_name: $tag, name: $name, target_commitish: $sha, draft: false, prerelease: $pre, make_latest: $latest}' \
    | gh api -X PATCH "repos/$R/releases/$DRAFT" --input - --jq .html_url)
log "published $URL"
[ -n "${GITHUB_STEP_SUMMARY:-}" ] && echo "Published [$TITLE]($URL) with $count files." >> "$GITHUB_STEP_SUMMARY"
exit 0
