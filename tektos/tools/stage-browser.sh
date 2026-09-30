#!/bin/sh
# stage-browser.sh <tag> — copy tektos into a browser-servable location with
# node:assert shimmed, so the suites can be run in WebKit.
#
# Needed because Node under iSH (the iOS sandbox) crashes in the heavier numeric
# loops, and because WebKit's module cache will serve a stale file for a URL that
# previously failed to load. Each run therefore gets a fresh directory, and each
# module URL is therefore new.
#
# Usage:  sh tools/stage-browser.sh t21
#         then open the staged directory through the browser harness.

set -e
tag="$1"
[ -n "$tag" ] || { echo "usage: stage-browser.sh <tag>"; exit 1; }
here=$(cd "$(dirname "$0")/.." && pwd)
out=${STAGE_ROOT:-/var/minis/workspace/dc}/"$tag"

rm -rf "$out"
mkdir -p "$out"
cp -r "$here"/src "$here"/tests "$here"/tools "$out"/

# node:assert shim for the browser
cat > "$out"/tests/_assert.js <<'SHIM'
class AssertionError extends Error {
  constructor(message) { super(message); this.name = "AssertionError"; }
}
const fail = message => { throw new AssertionError(message || "assertion failed"); };
const fmt = v => (typeof v === "object" ? JSON.stringify(v) : String(v));
const assert = (value, message) => { if (!value) fail(message); };
assert.ok = (value, message) => { if (!value) fail(message); };
assert.equal = (a, b, message) => { if (a != b) fail((message || "equal") + `: ${fmt(a)} != ${fmt(b)}`); };
assert.strictEqual = (a, b, message) => { if (a !== b) fail((message || "strictEqual") + `: ${fmt(a)} !== ${fmt(b)}`); };
assert.deepEqual = (a, b, message) => {
  const A = JSON.stringify(a), B = JSON.stringify(b);
  if (A !== B) fail((message || "deepEqual") + `: ${A} !== ${B}`);
};
assert.notEqual = (a, b, message) => { if (a == b) fail(message || "notEqual"); };
assert.throws = (fn, message) => { let threw = false; try { fn(); } catch { threw = true; } if (!threw) fail(message || "expected throw"); };
export default assert;
SHIM

# tests are .mjs on disk; the browser needs a JavaScript MIME type
for f in "$out"/tests/*.mjs; do
  sed 's#from "node:assert/strict"#from "./_assert.js"#' "$f" > "${f%.mjs}.js"
  rm -f "$f"
done

# process.env does not exist in the browser
sed -i 's/Number(process\.env\.GROUNDING_EXAMPLES || [0-9]*)/140/' "$out"/tests/grounding.js 2>/dev/null || true
sed -i 's/Number(process\.env\.GROUNDING_EPOCHS || [0-9]*)/40/' "$out"/tests/grounding.js 2>/dev/null || true

# diagnostics live in tools/ as .mjs and import ../src, so they can be copied as-is
for f in "$here"/tools/*.mjs; do
  [ -e "$f" ] || continue
  cp "$f" "$out"/tests/"$(basename "${f%.mjs}.js")"
done

echo "staged $out"
