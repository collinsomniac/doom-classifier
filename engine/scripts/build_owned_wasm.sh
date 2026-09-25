#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK="${ROOT}/engine/SOURCE.lock"
WORK="${DOOM_ENGINE_WORKDIR:-${ROOT}/.engine-work}"
DIST="${DOOM_ENGINE_DIST:-${ROOT}/engine/dist}"

UPSTREAM_REPO="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["upstream_repository"])' "${LOCK}")"
UPSTREAM_COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["upstream_commit"])' "${LOCK}")"

rm -rf "${WORK}" "${DIST}"
mkdir -p "${WORK}" "${DIST}"

git init "${WORK}/upstream" >/dev/null
git -C "${WORK}/upstream" remote add origin "${UPSTREAM_REPO}"
git -C "${WORK}/upstream" fetch --depth 1 origin "${UPSTREAM_COMMIT}"
git -C "${WORK}/upstream" checkout --detach FETCH_HEAD
git -C "${WORK}/upstream" submodule update --init --recursive --depth 1

python3 "${ROOT}/engine/scripts/instrument_engine.py" "${WORK}/upstream/vendor/chocolate-doom"
git -C "${WORK}/upstream" diff -- vendor/chocolate-doom/src/doom/browser_doom_bridge.c vendor/chocolate-doom/src/doom/p_inter.c vendor/chocolate-doom/src/doom/g_game.c > "${DIST}/telemetry.patch"

(
  cd "${WORK}/upstream"
  ./engine/scripts/build_wasm.sh
)

cp "${WORK}/upstream/public/engine/chocolate-doom.js" "${DIST}/"
cp "${WORK}/upstream/public/engine/chocolate-doom.wasm" "${DIST}/"
cp "${WORK}/upstream/public/engine/chocolate-doom.data" "${DIST}/"
cp "${LOCK}" "${DIST}/SOURCE.lock"
cp "${ROOT}/engine/scripts/instrument_engine.py" "${DIST}/INSTRUMENTATION.py"
cp "${WORK}/upstream/licenses/CHOCOLATE-DOOM-GPL-2.0.md" "${DIST}/CHOCOLATE-DOOM-GPL-2.0.md"
cp "${WORK}/upstream/licenses/FREEDOOM-COPYING.txt" "${DIST}/FREEDOOM-COPYING.txt"

cat > "${DIST}/SOURCE.txt" <<EOF
doom-classifier owned telemetry build

Corresponding upstream source:
${UPSTREAM_REPO}
commit ${UPSTREAM_COMMIT}

Instrumentation source:
engine/scripts/instrument_engine.py in the doom-classifier repository.

The exact applied diff is included as telemetry.patch.

Chocolate Doom is GPL-2.0-or-later. Freedoom content retains its own license.
EOF

test -s "${DIST}/chocolate-doom.js"
test -s "${DIST}/chocolate-doom.wasm"
test -s "${DIST}/chocolate-doom.data"
test -s "${DIST}/telemetry.patch"

sha256sum "${DIST}/chocolate-doom.js" "${DIST}/chocolate-doom.wasm" "${DIST}/chocolate-doom.data"
