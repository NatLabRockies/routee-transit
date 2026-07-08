#!/usr/bin/env bash
#
# Build and install `nrel.routee.compass` from its PyPI source distribution
# into the active environment.
#
# routee-transit's Python package depends on `nrel.routee.compass`, and simply
# `import nrel.routee.compass` eagerly loads compass's Rust extension (which
# links ONNX Runtime). The prebuilt PyPI wheels bundle a downloaded ONNX
# Runtime that needs a newer glibc than this host provides (they are tagged
# manylinux_2_28 but reference glibc 2.32+ symbols), so compass must be built
# from source here (--no-binary). Because the pixi hpc env sets ORT_LIB_PATH,
# the sdist build links the locally-built static ORT automatically, and
# because it runs inside this env it matches this env's Python version.
#
# The compass version is whatever routee-transit's own `nrel.routee.compass`
# constraint in pyproject.toml resolves to — the single place the compass
# Python version is specified.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ORT_LIB_PATH is normally exported by the pixi hpc env activation; derive it
# from the ONNX build dir as a fallback for use outside pixi.
if [[ -z "${ORT_LIB_PATH:-}" ]]; then
  : "${ONNXRUNTIME_DIR:?ONNXRUNTIME_DIR not set and ORT_LIB_PATH not provided}"
  : "${ORT_BUILD_CONFIG:?ORT_BUILD_CONFIG not set and ORT_LIB_PATH not provided}"
  ORT_LIB_PATH="${ONNXRUNTIME_DIR}/build/Linux/${ORT_BUILD_CONFIG}"
fi
if [[ ! -d "${ORT_LIB_PATH}" ]]; then
  echo "error: ORT_LIB_PATH=${ORT_LIB_PATH} does not exist" >&2
  echo "hint: build ONNX Runtime first: 'pixi run -e hpc build_hpc_ort'" >&2
  exit 1
fi
export ORT_LIB_PATH

for tool in python; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool '$tool' not found on PATH" >&2
    echo "hint: run inside the pixi hpc env (e.g. 'pixi run -e hpc install_compass')" >&2
    exit 1
  fi
done

# Pull the compass constraint out of routee-transit's own dependency list so
# the version is specified in exactly one place.
spec="$(python - "${REPO_ROOT}/pyproject.toml" <<'EOF'
import re, sys, tomllib
deps = tomllib.load(open(sys.argv[1], "rb"))["project"]["dependencies"]
print(next(d for d in deps if re.match(r"nrel[._-]routee[._-]compass\b", d)))
EOF
)"

echo "==> building nrel.routee.compass from sdist (static ORT): ${spec}"
echo "    ORT_LIB_PATH=${ORT_LIB_PATH}"

# --no-binary: the PyPI wheels bundle the wrong ORT; force a source build.
# --force-reinstall: recover even when the same version is already installed
#   (e.g. after a prebuilt wheel clobbered the env).
# --no-cache: don't reuse a wheel built against an older local ORT.
# pixi envs are uv-managed and often have no `pip`; prefer `python -m pip`,
# fall back to `uv pip` (targeting this env's python explicitly).
if python -m pip --version >/dev/null 2>&1; then
  python -m pip install --no-cache-dir --no-binary nrel-routee-compass \
    --force-reinstall --no-deps "${spec}"
elif command -v uv >/dev/null 2>&1; then
  uv pip install --no-cache --no-binary nrel-routee-compass \
    --force-reinstall --no-deps --python "$(command -v python)" "${spec}"
else
  echo "error: no installer found (need 'python -m pip' or 'uv')" >&2
  echo "hint: run inside the pixi hpc env (e.g. 'pixi run -e hpc install_compass')" >&2
  exit 1
fi

echo "==> installed. verify: python -c 'import nrel.routee.compass'"
