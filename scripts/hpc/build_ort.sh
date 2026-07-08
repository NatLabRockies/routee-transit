#!/usr/bin/env bash
#
# Clone and build ONNX Runtime as a static library for HPC systems where
# prebuilt ORT binaries can't be used (e.g. the host glibc is too old for
# crates.io's `ort` `download-binaries` feature, or the manylinux wheels
# target a newer glibc than the host provides).
#
# This is the SELF-CONTAINED path: routee-transit builds its own ONNX Runtime
# and does not depend on a side-by-side routee-compass checkout. The compass
# wheel that routee-transit installs (see install_compass.sh) reuses the ORT
# built here, so ORT is only compiled once.
#
# Configurable via environment variables (exported by the pixi `hpc`
# feature activation when run via `pixi run`):
#   ONNXRUNTIME_DIR    where to clone/build ORT
#   ONNXRUNTIME_TAG    git tag/branch to check out
#   ORT_BUILD_CONFIG   ORT build config (e.g. RelWithDebInfo, Release)
#   SKIP_ORT_BUILD     if set, skip the ORT build step and just validate that a
#                      pre-existing (e.g. precompiled) ORT is present. Combine
#                      with ORT_LIB_PATH to point at an ORT built elsewhere.

set -euo pipefail

: "${ONNXRUNTIME_DIR:?ONNXRUNTIME_DIR not set (expected from pixi hpc env)}"
: "${ONNXRUNTIME_TAG:?ONNXRUNTIME_TAG not set (expected from pixi hpc env)}"
: "${ORT_BUILD_CONFIG:?ORT_BUILD_CONFIG not set (expected from pixi hpc env)}"

ORT_LIB_DIR="${ONNXRUNTIME_DIR}/build/Linux/${ORT_BUILD_CONFIG}"

# Optional add-on: reuse an ORT built elsewhere (precompiled/static) rather
# than compiling from source here. Set SKIP_ORT_BUILD=1 and point ORT_LIB_PATH
# at the directory containing the static libs.
if [[ -n "${SKIP_ORT_BUILD:-}" ]]; then
  check_dir="${ORT_LIB_PATH:-${ORT_LIB_DIR}}"
  echo "==> SKIP_ORT_BUILD set, skipping ORT build; using ${check_dir}"
  if [[ ! -d "${check_dir}" ]]; then
    echo "error: ${check_dir} does not exist; cannot skip build" >&2
    echo "hint: unset SKIP_ORT_BUILD to build ORT from source, or point" >&2
    echo "      ORT_LIB_PATH at an existing static ONNX Runtime build." >&2
    exit 1
  fi
  exit 0
fi

for tool in cmake git; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool '$tool' not found on PATH" >&2
    echo "hint: run inside the pixi hpc env (e.g. 'pixi run -e hpc build_hpc_ort')" >&2
    exit 1
  fi
done

# Pin the C/C++ compiler for the ONNX build to one targeting the HOST's glibc.
# On Cray/HPC systems the default `cc`/`CC` wrappers can compile against a newer
# glibc than the runtime host provides, so the ONNX objects (which get archived
# into the ort-sys rlib and thus the final extension) end up referencing symbols
# like `__isoc23_strto*` (glibc 2.38) or `__libc_single_threaded` (glibc 2.32)
# that the host's older glibc lacks -- causing "undefined symbol" at import.
# Default to gcc-toolset-12 (uses the host glibc headers); override via CC/CXX.
if [[ -z "${CC:-}" || -z "${CXX:-}" ]]; then
  if [[ -x /opt/rh/gcc-toolset-12/root/usr/bin/gcc ]]; then
    : "${CC:=/opt/rh/gcc-toolset-12/root/usr/bin/gcc}"
    : "${CXX:=/opt/rh/gcc-toolset-12/root/usr/bin/g++}"
  elif command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1; then
    : "${CC:=$(command -v x86_64-conda-linux-gnu-gcc)}"
    : "${CXX:=$(command -v x86_64-conda-linux-gnu-g++)}"
  else
    echo "warning: no glibc-pinned compiler found (gcc-toolset-12 / conda);" >&2
    echo "         falling back to cmake's default, which on Cray may target a" >&2
    echo "         newer glibc than the host. Set CC/CXX to override." >&2
  fi
fi
export CC CXX

if [[ ! -d "${ONNXRUNTIME_DIR}/.git" ]]; then
  echo "==> cloning onnxruntime ${ONNXRUNTIME_TAG} into ${ONNXRUNTIME_DIR}"
  mkdir -p "$(dirname "${ONNXRUNTIME_DIR}")"
  git clone --depth 1 --branch "${ONNXRUNTIME_TAG}" --recurse-submodules \
    https://github.com/microsoft/onnxruntime.git "${ONNXRUNTIME_DIR}"
else
  echo "==> reusing existing onnxruntime checkout at ${ONNXRUNTIME_DIR}"
fi

# If a prior configure cached a different compiler, wipe the build dir so cmake
# reconfigures with the pinned one (CMakeCache pins the compiler otherwise).
CACHE="${ONNXRUNTIME_DIR}/build/Linux/${ORT_BUILD_CONFIG}/CMakeCache.txt"
if [[ -n "${CXX:-}" && -f "${CACHE}" ]]; then
  cached_cxx="$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "${CACHE}")"
  if [[ "${cached_cxx}" != "${CXX}" ]]; then
    echo "==> compiler changed (${cached_cxx:-none} -> ${CXX}); clearing ONNX build dir"
    rm -rf "${ONNXRUNTIME_DIR}/build/Linux"
  fi
fi

cmake_defs=()
[[ -n "${CC:-}" ]] && cmake_defs+=("CMAKE_C_COMPILER=${CC}")
[[ -n "${CXX:-}" ]] && cmake_defs+=("CMAKE_CXX_COMPILER=${CXX}")
extra_args=()
if [[ ${#cmake_defs[@]} -gt 0 ]]; then
  extra_args=(--cmake_extra_defines "${cmake_defs[@]}")
fi

echo "==> building onnxruntime (${ORT_BUILD_CONFIG}); CC=${CC:-<default>} CXX=${CXX:-<default>}"
( cd "${ONNXRUNTIME_DIR}" && \
  ./build.sh \
    --config "${ORT_BUILD_CONFIG}" \
    --parallel \
    --compile_no_warning_as_error \
    --skip_submodule_sync \
    --skip_tests \
    "${extra_args[@]}" )

echo "==> ORT static libs at ${ORT_LIB_DIR}"
