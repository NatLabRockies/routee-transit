# HPC build

On HPC hosts the prebuilt ("downloaded") ONNX Runtime that crates.io's `ort`
crate fetches can't be used — it requires a newer glibc (e.g.
`__libc_single_threaded`, glibc 2.32) than the host provides (RHEL 8 → glibc
2.28). The same applies to the PyPI/manylinux `nrel.routee.compass` wheel. Both
routee-transit's own Rust extension **and** the compass wheel it depends on are
instead built against an ONNX Runtime compiled from source and linked
statically.

The mechanism is simple: the `ort` crate's build script links whatever
`ORT_LIB_PATH` points at, **before** considering its download-binaries path and
regardless of cargo features. The pixi `hpc` env activation sets `ORT_LIB_PATH`
to this repo's from-source ORT build, so every cargo/maturin build run under
`-e hpc` links the local static ORT automatically. No feature flags, no shims.

This build is **self-contained**: from a fresh routee-transit clone it

1. builds ONNX Runtime from source (`build/onnxruntime`),
2. builds `nrel.routee.compass` from its PyPI **sdist** against that same ORT
   and installs it (`install_compass.sh`), then
3. builds and installs routee-transit's extension via `maturin develop`.

ORT is compiled only once (the compass build reuses it), and the compass
version is specified in exactly one place: routee-transit's own
`nrel.routee.compass` dependency constraint in `pyproject.toml`.

## Why two ONNX linkages

- `routee-transit`'s Python package depends on `nrel.routee.compass`, and
  `import nrel.routee.compass` eagerly loads compass's Rust extension, which
  links ONNX Runtime. The prebuilt PyPI wheels are tagged `manylinux_2_28` but
  actually reference glibc 2.32+ symbols, so installers accept them and the
  import then fails — compass must be built from source here, which is what
  `install_compass.sh` does (`pip/uv install --no-binary` from the sdist).
- `routee-transit`'s own extension (`routee_transit_py`) compiles the
  routee-compass crates from crates.io directly (including the ORT-consuming
  `routee-compass-powertrain`), so it has an independent ONNX linkage. Building
  routee-transit's Rust does **not** require a compass checkout.

## Build

From a fresh clone, on the HPC host:

```bash
pixi run -e hpc build_hpc
```

This runs `build_hpc_ort` → `install_compass` → `maturin develop --uv
--release`. Building ONNX Runtime from source is the slow step (tens of
minutes); reruns reuse the existing `build/onnxruntime` checkout, or skip the
ORT step entirely with `SKIP_ORT_BUILD=1`.

Verify:

```bash
pixi run -e hpc check_hpc   # imports both extensions, prints "ok"
```

## Day-to-day use

Enter the hpc env once per shell, then use plain commands:

```bash
pixi shell -e hpc
python -c "import routee.transit"
pytest tests/
```

> **Warning — always use the `hpc` env on HPC.** Running default-env pixi
> commands (`pixi run python`, `pixi install`, `pixi run test`, …) makes
> pixi/uv rebuild the editable install against the downloaded ONNX Runtime,
> which needs glibc ≥ 2.32 and fails to import (`undefined symbol:
> __libc_single_threaded`). The `hpc` env deliberately excludes the editable
> install; `build_hpc` provides the package instead. If a default-env run
> clobbers things, recover with:
> `SKIP_ORT_BUILD=1 pixi run -e hpc build_hpc`

## Optional add-ons

**Reuse a precompiled static ORT** (skip building ONNX Runtime from source):

```bash
SKIP_ORT_BUILD=1 pixi run -e hpc build_hpc
```

`build_hpc_ort` then only validates that `ORT_LIB_PATH` exists. To point at an
ORT built elsewhere, override `ORT_LIB_PATH` too.

**Distributable routee-transit wheel:**

```bash
pixi run -e hpc build_hpc_wheel   # lands in rust/target/wheels/
```

## Overrides

The `hpc` feature activation sets these (edit in `pyproject.toml`, or export
before running):

| var                | default                                       | notes                                                   |
| ------------------ | --------------------------------------------- | ------------------------------------------------------- |
| `ONNXRUNTIME_DIR`  | `$PIXI_PROJECT_ROOT/build/onnxruntime`        | where ORT is cloned/built                               |
| `ONNXRUNTIME_TAG`  | `v1.20.1`                                     | ONNX Runtime git tag                                    |
| `ORT_BUILD_CONFIG` | `RelWithDebInfo`                              | ORT build config                                        |
| `ORT_LIB_PATH`     | `$ONNXRUNTIME_DIR/build/Linux/RelWithDebInfo` | static ORT libs every `-e hpc` build links against      |
| `SKIP_ORT_BUILD`   | (unset)                                       | if set, don't build ORT; validate `ORT_LIB_PATH` exists |

The compass **version** is not an override — it always follows the
`nrel.routee.compass` constraint in `pyproject.toml` (`[project.dependencies]`).

## Diagnostics

If an import fails with `undefined symbol`, check which glibc symbols an
extension actually needs:

```bash
nm -D <the .so> | grep -E 'isoc23|single_threaded'   # any output = bad link
objdump -T <the .so> | grep -oE 'GLIBC_2\.[0-9]+' | sort -uV | tail -1  # must be <= host glibc
```

`build_ort.sh` pins the ONNX compiler to gcc-toolset-12 (overridable via
`CC`/`CXX`) because the Cray `cc`/`CC` wrappers target a newer glibc than the
host runtime. Note that after rebuilding ONNX Runtime, cargo does not notice
the external `.a` files changed — force it with
`cargo clean -p ort-sys --release` in `rust/`.
