# Maintained RT image-regression inputs

`SaschaWillems_Vulkan/` is a prepared, user-owned SaschaWillems Vulkan workload
tree, not a complete vendored upstream checkout. Versioned files here preserve
the regression manifest, software storage-image readback hooks, the five tested
GLSL shader sets, shadows' per-primitive BLAS/instance mapping, and selected
scene assets. Preserve these overrides when preparing the remaining upstream
source and dependencies. Upstream code is covered by the included LICENSE.md.

The separate ventus-env `workloads/rt/prepare.sh` prepares an older shadows-only
overlay; it does not prepare this all-example tree. These overrides were tested
in the existing configured workload. A clean-checkout preparation workflow for
all examples is not yet automated.

Builds, generated SPIR-V, screenshots, diagnostic shader variants and large
external workload collections are not source inputs to commit. Before image
testing, regenerate SPIR-V after every GLSL change: the upstream CMake app
target alone does not compile shaders. For each selected case, for example:

```sh
cd SaschaWillems_Vulkan
for shader in shaders/glsl/raytracingshadows/*.rgen \
              shaders/glsl/raytracingshadows/*.rchit \
              shaders/glsl/raytracingshadows/*.rmiss; do
    glslangValidator -V --target-env vulkan1.2 "$shader" -o "$shader.spv" || exit 1
done
```

Include `.rahit` and `.rint` stages for textures and intersection, respectively.
The maintained shader path is GLSL; HLSL/Slang variants are not validated here.
Then use `python3 tools/rtcore.py regression` from the ventus-env root. Its
manifest describes capture variables and pixel tolerances. `--include-candidates`
permits, but does not select, candidate cases: name them with `--case`.

The 2026-09-11 acceptance is five cases at 4×4: shadows, sbtdata, intersection,
reflections and textures. Software/Spike/Verilator comparisons pass; RTL equals
Spike exactly. Textures allows the existing RGB8 tolerance of 1 against software.
Basic has no software readback hook and is not part of this acceptance.
