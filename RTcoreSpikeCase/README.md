# Maintained RT image-regression inputs

This directory is an overlay, not a complete upstream checkout. It preserves
the five maintained application/GLSL sets, storage-image readback hooks,
regression manifest and their required scene assets. Upstream code is covered
by the included LICENSE.md; asset provenance is recorded under assets/models.

From the ventus-env root:

```sh
git submodule update --init testcases workloads/rt/SaschaWillems_Vulkan
bash build-ventus.sh --build rt-workload
python3 tools/rtcore.py regression
```

The preparation script combines pinned upstream sources with only tracked
files from this overlay in build/rt-workload/source and regenerates GLSL
SPIR-V. The application build is build/rt-workload/build. Repeat the build
after changing application or shader sources; CMake app targets alone do not
compile GLSL.

Maintained cases: shadows, sbtdata, intersection, reflections and textures.
All five run by default at 16×9, comparing software against Spike.
Use --backend software|spike|rtl|all and --width/--height to select renderers
and dimensions. Hardware selections include a software reference; software-only
captures images without a hardware comparison. Use --case raytracingintersection
to select one example, or repeat --case for several.
Basic has no software image-readback hook and is not maintained here.
Historical acceptance is not a claim that every current case or image size
passes: each invocation must compare fresh software/Spike/Verilator images.

Generated SPIR-V, binaries, screenshots, logs, diagnostic shader variants and
external workload collections are not source inputs to commit. Do not alter
the original OpenCL testcase directories as part of RT maintenance.
