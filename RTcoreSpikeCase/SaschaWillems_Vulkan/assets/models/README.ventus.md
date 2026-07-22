Ventus ray-tracing assets
================================

Vendored upstream ray-tracing assets
-----------------------------------

The Ventus testcase intentionally vendors the small set of upstream assets
needed by its checked-in ray-tracing examples.  This avoids a nested assets
submodule and makes a checkout sufficient for deterministic full-app tests.

`reflection_scene.gltf` is the complete scene used by the real
`raytracingreflections` sample.  It was restored from:

- `SaschaWillems/Vulkan-Assets` commit
  `a27c0e584434d59b7c7a714e9180eefca6f0ec4b`
- source `https://github.com/SaschaWillems/Vulkan-Assets`
- upstream path `models/reflection_scene.gltf`
- git blob `86526ae4184c594556a1bbe49dc6c044667ee2bf`
- SHA-256 `d20d0fb6ba02333b37f77f4fef576c1748eaa138a588654643196fb859325585`
- byte size `1095444`

The glTF embeds its geometry and images in data URIs, so this one tracked file
is the full dependency closure.  If it is refreshed, update all five
provenance fields above and verify it before committing:

```sh
sha256sum assets/models/reflection_scene.gltf
test "$(wc -c < assets/models/reflection_scene.gltf)" = 1095444
test "$(git hash-object assets/models/reflection_scene.gltf)" = \
  86526ae4184c594556a1bbe49dc6c044667ee2bf
```

Do not replace this scene with a minimal hand-authored asset: its recursive
reflection loop is the full-app CPS/continuation regression target.

raytracingshadows assets
===============================

`vulkanscene_shadow.gltf` is the upstream SaschaWillems shadow scene used by
the real `raytracingshadows` sample. It was restored from:

- `SaschaWillems/Vulkan-Assets` commit
  `a27c0e584434d59b7c7a714e9180eefca6f0ec4b`

`vulkanscene_shadow_minimal.gltf` is the small Ventus smoke-test asset that was
previously installed as `vulkanscene_shadow.gltf`. It contains only a single
two-triangle rectangle, so reference images from this asset are expected to be a
dark blue background with a flat colored rectangle rather than the full shadow
scene.

Use the smoke asset explicitly when debugging ABI, SBT, payload, image format,
or basic hit/miss behavior:

```sh
RT_SHADOW_ASSET=/path/to/vulkanscene_shadow_minimal.gltf ./raytracingshadows
```

Leave `RT_SHADOW_ASSET` unset for the restored upstream scene.
