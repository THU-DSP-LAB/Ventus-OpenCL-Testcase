Ventus raytracingshadows assets
================================

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
