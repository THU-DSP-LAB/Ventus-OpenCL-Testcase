# Maintained RT assets

Only assets required by the maintained five-case image regression belong here.

| Case | Runtime asset |
| --- | --- |
| raytracingshadows | models/vulkanscene_shadow.gltf |
| raytracingreflections | models/reflection_scene.gltf |
| raytracingtextures | textures/gratefloor_rgba.ktx |

Intersection and sbtdata construct their geometry procedurally.
FlightHelmet and treasure_smooth belong to examples outside this regression
and are not maintained inputs.

Upstream assets originate from SaschaWillems/Vulkan-Assets commit
`a27c0e584434d59b7c7a714e9180eefca6f0ec4b`.
Keep the associated license notices when updating assets.

The reflections glTF embeds its dependencies:
SHA-256 `d20d0fb6ba02333b37f77f4fef576c1748eaa138a588654643196fb859325585`,
size 1095444 bytes. Do not replace it with a simplified scene when measuring
recursive reflection behavior.

The optional Ventus smoke fixture vulkanscene_shadow_minimal.gltf contains
only a two-triangle rectangle. Select it explicitly with RT_SHADOW_ASSET;
leave that variable unset for the full default scene.

Run regressions from the ventus-env root using tools/rtcore.py regression.
Preparation, application builds and generated shaders are managed by
build-ventus.sh --build rt-workload.
