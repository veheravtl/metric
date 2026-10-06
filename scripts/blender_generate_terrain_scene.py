# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Строит G9-сцену с гладким рельефом и плотной скрытой истиной."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Vector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from blender_generate_scene import (  # noqa: E402
    add_lighting,
    build_cameras,
    camera_metadata,
    configure_render,
    make_collections,
    make_metric_texture,
    move_to_collection,
    parse_script_arguments,
    project_world_point_to_pixel,
    render_all_cameras,
    reset_scene,
)

from aerial_mapper.terrain_geometry import (  # noqa: E402
    TerrainSpecification,
    terrain_vertices,
    triangle_centroid_samples,
)


def build_terrain(
    description: dict[str, Any],
    collections: dict[str, bpy.types.Collection],
    output_directory: Path,
) -> tuple[bpy.types.Object, np.ndarray]:
    """Создаёт текстурированную mesh-поверхность и контрольные точки на ней."""

    world = description["world"]
    terrain = world["terrain"]
    specification = TerrainSpecification.from_mapping(terrain)
    width_m, height_m = (float(value) for value in world["ground_size_m"])
    columns, rows = (int(value) for value in terrain["grid_vertices_xy"])
    vertices, uv_coordinates, faces = terrain_vertices(
        width_m=width_m,
        height_m=height_m,
        columns=columns,
        rows=rows,
        specification=specification,
    )

    mesh = bpy.data.meshes.new("metric_terrain_mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    ground = bpy.data.objects.new("metric_terrain_ground", mesh)
    collections["Ground"].objects.link(ground)
    ground.pass_index = 1

    uv_layer = mesh.uv_layers.new(name="metric_uv")
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            uv_layer.data[loop_index].uv = uv_coordinates[vertex_index].tolist()

    image = make_metric_texture(description, output_directory)
    material = bpy.data.materials.new(name="mat_metric_terrain")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.interpolation = "Linear"
    links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    principled.inputs["Roughness"].default_value = 1.0
    ground.data.materials.append(material)
    move_to_collection(ground, collections["Ground"])

    samples = triangle_centroid_samples(
        vertices,
        faces,
        stride=int(terrain["truth_sample_stride"]),
    )
    return ground, samples


def visible_from_camera(
    camera: bpy.types.Object,
    point_world_m: np.ndarray,
    *,
    ground: bpy.types.Object,
    depsgraph: bpy.types.Depsgraph,
) -> bool:
    """Проверяет, является ли точка первым пересечением луча от камеры."""

    origin = camera.matrix_world.translation
    point = Vector(point_world_m.tolist())
    direction = point - origin
    distance = direction.length
    if distance <= 1e-9:
        return False
    hit, location, _, _, hit_object, _ = bpy.context.scene.ray_cast(
        depsgraph,
        origin,
        direction.normalized(),
        distance=distance + 0.01,
    )
    return bool(
        hit
        and hit_object is not None
        and hit_object.name == ground.name
        and (location - point).length <= 0.01
    )


def save_terrain_truth(
    description: dict[str, Any],
    cameras: dict[str, bpy.types.Object],
    ground: bpy.types.Object,
    samples_world_m: np.ndarray,
    output_directory: Path,
) -> dict[str, Any]:
    """Сохраняет проекции и окклюзионно корректную видимость всех точек."""

    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera_ids = [item["id"] for item in description["cameras"]]
    pixel_xy = np.empty((len(camera_ids), samples_world_m.shape[0], 2))
    in_frame = np.zeros((len(camera_ids), samples_world_m.shape[0]), dtype=bool)
    visible = np.zeros_like(in_frame)

    for camera_index, camera_id in enumerate(camera_ids):
        camera = cameras[camera_id]
        for point_index, point in enumerate(samples_world_m):
            projection = project_world_point_to_pixel(
                camera, point.tolist(), description
            )
            pixel_xy[camera_index, point_index] = projection["pixel_xy"]
            in_frame[camera_index, point_index] = projection["visible"]
            if projection["visible"]:
                visible[camera_index, point_index] = visible_from_camera(
                    camera,
                    point,
                    ground=ground,
                    depsgraph=depsgraph,
                )

    truth_path = output_directory / "terrain_truth.npz"
    np.savez_compressed(
        truth_path,
        world_xyz_m=samples_world_m.astype(np.float64),
        camera_ids=np.asarray(camera_ids, dtype="U64"),
        pixel_xy=pixel_xy.astype(np.float64),
        in_frame=in_frame,
        visible=visible,
    )
    return {
        "path": truth_path.name,
        "point_count": int(samples_world_m.shape[0]),
        "camera_ids": camera_ids,
        "in_frame_counts": {
            camera_id: int(np.count_nonzero(in_frame[index]))
            for index, camera_id in enumerate(camera_ids)
        },
        "visible_counts": {
            camera_id: int(np.count_nonzero(visible[index]))
            for index, camera_id in enumerate(camera_ids)
        },
        "visibility_method": "Blender Scene.ray_cast to first terrain hit",
    }


def main() -> None:
    """Генерирует одну поверхность G9 и все связанные Teach/Repeat-виды."""

    started_at = time.perf_counter()
    description_path, output_directory = parse_script_arguments()
    description = json.loads(description_path.read_text(encoding="utf-8"))
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "scene_description.json").write_text(
        json.dumps(description, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    reset_scene()
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    collections = make_collections()
    ground, samples = build_terrain(description, collections, output_directory)
    cameras = build_cameras(description, collections)
    add_lighting(collections)
    configure_render(description)
    truth = save_terrain_truth(
        description,
        cameras,
        ground,
        samples,
        output_directory,
    )

    bpy.context.preferences.filepaths.save_version = 0
    rendered_images, rendered_control_layers = render_all_cameras(
        description, cameras, output_directory
    )
    scene_path = output_directory / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(scene_path))

    metadata = {
        "schema_version": 1,
        "scenario_id": description["scenario_id"],
        "blender_version": bpy.app.version_string,
        "seed": description["seed"],
        "world_units": description["world"]["units"],
        "world_axis_convention": description["world"]["axis_convention"],
        "render_engine": scene.render.engine,
        "rendered_images": rendered_images,
        "rendered_control_layers": rendered_control_layers,
        "scene_file": scene_path.name,
        "control_points_world_m": [],
        "reference_map": None,
        "metric_controls": [],
        "terrain_truth": truth,
        "cameras": [
            camera_metadata(cameras[item["id"]], item)
            for item in description["cameras"]
        ],
        "elapsed_seconds": time.perf_counter() - started_at,
        "limitations": [
            "terrain_truth и контрольные маски доступны только оценщику.",
            "Поверхность является гладкой однозначной функцией Z=f(X,Y).",
            "Поза записана в соглашении осей Blender, а не OpenCV.",
        ],
    }
    (output_directory / "generation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SYNTHETIC_3D_METADATA=" + json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
