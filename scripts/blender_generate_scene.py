# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Строит и рендерит минимальную метрическую сцену внутри Blender.

Файл запускается встроенным Python Blender, а не обычным Python проекта:

    blender --background --python scripts/blender_generate_scene.py -- CONFIG OUTPUT

На вход подаётся JSON-описание, на выходе появляются scene.blend, три RGB-вида
и метаданные. Здесь нет случайного размещения: seed сохраняется для будущих
ступеней, а G0 должен при одинаковом Blender давать одну и ту же геометрию.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

COLLECTION_NAMES = (
    "Ground",
    "Objects",
    "Control Points",
    "Teach Cameras",
    "Repeat Cameras",
    "Truth Helpers",
)


def parse_script_arguments() -> tuple[Path, Path]:
    """Возвращает пути после обязательного разделителя аргументов --."""

    if "--" not in sys.argv:
        raise ValueError("Нужны аргументы после '--': CONFIG_PATH OUTPUT_DIRECTORY")
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    if len(arguments) != 2:
        raise ValueError("Ожидались ровно CONFIG_PATH и OUTPUT_DIRECTORY")
    return Path(arguments[0]), Path(arguments[1])


def reset_scene() -> None:
    """Удаляет стартовые объекты, чтобы результат не зависел от startup-файла."""

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def make_collections() -> dict[str, bpy.types.Collection]:
    """Создаёт плоскую и понятную структуру Outliner для ручной проверки."""

    collections: dict[str, bpy.types.Collection] = {}
    root = bpy.context.scene.collection
    for name in COLLECTION_NAMES:
        collection = bpy.data.collections.new(name)
        root.children.link(collection)
        collections[name] = collection
    return collections


def move_to_collection(
    object_: bpy.types.Object,
    collection: bpy.types.Collection,
) -> None:
    """Перемещает объект из временной активной коллекции в смысловую."""

    for current_collection in tuple(object_.users_collection):
        current_collection.objects.unlink(object_)
    collection.objects.link(object_)


def make_material(name: str, color_srgb: list[float]) -> bpy.types.Material:
    """Создаёт простой матовый материал с явно заданным sRGB-цветом."""

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.diffuse_color = (*color_srgb, 1.0)
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (*color_srgb, 1.0)
    principled.inputs["Roughness"].default_value = 0.78
    return material


def make_procedural_object_material(
    name: str,
    color_srgb: list[float],
    *,
    texture_seed: float,
) -> bpy.types.Material:
    """Создаёт контрастную детерминированную текстуру на объектах G3--G6.

    Обычный однотонный параллелепипед почти не даёт локальных признаков внутри
    крыши. Тогда sweep доли земли проверял бы лишь закрытие фона, но не опасный
    случай, когда RANSAC получает много согласованных точек другой плоскости.
    Процедурный Noise Texture встроен в Blender, не требует внешнего ассета и
    сохраняется в scene.blend. Значение W разносит узор разных объектов.
    """

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.diffuse_color = (*color_srgb, 1.0)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    coordinates = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    noise.inputs["Scale"].default_value = 7.5
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.7
    noise.inputs["W"].default_value = texture_seed
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.42
    ramp.color_ramp.elements[0].color = (
        0.12 * color_srgb[0],
        0.12 * color_srgb[1],
        0.12 * color_srgb[2],
        1.0,
    )
    ramp.color_ramp.elements[1].position = 0.58
    ramp.color_ramp.elements[1].color = (
        min(1.0, 0.45 + 0.65 * color_srgb[0]),
        min(1.0, 0.45 + 0.65 * color_srgb[1]),
        min(1.0, 0.45 + 0.65 * color_srgb[2]),
        1.0,
    )
    links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], principled.inputs["Base Color"])
    principled.inputs["Roughness"].default_value = 0.82
    return material


def make_metric_texture(
    description: dict[str, Any],
    output_directory: Path,
) -> bpy.types.Image:
    """Создаёт детерминированную псевдоаэросъёмку для плоского опыта.

    Текстура содержит признаки разных масштабов: плавный фон, мелкие ячейки,
    дороги, штрихи и уникальные цветные площадки. Она не пытается выглядеть как
    реальный город. Её задача — дать SIFT достаточно неповторяющихся деталей,
    не нарушая главное допущение первого опыта: все точки лежат в одной
    плоскости Z=0.
    """

    texture_specification = description["world"]["metric_texture"]
    width = int(texture_specification["width_pixels"])
    height = int(texture_specification["height_pixels"])
    if width < 64 or height < 64:
        raise ValueError("Метрическая текстура должна быть не меньше 64 x 64")

    random_generator = np.random.default_rng(description["seed"])
    y_coordinates, x_coordinates = np.indices((height, width), dtype=np.float32)
    x_fraction = x_coordinates / max(width - 1, 1)
    y_fraction = y_coordinates / max(height - 1, 1)

    # Низкочастотный фон напоминает неоднородность поля или грунта. Он нужен
    # вместе с геометрическими элементами, потому что один регулярный узор дал
    # бы множество неоднозначных соответствий.
    background = np.empty((height, width, 3), dtype=np.float32)
    background[..., 0] = 0.18 + 0.07 * np.sin(17.0 * x_fraction + 3.0 * y_fraction)
    background[..., 1] = 0.34 + 0.09 * np.sin(11.0 * y_fraction - 5.0 * x_fraction)
    background[..., 2] = 0.14 + 0.05 * np.cos(13.0 * (x_fraction + y_fraction))

    # Случайные, но фиксированные клетки создают углы и локальную текстуру.
    cell_size = 12
    grid_height = math.ceil(height / cell_size)
    grid_width = math.ceil(width / cell_size)
    cell_noise = random_generator.normal(0.0, 0.055, (grid_height, grid_width, 1))
    cell_noise = np.repeat(np.repeat(cell_noise, cell_size, axis=0), cell_size, axis=1)
    background += cell_noise[:height, :width]
    image_rgb = np.clip(background, 0.03, 0.92)

    # Две дороги и разметка дают хорошо различимые длинные структуры, но их
    # пересечение смещено от центра, чтобы не создавать лишнюю симметрию.
    horizontal_road = np.abs(y_fraction - 0.37) < 0.055
    diagonal_road = np.abs(y_fraction - (0.83 * x_fraction + 0.05)) < 0.035
    image_rgb[horizontal_road | diagonal_road] = (0.115, 0.125, 0.14)
    horizontal_marking = (np.abs(y_fraction - 0.37) < 0.004) & (
        (x_coordinates.astype(np.int32) // 55) % 2 == 0
    )
    diagonal_distance = np.abs(y_fraction - (0.83 * x_fraction + 0.05))
    diagonal_marking = (diagonal_distance < 0.003) & (
        ((x_coordinates + y_coordinates).astype(np.int32) // 70) % 2 == 0
    )
    image_rgb[horizontal_marking | diagonal_marking] = (0.92, 0.78, 0.18)

    # Уникальные площадки разных размеров работают как локальные ориентиры.
    for index in range(34):
        center_x = int(random_generator.integers(35, width - 35))
        center_y = int(random_generator.integers(35, height - 35))
        half_width = int(random_generator.integers(9, 31))
        half_height = int(random_generator.integers(8, 27))
        color = random_generator.uniform(0.12, 0.92, size=3)
        x_start = max(0, center_x - half_width)
        x_stop = min(width, center_x + half_width)
        y_start = max(0, center_y - half_height)
        y_stop = min(height, center_y + half_height)
        image_rgb[y_start:y_stop, x_start:x_stop] = color
        border = 3 + index % 4
        image_rgb[y_start : min(y_stop, y_start + border), x_start:x_stop] = 0.96
        image_rgb[max(y_start, y_stop - border) : y_stop, x_start:x_stop] = 0.04

    alpha = np.ones((height, width, 1), dtype=np.float32)
    image_rgba = np.concatenate((image_rgb, alpha), axis=2)
    image = bpy.data.images.new(
        "metric_ground_texture",
        width=width,
        height=height,
        alpha=True,
        float_buffer=False,
    )
    image.pixels.foreach_set(image_rgba.ravel())
    image.filepath_raw = str(output_directory / "ground_texture.png")
    image.file_format = "PNG"
    image.save()
    image.pack()
    return image


def build_planar_metric_ground(
    description: dict[str, Any],
    collections: dict[str, bpy.types.Collection],
    output_directory: Path,
) -> None:
    """Создаёт одну текстурированную плоскость без параллакса и рельефа."""

    width_m, height_m = description["world"]["ground_size_m"]
    image = make_metric_texture(description, output_directory)

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    ground = bpy.context.object
    ground.name = "metric_ground_plane"
    ground.dimensions = (width_m, height_m, 0.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    move_to_collection(ground, collections["Ground"])

    material = bpy.data.materials.new(name="mat_metric_ground_texture")
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


def add_box(
    *,
    name: str,
    center_m: list[float] | tuple[float, float, float],
    size_m: list[float] | tuple[float, float, float],
    material: bpy.types.Material,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Добавляет параллелепипед; координаты центра и размеры заданы в метрах."""

    if any(value <= 0.0 for value in size_m):
        raise ValueError(f"Размеры объекта {name} должны быть положительными")
    bpy.ops.mesh.primitive_cube_add(location=center_m)
    object_ = bpy.context.object
    object_.name = name
    object_.dimensions = size_m
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    object_.data.materials.append(material)
    move_to_collection(object_, collection)
    return object_


def add_cylinder(
    *,
    name: str,
    center_m: list[float] | tuple[float, float, float],
    radius_m: float,
    height_m: float,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Добавляет вертикальный цилиндр с осью вдоль мировой Z."""

    if radius_m <= 0.0 or height_m <= 0.0:
        raise ValueError(f"Размеры объекта {name} должны быть положительными")
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=48,
        radius=radius_m,
        depth=height_m,
        location=center_m,
    )
    object_ = bpy.context.object
    object_.name = name
    object_.data.materials.append(material)
    move_to_collection(object_, collection)
    return object_


def build_ground(
    description: dict[str, Any],
    collections: dict[str, bpy.types.Collection],
    output_directory: Path,
) -> None:
    """Создаёт землю, дороги и разметку с известными метрическими размерами."""

    if description["world"].get("appearance") == "planar_metric_texture":
        build_planar_metric_ground(description, collections, output_directory)
        return

    width_m, height_m = description["world"]["ground_size_m"]
    ground_collection = collections["Ground"]
    grass = make_material("mat_ground_grass", [0.19, 0.42, 0.16])
    dark_grass = make_material("mat_ground_dark", [0.12, 0.31, 0.10])
    road = make_material("mat_road", [0.12, 0.13, 0.15])
    marking = make_material("mat_road_marking", [0.92, 0.90, 0.68])

    add_box(
        name="ground_base",
        center_m=(0.0, 0.0, -0.10),
        size_m=(width_m, height_m, 0.20),
        material=grass,
        collection=ground_collection,
    )
    add_box(
        name="field_patch_north_west",
        center_m=(-12.0, 8.0, 0.015),
        size_m=(12.0, 9.0, 0.03),
        material=dark_grass,
        collection=ground_collection,
    )
    add_box(
        name="road_north_south",
        center_m=(0.0, 0.0, 0.035),
        size_m=(7.0, height_m, 0.07),
        material=road,
        collection=ground_collection,
    )
    add_box(
        name="road_east_west",
        center_m=(0.0, -6.0, 0.045),
        size_m=(width_m, 5.0, 0.09),
        material=road,
        collection=ground_collection,
    )
    for index, y_m in enumerate(range(-13, 14, 4)):
        add_box(
            name=f"road_marking_{index:02d}",
            center_m=(0.0, float(y_m), 0.09),
            size_m=(0.28, 1.8, 0.04),
            material=marking,
            collection=ground_collection,
        )


def build_control_points(
    collections: dict[str, bpy.types.Collection],
) -> list[list[float]]:
    """Добавляет четыре независимые цветные точки на плоскости земли."""

    locations = [
        [-15.0, -10.0, 0.12],
        [15.0, -10.0, 0.12],
        [15.0, 10.0, 0.12],
        [-15.0, 10.0, 0.12],
    ]
    colors = (
        [0.90, 0.08, 0.06],
        [0.05, 0.35, 0.95],
        [0.96, 0.75, 0.05],
        [0.65, 0.08, 0.82],
    )
    for index, (location, color) in enumerate(zip(locations, colors, strict=True)):
        add_cylinder(
            name=f"control_point_{index:02d}",
            center_m=location,
            radius_m=0.55,
            height_m=0.20,
            material=make_material(f"mat_control_{index:02d}", color),
            collection=collections["Control Points"],
        )
    return locations


def build_configured_objects(
    description: dict[str, Any],
    collections: dict[str, bpy.types.Collection],
) -> None:
    """Строит только поддерживаемые G0-примитивы и отвергает неизвестные."""

    for object_index, specification in enumerate(description["objects"]):
        material_name = f"mat_{specification['id']}"
        if specification.get("appearance") == "procedural_texture":
            material = make_procedural_object_material(
                material_name,
                specification["color_srgb"],
                texture_seed=float(description["seed"] % 997 + object_index * 17),
            )
        else:
            material = make_material(material_name, specification["color_srgb"])
        common_arguments = {
            "name": specification["id"],
            "center_m": specification["center_m"],
            "material": material,
            "collection": collections["Objects"],
        }
        if specification["kind"] == "box":
            add_box(size_m=specification["size_m"], **common_arguments)
        elif specification["kind"] == "cylinder":
            add_cylinder(
                radius_m=specification["radius_m"],
                height_m=specification["height_m"],
                **common_arguments,
            )
        else:
            raise ValueError(
                f"Неподдерживаемый kind={specification['kind']!r} "
                f"у объекта {specification['id']}"
            )


def orient_camera(
    camera_object: bpy.types.Object,
    target_m: list[float],
) -> None:
    """Направляет локальную ось -Z камеры на цель, сохраняя локальную Y вверх."""

    direction = Vector(target_m) - camera_object.location
    if direction.length < 1e-9:
        raise ValueError(f"Камера {camera_object.name} совпадает со своей целью")
    camera_object.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def build_cameras(
    description: dict[str, Any],
    collections: dict[str, bpy.types.Collection],
) -> dict[str, bpy.types.Object]:
    """Создаёт эталонную, Teach- и Repeat-камеры из одного описания."""

    cameras: dict[str, bpy.types.Object] = {}
    collection_by_role = {
        "reference": collections["Truth Helpers"],
        "teach": collections["Teach Cameras"],
        "repeat": collections["Repeat Cameras"],
    }
    for specification in description["cameras"]:
        camera_data = bpy.data.cameras.new(specification["id"])
        camera_data.display_size = 2.0
        camera_data.clip_start = 0.1
        camera_data.clip_end = 250.0
        if specification["projection"] == "orthographic":
            camera_data.type = "ORTHO"
            camera_data.ortho_scale = specification["orthographic_scale_m"]
        elif specification["projection"] == "perspective":
            camera_data.type = "PERSP"
            camera_data.lens = specification["focal_length_mm"]
            camera_data.sensor_width = specification["sensor_width_mm"]
        else:
            raise ValueError(f"Неизвестная проекция {specification['projection']!r}")

        camera_object = bpy.data.objects.new(specification["id"], camera_data)
        camera_object.location = specification["location_m"]
        orient_camera(camera_object, specification["target_m"])
        collection_by_role[specification["role"]].objects.link(camera_object)
        camera_object["camera_role"] = specification["role"]
        camera_object["target_m"] = specification["target_m"]
        cameras[specification["id"]] = camera_object
    return cameras


def add_lighting(collections: dict[str, bpy.types.Collection]) -> None:
    """Задаёт постоянное солнце и нейтральное окружение без внешних текстур."""

    sun_data = bpy.data.lights.new(name="sun_key", type="SUN")
    sun_data.energy = 3.0
    sun_data.angle = math.radians(12.0)
    sun = bpy.data.objects.new(name="sun_key", object_data=sun_data)
    sun.rotation_euler = (math.radians(28.0), math.radians(-18.0), math.radians(28.0))
    collections["Truth Helpers"].objects.link(sun)

    world = bpy.context.scene.world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.055, 0.08, 0.12, 1.0)
    background.inputs["Strength"].default_value = 0.35


def configure_render(description: dict[str, Any]) -> None:
    """Фиксирует движок, разрешение, формат и управление цветом."""

    scene = bpy.context.scene
    render = description["render"]
    scene.render.engine = render["engine"]
    scene.render.resolution_x = render["width_pixels"]
    scene.render.resolution_y = render["height_pixels"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.image_settings.compression = 15
    scene.view_settings.look = "AgX - Medium High Contrast"

    # Название свойства сэмплов EEVEE менялось между версиями Blender.
    # G0 не должен падать только из-за необязательной оптимизации качества.
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_samples"):
        scene.eevee.taa_samples = render["samples"]


def camera_metadata(
    camera_object: bpy.types.Object,
    specification: dict[str, Any],
) -> dict[str, Any]:
    """Сериализует позу Blender camera-to-world без смены соглашения осей."""

    matrix_world = [
        [float(value) for value in row] for row in camera_object.matrix_world
    ]
    return {
        "id": specification["id"],
        "role": specification["role"],
        "projection": specification["projection"],
        "matrix_camera_to_world_blender": matrix_world,
        "location_m": [float(value) for value in camera_object.location],
        "target_m": specification["target_m"],
        "axis_note": "Blender camera looks along local -Z with local +Y up",
    }


def project_world_point_to_pixel(
    camera_object: bpy.types.Object,
    point_world_m: list[float],
    description: dict[str, Any],
) -> dict[str, Any]:
    """Проецирует скрытую мировую точку в координаты центров пикселей OpenCV.

    world_to_camera_view возвращает нормированные координаты, отсчитанные
    от нижнего левого угла кадра. OpenCV отсчитывает Y сверху, а целое значение
    обозначает центр пикселя. Поэтому Y отражается и обе координаты сдвигаются
    на половину пикселя относительно границ растра.
    """

    normalized = world_to_camera_view(
        bpy.context.scene,
        camera_object,
        Vector(point_world_m),
    )
    width = description["render"]["width_pixels"]
    height = description["render"]["height_pixels"]
    return {
        "pixel_xy": [
            float(normalized.x * width - 0.5),
            float((1.0 - normalized.y) * height - 0.5),
        ],
        "normalized_xy": [float(normalized.x), float(normalized.y)],
        "depth": float(normalized.z),
        "visible": bool(
            0.0 <= normalized.x <= 1.0
            and 0.0 <= normalized.y <= 1.0
            and normalized.z > 0.0
        ),
    }


def reference_map_metadata(
    description: dict[str, Any],
    cameras: dict[str, bpy.types.Object],
) -> dict[str, Any] | None:
    """Строит известную affine-привязку ортофото из его размеченной позы.

    Результат переводит координаты углов пикселей в проецированную систему
    координат карты. Это именно разрешённая информация первой, полностью
    размеченной позы; параметры Repeat-камеры в рабочую ветку не передаются.
    """

    map_specification = description.get("reference_map")
    if map_specification is None:
        return None
    reference_specification = next(
        item for item in description["cameras"] if item["role"] == "reference"
    )
    if reference_specification["projection"] != "orthographic":
        raise ValueError("Метрический эталон должен сниматься orthographic-камерой")

    camera = cameras[reference_specification["id"]]
    origin_pixel = np.asarray(
        project_world_point_to_pixel(camera, [0.0, 0.0, 0.0], description)["pixel_xy"],
        dtype=np.float64,
    )
    x_pixel = np.asarray(
        project_world_point_to_pixel(camera, [1.0, 0.0, 0.0], description)["pixel_xy"],
        dtype=np.float64,
    )
    y_pixel = np.asarray(
        project_world_point_to_pixel(camera, [0.0, 1.0, 0.0], description)["pixel_xy"],
        dtype=np.float64,
    )
    world_to_pixel = np.column_stack((x_pixel - origin_pixel, y_pixel - origin_pixel))
    if abs(float(np.linalg.det(world_to_pixel))) < 1e-12:
        raise ValueError("Ортофото не задаёт обратимое преобразование земли в пиксели")
    pixel_to_world = np.linalg.inv(world_to_pixel)

    map_origin = np.asarray(
        [
            map_specification["origin_easting_m"],
            map_specification["origin_northing_m"],
        ],
        dtype=np.float64,
    )
    # Rasterio передаёт affine-преобразованию угол пикселя, тогда как выше
    # использованы координаты его центра. Компенсируем ровно +0.5 по обеим осям.
    translation = map_origin - pixel_to_world @ (
        origin_pixel + np.asarray([0.5, 0.5], dtype=np.float64)
    )
    return {
        "crs": map_specification["crs"],
        "affine": [
            float(pixel_to_world[0, 0]),
            float(pixel_to_world[0, 1]),
            float(translation[0]),
            float(pixel_to_world[1, 0]),
            float(pixel_to_world[1, 1]),
            float(translation[1]),
        ],
        "origin_world_xy_in_crs_m": map_origin.tolist(),
        "pixel_coordinate_note": (
            "OpenCV pixel centers; raster affine receives pixel corners"
        ),
    }


def metric_control_metadata(
    description: dict[str, Any],
    cameras: dict[str, bpy.types.Object],
) -> list[dict[str, Any]]:
    """Сохраняет скрытые проекции контрольных отрезков для итоговой оценки."""

    controls: list[dict[str, Any]] = []
    for control in description.get("metric_controls", []):
        points_world = control["points_world_m"]
        if len(points_world) != 2:
            raise ValueError(f"Контроль {control['id']} должен иметь ровно две точки")
        projections = {
            camera_id: [
                project_world_point_to_pixel(camera, point, description)
                for point in points_world
            ]
            for camera_id, camera in cameras.items()
        }
        controls.append(
            {
                "id": control["id"],
                "surface": control.get("surface", "ground"),
                "purpose": control.get("purpose", "evaluation"),
                "points_world_m": points_world,
                "projections": projections,
            }
        )
    return controls


def configure_depth_output(output_directory: Path) -> None:
    """Направляет стандартный Z-pass в отдельный 32-битный EXR.

    Z-pass хранит расстояние от камеры до ближайшей видимой поверхности. Слой
    нужен только оценщику и принципиально не передаётся RGB-алгоритму.
    """

    scene = bpy.context.scene
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    scene.use_nodes = True
    node_tree = scene.node_tree
    node_tree.nodes.clear()

    render_layers = node_tree.nodes.new("CompositorNodeRLayers")
    composite = node_tree.nodes.new("CompositorNodeComposite")
    node_tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])

    depth_output = node_tree.nodes.new("CompositorNodeOutputFile")
    depth_output.base_path = str(output_directory)
    depth_output.file_slots[0].path = "depth"
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_mode = "BW"
    depth_output.format.color_depth = "32"
    node_tree.links.new(render_layers.outputs["Depth"], depth_output.inputs[0])


def make_emission_material(
    name: str,
    value: float,
) -> bpy.types.Material:
    """Создаёт материал постоянной яркости для бинарного mask-рендера."""

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (value, value, value, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def render_ground_mask(
    output_path: Path,
) -> None:
    """Рендерит видимую землю белой, а объекты и фон чёрными.

    Отдельный бинарный рендер надёжнее legacy ID Mask compositor node в
    Blender 4.5. Материалы и настройки полностью восстанавливаются, поэтому
    следующий RGB-кадр остаётся идентичным сцене из описания.
    """

    scene = bpy.context.scene
    white = make_emission_material("mask_ground_white", 1.0)
    black = make_emission_material("mask_non_ground_black", 0.0)
    saved_materials: list[tuple[bpy.types.Object, list[bpy.types.Material]]] = []

    for object_ in scene.objects:
        if not hasattr(object_.data, "materials"):
            continue
        saved_materials.append((object_, list(object_.data.materials)))
        object_.data.materials.clear()
        object_.data.materials.append(white if object_.pass_index == 1 else black)

    background = scene.world.node_tree.nodes.get("Background")
    saved_background_color = background.inputs["Color"].default_value[:]
    saved_background_strength = background.inputs["Strength"].default_value
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0

    saved_use_nodes = scene.use_nodes
    saved_filepath = scene.render.filepath
    saved_color_mode = scene.render.image_settings.color_mode
    saved_color_depth = scene.render.image_settings.color_depth
    scene.use_nodes = False
    scene.render.filepath = str(output_path)
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "8"
    bpy.ops.render.render(write_still=True)

    scene.use_nodes = saved_use_nodes
    scene.render.filepath = saved_filepath
    scene.render.image_settings.color_mode = saved_color_mode
    scene.render.image_settings.color_depth = saved_color_depth
    background.inputs["Color"].default_value = saved_background_color
    background.inputs["Strength"].default_value = saved_background_strength
    for object_, materials in saved_materials:
        object_.data.materials.clear()
        for material in materials:
            object_.data.materials.append(material)


def find_single_control_pass(
    directory: Path,
    *,
    pattern: str,
    label: str,
) -> Path:
    """Находит единственный файл compositor-а или явно сообщает о сбое."""

    candidates = sorted(directory.glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Для {label} ожидался один файл {pattern!r}, найдено {len(candidates)}"
        )
    return candidates[0]


def render_all_cameras(
    description: dict[str, Any],
    cameras: dict[str, bpy.types.Object],
    output_directory: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """Рендерит RGB и опциональные контрольные слои каждой камеры."""

    relative_paths: list[str] = []
    control_layers: list[dict[str, str]] = []
    write_control_passes = bool(description["render"].get("control_passes", False))
    for specification in description["cameras"]:
        camera_id = specification["id"]
        role = specification["role"]
        if role == "reference":
            projection = specification["projection"]
            relative_path = Path("reference") / f"{projection}_rgb.png"
        else:
            relative_path = Path(role) / camera_id / "rgb.png"
        absolute_path = output_directory / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)

        bpy.context.scene.camera = cameras[camera_id]
        bpy.context.scene.render.filepath = str(absolute_path)
        if write_control_passes:
            configure_depth_output(absolute_path.parent)
        bpy.ops.render.render(write_still=True)
        relative_paths.append(relative_path.as_posix())
        if write_control_passes:
            depth_path = find_single_control_pass(
                absolute_path.parent, pattern="depth*.exr", label="depth"
            )
            mask_path = absolute_path.parent / "ground_mask.png"
            render_ground_mask(mask_path)
            control_layers.append(
                {
                    "camera_id": camera_id,
                    "depth": depth_path.relative_to(output_directory).as_posix(),
                    "ground_mask": (mask_path.relative_to(output_directory).as_posix()),
                }
            )
    return relative_paths, control_layers


def main() -> None:
    """Выполняет полный детерминированный G0-проход."""

    started_at = time.perf_counter()
    description_path, output_directory = parse_script_arguments()
    description = json.loads(description_path.read_text(encoding="utf-8"))
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "scene_description.json").write_text(
        json.dumps(description, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    reset_scene()
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.length_unit = "METERS"
    collections = make_collections()
    build_ground(description, collections, output_directory)
    control_points = (
        build_control_points(collections)
        if description["world"].get("show_control_points", True)
        else []
    )
    build_configured_objects(description, collections)
    for object_ in collections["Ground"].objects:
        object_.pass_index = 1
    for object_ in collections["Objects"].objects:
        object_.pass_index = 2
    for object_ in collections["Control Points"].objects:
        object_.pass_index = 3
    cameras = build_cameras(description, collections)
    add_lighting(collections)
    configure_render(description)

    # Резервный .blend1 полезен при ручном моделировании, но в детерминированном
    # генераторе лишь создаёт неописанный артефакт рядом с итоговой сценой.
    bpy.context.preferences.filepaths.save_version = 0
    scene_path = output_directory / "scene.blend"
    rendered_images, rendered_control_layers = render_all_cameras(
        description, cameras, output_directory
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(scene_path))

    metadata = {
        "schema_version": 1,
        "scenario_id": description["scenario_id"],
        "blender_version": bpy.app.version_string,
        "seed": description["seed"],
        "world_units": description["world"]["units"],
        "world_axis_convention": description["world"]["axis_convention"],
        "render_engine": bpy.context.scene.render.engine,
        "rendered_images": rendered_images,
        "rendered_control_layers": rendered_control_layers,
        "scene_file": "scene.blend",
        "control_points_world_m": control_points,
        "reference_map": reference_map_metadata(description, cameras),
        "metric_controls": metric_control_metadata(description, cameras),
        "cameras": [
            camera_metadata(cameras[item["id"]], item)
            for item in description["cameras"]
        ],
        "elapsed_seconds": time.perf_counter() - started_at,
        "limitations": [
            (
                "Z-pass и ground mask являются скрытой истиной и не входят в "
                "RGB-алгоритм."
                if description["render"].get("control_passes", False)
                else "Depth и ground mask для этого сценария не сформированы."
            ),
            "Поза записана в соглашении осей Blender, а не OpenCV.",
            "Метрики smoke-теста не проверяют метрическую правильность.",
        ],
    }
    (output_directory / "generation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SYNTHETIC_3D_METADATA=" + json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
