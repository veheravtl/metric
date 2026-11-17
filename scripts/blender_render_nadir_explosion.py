# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Рендерит согласованный во времени взрыв камерой строго сверху.

Скрипт запускается встроенным Python Blender:

    blender --background --python scripts/blender_render_nadir_explosion.py \
        -- OUTPUT_DIRECTORY FIRST_FRAME LAST_FRAME

RGBA-кадры содержат только эффект и мягкую тень. Горизонта и плоскости
исходного футажа здесь нет: геометрия строится в метрах, камера смотрит вдоль
мировой оси Z, а прозрачный слой затем можно совместить с надирной съёмкой.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector

RANDOM_SEED = 20261113
CANVAS_SIZE_METERS = 32.0
OUTPUT_SIZE_PIXELS = 512
ANIMATION_LAST_FRAME = 76


def parse_arguments() -> tuple[Path, int, int]:
    """Возвращает каталог и включительный диапазон кадров после ``--``."""

    if "--" not in sys.argv:
        raise ValueError("Нужны аргументы: OUTPUT_DIRECTORY FIRST_FRAME LAST_FRAME")
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    if len(arguments) != 3:
        raise ValueError("Ожидались OUTPUT_DIRECTORY FIRST_FRAME LAST_FRAME")
    output = Path(arguments[0]).expanduser().resolve()
    first_frame = int(arguments[1])
    last_frame = int(arguments[2])
    if first_frame < 1 or last_frame < first_frame:
        raise ValueError("Некорректный диапазон кадров Blender")
    if last_frame > ANIMATION_LAST_FRAME:
        raise ValueError(f"Максимальный кадр сцены: {ANIMATION_LAST_FRAME}")
    return output, first_frame, last_frame


def reset_scene() -> None:
    """Удаляет стартовые объекты и не зависит от пользовательского startup."""

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)


def set_linear_animation(target: object) -> None:
    """Убирает автоматические ускорения между физическими фазами эффекта."""

    animation_data = getattr(target, "animation_data", None)
    action = getattr(animation_data, "action", None)
    if action is None:
        return
    for curve in action.fcurves:
        for keyframe in curve.keyframe_points:
            keyframe.interpolation = "LINEAR"


def animate_vector(
    object_: bpy.types.Object,
    property_name: str,
    values: list[tuple[int, tuple[float, float, float]]],
) -> None:
    """Ставит линейные ключи для координаты, масштаба или вращения."""

    for frame, value in values:
        setattr(object_, property_name, value)
        object_.keyframe_insert(data_path=property_name, frame=frame)
    set_linear_animation(object_)


def make_cloud_material(
    name: str,
    *,
    dark_color: tuple[float, float, float, float],
    light_color: tuple[float, float, float, float],
    maximum_opacity: float,
    emission_strength: float = 0.0,
) -> tuple[bpy.types.Material, bpy.types.NodeSocket]:
    """Создаёт объёмный клубок с шумовой плотностью и мягкой границей.

    Плотность внутри меша равна произведению трёх величин: трёхмерного шума,
    радиального затухания к границе и анимируемой общей плотности. Поэтому
    поверхность сферы не появляется в кадре отдельной круглой оболочкой.
    """

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeVolumePrincipled")
    principled.inputs["Anisotropy"].default_value = 0.18
    principled.inputs["Emission Color"].default_value = light_color
    principled.inputs["Emission Strength"].default_value = emission_strength

    coordinates = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    noise.inputs["Scale"].default_value = 2.7
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.72
    noise.inputs["Distortion"].default_value = 0.28
    noise.inputs["W"].default_value = random.random() * 20.0
    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].position = 0.25
    color_ramp.color_ramp.elements[0].color = dark_color
    color_ramp.color_ramp.elements[1].position = 0.78
    color_ramp.color_ramp.elements[1].color = light_color

    density_ramp = nodes.new("ShaderNodeValToRGB")
    density_ramp.color_ramp.elements[0].position = 0.30
    density_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    density_ramp.color_ramp.elements[1].position = 0.58
    density_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    center = nodes.new("ShaderNodeCombineXYZ")
    center.inputs["X"].default_value = 0.5
    center.inputs["Y"].default_value = 0.5
    center.inputs["Z"].default_value = 0.5
    distance = nodes.new("ShaderNodeVectorMath")
    distance.operation = "DISTANCE"
    radial_falloff = nodes.new("ShaderNodeMapRange")
    radial_falloff.clamp = True
    radial_falloff.inputs["From Min"].default_value = 0.12
    radial_falloff.inputs["From Max"].default_value = 0.50
    radial_falloff.inputs["To Min"].default_value = 1.0
    radial_falloff.inputs["To Max"].default_value = 0.0
    noise_radial = nodes.new("ShaderNodeMath")
    noise_radial.operation = "MULTIPLY"
    opacity = nodes.new("ShaderNodeValue")
    opacity.name = "animated_opacity"
    opacity.outputs[0].default_value = maximum_opacity
    opacity_density = nodes.new("ShaderNodeMath")
    opacity_density.operation = "MULTIPLY"
    base_density = nodes.new("ShaderNodeMath")
    base_density.operation = "MULTIPLY"
    base_density.inputs[1].default_value = 2.0

    links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], principled.inputs["Color"])
    links.new(noise.outputs["Fac"], density_ramp.inputs["Fac"])
    links.new(coordinates.outputs["Generated"], distance.inputs[0])
    links.new(center.outputs["Vector"], distance.inputs[1])
    links.new(distance.outputs["Value"], radial_falloff.inputs["Value"])
    links.new(density_ramp.outputs["Color"], noise_radial.inputs[0])
    links.new(radial_falloff.outputs["Result"], noise_radial.inputs[1])
    links.new(noise_radial.outputs[0], opacity_density.inputs[0])
    links.new(opacity.outputs[0], opacity_density.inputs[1])
    links.new(opacity_density.outputs[0], base_density.inputs[0])
    links.new(base_density.outputs[0], principled.inputs["Density"])
    links.new(principled.outputs["Volume"], output.inputs["Volume"])
    return material, opacity.outputs[0]


def animate_opacity(
    socket: bpy.types.NodeSocket,
    values: list[tuple[int, float]],
) -> None:
    """Анимирует общую непрозрачность конкретного клубка."""

    for frame, value in values:
        socket.default_value = value
        socket.keyframe_insert(data_path="default_value", frame=frame)
    set_linear_animation(socket.id_data)


def add_irregular_sphere(
    name: str,
    *,
    material: bpy.types.Material,
    location: tuple[float, float, float],
    displacement_strength: float,
) -> bpy.types.Object:
    """Создаёт сглаженный клубок с нерегулярным, но детерминированным краем."""

    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=3,
        radius=1.0,
        location=location,
    )
    object_ = bpy.context.object
    object_.name = name
    object_.data.materials.append(material)
    for polygon in object_.data.polygons:
        polygon.use_smooth = True
    texture = bpy.data.textures.new(f"tex_{name}", type="CLOUDS")
    texture.noise_scale = 0.42
    texture.noise_depth = 2
    modifier = object_.modifiers.new(f"displace_{name}", type="DISPLACE")
    modifier.texture = texture
    modifier.texture_coords = "GLOBAL"
    modifier.strength = displacement_strength
    modifier.mid_level = 0.48
    bpy.context.view_layer.objects.active = object_
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    return object_


def add_fire_core(random_generator: random.Random) -> None:
    """Добавляет короткоживущие горячие клубки в центре события."""

    fire_palettes = (
        ((0.24, 0.008, 0.001, 1.0), (1.0, 0.18, 0.004, 1.0), 0.65),
        ((0.42, 0.025, 0.002, 1.0), (1.0, 0.42, 0.020, 1.0), 0.90),
        ((0.16, 0.004, 0.001, 1.0), (0.72, 0.07, 0.002, 1.0), 0.45),
    )
    for index in range(14):
        angle = random_generator.uniform(0.0, math.tau)
        radius = random_generator.uniform(0.0, 2.0)
        birth = random_generator.randint(1, 2)
        palette = fire_palettes[index % len(fire_palettes)]
        material, opacity = make_cloud_material(
            f"mat_fire_{index:02d}",
            dark_color=palette[0],
            light_color=palette[1],
            maximum_opacity=0.92,
            emission_strength=palette[2],
        )
        start = (
            math.cos(angle) * radius,
            math.sin(angle) * radius,
            random_generator.uniform(0.45, 1.8),
        )
        object_ = add_irregular_sphere(
            f"fire_{index:02d}",
            material=material,
            location=start,
            displacement_strength=random_generator.uniform(0.48, 0.82),
        )
        aspect = random_generator.uniform(0.62, 1.48)
        peak_radius = random_generator.uniform(1.35, 2.45)
        animate_vector(
            object_,
            "scale",
            [
                (birth - 1, (0.01, 0.01, 0.01)),
                (birth, (0.65, 0.65, 0.72)),
                (3, (peak_radius * aspect, peak_radius / aspect, peak_radius)),
                (6, (peak_radius * 1.25, peak_radius * 1.15, peak_radius * 1.35)),
            ],
        )
        end = (
            start[0] * random_generator.uniform(1.8, 3.0),
            start[1] * random_generator.uniform(1.8, 3.0),
            start[2] + random_generator.uniform(1.5, 3.8),
        )
        animate_vector(object_, "location", [(birth, start), (6, end)])
        animate_opacity(
            opacity,
            [
                (birth - 1, 0.0),
                (birth, 0.85),
                (birth + 1, 0.98),
                (birth + 3, 0.55),
                (birth + 6, 0.0),
            ],
        )


def add_smoke_billows(random_generator: random.Random) -> None:
    """Добавляет тёмные объёмные клубы, которые растут и поднимаются."""

    palettes = (
        ((0.025, 0.022, 0.020, 1.0), (0.17, 0.15, 0.13, 1.0)),
        ((0.045, 0.040, 0.036, 1.0), (0.29, 0.25, 0.21, 1.0)),
        ((0.07, 0.065, 0.06, 1.0), (0.40, 0.37, 0.33, 1.0)),
    )
    for index in range(22):
        angle = random_generator.uniform(0.0, math.tau)
        radius = random_generator.uniform(0.0, 2.4)
        birth = random_generator.randint(1, 3)
        dark, light = palettes[index % len(palettes)]
        maximum_opacity = random_generator.uniform(0.62, 0.88)
        material, opacity = make_cloud_material(
            f"mat_smoke_{index:02d}",
            dark_color=dark,
            light_color=light,
            maximum_opacity=maximum_opacity,
        )
        start = (
            math.cos(angle) * radius,
            math.sin(angle) * radius,
            random_generator.uniform(0.7, 2.2),
        )
        object_ = add_irregular_sphere(
            f"smoke_{index:02d}",
            material=material,
            location=start,
            displacement_strength=random_generator.uniform(0.30, 0.58),
        )
        initial = random_generator.uniform(0.35, 0.65)
        middle = random_generator.uniform(1.2, 2.4)
        final = random_generator.uniform(2.4, 4.2)
        aspect = random_generator.uniform(0.72, 1.35)
        animate_vector(
            object_,
            "scale",
            [
                (birth - 1, (0.01, 0.01, 0.01)),
                (birth, (initial, initial, initial)),
                (birth + 2, (middle * 0.8 * aspect, middle * 0.8 / aspect, middle)),
                (6, (middle * 1.15 * aspect, middle * 1.15 / aspect, middle * 1.3)),
                (15, (middle * 1.5 * aspect, middle * 1.5 / aspect, middle * 1.7)),
                (ANIMATION_LAST_FRAME, (final * aspect, final / aspect, final * 1.4)),
            ],
        )
        end_radius = radius + random_generator.uniform(3.0, 7.0)
        middle = (
            math.cos(angle) * (radius + random_generator.uniform(0.8, 2.5)),
            math.sin(angle) * (radius + random_generator.uniform(0.8, 2.5)),
            start[2] + random_generator.uniform(2.0, 5.0),
        )
        end = (
            math.cos(angle) * end_radius,
            math.sin(angle) * end_radius,
            start[2] + random_generator.uniform(7.0, 12.0),
        )
        animate_vector(
            object_,
            "location",
            [(birth, start), (6, middle), (ANIMATION_LAST_FRAME, end)],
        )
        animate_opacity(
            opacity,
            [
                (birth - 1, 0.0),
                (birth, 0.20),
                (birth + 2, maximum_opacity),
                (15, maximum_opacity * 0.82),
                (35, maximum_opacity * 0.70),
                (ANIMATION_LAST_FRAME, maximum_opacity * 0.18),
            ],
        )


def add_dust_ring(random_generator: random.Random) -> None:
    """Добавляет расширяющееся низкое кольцо пыли в плоскости земли."""

    palettes = (
        ((0.075, 0.050, 0.032, 1.0), (0.36, 0.23, 0.12, 1.0)),
        ((0.06, 0.048, 0.040, 1.0), (0.29, 0.23, 0.18, 1.0)),
        ((0.10, 0.07, 0.045, 1.0), (0.46, 0.32, 0.18, 1.0)),
    )
    count = 24
    for index in range(count):
        angle = math.tau * index / count + random_generator.uniform(-0.10, 0.10)
        birth = random_generator.randint(1, 2)
        dark, light = palettes[index % len(palettes)]
        maximum_opacity = random_generator.uniform(0.32, 0.58)
        material, opacity = make_cloud_material(
            f"mat_dust_{index:02d}",
            dark_color=dark,
            light_color=light,
            maximum_opacity=maximum_opacity,
        )
        start_radius = random_generator.uniform(0.7, 1.5)
        middle_radius = random_generator.uniform(5.0, 7.5)
        end_radius = random_generator.uniform(11.0, 14.0)
        start = (
            math.cos(angle) * start_radius,
            math.sin(angle) * start_radius,
            random_generator.uniform(0.18, 0.55),
        )
        object_ = add_irregular_sphere(
            f"dust_{index:02d}",
            material=material,
            location=start,
            displacement_strength=random_generator.uniform(0.28, 0.52),
        )
        middle_scale = random_generator.uniform(1.0, 2.0)
        final_scale = random_generator.uniform(2.0, 3.6)
        tangential = random_generator.uniform(0.8, 1.6)
        animate_vector(
            object_,
            "scale",
            [
                (birth - 1, (0.01, 0.01, 0.01)),
                (birth, (0.25, 0.25, 0.18)),
                (3, (middle_scale * 1.2, tangential, middle_scale * 0.70)),
                (7, (middle_scale * 1.8, tangential * 1.4, middle_scale * 0.85)),
                (15, (final_scale * 1.4, final_scale, final_scale * 0.60)),
                (35, (final_scale * 1.7, final_scale * 1.2, final_scale * 0.52)),
            ],
        )
        middle = (
            math.cos(angle) * middle_radius,
            math.sin(angle) * middle_radius,
            random_generator.uniform(0.35, 1.0),
        )
        end = (
            math.cos(angle) * end_radius,
            math.sin(angle) * end_radius,
            random_generator.uniform(0.45, 1.5),
        )
        animate_vector(object_, "location", [(birth, start), (3, middle), (8, end)])
        animate_opacity(
            opacity,
            [
                (birth - 1, 0.0),
                (birth, 0.18),
                (birth + 1, maximum_opacity),
                (8, maximum_opacity * 0.72),
                (15, maximum_opacity * 0.42),
                (35, 0.0),
            ],
        )


def add_soft_shadow() -> None:
    """Создаёт мягкую локальную тень без непрозрачной наземной плоскости."""

    material, opacity = make_cloud_material(
        "mat_shadow",
        dark_color=(0.002, 0.002, 0.002, 1.0),
        light_color=(0.012, 0.010, 0.008, 1.0),
        maximum_opacity=0.24,
    )
    shadow = add_irregular_sphere(
        "soft_ground_shadow",
        material=material,
        location=(0.8, -0.6, 0.12),
        displacement_strength=0.18,
    )
    animate_vector(
        shadow,
        "scale",
        [
            (1, (0.8, 0.8, 0.03)),
            (8, (3.6, 3.0, 0.06)),
            (15, (5.8, 4.7, 0.08)),
            (ANIMATION_LAST_FRAME, (9.0, 7.2, 0.10)),
        ],
    )
    animate_opacity(
        opacity,
        [(1, 0.08), (8, 0.24), (25, 0.18), (ANIMATION_LAST_FRAME, 0.03)],
    )


def add_shockwave_ring() -> None:
    """Добавляет короткую расширяющуюся кольцевую вспышку на земле."""

    material, opacity = make_cloud_material(
        "mat_shockwave",
        dark_color=(0.20, 0.045, 0.005, 1.0),
        light_color=(1.0, 0.34, 0.025, 1.0),
        maximum_opacity=0.72,
        emission_strength=2.2,
    )
    bpy.ops.mesh.primitive_torus_add(
        major_radius=1.0,
        minor_radius=0.12,
        major_segments=96,
        minor_segments=16,
        location=(0.0, 0.0, 0.25),
    )
    ring = bpy.context.object
    ring.name = "shockwave_ring"
    ring.data.materials.append(material)
    animate_vector(
        ring,
        "scale",
        [(1, (0.4, 0.4, 0.6)), (5, (4.8, 4.8, 1.0)), (9, (7.0, 7.0, 1.2))],
    )
    animate_opacity(opacity, [(1, 0.72), (5, 0.28), (9, 0.0)])


def configure_camera_and_lighting() -> None:
    """Настраивает надирную ортографическую камеру и нейтральный свет."""

    bpy.ops.object.camera_add(location=(0.0, 0.0, 40.0))
    camera = bpy.context.object
    camera.name = "nadir_camera"
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera.rotation_euler[0] = 0.0
    direction = Vector((0.0, 0.0, 0.0)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = CANVAS_SIZE_METERS
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 20.0))
    sun = bpy.context.object
    sun.name = "sun"
    sun.data.energy = 3.0
    sun.data.angle = math.radians(10.0)
    sun.rotation_euler = (math.radians(28.0), math.radians(-24.0), math.radians(18.0))

    bpy.ops.object.light_add(type="AREA", location=(-7.0, -8.0, 18.0))
    area = bpy.context.object
    area.name = "warm_fill"
    area.data.energy = 420.0
    area.data.shape = "DISK"
    area.data.size = 8.0
    area.data.color = (1.0, 0.34, 0.08)
    area.rotation_euler = (
        Vector((0.0, 0.0, 2.0)) - area.location
    ).to_track_quat("-Z", "Y").to_euler()


def configure_render(output_directory: Path, first_frame: int, last_frame: int) -> None:
    """Фиксирует движок, прозрачность, цвет и выходной шаблон PNG."""

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = OUTPUT_SIZE_PIXELS
    scene.render.resolution_y = OUTPUT_SIZE_PIXELS
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.filepath = str(output_directory / "rgba" / "effect_")
    scene.render.fps = 10
    scene.frame_start = first_frame
    scene.frame_end = last_frame
    scene.render.image_settings.compression = 15
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.volumetric_tile_size = "2"
    scene.eevee.volumetric_samples = 96
    scene.eevee.volumetric_shadow_samples = 32
    scene.eevee.use_volumetric_shadows = True
    scene.render.film_transparent = True
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.world.color = (0.025, 0.025, 0.025)
    if scene.world.use_nodes:
        background = scene.world.node_tree.nodes.get("Background")
        if background is not None:
            background.inputs["Color"].default_value = (0.025, 0.025, 0.025, 1.0)
            background.inputs["Strength"].default_value = 0.32


def main() -> None:
    """Строит сцену, сохраняет её и рендерит запрошенный диапазон."""

    output_directory, first_frame, last_frame = parse_arguments()
    rgba_directory = output_directory / "rgba"
    rgba_directory.mkdir(parents=True, exist_ok=True)
    reset_scene()
    random.seed(RANDOM_SEED)
    random_generator = random.Random(RANDOM_SEED)
    configure_render(output_directory, first_frame, last_frame)
    configure_camera_and_lighting()
    add_soft_shadow()
    add_fire_core(random_generator)
    add_smoke_billows(random_generator)
    add_dust_ring(random_generator)

    scene = bpy.context.scene
    scene.frame_set(first_frame)
    bpy.ops.wm.save_as_mainfile(
        filepath=str(output_directory / "nadir_explosion.blend")
    )
    metadata = {
        "schema_version": 1,
        "random_seed": RANDOM_SEED,
        "projection": "orthographic_nadir",
        "canvas_size_meters": CANVAS_SIZE_METERS,
        "output_size_pixels": OUTPUT_SIZE_PIXELS,
        "render_frame_range_inclusive": [first_frame, last_frame],
        "animation_last_frame": ANIMATION_LAST_FRAME,
        "render_engine": scene.render.engine,
        "blender_version": bpy.app.version_string,
    }
    (output_directory / "render_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
