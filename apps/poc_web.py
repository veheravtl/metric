#!/usr/bin/env python3
"""Локальный Gradio-интерфейс для визуальной проверки первого PoC."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import rasterio
from numpy.typing import NDArray

from aerial_mapper.synthetic import SyntheticFrameSpec, generate_synthetic_frame
from aerial_mapper.visualization import (
    build_reverse_overlay,
    draw_reference_footprint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = PROJECT_ROOT / "data/reference/cherkasy_2021_poc.tif"
DEFAULT_ROTATION_DEGREES = 12.0
DEFAULT_PERSPECTIVE_STRENGTH = 0.35


@lru_cache(maxsize=1)
def load_reference() -> tuple[NDArray[np.uint8], float, str]:
    """Загружает RGB-эталон и его метрическое разрешение один раз за процесс.

    Кэш нужен только для отзывчивости интерфейса: повторное нажатие кнопки не
    должно заново распаковывать многомегабайтный GeoTIFF. Математические функции
    по-прежнему получают обычный NumPy-массив и не зависят от Gradio.
    """

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            "Эталонный GeoTIFF не найден. Сначала выполните: "
            "uv run python scripts/download_reference.py"
        )

    with rasterio.open(REFERENCE_PATH) as dataset:
        if dataset.count < 3:
            raise ValueError("Эталон должен содержать не менее трёх RGB-каналов")
        if dataset.crs is None:
            raise ValueError("В эталонном GeoTIFF отсутствует система координат")

        resolution_x = float(dataset.res[0])
        resolution_y = float(dataset.res[1])
        if not np.isclose(resolution_x, resolution_y, atol=1e-9):
            raise ValueError("Для PoC ожидаются квадратные пиксели")

        # Rasterio возвращает массив в порядке каналы, строки, столбцы.
        # Gradio и OpenCV в этом проекте используют строки, столбцы, каналы.
        bands_first = dataset.read((1, 2, 3))
        reference_rgb = np.moveaxis(bands_first, 0, -1)
        return reference_rgb, resolution_x, dataset.crs.to_string()


def render_experiment(
    rotation_degrees: float = DEFAULT_ROTATION_DEGREES,
    perspective_strength: float = DEFAULT_PERSPECTIVE_STRENGTH,
) -> tuple[
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.uint8],
    dict[str, Any],
]:
    """Создаёт изображения и диагностику для выбранной плоской деформации.

    Аргументы приходят из компонентов Gradio, поэтому мы явно приводим их к
    ``float`` перед передачей в математическое ядро. Интерфейс отвечает только
    за ввод параметров и отображение результата; сама генерация синтетического
    кадра остаётся в пакете ``aerial_mapper`` и проверяется отдельно тестами.
    """

    reference_rgb, resolution_m_per_pixel, reference_crs = load_reference()
    spec = SyntheticFrameSpec(
        rotation_degrees=float(rotation_degrees),
        perspective_strength=float(perspective_strength),
    )
    synthetic = generate_synthetic_frame(
        reference_rgb,
        reference_resolution_m_per_pixel=resolution_m_per_pixel,
        spec=spec,
    )

    annotated_reference = draw_reference_footprint(
        reference_rgb,
        synthetic.source_corners_reference_px,
    )
    reverse_overlay = build_reverse_overlay(
        reference_rgb,
        synthetic.image_rgb,
        synthetic.homography_frame_to_reference,
        synthetic.source_corners_reference_px,
    )

    diagnostics: dict[str, Any] = {
        "reference": {
            "path": str(REFERENCE_PATH.relative_to(PROJECT_ROOT)),
            "shape_pixels": list(reference_rgb.shape),
            "crs": reference_crs,
            "resolution_m_per_pixel": resolution_m_per_pixel,
        },
        "synthetic_frame": {
            "ground_footprint_m": [
                spec.footprint_width_m,
                spec.footprint_height_m,
            ],
            "shape_pixels": [
                spec.output_height_pixels,
                spec.output_width_pixels,
                3,
            ],
            "rotation_degrees": spec.rotation_degrees,
            "perspective_strength": spec.perspective_strength,
        },
        "source_corners_reference_px": np.round(
            synthetic.source_corners_reference_px,
            3,
        ).tolist(),
        "homography_reference_to_frame": np.round(
            synthetic.homography_reference_to_frame,
            8,
        ).tolist(),
        "homography_frame_to_reference": np.round(
            synthetic.homography_frame_to_reference,
            8,
        ).tolist(),
        "reverse_overlay": {
            "covered_pixels": reverse_overlay.covered_pixels,
            "mean_absolute_error_0_255": round(
                reverse_overlay.mean_absolute_error,
                4,
            ),
            "note": (
                "Ошибка не обязана быть нулевой из-за двух последовательных "
                "интерполяций изображения."
            ),
        },
    }

    return (
        annotated_reference,
        synthetic.image_rgb,
        reverse_overlay.comparison_rgb,
        diagnostics,
    )


def build_app() -> gr.Blocks:
    """Собирает декларативное дерево компонентов Gradio."""

    (
        initial_reference,
        initial_frame,
        initial_overlay,
        initial_diagnostics,
    ) = render_experiment()

    with gr.Blocks(
        title="Aerial Map Measurement — PoC",
        fill_width=True,
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            """
# Визуальная проверка синтетической гомографии

Слева показан полный эталон и оранжевый след виртуальной камеры. Справа —
прямоугольный кадр, полученный перспективным преобразованием этого участка.
Ниже кадр преобразован обратно: красно-бирюзовые двойные контуры означают
несовпадение, серые области — геометрическое согласие.

Это пока контрольный пример с известной истинной гомографией. Здесь ещё нет
оценивания гомографии по выбранным или автоматически найденным точкам. Важно:
слайдер «перспектива» деформирует одну плоскость и не моделирует высоту зданий,
видимые фасады, деревья, взаимные перекрытия объектов и параллакс.
"""
        )

        with gr.Row():
            reference_image = gr.Image(
                value=initial_reference,
                label="Полный эталон и след камеры",
                format="png",
                image_mode="RGB",
                interactive=False,
                height=520,
                buttons=["fullscreen", "download"],
            )
            frame_image = gr.Image(
                value=initial_frame,
                label="Синтетический кадр виртуальной камеры",
                format="png",
                image_mode="RGB",
                interactive=False,
                height=520,
                buttons=["fullscreen", "download"],
            )

        with gr.Row():
            rotation_slider = gr.Slider(
                minimum=-35.0,
                maximum=35.0,
                value=DEFAULT_ROTATION_DEGREES,
                step=1.0,
                label="Поворот участка, градусы",
                info=(
                    "Поворачивает след виртуальной камеры относительно "
                    "эталонного снимка."
                ),
            )
            perspective_slider = gr.Slider(
                minimum=0.0,
                maximum=0.8,
                value=DEFAULT_PERSPECTIVE_STRENGTH,
                step=0.05,
                label="Сила плоского перспективного перекоса",
                info=(
                    "0 — прямоугольный участок; большие значения сильнее "
                    "сужают одну сторону четырёхугольника. Это не 3D-модель."
                ),
            )

        recompute_button = gr.Button(
            "Пересчитать контрольный пример",
            variant="primary",
        )

        with gr.Row():
            overlay_image = gr.Image(
                value=initial_overlay,
                label="Обратное наложение: красный — кадр, бирюзовый — эталон",
                format="png",
                image_mode="RGB",
                interactive=False,
                height=560,
                buttons=["fullscreen", "download"],
                scale=2,
            )
            diagnostics = gr.JSON(
                value=initial_diagnostics,
                label="Параметры и геометрическая истина",
                open=True,
                scale=1,
            )

        recompute_button.click(
            fn=render_experiment,
            inputs=[rotation_slider, perspective_slider],
            outputs=[
                reference_image,
                frame_image,
                overlay_image,
                diagnostics,
            ],
            show_progress="minimal",
        )

    return demo


def main() -> None:
    """Запускает только локальный сервер без публичной ссылки Gradio."""

    demo = build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
