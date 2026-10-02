#!/usr/bin/env python3
"""Локальный Gradio-интерфейс для визуальной проверки первого PoC."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import rasterio
from numpy.typing import NDArray

from aerial_mapper.alignment import SiftRansacConfig, align_frame_to_reference
from aerial_mapper.evaluation import evaluate_homography
from aerial_mapper.synthetic import SyntheticFrameSpec, generate_synthetic_frame
from aerial_mapper.visualization import (
    build_reverse_overlay,
    draw_alignment_footprints,
    draw_alignment_matches,
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
    NDArray[np.uint8],
    dict[str, Any],
]:
    """Генерирует кадр, независимо привязывает его и оценивает результат.

    Аргументы приходят из компонентов Gradio, поэтому мы явно приводим их к
    ``float`` перед передачей в математическое ядро. Важная граница эксперимента:
    функция ``align_frame_to_reference`` получает только два изображения. Лишь
    после её завершения истинная матрица генератора передаётся оценщику ошибки.
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

    alignment_config = SiftRansacConfig()
    alignment = align_frame_to_reference(
        reference_rgb,
        synthetic.image_rgb,
        config=alignment_config,
    )
    evaluation = evaluate_homography(
        alignment.homography_frame_to_reference,
        synthetic.homography_frame_to_reference,
        frame_width_pixels=spec.output_width_pixels,
        frame_height_pixels=spec.output_height_pixels,
        reference_resolution_m_per_pixel=resolution_m_per_pixel,
    )

    # Эти углы вычисляются из найденной матрицы. Истинные углы генератора не
    # участвуют в привязке и нужны только для последующего сравнения контуров.
    estimated_corners_reference_px = cv2.perspectiveTransform(
        synthetic.destination_corners_frame_px.reshape(1, -1, 2),
        alignment.homography_frame_to_reference,
    ).reshape(-1, 2)

    annotated_reference = draw_alignment_footprints(
        reference_rgb,
        synthetic.source_corners_reference_px,
        estimated_corners_reference_px,
    )
    match_visualization = draw_alignment_matches(
        reference_rgb,
        synthetic.image_rgb,
        alignment,
    )
    reverse_overlay = build_reverse_overlay(
        reference_rgb,
        synthetic.image_rgb,
        alignment.homography_frame_to_reference,
        estimated_corners_reference_px,
    )
    normalized_true_homography = synthetic.homography_frame_to_reference.copy()
    normalized_true_homography /= normalized_true_homography[2, 2]

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
        "independent_alignment": {
            "method": "SIFT + brute-force kNN + Lowe ratio test + RANSAC",
            "reference_keypoints": alignment.reference_keypoint_count,
            "frame_keypoints": alignment.frame_keypoint_count,
            "candidate_knn_pairs": alignment.candidate_match_count,
            "matches_after_ratio_test": alignment.ratio_match_count,
            "ransac_inliers": alignment.inlier_count,
            "ransac_inlier_ratio": round(alignment.inlier_ratio, 4),
            "inlier_spatial_coverage_fraction": round(
                alignment.inlier_spatial_coverage_fraction,
                4,
            ),
            "processing_time_seconds": round(
                alignment.processing_time_seconds,
                4,
            ),
            "ratio_threshold": alignment_config.ratio_threshold,
            "ransac_reprojection_threshold_reference_px": (
                alignment_config.ransac_reprojection_threshold_px
            ),
            "estimated_homography_frame_to_reference": np.round(
                alignment.homography_frame_to_reference,
                8,
            ).tolist(),
        },
        "evaluation_after_alignment": {
            "control_grid_points": evaluation.control_point_count,
            "mean_transfer_error_reference_px": round(
                evaluation.mean_error_pixels,
                4,
            ),
            "median_transfer_error_reference_px": round(
                evaluation.median_error_pixels,
                4,
            ),
            "max_transfer_error_reference_px": round(
                evaluation.max_error_pixels,
                4,
            ),
            "rmse_transfer_error_reference_px": round(
                evaluation.root_mean_square_error_pixels,
                4,
            ),
            "mean_transfer_error_meters": round(
                evaluation.mean_error_meters,
                6,
            ),
            "median_transfer_error_meters": round(
                evaluation.median_error_meters,
                6,
            ),
            "max_transfer_error_meters": round(
                evaluation.max_error_meters,
                6,
            ),
            "corner_errors_reference_px": np.round(
                evaluation.corner_errors_pixels,
                4,
            ).tolist(),
            "true_homography_frame_to_reference": np.round(
                normalized_true_homography,
                8,
            ).tolist(),
            "note": (
                "Истинная матрица открывается только оценщику после того, "
                "как SIFT/RANSAC завершили независимый расчёт."
            ),
        },
        "estimated_reverse_overlay": {
            "covered_pixels": reverse_overlay.covered_pixels,
            "mean_absolute_error_0_255": round(
                reverse_overlay.mean_absolute_error,
                4,
            ),
            "note": (
                "Наложение построено найденной, а не истинной матрицей. "
                "Фотометрическая ошибка включает интерполяцию изображения."
            ),
        },
    }

    return (
        annotated_reference,
        synthetic.image_rgb,
        match_visualization,
        reverse_overlay.comparison_rgb,
        diagnostics,
    )


def build_app() -> gr.Blocks:
    """Собирает декларативное дерево компонентов Gradio."""

    (
        initial_reference,
        initial_frame,
        initial_matches,
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
# Честная плоская привязка: SIFT + RANSAC

Генератор знает истинное положение участка, но алгоритм привязки получает только
полный эталон и готовый кадр. SIFT самостоятельно находит локальные признаки,
а RANSAC выбирает пары, согласующиеся с одной гомографией.

На эталоне оранжевый контур — скрытая от алгоритма истина, бирюзовый — независимо
найденное положение. Красно-бирюзовое наложение ниже построено уже оценённой
матрицей. Слайдер «перспектива» по-прежнему деформирует только одну плоскость и
не моделирует высоту зданий, фасады, окклюзии и параллакс.
"""
        )

        with gr.Row():
            reference_image = gr.Image(
                value=initial_reference,
                label="Истинный (оранжевый) и найденный (бирюзовый) след",
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
            "Создать кадр и найти его без подсказки",
            variant="primary",
        )

        match_image = gr.Image(
            value=initial_matches,
            label=(
                "SIFT-пары: зелёные приняты RANSAC, красные отвергнуты "
                "(показана выборка)"
            ),
            format="png",
            image_mode="RGB",
            interactive=False,
            height=540,
            buttons=["fullscreen", "download"],
        )

        with gr.Row():
            overlay_image = gr.Image(
                value=initial_overlay,
                label=("Наложение по найденной H: красный — кадр, бирюзовый — эталон"),
                format="png",
                image_mode="RGB",
                interactive=False,
                height=560,
                buttons=["fullscreen", "download"],
                scale=2,
            )
            diagnostics = gr.JSON(
                value=initial_diagnostics,
                label="Признаки, RANSAC и независимая ошибка",
                open=True,
                scale=1,
            )

        recompute_button.click(
            fn=render_experiment,
            inputs=[rotation_slider, perspective_slider],
            outputs=[
                reference_image,
                frame_image,
                match_image,
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
