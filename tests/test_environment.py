"""Дымовые тесты минимального окружения проекта.

Эти тесты пока не проверяют предметные алгоритмы. Их задача проще: быстро
обнаружить неполную установку окружения до начала экспериментов с гомографией.
"""

import cv2
import matplotlib
import numpy as np

import aerial_mapper


def test_project_package_is_importable() -> None:
    """Проверяем, что пакет установлен из ``src`` и доступен тестам."""
    assert aerial_mapper.__version__ == "0.1.0"


def test_scientific_dependencies_are_importable() -> None:
    """Проверяем импорт библиотек, необходимых для первого дня spike.

    Сам факт импорта важен для бинарных зависимостей NumPy и OpenCV: пакет мог
    установиться, но оказаться несовместимым с используемой версией Python или
    системными библиотеками. Создание маленького массива дополнительно
    подтверждает, что базовая численная операция действительно выполняется.
    """
    sample_image = np.zeros((2, 2, 3), dtype=np.uint8)

    assert sample_image.shape == (2, 2, 3)
    assert cv2.__version__
    assert matplotlib.__version__
