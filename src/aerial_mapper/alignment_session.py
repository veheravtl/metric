"""Неразрушающая память операторской привязки кадров к карте.

Маяк принадлежит неподвижной карте и потому живёт дольше одного видеокадра.
На каждом кадре хранится только наблюдение маяка — его текущая пиксельная
координата. Такое разделение позволяет потерять точку на одном кадре, не
забывая её координату на карте и разметку предыдущих кадров.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from aerial_mapper.frame_quality import FrameImageQuality
from aerial_mapper.manual_alignment import ManualAlignmentResult

Point = tuple[float, float]
AnnotationStatus = Literal[
    "draft", "tracked", "keyframe", "rejected", "excluded", "stale"
]


@dataclass(frozen=True)
class MapLandmark:
    """Постоянный маяк с уникальным номером и координатой на карте."""

    landmark_id: int
    map_point_px: Point


@dataclass(frozen=True)
class FrameObservation:
    """Положение известного картографического маяка на одном видеокадре."""

    landmark_id: int
    frame_point_px: Point


@dataclass(frozen=True)
class MachineObservation:
    """Скрытая автоматическая опора одного перехода между кадрами.

    Координата карты наследуется от решения исходного кадра. Поэтому такая
    точка улучшает временную устойчивость, но не является независимой наземной
    контрольной точкой и не должна подменять ручные маяки.
    """

    source_frame_index: int
    source_frame_point_px: Point
    frame_point_px: Point
    map_point_px: Point
    forward_backward_error_px: float
    used_for_alignment: bool


@dataclass(frozen=True)
class FrameAnnotation:
    """Снимок всех известных наблюдений и совмещения одного кадра.

    ``draft`` означает незавершённую операторскую правку, ``tracked`` —
    автоматический перенос с соседнего кадра, а ``keyframe`` — вручную
    подтверждённую опору. ``rejected`` означает проверенный, но не допущенный
    к измерениям переход, который ещё можно исправить. ``excluded`` означает
    сознательно исключённый оператором мусорный кадр. Черновик, отказ и
    исключение намеренно могут не иметь гомографии.

    ``suspect_landmark_ids`` перечисляет сохранённые наблюдения, которые не
    участвовали в гомографии этого кадра и потому требуют внимания оператора.
    """

    frame_index: int
    observations: tuple[FrameObservation, ...]
    alignment: ManualAlignmentResult | None
    status: AnnotationStatus
    suspect_landmark_ids: frozenset[int] = frozenset()
    machine_observations: tuple[MachineObservation, ...] = ()
    image_quality: FrameImageQuality | None = None
    rejection_reasons: tuple[str, ...] = ()

    @property
    def is_keyframe(self) -> bool:
        """Подтверждён ли этот кадр оператором."""

        return self.status == "keyframe"

    @property
    def is_rejected(self) -> bool:
        """Был ли кадр явно отклонён автоматической проверкой."""

        return self.status == "rejected"

    @property
    def is_excluded(self) -> bool:
        """Исключил ли оператор кадр из рабочего набора измерений."""

        return self.status == "excluded"

    @property
    def is_stale(self) -> bool:
        """Устарело ли решение после коррекции более ранней опоры."""

        return self.status == "stale"


class AlignmentSession:
    """Хранить маяки и независимые снимки разметки обработанных кадров."""

    def __init__(self) -> None:
        self._landmarks: dict[int, MapLandmark] = {}
        self._frames: dict[int, FrameAnnotation] = {}
        self._next_landmark_id = 0

    @property
    def frame_indices(self) -> tuple[int, ...]:
        """Отсортированные индексы кадров, для которых сохранена разметка."""

        return tuple(sorted(self._frames))

    @property
    def landmark_count(self) -> int:
        """Число когда-либо созданных картографических маяков."""

        return len(self._landmarks)

    @property
    def landmark_ids(self) -> tuple[int, ...]:
        """Стабильные номера всех созданных картографических маяков."""

        return tuple(sorted(self._landmarks))

    @property
    def next_landmark_id(self) -> int:
        """Номер, который получит следующий новый маяк.

        Значение сохраняется в файле проекта отдельно от числа маяков. Это
        предотвращает повторное использование старого номера, если позднее в
        редакторе появится удаление картографических маяков целиком.
        """

        return self._next_landmark_id

    def add_landmark(self, map_point_px: Point) -> int:
        """Создать маяк и вернуть его стабильный целочисленный идентификатор."""

        point = _validated_point(map_point_px, name="Координата маяка на карте")
        landmark_id = self._next_landmark_id
        self._next_landmark_id += 1
        self._landmarks[landmark_id] = MapLandmark(landmark_id, point)
        return landmark_id

    def restore_landmark(self, landmark_id: int, map_point_px: Point) -> None:
        """Восстановить маяк с прежним стабильным номером из файла проекта.

        Обычная разметка должна использовать add_landmark. Этот метод
        предназначен только для проверяемой загрузки: он запрещает
        отрицательные и повторяющиеся номера и сдвигает счётчик вперёд.
        """

        restored_id = int(landmark_id)
        if restored_id < 0:
            raise ValueError("Номер маяка не может быть отрицательным")
        if restored_id in self._landmarks:
            raise ValueError(f"Маяк {restored_id} дважды записан в проекте")
        point = _validated_point(map_point_px, name="Координата маяка на карте")
        self._landmarks[restored_id] = MapLandmark(restored_id, point)
        self._next_landmark_id = max(self._next_landmark_id, restored_id + 1)

    def reserve_next_landmark_id(self, next_landmark_id: int) -> None:
        """Восстановить монотонный счётчик номеров после загрузки проекта."""

        requested = int(next_landmark_id)
        minimum = max(self._landmarks, default=-1) + 1
        if requested < minimum:
            raise ValueError(
                "Следующий номер маяка меньше уже существующего номера"
            )
        self._next_landmark_id = requested

    def landmark(self, landmark_id: int) -> MapLandmark:
        """Вернуть маяк либо явно сообщить о неизвестном идентификаторе."""

        try:
            return self._landmarks[int(landmark_id)]
        except KeyError as error:
            raise KeyError(f"Неизвестный маяк {landmark_id}") from error

    def remember_frame(
        self,
        frame_index: int,
        observations: list[FrameObservation] | tuple[FrameObservation, ...],
        *,
        alignment: ManualAlignmentResult | None,
        status: AnnotationStatus,
        suspect_landmark_ids: set[int] | frozenset[int] = frozenset(),
        machine_observations: tuple[MachineObservation, ...] = (),
        image_quality: FrameImageQuality | None = None,
        rejection_reasons: tuple[str, ...] = (),
    ) -> FrameAnnotation:
        """Запомнить полное состояние кадра, заменив его старую версию.

        Замена нужна после ручного уточнения: предыдущая автоматическая версия
        кадра больше не является источником истины, но записи других кадров не
        затрагиваются.
        """

        if frame_index < 0:
            raise ValueError("Индекс кадра не может быть отрицательным")
        if status not in (
            "draft",
            "tracked",
            "keyframe",
            "rejected",
            "excluded",
            "stale",
        ):
            raise ValueError(f"Неизвестное состояние разметки: {status}")

        validated: list[FrameObservation] = []
        seen_landmarks: set[int] = set()
        for observation in observations:
            landmark_id = int(observation.landmark_id)
            self.landmark(landmark_id)
            if landmark_id in seen_landmarks:
                raise ValueError(
                    f"Маяк {landmark_id} дважды присутствует на кадре {frame_index}"
                )
            seen_landmarks.add(landmark_id)
            validated.append(
                FrameObservation(
                    landmark_id,
                    _validated_point(
                        observation.frame_point_px,
                        name=f"Наблюдение маяка {landmark_id}",
                    ),
                )
            )

        validated_machine: list[MachineObservation] = []
        for observation in machine_observations:
            if observation.source_frame_index < 0:
                raise ValueError("Индекс исходного кадра машинной точки отрицателен")
            error = float(observation.forward_backward_error_px)
            if not np.isfinite(error) or error < 0.0:
                raise ValueError("Ошибка машинной точки должна быть неотрицательной")
            validated_machine.append(
                MachineObservation(
                    source_frame_index=int(observation.source_frame_index),
                    source_frame_point_px=_validated_point(
                        observation.source_frame_point_px,
                        name="Исходная машинная точка",
                    ),
                    frame_point_px=_validated_point(
                        observation.frame_point_px,
                        name="Текущая машинная точка",
                    ),
                    map_point_px=_validated_point(
                        observation.map_point_px,
                        name="Координата машинной точки на карте",
                    ),
                    forward_backward_error_px=error,
                    used_for_alignment=bool(observation.used_for_alignment),
                )
            )

        suspects = frozenset(int(value) for value in suspect_landmark_ids)
        unknown_suspects = suspects.difference(seen_landmarks)
        if unknown_suspects:
            raise ValueError(
                "Сомнительными могут быть только наблюдаемые на кадре маяки"
            )

        if alignment is not None and len(validated) < 4:
            raise ValueError("Гомография не может храниться менее чем с 4 маяками")
        if status in ("tracked", "keyframe") and alignment is None:
            raise ValueError(
                "Подтверждённая или автоматически прослеженная разметка "
                "должна содержать гомографию"
            )
        reasons = tuple(str(reason).strip() for reason in rejection_reasons)
        if any(not reason for reason in reasons):
            raise ValueError("Причина отказа не может быть пустой")
        if status == "rejected":
            if alignment is not None:
                raise ValueError("Отклонённый кадр не должен содержать гомографию")
            if not reasons:
                raise ValueError("Отклонённый кадр должен содержать причину отказа")
        elif status == "excluded":
            if alignment is not None:
                raise ValueError("Исключённый кадр не должен содержать гомографию")
            if not reasons:
                raise ValueError("Исключённый кадр должен содержать причину")
        elif status == "stale":
            if alignment is not None:
                raise ValueError("Устаревший кадр не должен содержать гомографию")
        elif reasons:
            raise ValueError(
                "Причины допустимы только для отклонённого или исключённого кадра"
            )

        annotation = FrameAnnotation(
            frame_index=int(frame_index),
            observations=tuple(validated),
            alignment=alignment,
            status=status,
            suspect_landmark_ids=suspects,
            machine_observations=tuple(validated_machine),
            image_quality=image_quality,
            rejection_reasons=reasons,
        )
        self._frames[frame_index] = annotation
        return annotation

    def frame(self, frame_index: int) -> FrameAnnotation | None:
        """Вернуть сохранённую разметку кадра или ``None``."""

        return self._frames.get(int(frame_index))

    def invalidate_derived_frames_after(self, frame_index: int) -> tuple[int, ...]:
        """Пометить зависимое будущее устаревшим до следующей ручной опоры.

        Подтверждённый оператором keyframe начинает независимый участок и
        останавливает распространение. Исключённые кадры сохраняют операторское
        решение. Наблюдения оставляются для сравнения, но старая гомография и
        машинные опоры удаляются.
        """

        origin = int(frame_index)
        invalidated: list[int] = []
        for candidate_index in self.frame_indices:
            if candidate_index <= origin:
                continue
            annotation = self._frames[candidate_index]
            if annotation.is_keyframe:
                break
            if annotation.is_excluded or annotation.is_stale:
                continue
            self.remember_frame(
                candidate_index,
                annotation.observations,
                alignment=None,
                status="stale",
                suspect_landmark_ids=annotation.suspect_landmark_ids,
                image_quality=annotation.image_quality,
            )
            invalidated.append(candidate_index)
        return tuple(invalidated)

    def clear(self) -> None:
        """Удалить всю операторскую сессию, включая историю и маяки."""

        self._landmarks.clear()
        self._frames.clear()
        self._next_landmark_id = 0


def _validated_point(point: Point, *, name: str) -> Point:
    """Проверить форму и конечность одной пиксельной координаты."""

    array = np.asarray(point, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} должна содержать две конечные координаты")
    return float(array[0]), float(array[1])
