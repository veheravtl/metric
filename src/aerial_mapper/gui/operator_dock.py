"""Компактная плавающая панель ежедневных операторских действий.

Панель не владеет моделью и не дублирует команды: главное окно связывает её
кнопки с теми же QAction и контроллерами, что используются меню. Поэтому
галочка в панели и соответствующий пункт меню всегда отражают одно состояние.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class OperatorDock(QDockWidget):
    """Собрать навигацию, маяки и сохранение в перемещаемой панели."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Операторская панель", parent)
        self.setObjectName("operatorDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        body = QWidget(self)
        root = QVBoxLayout(body)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(self._make_project_group())
        root.addWidget(self._make_navigation_group())
        root.addWidget(self._make_view_group())
        root.addWidget(self._make_tracking_group())
        root.addWidget(self._make_landmark_group())
        root.addStretch(1)
        self.setWidget(body)
        self.setMinimumWidth(300)

    def _make_project_group(self) -> QGroupBox:
        group = QGroupBox("Проект")
        layout = QVBoxLayout(group)
        buttons = QHBoxLayout()
        self.open_project_button = QPushButton("Открыть")
        self.save_project_button = QPushButton("Сохранить")
        self.save_as_project_button = QPushButton("Сохранить как…")
        buttons.addWidget(self.open_project_button)
        buttons.addWidget(self.save_project_button)
        buttons.addWidget(self.save_as_project_button)
        layout.addLayout(buttons)
        self.project_status_label = QLabel("Проект не сохранён")
        self.project_status_label.setWordWrap(True)
        layout.addWidget(self.project_status_label)
        return group

    def _make_navigation_group(self) -> QGroupBox:
        group = QGroupBox("Видео")
        layout = QVBoxLayout(group)
        step = QHBoxLayout()
        self.previous_frame_button = QPushButton("◀")
        self.next_frame_button = QPushButton("▶")
        self.frame_number_spin = QSpinBox()
        self.frame_number_spin.setRange(1, 1)
        self.frame_number_spin.setPrefix("Кадр ")
        self.go_to_frame_button = QPushButton("Перейти")
        step.addWidget(self.previous_frame_button)
        step.addWidget(self.next_frame_button)
        step.addWidget(self.frame_number_spin, 1)
        step.addWidget(self.go_to_frame_button)
        layout.addLayout(step)

        self.frame_status_label = QLabel("Кадр не загружен")
        layout.addWidget(self.frame_status_label)
        resume = QHBoxLayout()
        self.resume_button = QPushButton("К месту остановки")
        self.next_attention_button = QPushButton("Следующий требующий внимания")
        self.next_attention_button.setToolTip(
            "Ищет отсутствующий, черновой или отклонённый кадр"
        )
        resume.addWidget(self.resume_button)
        resume.addWidget(self.next_attention_button)
        layout.addLayout(resume)
        return group

    def _make_view_group(self) -> QGroupBox:
        group = QGroupBox("Отображение")
        layout = QVBoxLayout(group)
        self.overlay_check = QCheckBox("Показать оверлей")
        layout.addWidget(self.overlay_check)
        return group

    def _make_tracking_group(self) -> QGroupBox:
        group = QGroupBox("Сопровождение и коррекция")
        layout = QVBoxLayout(group)
        self.tracking_check = QCheckBox("Сопровождать маяки")
        layout.addWidget(self.tracking_check)
        self.tracking_status_label = QLabel("Сопровождение выключено")
        self.tracking_status_label.setWordWrap(True)
        layout.addWidget(self.tracking_status_label)
        buttons = QHBoxLayout()
        self.correct_frame_button = QPushButton("Исправить этот кадр")
        self.confirm_alignment_button = QPushButton("Подтвердить совмещение")
        buttons.addWidget(self.correct_frame_button)
        buttons.addWidget(self.confirm_alignment_button)
        layout.addLayout(buttons)
        return group

    def _make_landmark_group(self) -> QGroupBox:
        group = QGroupBox("Маяки")
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.landmark_mode_combo = QComboBox()
        self.landmark_mode_combo.addItems(
            ["Перетаскивание", "Установка новых точек"]
        )
        form.addRow("Режим:", self.landmark_mode_combo)
        choose = QHBoxLayout()
        self.landmark_number_spin = QSpinBox()
        self.landmark_number_spin.setRange(1, 1)
        self.focus_landmark_button = QPushButton("Найти и подсветить")
        choose.addWidget(self.landmark_number_spin)
        choose.addWidget(self.focus_landmark_button)
        form.addRow("Маяк №:", choose)
        layout.addLayout(form)

        buttons = QGridLayout()
        self.undo_point_button = QPushButton("Отменить последнюю точку")
        self.remove_selected_button = QPushButton("Удалить выбранный маяк")
        buttons.addWidget(self.undo_point_button, 0, 0)
        buttons.addWidget(self.remove_selected_button, 0, 1)
        layout.addLayout(buttons)
        self.landmark_status_label = QLabel("Маяков на кадре: 0")
        self.landmark_status_label.setWordWrap(True)
        layout.addWidget(self.landmark_status_label)
        return group

    def update_frame(self, frame_index: int | None, frame_count: int) -> None:
        """Отразить внутренний индекс как понятный номер, начинающийся с 1."""

        loaded = frame_index is not None
        maximum = max(1, frame_count)
        self.frame_number_spin.setRange(1, maximum)
        if loaded:
            self.frame_number_spin.setValue(frame_index + 1)
            self.frame_status_label.setText(
                f"Кадр {frame_index + 1} из {frame_count or '?'}"
            )
        else:
            self.frame_status_label.setText("Кадр не загружен")
        for widget in (
            self.previous_frame_button,
            self.next_frame_button,
            self.frame_number_spin,
            self.go_to_frame_button,
            self.resume_button,
            self.next_attention_button,
        ):
            widget.setEnabled(loaded)
        self.previous_frame_button.setEnabled(loaded and frame_index > 0)
        self.next_frame_button.setEnabled(
            loaded and (frame_count <= 0 or frame_index < frame_count - 1)
        )

    def update_alignment(
        self,
        *,
        pair_count: int,
        pending_point: bool,
        tracking_active: bool,
        tracking_available: bool,
        point_editing_active: bool,
        annotation_status: str,
    ) -> None:
        """Показать причину блокировки вместо молчаливо неподвижного маяка."""

        self.tracking_check.setEnabled(tracking_available or tracking_active)
        self.correct_frame_button.setEnabled(pair_count > 0)
        self.confirm_alignment_button.setEnabled(pair_count >= 4 and not pending_point)
        self.undo_point_button.setEnabled(pair_count > 0 or pending_point)
        self.remove_selected_button.setEnabled(pair_count > 0)
        self.focus_landmark_button.setEnabled(pair_count > 0)
        self.landmark_number_spin.setEnabled(pair_count > 0)
        self.landmark_status_label.setText(
            f"Маяков на кадре: {pair_count}"
            + (" | ожидается точка на карте" if pending_point else "")
        )
        self.landmark_mode_combo.setCurrentIndex(1 if point_editing_active else 0)
        if tracking_active:
            state = "Сопровождение активно — маяки защищены от перемещения"
        elif annotation_status in ("draft", "rejected", "stale"):
            state = "Коррекция/черновик — подтвердите совмещение"
        else:
            state = "Сопровождение выключено — маяки можно перетаскивать"
        self.tracking_status_label.setText(state)

    def set_project_status(self, text: str, *, dirty: bool) -> None:
        """Обозначить несохранённые изменения заметным префиксом."""

        self.project_status_label.setText(
            ("● Не сохранено | " if dirty else "✓ ") + text
        )
