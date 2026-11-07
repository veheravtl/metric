"""Главное окно и будущая логика взаимодействия с оператором."""

import os
import time
from collections.abc import Mapping
from pathlib import Path

import cv2
from PySide6.QtCore import QSettings, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QKeySequence,
    QPixmap,
    QResizeEvent,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
)

from aerial_mapper.alignment_report import export_alignment_comparison_package
from aerial_mapper.automatic_map_alignment import (
    AutomaticMapAlignmentDiagnostics,
    AutomaticMapAlignmentFailure,
    AutomaticMapAlignmentResult,
    align_frame_to_map_automatically,
)
from aerial_mapper.camera_distortion import undistort_points
from aerial_mapper.live_debug import LiveDebugSession
from aerial_mapper.mars_lvig_preprocessing import (
    mars_lvig_camera_calibration,
    prepare_mars_lvig_image,
)
from aerial_mapper.operator_project import (
    PROJECT_FILE_SUFFIX,
    load_operator_project,
    save_operator_project,
    source_matches_fingerprint,
)

from .correspondence_editor import CorrespondenceEditor
from .generated.ui_dron import Ui_MainWindow
from .map_image import MapImageController
from .operator_dock import OperatorDock
from .video_playback import VideoPlaybackController

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FLIGHTS_DIRECTORY = PROJECT_ROOT / "data/queries/mars_lvig/extracted"
DEFAULT_MAPS_DIRECTORY = PROJECT_ROOT / "data/reference"
DEFAULT_REPORTS_DIRECTORY = PROJECT_ROOT / "outputs/alignment_reports"
DEFAULT_PROJECTS_DIRECTORY = PROJECT_ROOT / "outputs/operator_projects"
SETTINGS_ORGANIZATION = "Aerial Map Measurement"
SETTINGS_APPLICATION = "desktop"
RECENT_SOURCE_PATH_KEY = "recent/source_path"
RECENT_SOURCE_TYPE_KEY = "recent/source_type"
RECENT_MAP_PATH_KEY = "recent/map_path"
WINDOW_STATE_KEY = "operator/window_state"
DOCK_GEOMETRY_KEY = "operator/dock_geometry"


class MainWindow(QMainWindow):
    """Связать форму с загрузкой и покадровым просмотром видео.

    Внешний вид меняется в ``dron.ui``. Обработчики загрузки видео, карты и
    измерительных инструментов будут добавляться в этот класс, поэтому
    повторная генерация формы не уничтожит написанную вручную логику.
    """

    def __init__(
        self,
        *,
        remember_recent_sources: bool = False,
        settings: QSettings | None = None,
        live_debug: LiveDebugSession | None = None,
        protect_unsaved_progress: bool = False,
    ) -> None:
        """Создать окно и при запуске приложения восстановить прошлые данные.

        ``remember_recent_sources`` намеренно включается точкой входа
        приложения, а не каждым тестом или встраиваемым экземпляром окна.
        ``settings`` позволяет тестам использовать отдельный INI-файл вместо
        пользовательского хранилища Qt.
        """

        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._live_debug = live_debug
        self._live_debug_closed = False
        self._last_live_debug_capture_monotonic = 0.0

        self._remember_recent_sources = remember_recent_sources
        self._settings = (
            settings
            if settings is not None
            else QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)
        )
        self._restoring_recent_sources = False
        self._protect_unsaved_progress = bool(protect_unsaved_progress)
        self._automatic_alignment_busy = False
        self._last_automatic_candidate: AutomaticMapAlignmentResult | None = None
        self._last_automatic_diagnostics: AutomaticMapAlignmentDiagnostics | None = None
        self._last_automatic_status = "not_run"
        self._last_automatic_message = ""
        self._last_automatic_frame_bgr = None
        self._last_automatic_correction_note = ""
        self._last_automatic_alignment: AutomaticMapAlignmentResult | None = None
        self._project_path: Path | None = None
        self._project_dirty = False
        self._loading_project = False
        self._closing_window = False
        self._resume_frame_index: int | None = None
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(1500)
        self._autosave_timer.timeout.connect(self._autosave_project)

        self.map_image = MapImageController(self.ui.mapView, self)
        self.map_image.map_loaded.connect(self._on_map_loaded)
        self.map_image.error_occurred.connect(self._show_map_error)

        self.video = VideoPlaybackController(self.ui.videoView, self)
        self.video.frame_changed.connect(self._on_frame_changed)
        self.video.video_loaded.connect(self._on_video_loaded)
        self.video.playback_changed.connect(self._on_playback_changed)
        self.video.error_occurred.connect(self._show_video_error)

        self.alignment = CorrespondenceEditor(
            self.ui.videoView,
            self.ui.mapView,
            self.video,
            self.map_image,
            self,
        )
        self.alignment.message.connect(self.ui.statusbar.showMessage)
        self.alignment.error_occurred.connect(self._show_alignment_error)
        self.alignment.state_changed.connect(self._on_alignment_state_changed)
        self.alignment.tracking_changed.connect(self._on_tracking_changed)
        self.video.frame_changed.connect(self.alignment.on_frame_changed)
        self.alignment.debug_event.connect(self._on_live_debug_event)
        self.video.video_loaded.connect(self.alignment.on_video_loaded)
        self.map_image.map_loaded.connect(self.alignment.on_map_loaded)
        self._create_operator_workspace()

        self.ui.actionLoad_video.triggered.connect(self._choose_video)
        self.ui.actionLoad_flight.triggered.connect(self._choose_flight)
        self.ui.actionLoad_map.triggered.connect(self._choose_map)
        self.ui.pbPlay.clicked.connect(self.video.play)
        self.ui.pbStop.clicked.connect(self.video.stop)
        self.ui.pbBack.clicked.connect(self.video.step_backward)
        self.ui.pbForward.clicked.connect(self.video.step_forward)
        self.ui.pbExcludeFrame.clicked.connect(self._exclude_current_frame_and_advance)
        self.ui.frameSlider.valueChanged.connect(self._seek_from_slider)

        self.ui.actionSet_points.toggled.connect(self._set_point_editing_active)
        self.ui.actionUndo_point.triggered.connect(self.alignment.undo)
        self.ui.actionRemove_selected.triggered.connect(self.alignment.remove_selected)
        self.ui.actionClear_points.triggered.connect(self.alignment.clear)
        self.ui.actionCompute_alignment.triggered.connect(
            self.alignment.compute_alignment
        )
        self.ui.actionTrack_points.toggled.connect(self._set_beacon_tracking)
        self.ui.actionExclude_frame.triggered.connect(
            self._exclude_current_frame_and_advance
        )
        self.ui.actionExclude_frame_range.triggered.connect(self._exclude_frame_range)
        self.ui.actionShow_overlay.toggled.connect(self.alignment.set_overlay_visible)
        self.ui.actionAuto_align.triggered.connect(self._run_automatic_alignment)
        self.ui.actionExport_alignment_report.triggered.connect(
            self._export_alignment_report
        )

        self._incident_shortcut = QShortcut(QKeySequence("F12"), self)
        self._incident_shortcut.activated.connect(self._mark_live_debug_incident)
        self._cancel_point_shortcut = QShortcut(QKeySequence("Esc"), self)
        self._cancel_point_shortcut.activated.connect(self.alignment.cancel_pending_point)

        # Поток с камеры будет отдельным источником кадров. Пока действие
        # отключено явно, чтобы нажатие не выглядело как незамеченная ошибка.
        self.ui.actionStream_from_Camera.setEnabled(False)
        self.ui.actionStream_from_Camera.setToolTip(
            "Поток с камеры будет добавлен после файлового плеера"
        )

        self._set_video_controls_enabled(False)
        self._on_alignment_state_changed(0, False, False)
        self.ui.statusbar.showMessage(
            "Выберите Video → Load video или Load flight folder | F12 — инцидент"
        )
        if self._remember_recent_sources:
            self._restore_recent_sources()

        if self._live_debug is not None:
            self._live_debug.record_event(
                "window_created",
                {"f12_marks_incident": True},
            )
            self._write_live_debug_state()

    def _create_operator_workspace(self) -> None:
        """Создать одну панель над существующими действиями меню."""

        self.action_open_project = QAction("Открыть проект…", self)
        self.action_open_project.setShortcut(QKeySequence.StandardKey.Open)
        self.action_save_project = QAction("Сохранить проект", self)
        self.action_save_project.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save_project_as = QAction("Сохранить проект как…", self)
        self.action_save_project_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        project_menu = self.ui.menubar.addMenu("Проект")
        project_menu.addAction(self.action_open_project)
        project_menu.addAction(self.action_save_project)
        project_menu.addAction(self.action_save_project_as)

        self.operator_dock = OperatorDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.operator_dock)
        view_menu = self.ui.menubar.addMenu("Вид")
        self.action_toggle_operator_dock = self.operator_dock.toggleViewAction()
        self.action_toggle_operator_dock.setText("Операторская панель")
        self.action_toggle_operator_dock.setShortcut(QKeySequence("Ctrl+Shift+O"))
        view_menu.addAction(self.action_toggle_operator_dock)
        if (
            self._remember_recent_sources
            and self._settings.contains(WINDOW_STATE_KEY)
        ):
            self.restoreState(self._settings.value(WINDOW_STATE_KEY))
        else:
            self.operator_dock.setFloating(True)
            self.operator_dock.resize(360, 720)
        if (
            self._remember_recent_sources
            and self._settings.contains(DOCK_GEOMETRY_KEY)
        ):
            self.operator_dock.restoreGeometry(
                self._settings.value(DOCK_GEOMETRY_KEY)
            )
        # restoreState() честно восстанавливает и скрытое состояние панели.
        # Для основного рабочего инструмента это опасно: оператор остаётся без
        # управления и без очевидного способа понять, куда исчезла панель.
        # Отложенная проверка выполняется уже после показа главного окна.
        QTimer.singleShot(0, self._ensure_operator_dock_visible)

        panel = self.operator_dock
        self.action_open_project.triggered.connect(self._choose_operator_project)
        self.action_save_project.triggered.connect(self._save_operator_project)
        self.action_save_project_as.triggered.connect(
            lambda: self._save_operator_project(save_as=True)
        )
        panel.open_project_button.clicked.connect(self.action_open_project.trigger)
        panel.save_project_button.clicked.connect(self.action_save_project.trigger)
        panel.save_as_project_button.clicked.connect(
            self.action_save_project_as.trigger
        )

        panel.previous_frame_button.clicked.connect(self.video.step_backward)
        panel.next_frame_button.clicked.connect(self.video.step_forward)
        panel.go_to_frame_button.clicked.connect(self._seek_from_operator_panel)
        panel.frame_number_spin.editingFinished.connect(
            self._seek_from_operator_panel
        )
        panel.resume_button.clicked.connect(self._resume_saved_position)
        panel.next_attention_button.clicked.connect(
            self._seek_next_frame_requiring_attention
        )

        panel.overlay_check.setChecked(self.ui.actionShow_overlay.isChecked())
        panel.overlay_check.toggled.connect(
            self.ui.actionShow_overlay.setChecked
        )
        self.ui.actionShow_overlay.toggled.connect(panel.overlay_check.setChecked)
        self.ui.actionShow_overlay.toggled.connect(self._mark_project_dirty)
        panel.tracking_check.toggled.connect(
            self.ui.actionTrack_points.setChecked
        )
        self.ui.actionTrack_points.toggled.connect(panel.tracking_check.setChecked)
        panel.landmark_mode_combo.currentIndexChanged.connect(
            lambda index: self.ui.actionSet_points.setChecked(index == 1)
        )
        self.ui.actionSet_points.toggled.connect(
            lambda active: panel.landmark_mode_combo.setCurrentIndex(
                1 if active else 0
            )
        )
        panel.undo_point_button.clicked.connect(self.ui.actionUndo_point.trigger)
        panel.remove_selected_button.clicked.connect(
            self.ui.actionRemove_selected.trigger
        )
        panel.confirm_alignment_button.clicked.connect(
            self.ui.actionCompute_alignment.trigger
        )
        panel.correct_frame_button.clicked.connect(
            self._begin_current_frame_correction
        )
        panel.focus_landmark_button.clicked.connect(self._focus_selected_landmark)
        self._update_project_status()

    def _ensure_operator_dock_visible(self) -> None:
        """Показать панель и вернуть её с отключённого монитора при необходимости.

        Qt сохраняет абсолютные координаты плавающего ``QDockWidget``. После
        смены монитора сохранённый прямоугольник может целиком оказаться вне
        доступной области экранов. В таком случае панель безопаснее пристыковать
        справа: пользователь сразу её увидит и при желании снова открепит.
        """

        self.operator_dock.show()
        if self.operator_dock.isFloating():
            dock_geometry = self.operator_dock.frameGeometry()
            is_on_screen = any(
                dock_geometry.intersects(screen.availableGeometry())
                for screen in QApplication.screens()
            )
            if not is_on_screen:
                self.operator_dock.setFloating(False)
                self.addDockWidget(
                    Qt.DockWidgetArea.RightDockWidgetArea,
                    self.operator_dock,
                )
                self.operator_dock.show()
        # Плавающее окно могло открыться раньше главного и остаться под ним.
        self.operator_dock.raise_()

    def _seek_from_operator_panel(self) -> None:
        """Перейти к введённому оператором номеру кадра, начинающемуся с 1."""

        self._seek_from_slider(self.operator_dock.frame_number_spin.value() - 1)

    def _resume_saved_position(self) -> None:
        """Вернуться к кадру, который был активен при последнем сохранении."""

        if self._resume_frame_index is None:
            self.ui.statusbar.showMessage("У проекта ещё нет сохранённой позиции")
            return
        self._seek_from_slider(self._resume_frame_index)

    def _seek_next_frame_requiring_attention(self) -> None:
        """Открыть первый неготовый кадр от текущей позиции включительно."""

        current = self.video.current_frame_index
        if current is None:
            return
        target = self.alignment.first_frame_requiring_attention(
            start_frame_index=current,
            frame_count=self.video.frame_count,
        )
        if target is None:
            self.ui.statusbar.showMessage(
                "После текущей позиции нет кадров, требующих внимания"
            )
            return
        self._seek_from_slider(target)

    def _begin_current_frame_correction(self) -> None:
        """Сделать явным переход из сопровождения в ручную коррекцию."""

        if self.ui.actionSet_points.isChecked():
            self.ui.actionSet_points.setChecked(False)
        if self.ui.actionTrack_points.isChecked():
            self.ui.actionTrack_points.setChecked(False)
        self.alignment.begin_current_frame_correction()

    def _focus_selected_landmark(self) -> None:
        """Подсветить введённый видимый номер маяка в обеих сценах."""

        self.alignment.focus_landmark(
            self.operator_dock.landmark_number_spin.value() - 1
        )

    def _choose_operator_project(self, _checked: bool = False) -> None:
        """Выбрать сохранённый JSON-проект и восстановить рабочее место."""

        if not self._confirm_unsaved_progress():
            return
        start = (
            self._project_path.parent
            if self._project_path is not None
            else DEFAULT_PROJECTS_DIRECTORY
        )
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть операторский проект",
            str(start),
            f"Проект Aerial Mapper (*{PROJECT_FILE_SUFFIX});;JSON (*.json)",
        )
        if filename:
            self._load_operator_project(Path(filename))

    def _load_operator_project(self, path: Path) -> bool:
        """Проверить источники, загрузить их и только затем принять разметку."""

        try:
            project = load_operator_project(path)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Проект не открыт", str(error))
            return False
        for label, source_path, fingerprint in (
            ("видео/пролёт", project.video_source_path, project.video_fingerprint),
            ("карта", project.map_source_path, project.map_fingerprint),
        ):
            matches, reason = source_matches_fingerprint(source_path, fingerprint)
            if not matches:
                QMessageBox.critical(
                    self,
                    "Исходные материалы проекта не совпадают",
                    f"{label}: {source_path}\n{reason}",
                )
                return False

        self._loading_project = True
        try:
            source_loaded = (
                self.video.load_flight(project.video_source_path)
                if project.video_source_type == "flight"
                else self.video.load_video(project.video_source_path)
            )
            if not source_loaded or not self.map_image.load_image(
                project.map_source_path
            ):
                return False
            if project.current_frame_index >= self.video.frame_count:
                QMessageBox.critical(
                    self,
                    "Проект не открыт",
                    "Сохранённый номер кадра выходит за пределы источника",
                )
                return False
            if not self.video.seek(project.current_frame_index):
                return False
            self.alignment.restore_session(
                project.session,
                current_frame_index=project.current_frame_index,
            )
            self.ui.actionShow_overlay.setChecked(project.overlay_visible)
            self.alignment.set_overlay_visible(project.overlay_visible)
            self._project_path = path.expanduser().resolve()
            self._resume_frame_index = project.current_frame_index
            self._project_dirty = False
            self._autosave_timer.stop()
            self.setWindowTitle(
                f"Aerial Map Measurement — {self._project_path.name}"
            )
            self.ui.statusbar.showMessage(
                f"Проект открыт: {self._project_path} | "
                f"продолжение с кадра {project.current_frame_index + 1}"
            )
            return True
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.critical(self, "Проект не открыт", str(error))
            return False
        finally:
            self._loading_project = False
            self._update_project_status()

    def _save_operator_project(
        self,
        _checked: bool = False,
        *,
        save_as: bool = False,
    ) -> bool:
        """Сохранить проект, запросив имя только при первой записи."""

        if self.alignment.has_pending_point:
            QMessageBox.warning(
                self,
                "Незавершённая точка",
                "Сначала поставьте парную точку на карте либо нажмите "
                "«Отменить последнюю точку»/Esc.",
            )
            return False
        if (
            self.video.source_path is None
            or self.map_image.source_path is None
            or self.video.current_frame_index is None
        ):
            QMessageBox.warning(
                self,
                "Проект не сохранён",
                "Сначала загрузите источник кадров и карту.",
            )
            return False

        output = self._project_path
        if save_as or output is None:
            DEFAULT_PROJECTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить операторский проект",
                str(DEFAULT_PROJECTS_DIRECTORY / f"project{PROJECT_FILE_SUFFIX}"),
                f"Проект Aerial Mapper (*{PROJECT_FILE_SUFFIX});;JSON (*.json)",
            )
            if not filename:
                return False
            output = Path(filename)
            if not str(output).endswith(PROJECT_FILE_SUFFIX):
                output = Path(str(output) + PROJECT_FILE_SUFFIX)
        return self._write_operator_project(output, automatic=False)

    def _write_operator_project(self, path: Path, *, automatic: bool) -> bool:
        """Записать модель и обновить индикатор только после успешной замены."""

        assert self.video.source_path is not None
        assert self.map_image.source_path is not None
        assert self.video.current_frame_index is not None
        try:
            self.alignment.commit_visible_state()
            saved = save_operator_project(
                path,
                video_source_path=self.video.source_path,
                video_source_type="flight" if self.video.is_flight else "video",
                map_source_path=self.map_image.source_path,
                current_frame_index=self.video.current_frame_index,
                overlay_visible=self.ui.actionShow_overlay.isChecked(),
                session=self.alignment.session,
            )
        except (OSError, ValueError) as error:
            if automatic:
                self.ui.statusbar.showMessage(f"Ошибка автосохранения: {error}")
            else:
                QMessageBox.critical(self, "Проект не сохранён", str(error))
            self._project_dirty = True
            self._update_project_status()
            return False

        self._project_path = saved
        self._resume_frame_index = self.video.current_frame_index
        self._project_dirty = False
        self._autosave_timer.stop()
        operation = "Автосохранение" if automatic else "Проект сохранён"
        self.ui.statusbar.showMessage(f"{operation}: {saved}")
        self._update_project_status()
        return True

    def _autosave_project(self) -> None:
        """После паузы сохранить именованный проект, не открывая диалог."""

        if (
            not self._project_dirty
            or self._project_path is None
            or self.alignment.has_pending_point
            or self._loading_project
            or self._closing_window
        ):
            return
        self._write_operator_project(self._project_path, automatic=True)

    def _mark_project_dirty(self) -> None:
        """Отметить изменение данных или позиции и отложить запись на диск."""

        if (
            self._loading_project
            or self._closing_window
            or (self._project_path is None and not self.alignment.has_operator_progress)
        ):
            return
        self._project_dirty = True
        self._update_project_status()
        if self._project_path is not None and not self.alignment.has_pending_point:
            self._autosave_timer.start()

    def _update_project_status(self) -> None:
        """Показать оператору имя проекта и наличие несохранённых изменений."""

        if not hasattr(self, "operator_dock"):
            return
        if self._project_path is None:
            text = "Проект ещё не имеет файла"
        else:
            text = self._project_path.name
        self.operator_dock.set_project_status(text, dirty=self._project_dirty)

    def _confirm_unsaved_progress(self) -> bool:
        """Предложить сохранить, отбросить или продолжить работу."""

        if not self._project_dirty:
            return True
        answer = QMessageBox.warning(
            self,
            "Есть несохранённый прогресс",
            "Изменения разметки и текущая позиция не сохранены.",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self._save_operator_project()
        return True

    def _reset_project_identity(self) -> None:
        """Начать новый безымянный проект после явной замены материала."""

        self._autosave_timer.stop()
        self._project_path = None
        self._resume_frame_index = None
        self._project_dirty = self.alignment.has_operator_progress
        self._update_project_status()

    def _choose_video(self) -> None:
        """Открыть системный диалог и передать выбранный файл декодеру."""

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео пролёта",
            str(Path.home()),
            "Видео (*.mp4 *.avi *.mov *.mkv *.m4v);;Все файлы (*)",
        )
        if (
            filename
            and self._confirm_unsaved_progress()
            and self.video.load_video(filename)
        ):
            self._reset_project_identity()

    def _choose_flight(self) -> None:
        """Выбрать каталог извлечённого пролёта с кадрами и телеметрией."""

        start_directory = (
            DEFAULT_FLIGHTS_DIRECTORY
            if DEFAULT_FLIGHTS_DIRECTORY.is_dir()
            else Path.home()
        )
        directory = QFileDialog.getExistingDirectory(
            self,
            "Выберите каталог пролёта (например, HKairport01)",
            str(start_directory),
        )
        if (
            directory
            and self._confirm_unsaved_progress()
            and self.video.load_flight(directory)
        ):
            self._reset_project_identity()

    def _choose_map(self) -> None:
        """Выбрать локальный снимок карты для правой области."""

        start_directory = (
            DEFAULT_MAPS_DIRECTORY if DEFAULT_MAPS_DIRECTORY.is_dir() else Path.home()
        )
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение карты",
            str(start_directory),
            "Изображения (*.png *.jpg *.jpeg *.tif *.tiff);;Все файлы (*)",
        )
        if (
            filename
            and self._confirm_unsaved_progress()
            and self.map_image.load_image(filename)
        ):
            self._reset_project_identity()

    def _on_video_loaded(self, path: str, frame_count: int, fps: float) -> None:
        self.ui.pbPlay.setEnabled(True)
        self.ui.pbStop.setEnabled(True)
        self._configure_frame_slider(frame_count)
        self.setWindowTitle(f"Aerial Map Measurement — {Path(path).name}")
        self._on_frame_changed(self.video.current_frame_index or 0, frame_count)
        frame_status = self.ui.statusbar.currentMessage()
        self.ui.statusbar.showMessage(
            f"{Path(path).name} | {frame_status} | {fps:.3f} FPS"
        )
        self._update_alignment_source_state()
        self._remember_video_source(path)
        self._record_live_debug_event(
            "video_loaded",
            {
                "path": path,
                "frame_count": frame_count,
                "fps": fps,
                "is_flight": self.video.is_flight,
            },
        )
        self._write_live_debug_state()

    def _on_frame_changed(self, frame_index: int, frame_count: int) -> None:
        visible_number = frame_index + 1
        total = str(frame_count) if frame_count > 0 else "?"
        self._reset_automatic_attempt()
        status_parts = [f"Кадр {visible_number}/{total}", f"индекс {frame_index}"]

        telemetry = self.video.current_frame_telemetry
        if telemetry is not None:
            status_parts.extend(
                [
                    f"Image Id {telemetry.image_id}",
                    f"t +{telemetry.elapsed_seconds:.3f} с",
                ]
            )
            if telemetry.has_position:
                status_parts.append(
                    f"GPS {telemetry.latitude_degrees:.7f}, "
                    f"{telemetry.longitude_degrees:.7f}, "
                    f"h_ell={telemetry.altitude_ellipsoid_meters:.2f} м"
                )

        self.ui.statusbar.showMessage(" | ".join(status_parts))

        self._synchronise_frame_slider(frame_index, frame_count)
        self._update_frame_navigation(frame_index, frame_count)
        self.operator_dock.update_frame(frame_index, frame_count)
        if self._project_path is not None or self.alignment.has_operator_progress:
            self._mark_project_dirty()
        self._record_live_debug_event(
            "frame_changed",
            {
                "frame_index": frame_index,
                "frame_count": frame_count,
                "is_playing": self.video.is_playing,
            },
        )

    def _on_playback_changed(self, is_playing: bool) -> None:
        state = "Воспроизведение" if is_playing else "Остановлено"
        self.ui.pbPlay.setEnabled(self.video.is_loaded and not is_playing)
        self.ui.statusbar.showMessage(state)

        self._record_live_debug_event(
            "playback_changed",
            {"is_playing": is_playing, "frame_index": self.video.current_frame_index},
        )
        self._write_live_debug_state()

    def _show_video_error(self, message: str) -> None:
        self._record_live_debug_event("video_error", {"message": message})
        self._set_video_controls_enabled(self.video.is_loaded)
        if self._restoring_recent_sources:
            self._forget_recent_video_source()
            self.ui.statusbar.showMessage(
                f"Последний источник кадров не восстановлен: {message}"
            )
            return
        QMessageBox.critical(self, "Ошибка источника кадров", message)
        self.ui.statusbar.showMessage(message)

    def _on_map_loaded(self, path: str, width: int, height: int) -> None:
        """Подтвердить загрузку и показать исходный размер карты."""

        self._reset_automatic_attempt()
        self.ui.statusbar.showMessage(
            f"Карта: {Path(path).name} | {width}×{height} пикселей"
        )
        self._update_alignment_source_state()
        self._remember_map_source(path)
        self._record_live_debug_event(
            "map_loaded",
            {
                "path": path,
                "width_px": width,
                "height_px": height,
            },
        )
        self._write_live_debug_state()

    def _show_map_error(self, message: str) -> None:
        self._record_live_debug_event("map_error", {"message": message})
        if self._restoring_recent_sources:
            self._settings.remove(RECENT_MAP_PATH_KEY)
            self._settings.sync()
            self.ui.statusbar.showMessage(
                f"Последняя карта не восстановлена: {message}"
            )
            return
        QMessageBox.critical(self, "Ошибка карты", message)
        self.ui.statusbar.showMessage(message)

    def _restore_recent_sources(self) -> None:
        """Загрузить последний пролёт/видеофайл и карту из настроек Qt.

        Хранятся только локальные пути. Сами видео и снимки не копируются в
        конфигурацию. Если файл был перемещён или удалён, устаревшая запись
        удаляется без модального окна, чтобы запуск приложения не блокировался.
        """

        self._restoring_recent_sources = True
        try:
            source_value = self._settings.value(RECENT_SOURCE_PATH_KEY, "")
            source_type = str(self._settings.value(RECENT_SOURCE_TYPE_KEY, "") or "")
            if source_value:
                source_path = Path(str(source_value)).expanduser()
                if not source_path.exists():
                    self._forget_recent_video_source()
                    self.ui.statusbar.showMessage(
                        f"Последний источник кадров не найден: {source_path}"
                    )
                elif source_type == "flight":
                    self.video.load_flight(source_path)
                elif source_type == "video":
                    self.video.load_video(source_path)
                else:
                    self._forget_recent_video_source()

            map_value = self._settings.value(RECENT_MAP_PATH_KEY, "")
            if map_value:
                map_path = Path(str(map_value)).expanduser()
                if map_path.is_file():
                    self.map_image.load_image(map_path)
                else:
                    self._settings.remove(RECENT_MAP_PATH_KEY)
                    self._settings.sync()
                    self.ui.statusbar.showMessage(
                        f"Последняя карта не найдена: {map_path}"
                    )
        finally:
            self._restoring_recent_sources = False

    def _remember_video_source(self, path: str) -> None:
        """Сохранить путь и вид успешно открытого источника кадров."""

        if not self._remember_recent_sources:
            return
        self._settings.setValue(RECENT_SOURCE_PATH_KEY, path)
        self._settings.setValue(
            RECENT_SOURCE_TYPE_KEY,
            "flight" if self.video.is_flight else "video",
        )
        self._settings.sync()

    def _remember_map_source(self, path: str) -> None:
        """Сохранить путь к успешно открытому локальному снимку карты."""

        if not self._remember_recent_sources:
            return
        self._settings.setValue(RECENT_MAP_PATH_KEY, path)
        self._settings.sync()

    def _forget_recent_video_source(self) -> None:
        """Удалить обе связанные настройки несуществующего источника."""

        self._settings.remove(RECENT_SOURCE_PATH_KEY)
        self._settings.remove(RECENT_SOURCE_TYPE_KEY)
        self._settings.sync()

    def _run_automatic_alignment(self) -> None:
        """Найти текущий кадр на карте экспериментальным SIFT + MAGSAC.

        Первый вариант выполняется синхронно: окно не принимает действия во
        время нескольких секунд расчёта. Это сохраняет простой и проверяемый
        поток данных. Фоновое выполнение и кэш признаков карты будут отдельной
        оптимизацией после проверки качества на реальных кадрах.
        """

        if self._automatic_alignment_busy:
            return
        frame_bgr = self.video.current_frame_bgr
        map_bgr = self.map_image.current_image_bgr
        if frame_bgr is None or map_bgr is None:
            QMessageBox.warning(
                self,
                "Автосовмещение недоступно",
                "Сначала загрузите кадр и карту",
            )
            return

        self._reset_automatic_attempt()
        self.video.pause()
        self._automatic_alignment_busy = True
        self._update_alignment_source_state()
        self.ui.statusbar.showMessage(
            "Экспериментальное автосовмещение: поиск SIFT и проверка MAGSAC…"
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        correction_note = "без известной коррекции объектива"
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if self.video.is_flight:
                intrinsics, distortion = mars_lvig_camera_calibration()
                frame_rgb = prepare_mars_lvig_image(
                    frame_rgb,
                    intrinsics=intrinsics,
                    distortion=distortion,
                    maximum_side_pixels=max(frame_rgb.shape[:2]),
                )
                correction_note = "дисторсия MARS-LVIG исправлена"
            self._last_automatic_frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self._last_automatic_correction_note = correction_note
            map_rgb = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB)
            result = align_frame_to_map_automatically(map_rgb, frame_rgb)
        except AutomaticMapAlignmentFailure as error:
            self._last_automatic_alignment = None
            self._last_automatic_status = "rejected"
            self._last_automatic_message = str(error)
            self._last_automatic_diagnostics = error.diagnostics
            self._last_automatic_candidate = (
                error.diagnostics.best_candidate
                if error.diagnostics is not None
                else None
            )
            QMessageBox.information(self, "Автосовмещение отклонено", str(error))
            self.ui.statusbar.showMessage(str(error))
            return
        except (ValueError, cv2.error) as error:
            self._last_automatic_alignment = None
            self._last_automatic_status = "error"
            self._last_automatic_message = str(error)
            QMessageBox.critical(self, "Ошибка автосовмещения", str(error))
            self.ui.statusbar.showMessage(str(error))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._automatic_alignment_busy = False
            self._update_alignment_source_state()

        aligned_frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if not self.alignment.show_automatic_alignment_overlay(
            aligned_frame_bgr,
            result.homography_frame_to_reference,
        ):
            return
        self._last_automatic_alignment = result
        self._last_automatic_candidate = result
        self._last_automatic_status = "accepted"
        summary = self._automatic_alignment_summary(result, correction_note)
        self._last_automatic_message = summary
        self.ui.statusbar.showMessage(summary)

    def _export_alignment_report(self) -> None:
        """Запустить недостающую автоматику и сохранить один передаваемый ZIP."""

        manual_result = self.alignment.result
        raw_frame_bgr = self.video.current_frame_bgr
        map_bgr = self.map_image.current_image_bgr
        frame_index = self.video.current_frame_index
        if manual_result is None or raw_frame_bgr is None or map_bgr is None:
            QMessageBox.warning(
                self,
                "Отчёт недоступен",
                "Сначала разметьте минимум четыре пары и нажмите Compute alignment",
            )
            return
        if frame_index is None:
            QMessageBox.warning(self, "Отчёт недоступен", "Текущий кадр не определён")
            return

        # Один операторский шаг должен дать полный пакет. Если автоматика на
        # этом кадре ещё не запускалась, действие экспорта запускает её само.
        if self._last_automatic_status == "not_run":
            self._run_automatic_alignment()
        automatic_frame_bgr = self._last_automatic_frame_bgr
        if automatic_frame_bgr is None:
            automatic_frame_bgr = raw_frame_bgr.copy()

        try:
            manual_points_for_automatic = manual_result.frame_points_px.copy()
            if self.video.is_flight:
                intrinsics, distortion = mars_lvig_camera_calibration()
                manual_points_for_automatic = undistort_points(
                    manual_points_for_automatic,
                    intrinsics=intrinsics,
                    distortion=distortion,
                )
            DEFAULT_REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Ошибка подготовки отчёта", str(error))
            self.ui.statusbar.showMessage(str(error))
            return

        default_name = DEFAULT_REPORTS_DIRECTORY / (
            f"alignment-frame-{frame_index:06d}.zip"
        )
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить пакет сравнения",
            str(default_name),
            "ZIP archive (*.zip)",
        )
        if not filename:
            return

        try:
            output_path = export_alignment_comparison_package(
                filename,
                raw_frame_bgr=raw_frame_bgr,
                automatic_frame_bgr=automatic_frame_bgr,
                reference_map_bgr=map_bgr,
                manual_result=manual_result,
                manual_points_in_automatic_frame_px=manual_points_for_automatic,
                automatic_status=self._last_automatic_status,
                automatic_candidate=self._last_automatic_candidate,
                automatic_diagnostics=self._last_automatic_diagnostics,
                automatic_message=self._last_automatic_message,
                correction_note=self._last_automatic_correction_note,
                video_source_name=(
                    self.video.source_path.name if self.video.source_path else "unknown"
                ),
                map_source_name=(
                    self.map_image.source_path.name
                    if self.map_image.source_path
                    else "unknown"
                ),
                frame_index=frame_index,
            )
        except (OSError, ValueError, RuntimeError, cv2.error) as error:
            QMessageBox.critical(self, "Ошибка экспорта", str(error))
            self.ui.statusbar.showMessage(str(error))
            return

        self.ui.statusbar.showMessage(f"Диагностический пакет: {output_path}")
        QMessageBox.information(
            self,
            "Отчёт сохранён",
            f"Пришлите этот файл целиком:\n{output_path}",
        )

    @staticmethod
    def _automatic_alignment_summary(
        result: AutomaticMapAlignmentResult, correction_note: str
    ) -> str:
        """Кратко показать оператору основания принятого решения."""

        return (
            "АВТО-ЭКСПЕРИМЕНТ: пары "
            f"{result.unfiltered_mutual_match_count}→{result.mutual_match_count}"
            f"→{result.inlier_count} inlier | "
            f"покрытие {result.frame_coverage_fraction:.1%} | "
            f"p95 {result.inlier_reprojection_p95_reference_px:.2f} px | "
            f"проекция {result.projected_area_fraction:.1%} карты | "
            f"{result.processing_time_seconds:.2f} с | {correction_note}"
        )

    def _set_point_editing_active(self, active: bool) -> None:
        """Не позволять одновременно редактировать и сопровождать маяки."""

        if active and self.ui.actionTrack_points.isChecked():
            self.ui.actionTrack_points.setChecked(False)
        self.alignment.set_active(active)

    def _set_beacon_tracking(self, active: bool) -> None:
        """Передать операторский переключатель контроллеру сопровождения."""

        if active and self.ui.actionSet_points.isChecked():
            self.ui.actionSet_points.setChecked(False)
        if not self.alignment.set_tracking_active(active) and active:
            previous = self.ui.actionTrack_points.blockSignals(True)
            self.ui.actionTrack_points.setChecked(False)
            self.ui.actionTrack_points.blockSignals(previous)

    def _on_tracking_changed(self, active: bool) -> None:
        """Синхронизировать меню, когда алгоритм сам остановил сопровождение."""

        previous = self.ui.actionTrack_points.blockSignals(True)
        self.ui.actionTrack_points.setChecked(active)
        self.ui.actionTrack_points.blockSignals(previous)
        self.operator_dock.tracking_check.setChecked(active)

    def _exclude_current_frame_and_advance(self) -> None:
        """Исключить текущий кадр и открыть следующий одним действием."""

        frame_index = self.video.current_frame_index
        if frame_index is None:
            return
        if self.ui.actionSet_points.isChecked():
            self.ui.actionSet_points.setChecked(False)
        if not self.alignment.exclude_current_frame(
            "оператор исключил кадр как непригодный для совмещения"
        ):
            return

        advanced = self.video.step_forward()
        if advanced:
            destination = self.video.current_frame_index
            self.ui.statusbar.showMessage(
                f"Кадр {frame_index} исключён; открыт индекс {destination}. "
                "После мусорного участка подтвердите новую ручную опору"
            )
        else:
            self.ui.statusbar.showMessage(
                f"Последний кадр {frame_index} исключён из измерений"
            )

    def _exclude_frame_range(self) -> None:
        """Спросить последний индекс и исключить диапазон без покадровых кликов."""

        current_index = self.video.current_frame_index
        if current_index is None:
            return
        frame_count = self.video.frame_count
        maximum_index = max(current_index, frame_count - 1)
        last_index, accepted = QInputDialog.getInt(
            self,
            "Исключить диапазон кадров",
            f"Текущий индекс: {current_index}. Последний исключаемый индекс:",
            current_index,
            current_index,
            maximum_index,
            1,
        )
        if not accepted:
            return
        if self.ui.actionSet_points.isChecked():
            self.ui.actionSet_points.setChecked(False)

        excluded_count = self.alignment.exclude_frame_range(
            current_index,
            last_index,
            reason="оператор исключил диапазон как непригодный для совмещения",
        )
        next_index = last_index + 1
        if frame_count > 0 and next_index < frame_count:
            self.video.seek(next_index)
            destination_note = f"открыт индекс {next_index}"
        else:
            if last_index != current_index:
                self.video.seek(last_index)
            destination_note = "достигнут конец источника"
        self.ui.statusbar.showMessage(
            f"Исключено кадров: {excluded_count} ({current_index}…{last_index}); "
            f"{destination_note}. На следующем пригодном кадре создайте новую опору"
        )

    def _show_alignment_error(self, message: str) -> None:
        """Явно показать, почему ручное совмещение не было рассчитано."""

        QMessageBox.warning(self, "Ручное совмещение не рассчитано", message)
        self.ui.statusbar.showMessage(message)

        self._record_live_debug_event("alignment_error", {"message": message})

    def _on_alignment_state_changed(
        self,
        pair_count: int,
        has_pending_point: bool,
        overlay_available: bool,
    ) -> None:
        """Синхронизировать доступность действий с состоянием разметки."""

        has_points = pair_count > 0 or has_pending_point
        self.ui.actionUndo_point.setEnabled(has_points)
        self.ui.actionRemove_selected.setEnabled(pair_count > 0)
        self.ui.actionClear_points.setEnabled(
            has_points
            or overlay_available
            or bool(self.alignment.remembered_frame_indices)
        )
        self.ui.actionCompute_alignment.setEnabled(
            pair_count >= 4 and not has_pending_point
        )
        self.ui.actionTrack_points.setEnabled(self.alignment.tracking_available)
        self.ui.actionShow_overlay.setEnabled(overlay_available)
        landmark_ids = self.alignment.current_landmark_ids
        self.operator_dock.landmark_number_spin.setRange(
            1,
            max((landmark_id + 1 for landmark_id in landmark_ids), default=1),
        )
        self.operator_dock.update_alignment(
            pair_count=pair_count,
            pending_point=has_pending_point,
            tracking_active=self.alignment.tracking_active,
            tracking_available=self.alignment.tracking_available,
            point_editing_active=self.alignment.point_editing_active,
            annotation_status=self.alignment.current_annotation_status,
        )
        if overlay_available and not self.ui.actionShow_overlay.isChecked():
            self.ui.actionShow_overlay.setChecked(True)
        if self.alignment.has_operator_progress or has_pending_point:
            self._mark_project_dirty()
        self._update_alignment_source_state()

    def _update_alignment_source_state(self) -> None:
        """Разрешить разметку только при загруженных кадре и карте."""
        self._write_live_debug_state()
        if (
            self.video.current_frame_bgr is not None
            or self.map_image.current_image_bgr is not None
        ):
            self._capture_live_debug_artifacts(
                "state-update",
                create_incident=False,
                force=False,
            )

        has_frame = self.video.current_frame_bgr is not None
        self.ui.actionExclude_frame.setEnabled(has_frame)
        self.ui.actionExclude_frame_range.setEnabled(has_frame)
        self.ui.pbExcludeFrame.setEnabled(has_frame)
        ready = self.video.is_loaded and self.map_image.is_loaded
        self.ui.actionSet_points.setEnabled(ready)
        self.ui.actionAuto_align.setEnabled(
            ready and not self._automatic_alignment_busy
        )
        self.ui.actionExport_alignment_report.setEnabled(
            ready
            and self.alignment.result is not None
            and not self._automatic_alignment_busy
        )
        if not ready and self.ui.actionSet_points.isChecked():
            self.ui.actionSet_points.setChecked(False)
        if not ready and self.ui.actionTrack_points.isChecked():
            self.ui.actionTrack_points.setChecked(False)

    def _on_live_debug_event(
        self,
        event_type: str,
        payload: object,
        capture_incident: bool,
    ) -> None:
        """Принять структурированное событие редактора без связи с Qt-типами."""

        event_payload = (
            dict(payload) if isinstance(payload, Mapping) else {"value": payload}
        )
        if not self._record_live_debug_event(event_type, event_payload):
            return
        self._write_live_debug_state()
        if capture_incident:
            self._capture_live_debug_artifacts(
                event_type,
                payload=event_payload,
                create_incident=True,
                force=True,
            )

    def _record_live_debug_event(
        self,
        event_type: str,
        payload: Mapping[str, object] | None = None,
    ) -> bool:
        """Записать событие, не позволяя отказу диагностики сломать GUI."""

        if (
            self._live_debug is None
            or self._live_debug_closed
            or self._live_debug.closed
        ):
            return False
        try:
            self._live_debug.record_event(event_type, payload)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self.ui.statusbar.showMessage(f"Живая диагностика недоступна: {error}")
            return False
        return True

    def _write_live_debug_state(self) -> None:
        """Обновить атомарный JSON текущего состояния приложения."""

        if (
            self._live_debug is None
            or self._live_debug_closed
            or self._live_debug.closed
        ):
            return
        try:
            self._live_debug.write_state(self._build_live_debug_state())
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self.ui.statusbar.showMessage(
                f"Снимок живой диагностики не записан: {error}"
            )

    def _build_live_debug_state(self) -> dict[str, object]:
        """Собрать небольшой снимок без копирования истории всех кадров."""

        frame_bgr = self.video.current_frame_bgr
        map_bgr = self.map_image.current_image_bgr
        telemetry = self.video.current_frame_telemetry
        telemetry_state: dict[str, object] | None = None
        if telemetry is not None:
            telemetry_state = {
                "image_id": telemetry.image_id,
                "elapsed_seconds": telemetry.elapsed_seconds,
                "latitude_degrees": telemetry.latitude_degrees,
                "longitude_degrees": telemetry.longitude_degrees,
                "altitude_ellipsoid_meters": telemetry.altitude_ellipsoid_meters,
                "has_position": telemetry.has_position,
            }

        return {
            "window": {
                "title": self.windowTitle(),
                "visible": self.isVisible(),
                "width_px": self.width(),
                "height_px": self.height(),
                "status_message": self.ui.statusbar.currentMessage(),
            },
            "video": {
                "source_path": self.video.source_path,
                "is_flight": self.video.is_flight,
                "is_playing": self.video.is_playing,
                "frame_index": self.video.current_frame_index,
                "frame_count": self.video.frame_count,
                "fps": self.video.fps,
                "frame_size_px": (
                    [frame_bgr.shape[1], frame_bgr.shape[0]]
                    if frame_bgr is not None
                    else None
                ),
                "telemetry": telemetry_state,
            },
            "map": {
                "source_path": self.map_image.source_path,
                "image_size_px": (
                    [map_bgr.shape[1], map_bgr.shape[0]]
                    if map_bgr is not None
                    else None
                ),
            },
            "ui": {
                "set_points_checked": self.ui.actionSet_points.isChecked(),
                "track_points_checked": self.ui.actionTrack_points.isChecked(),
                "show_overlay_checked": self.ui.actionShow_overlay.isChecked(),
                "frame_slider_value": self.ui.frameSlider.value(),
            },
            "alignment": self.alignment.debug_snapshot(),
            "automatic_alignment": {
                "status": self._last_automatic_status,
                "message": self._last_automatic_message,
                "correction_note": self._last_automatic_correction_note,
            },
        }

    def _capture_live_debug_artifacts(
        self,
        reason: str,
        *,
        payload: Mapping[str, object] | None = None,
        create_incident: bool,
        force: bool,
    ) -> Path | None:
        """Сохранить текущие изображения, а при сбое — отдельный инцидент.

        В обычном режиме сохраняется только уменьшенное JPEG-превью не чаще
        раза в секунду. Полноразмерные PNG создаются один раз внутри инцидента:
        прежняя запись шести PNG блокировала GUI примерно на секунду даже при
        простом перетаскивании маяка.
        """

        if (
            self._live_debug is None
            or self._live_debug_closed
            or self._live_debug.closed
        ):
            return None
        now = time.monotonic()
        if not force and now - self._last_live_debug_capture_monotonic < 1.0:
            return None
        self._last_live_debug_capture_monotonic = now

        state = self._build_live_debug_state()
        try:
            self._live_debug.write_state(state)
            self._save_live_debug_preview()
            incident_directory: Path | None = None
            if create_incident:
                incident_directory = self._live_debug.create_incident(
                    reason,
                    state,
                    payload=payload,
                )
                self._save_live_debug_images(incident_directory)
                self._live_debug.record_event(
                    "incident_captured",
                    {
                        "reason": reason,
                        "incident_directory": str(incident_directory),
                    },
                )
            return incident_directory
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._record_live_debug_event(
                "debug_capture_failed",
                {"reason": reason, "message": str(error)},
            )
            return None

    def _save_live_debug_preview(self) -> None:
        """Сохранить один лёгкий снимок окна для наблюдения «через плечо»."""

        assert self._live_debug is not None
        preview = self.grab()
        max_width_px = 1600
        if preview.width() > max_width_px:
            preview = preview.scaledToWidth(
                max_width_px,
                Qt.TransformationMode.SmoothTransformation,
            )
        _atomic_save_pixmap(
            self._live_debug.session_directory / "latest-window.jpg",
            preview,
            image_format="JPG",
            quality=70,
        )

    def _save_live_debug_images(self, directory: Path, *, prefix: str = "") -> None:
        """Сохранить исходные изображения и реальные представления Qt."""

        frame_bgr = self.video.current_frame_bgr
        if frame_bgr is not None:
            _atomic_save_bgr(directory / f"{prefix}frame.png", frame_bgr)
        previous_bgr = self.alignment.last_debug_previous_frame_bgr
        if previous_bgr is not None:
            _atomic_save_bgr(
                directory / f"{prefix}previous-frame.png",
                previous_bgr,
            )
        map_bgr = self.map_image.current_image_bgr
        if map_bgr is not None:
            _atomic_save_bgr(directory / f"{prefix}map.png", map_bgr)

        _atomic_save_pixmap(
            directory / f"{prefix}video-view.png",
            self.ui.videoView.grab(),
        )
        _atomic_save_pixmap(
            directory / f"{prefix}map-view.png",
            self.ui.mapView.grab(),
        )
        _atomic_save_pixmap(
            directory / f"{prefix}window.png",
            self.grab(),
        )

    def _mark_live_debug_incident(self) -> None:
        """F12: без диалога зафиксировать то, что сейчас видит оператор."""

        frame_index = self.video.current_frame_index
        payload = {"frame_index": frame_index, "source": "F12"}
        if not self._record_live_debug_event("incident_marked", payload):
            self.ui.statusbar.showMessage("Живая диагностика не включена")
            return
        incident_directory = self._capture_live_debug_artifacts(
            "manual-f12",
            payload=payload,
            create_incident=True,
            force=True,
        )
        if incident_directory is not None:
            self.ui.statusbar.showMessage(
                f"Инцидент зафиксирован: {incident_directory.name}"
            )

    def _reset_automatic_attempt(self) -> None:
        """Удалить результат, когда изменился хотя бы один исходный пиксель."""

        self._last_automatic_alignment = None
        self._last_automatic_candidate = None
        self._last_automatic_diagnostics = None
        self._last_automatic_status = "not_run"
        self._last_automatic_message = ""
        self._last_automatic_frame_bgr = None
        self._last_automatic_correction_note = ""
        if hasattr(self.ui, "actionExport_alignment_report"):
            self.ui.actionExport_alignment_report.setEnabled(False)

    def _seek_from_slider(self, frame_index: int) -> None:
        """Перейти к выбранному кадру и сбросить неприменимое сопровождение.

        Сопровождение маяков предполагает обработку каждого соседнего кадра.
        Скачок через слайдер нарушает это допущение, поэтому оператор должен
        заново выбрать опорную разметку на новом кадре.
        """

        if not self.video.is_loaded:
            return
        if frame_index == self.video.current_frame_index:
            return
        if self.alignment.tracking_active:
            self.ui.actionTrack_points.setChecked(False)
        if not self.video.seek(frame_index):
            self._synchronise_frame_slider(
                self.video.current_frame_index or 0,
                self.video.frame_count,
            )

    def _configure_frame_slider(self, frame_count: int) -> None:
        """Настроить целочисленный диапазон слайдера по числу кадров."""

        blocker = QSignalBlocker(self.ui.frameSlider)
        self.ui.frameSlider.setRange(0, max(0, frame_count - 1))
        self.ui.frameSlider.setSingleStep(1)
        self.ui.frameSlider.setPageStep(max(1, frame_count // 20))
        self.ui.frameSlider.setEnabled(frame_count > 1)
        del blocker

    def _synchronise_frame_slider(self, frame_index: int, frame_count: int) -> None:
        """Отразить программную смену кадра без повторного вызова seek."""

        blocker = QSignalBlocker(self.ui.frameSlider)
        if frame_count > 0 and self.ui.frameSlider.maximum() != frame_count - 1:
            self.ui.frameSlider.setRange(0, frame_count - 1)
        self.ui.frameSlider.setValue(frame_index)
        self.ui.frameSlider.setEnabled(frame_count > 1)
        del blocker

    def _set_video_controls_enabled(self, enabled: bool) -> None:
        self.ui.pbPlay.setEnabled(enabled)
        self.ui.pbStop.setEnabled(enabled)
        self.ui.pbBack.setEnabled(enabled)
        self.ui.pbExcludeFrame.setEnabled(enabled)
        self.ui.pbForward.setEnabled(enabled)
        self.ui.frameSlider.setEnabled(enabled and self.video.frame_count > 1)

    def _update_frame_navigation(self, frame_index: int, frame_count: int) -> None:
        """Включить покадровые кнопки только там, где есть соседний кадр."""

        self.ui.pbBack.setEnabled(frame_index > 0)
        self.ui.pbExcludeFrame.setEnabled(self.video.is_loaded)
        self.ui.pbForward.setEnabled(frame_count <= 0 or frame_index < frame_count - 1)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Повторно вписать кадр после изменения размеров окна."""

        super().resizeEvent(event)
        if hasattr(self, "video"):
            self.video.fit_to_view()
        if hasattr(self, "map_image"):
            self.map_image.fit_to_view()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Предотвратить потерю прогресса и затем освободить источники."""

        if (
            self._protect_unsaved_progress
            and not self._confirm_unsaved_progress()
        ):
            event.ignore()
            return
        self._closing_window = True
        self._autosave_timer.stop()
        if self._remember_recent_sources:
            self._settings.setValue(WINDOW_STATE_KEY, self.saveState())
            self._settings.setValue(
                DOCK_GEOMETRY_KEY,
                self.operator_dock.saveGeometry(),
            )
            self._settings.sync()
        if self._live_debug is not None and not self._live_debug_closed:
            self._record_live_debug_event(
                "window_closing",
                {"frame_index": self.video.current_frame_index},
            )
            self._write_live_debug_state()
            self._capture_live_debug_artifacts(
                "window-closing",
                create_incident=False,
                force=True,
            )
            self._live_debug_closed = True
            self._live_debug.close()
        self.alignment.clear()
        self.alignment.dispose()
        self.video.close()
        self.map_image.clear()
        super().closeEvent(event)


def _temporary_image_path(path: Path) -> Path:
    """Дать соседний PNG-путь, пригодный для последующей атомарной замены."""

    return path.with_name(
        f".{path.stem}.tmp-{os.getpid()}-{time.time_ns()}{path.suffix}"
    )


def _atomic_save_bgr(path: Path, image_bgr: object) -> None:
    """Не показывать читателю частично записанное изображение OpenCV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_image_path(path)
    try:
        if not cv2.imwrite(str(temporary), image_bgr):
            raise OSError(f"OpenCV не сохранил изображение {temporary}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_pixmap(
    path: Path,
    pixmap: QPixmap,
    *,
    image_format: str = "PNG",
    quality: int = -1,
) -> None:
    """Атомарно сохранить изображение Qt-сцены или всего окна."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_image_path(path)
    try:
        if pixmap.isNull():
            raise OSError(f"Qt вернул пустой снимок для {path.name}")
        if not pixmap.save(str(temporary), image_format, quality):
            raise OSError(f"Qt не сохранил изображение {temporary}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
