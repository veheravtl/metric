"""Проверки воспроизводимой генерации Python-класса из Qt Designer UI."""

from aerial_mapper.gui.build_ui import UI_SOURCE_PATH, build_ui, generated_ui_is_current


def test_build_ui_creates_importable_python_file(tmp_path):
    """Исправный ``dron.ui`` должен давать синтаксически корректный класс.

    Тест обнаружит повреждение XML, отсутствие ``pyside6-uic`` или изменение
    процедуры генерации, при котором приложение больше нельзя импортировать.
    """

    output_path = tmp_path / "ui_dron.py"

    assert build_ui(UI_SOURCE_PATH, output_path, force=True)

    generated_source = output_path.read_text(encoding="utf-8")
    compile(generated_source, str(output_path), "exec")
    assert "class Ui_MainWindow" in generated_source

    # Эти имена являются договором между формой и будущей логикой окна.
    # QSplitter важен отдельно: без него области видео и карты снова станут
    # виджетами с фиксированными координатами и не будут адаптироваться к окну.
    assert "self.viewSplitter = QSplitter" in generated_source
    assert "self.mapView = QGraphicsView" in generated_source
    assert "self.pbForward = QPushButton" in generated_source
    assert "self.pbExcludeFrame = QPushButton" in generated_source
    assert "self.actionExclude_frame = QAction" in generated_source
    assert "self.actionExclude_frame_range = QAction" in generated_source
    assert "self.frameSlider = QSlider" in generated_source
    assert "self.actionLoad_flight = QAction" in generated_source
    assert "self.menuAlignment = QMenu" in generated_source
    assert "self.actionCompute_alignment = QAction" in generated_source
    assert "self.actionRemove_selected = QAction" in generated_source
    assert "self.actionAuto_align = QAction" in generated_source
    assert "self.actionExport_alignment_report = QAction" in generated_source

    assert generated_ui_is_current(UI_SOURCE_PATH, output_path)

    # Повторная сборка без изменения исходника не должна трогать файл.
    assert not build_ui(UI_SOURCE_PATH, output_path)
