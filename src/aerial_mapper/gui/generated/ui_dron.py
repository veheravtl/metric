# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dron.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QGraphicsView, QHBoxLayout, QMainWindow,
    QMenu, QMenuBar, QPushButton, QSizePolicy,
    QSlider, QSplitter, QStatusBar, QVBoxLayout,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1362, 871)
        self.actionLoad_video = QAction(MainWindow)
        self.actionLoad_video.setObjectName(u"actionLoad_video")
        self.actionLoad_flight = QAction(MainWindow)
        self.actionLoad_flight.setObjectName(u"actionLoad_flight")
        self.actionStream_from_Camera = QAction(MainWindow)
        self.actionStream_from_Camera.setObjectName(u"actionStream_from_Camera")
        self.actionLoad_map = QAction(MainWindow)
        self.actionLoad_map.setObjectName(u"actionLoad_map")
        self.actionAuto_align = QAction(MainWindow)
        self.actionAuto_align.setObjectName(u"actionAuto_align")
        self.actionAuto_align.setEnabled(False)
        self.actionExport_alignment_report = QAction(MainWindow)
        self.actionExport_alignment_report.setObjectName(u"actionExport_alignment_report")
        self.actionExport_alignment_report.setEnabled(False)
        self.actionSet_points = QAction(MainWindow)
        self.actionSet_points.setObjectName(u"actionSet_points")
        self.actionSet_points.setCheckable(True)
        self.actionSet_points.setEnabled(False)
        self.actionUndo_point = QAction(MainWindow)
        self.actionUndo_point.setObjectName(u"actionUndo_point")
        self.actionUndo_point.setEnabled(False)
        self.actionRemove_selected = QAction(MainWindow)
        self.actionRemove_selected.setObjectName(u"actionRemove_selected")
        self.actionRemove_selected.setEnabled(False)
        self.actionClear_points = QAction(MainWindow)
        self.actionClear_points.setObjectName(u"actionClear_points")
        self.actionClear_points.setEnabled(False)
        self.actionCompute_alignment = QAction(MainWindow)
        self.actionCompute_alignment.setObjectName(u"actionCompute_alignment")
        self.actionCompute_alignment.setEnabled(False)
        self.actionTrack_points = QAction(MainWindow)
        self.actionTrack_points.setObjectName(u"actionTrack_points")
        self.actionTrack_points.setCheckable(True)
        self.actionTrack_points.setEnabled(False)
        self.actionExclude_frame = QAction(MainWindow)
        self.actionExclude_frame.setObjectName(u"actionExclude_frame")
        self.actionExclude_frame.setEnabled(False)
        self.actionExclude_frame_range = QAction(MainWindow)
        self.actionExclude_frame_range.setObjectName(u"actionExclude_frame_range")
        self.actionExclude_frame_range.setEnabled(False)
        self.actionShow_overlay = QAction(MainWindow)
        self.actionShow_overlay.setObjectName(u"actionShow_overlay")
        self.actionShow_overlay.setCheckable(True)
        self.actionShow_overlay.setEnabled(False)
        self.actionShow_overlay.setChecked(True)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralLayout = QVBoxLayout(self.centralwidget)
        self.centralLayout.setObjectName(u"centralLayout")
        self.viewSplitter = QSplitter(self.centralwidget)
        self.viewSplitter.setObjectName(u"viewSplitter")
        self.viewSplitter.setOrientation(Qt.Orientation.Horizontal)
        self.viewSplitter.setChildrenCollapsible(False)
        self.videoView = QGraphicsView(self.viewSplitter)
        self.videoView.setObjectName(u"videoView")
        self.videoView.setMinimumSize(QSize(320, 240))
        self.viewSplitter.addWidget(self.videoView)
        self.mapView = QGraphicsView(self.viewSplitter)
        self.mapView.setObjectName(u"mapView")
        self.mapView.setMinimumSize(QSize(320, 240))
        self.viewSplitter.addWidget(self.mapView)

        self.centralLayout.addWidget(self.viewSplitter)

        self.playbackLayout = QHBoxLayout()
        self.playbackLayout.setObjectName(u"playbackLayout")
        self.pbPlay = QPushButton(self.centralwidget)
        self.pbPlay.setObjectName(u"pbPlay")

        self.playbackLayout.addWidget(self.pbPlay)

        self.pbStop = QPushButton(self.centralwidget)
        self.pbStop.setObjectName(u"pbStop")

        self.playbackLayout.addWidget(self.pbStop)

        self.pbBack = QPushButton(self.centralwidget)
        self.pbBack.setObjectName(u"pbBack")

        self.playbackLayout.addWidget(self.pbBack)

        self.pbForward = QPushButton(self.centralwidget)
        self.pbForward.setObjectName(u"pbForward")

        self.playbackLayout.addWidget(self.pbForward)

        self.pbExcludeFrame = QPushButton(self.centralwidget)
        self.pbExcludeFrame.setObjectName(u"pbExcludeFrame")
        self.pbExcludeFrame.setEnabled(False)

        self.playbackLayout.addWidget(self.pbExcludeFrame)

        self.frameSlider = QSlider(self.centralwidget)
        self.frameSlider.setObjectName(u"frameSlider")
        self.frameSlider.setEnabled(False)
        self.frameSlider.setMinimumSize(QSize(240, 0))
        self.frameSlider.setMaximum(0)
        self.frameSlider.setOrientation(Qt.Orientation.Horizontal)
        self.frameSlider.setTracking(True)

        self.playbackLayout.addWidget(self.frameSlider)


        self.centralLayout.addLayout(self.playbackLayout)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1362, 22))
        self.menuVideo = QMenu(self.menubar)
        self.menuVideo.setObjectName(u"menuVideo")
        self.menuMap = QMenu(self.menubar)
        self.menuMap.setObjectName(u"menuMap")
        self.menuAlignment = QMenu(self.menubar)
        self.menuAlignment.setObjectName(u"menuAlignment")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuVideo.menuAction())
        self.menubar.addAction(self.menuMap.menuAction())
        self.menubar.addAction(self.menuAlignment.menuAction())
        self.menuVideo.addAction(self.actionLoad_video)
        self.menuVideo.addAction(self.actionLoad_flight)
        self.menuVideo.addAction(self.actionStream_from_Camera)
        self.menuMap.addAction(self.actionLoad_map)
        self.menuAlignment.addAction(self.actionAuto_align)
        self.menuAlignment.addAction(self.actionExport_alignment_report)
        self.menuAlignment.addSeparator()
        self.menuAlignment.addAction(self.actionSet_points)
        self.menuAlignment.addAction(self.actionUndo_point)
        self.menuAlignment.addAction(self.actionRemove_selected)
        self.menuAlignment.addAction(self.actionClear_points)
        self.menuAlignment.addSeparator()
        self.menuAlignment.addAction(self.actionCompute_alignment)
        self.menuAlignment.addAction(self.actionTrack_points)
        self.menuAlignment.addAction(self.actionShow_overlay)
        self.menuAlignment.addAction(self.actionExclude_frame)
        self.menuAlignment.addAction(self.actionExclude_frame_range)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionLoad_video.setText(QCoreApplication.translate("MainWindow", u"Load video", None))
        self.actionLoad_flight.setText(QCoreApplication.translate("MainWindow", u"Load flight folder", None))
        self.actionStream_from_Camera.setText(QCoreApplication.translate("MainWindow", u"Stream from Camera", None))
        self.actionLoad_map.setText(QCoreApplication.translate("MainWindow", u"Load map", None))
        self.actionAuto_align.setText(QCoreApplication.translate("MainWindow", u"Auto align current frame", None))
#if QT_CONFIG(tooltip)
        self.actionAuto_align.setToolTip(QCoreApplication.translate("MainWindow", u"Experimental SIFT + MAGSAC alignment", None))
#endif // QT_CONFIG(tooltip)
        self.actionExport_alignment_report.setText(QCoreApplication.translate("MainWindow", u"Export manual + auto report...", None))
#if QT_CONFIG(tooltip)
        self.actionExport_alignment_report.setToolTip(QCoreApplication.translate("MainWindow", u"Run auto alignment if needed and save a diagnostic ZIP", None))
#endif // QT_CONFIG(tooltip)
        self.actionSet_points.setText(QCoreApplication.translate("MainWindow", u"Set corresponding points", None))
        self.actionUndo_point.setText(QCoreApplication.translate("MainWindow", u"Undo last point", None))
        self.actionRemove_selected.setText(QCoreApplication.translate("MainWindow", u"Remove selected beacon", None))
#if QT_CONFIG(shortcut)
        self.actionRemove_selected.setShortcut(QCoreApplication.translate("MainWindow", u"Del", None))
#endif // QT_CONFIG(shortcut)
        self.actionClear_points.setText(QCoreApplication.translate("MainWindow", u"Clear points", None))
        self.actionCompute_alignment.setText(QCoreApplication.translate("MainWindow", u"Compute / confirm alignment", None))
#if QT_CONFIG(tooltip)
        self.actionCompute_alignment.setToolTip(QCoreApplication.translate("MainWindow", u"Confirm the current frame as a new operator keyframe", None))
#endif // QT_CONFIG(tooltip)
        self.actionTrack_points.setText(QCoreApplication.translate("MainWindow", u"Track beacons across frames", None))
#if QT_CONFIG(tooltip)
        self.actionTrack_points.setToolTip(QCoreApplication.translate("MainWindow", u"Track confirmed manual points on consecutive video frames", None))
#endif // QT_CONFIG(tooltip)
        self.actionExclude_frame.setText(QCoreApplication.translate("MainWindow", u"Exclude current frame and advance", None))
#if QT_CONFIG(tooltip)
        self.actionExclude_frame.setToolTip(QCoreApplication.translate("MainWindow", u"Mark the current frame unusable for measurements and open the next frame", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.actionExclude_frame.setShortcut(QCoreApplication.translate("MainWindow", u"Alt+X", None))
#endif // QT_CONFIG(shortcut)
        self.actionExclude_frame_range.setText(QCoreApplication.translate("MainWindow", u"Exclude frame range...", None))
#if QT_CONFIG(tooltip)
        self.actionExclude_frame_range.setToolTip(QCoreApplication.translate("MainWindow", u"Exclude the current frame through a chosen last frame index", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.actionExclude_frame_range.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+X", None))
#endif // QT_CONFIG(shortcut)
        self.actionShow_overlay.setText(QCoreApplication.translate("MainWindow", u"Show overlay", None))
        self.pbPlay.setText(QCoreApplication.translate("MainWindow", u"Play", None))
        self.pbStop.setText(QCoreApplication.translate("MainWindow", u"Stop", None))
        self.pbBack.setText(QCoreApplication.translate("MainWindow", u"<<", None))
        self.pbForward.setText(QCoreApplication.translate("MainWindow", u">>", None))
        self.pbExcludeFrame.setText(QCoreApplication.translate("MainWindow", u"Exclude + >>", None))
#if QT_CONFIG(tooltip)
        self.pbExcludeFrame.setToolTip(QCoreApplication.translate("MainWindow", u"Exclude current frame from measurements and advance", None))
#endif // QT_CONFIG(tooltip)
        self.menuVideo.setTitle(QCoreApplication.translate("MainWindow", u"Video", None))
        self.menuMap.setTitle(QCoreApplication.translate("MainWindow", u"Map", None))
        self.menuAlignment.setTitle(QCoreApplication.translate("MainWindow", u"Alignment", None))
    # retranslateUi
