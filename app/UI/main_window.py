from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QProgressBar,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QToolButton,
    QComboBox
)

from app.UI.widgets.scene_tree import SceneTree
from app.UI.widgets.viewer_3d import Viewer3D


class MainWindow(QMainWindow):
    def __init__(self, qss_path: str) -> None:
        super().__init__()

        self.setWindowTitle("Laser")
        self.resize(1400, 800)

        self._build_actions()
        self._build_ui()
        self._apply_qss(qss_path)

    def _build_actions(self) -> None:
        icons_path = os.path.join(os.path.dirname(__file__))
        self.act_new = QAction(QIcon(icons_path + "/icons/new.png"), "Новый\nдокумент", self)
        self.act_open = QAction(QIcon(icons_path + "/icons/open.png"), "Открыть\nдокумент", self)
        self.act_save = QAction(QIcon(icons_path + "./icons/save.png"), "Сохранить\nдокумент", self)

        self.act_import_model = QAction(QIcon(icons_path + "/icons/import-model.png"), "Импорт\nмодели", self)
        self.act_import_cloud = QAction(QIcon(icons_path + "/icons/import-cloud.png"), "Импорт\nоблака", self)

        self.act_filter = QAction(QIcon(icons_path + "/icons/filter.png"), "Фильтрация", self)
        self.act_downsample = QAction(QIcon(icons_path + "/icons/downsampling.png"), "Даунсемплинг", self)

        self.act_select = QAction(QIcon(icons_path + "/icons/segment.png"), "Выделить и\n изолировать", self)

        self.act_manual = QAction(QIcon(icons_path + "/icons/reg.png"), "Ручная", self)
        self.act_icp = QAction(QIcon(icons_path + "/icons/ICP.png"), "ICP", self)

        self.act_build_mesh = QAction(QIcon(icons_path + "/icons/mash.png"), "Построить\nмэш", self)
        self.act_calc = QAction(QIcon(icons_path + "/icons/calc.png"), "Рассчитать", self)
        self.act_compare = QAction(QIcon(icons_path + "/icons/compare.png"), "Сравнить", self)

        self.act_report = QAction(QIcon(icons_path + "/icons/new.png"), "Создать\nPDF", self)

    def _build_ui(self) -> None:
        # Toolbar
        tb = QToolBar()
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)

        self._toolbar_group(tb, "Проект", [self.act_new, self.act_open, self.act_save])
        self._toolbar_group(tb, "Данные", [self.act_import_model, self.act_import_cloud])
        self._toolbar_group(tb, "Обработка", [self.act_filter, self.act_downsample])
        self._toolbar_group(tb, "Сегментация", [self.act_select])
        self._toolbar_group(tb, "Регистрация", [self.act_manual, self.act_icp])
        self._toolbar_group(tb, "Объёмы", [self.act_build_mesh, self.act_calc, self.act_compare])
        self._toolbar_group(tb, "Отчёт", [self.act_report])

        # Central splitter
        splitter = QSplitter(Qt.Horizontal)

        self.tree = SceneTree()
        self.tree.setMinimumWidth(260)
        splitter.addWidget(self.tree)

        self.viewer = Viewer3D()
        splitter.addWidget(self.viewer)

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(360)
        self.tabs.addTab(self._tab_preprocess(), "Предобработка")
        self.tabs.addTab(self._tab_segmentation(), "Сегментация")
        self.tabs.addTab(self._tab_registration(), "Регистрация")
        self.tabs.addTab(self._tab_volumes(), "Объёмы")
        self.tabs.addTab(self._tab_report(), "Отчёт")
        splitter.addWidget(self.tabs)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

        # Status bar
        self.status = self.statusBar()
        self.status_label = QLabel("Готово")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setValue(0)

        self.status.addWidget(self.status_label, 1)
        self.status.addPermanentWidget(self.progress)

    def _toolbar_group(self, toolbar: QToolBar, title: str, actions: list[QAction]) -> None:
        lbl = QLabel(title)
        container_buttons = QWidget()
        button_layout = QHBoxLayout(container_buttons)
        lbl.setStyleSheet("color: #E6E8F0; font-size: 13px; padding: 0px;")
        for a in actions:
            button = QToolButton()
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setDefaultAction(a)
            button_layout.addWidget(button)
        container_toolbar = QWidget()
        main_layout = QVBoxLayout(container_toolbar)
        main_layout.addWidget(lbl, alignment=Qt.AlignHCenter)
        main_layout.addWidget(container_buttons)
        toolbar.addWidget(container_toolbar)
        toolbar.addSeparator()
        button_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setContentsMargins(0, 0, 0, 0)

    # ---- Tabs (пока заглушки, но структура как в ТЗ) ----
    def _tab_preprocess(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(12, 12, 12, 12)
       
        gb = QGroupBox("Фильтрация")
        g = QVBoxLayout(gb)
        g.addWidget(QLabel("Метод"))
        cb = QComboBox()
        cb.setPlaceholderText("Название метода")
        g.addWidget(cb)
        g.addWidget(QLabel("Количество соседей"))
        le1 = QLineEdit()
        le1.setPlaceholderText("Введите число")
        g.addWidget(le1)
        g.addWidget(QLabel("Порог"))
        le2 = QLineEdit()
        le2.setPlaceholderText("Введите число")
        g.addWidget(le2)

        l.addWidget(gb)
        l.addStretch(1)
        return w

    def _tab_segmentation(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(12, 12, 12, 12)

        gb = QGroupBox("Сегментация")
        g = QVBoxLayout(gb)
        g.addWidget(QPushButton("Выделить область"))
        g.addWidget(QPushButton("Изолировать выделенное"))
        g.addWidget(QPushButton("Удалить выделение"))

        l.addWidget(gb)
        l.addStretch(1)
        return w

    def _tab_registration(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(12, 12, 12, 12)

        gb = QGroupBox("Регистрация (ICP)")
        g = QVBoxLayout(gb)
        rms = QLineEdit()
        rms.setReadOnly(True)
        rms.setPlaceholderText("RMS: —")
        g.addWidget(rms)

        l.addWidget(gb)
        l.addStretch(1)
        return w

    def _tab_volumes(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(12, 12, 12, 12)

        gb = QGroupBox("Объёмы")
        g = QVBoxLayout(gb)
        g.addWidget(QPushButton("Построить мэш"))
        g.addWidget(QPushButton("Рассчитать объём"))
        g.addWidget(QPushButton("Сравнить объёмы"))

        l.addWidget(gb)
        l.addStretch(1)
        return w

    def _tab_report(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(12, 12, 12, 12)

        gb = QGroupBox("Отчёт")
        g = QVBoxLayout(gb)
        btn = QPushButton("Создать PDF")
        btn.setObjectName("primaryBtn")
        g.addWidget(btn)

        l.addWidget(gb)
        l.addStretch(1)
        return w

    # ---- Helpers ----
    def _apply_qss(self, qss_path: str) -> None:
        if qss_path and os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    def set_status(self, text: str, progress: int | None = None) -> None:
        self.status_label.setText(text)
        if progress is not None:
            self.progress.setValue(progress)
