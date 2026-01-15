from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication
from app.UI.resources import resource_path

from app.UI.main_window import MainWindow
from app.Controllers.app_controller import AppController
from app.Infrastructure.pointcloud_io import load_point_cloud_any
from PySide6.QtGui import QFontDatabase


def main() -> None:
    app = QApplication(sys.argv)

    font_path = resource_path("app", "UI", "fonts", "Montserrat.ttf")
    QFontDatabase.addApplicationFont(font_path)

    qss_path = resource_path("app", "UI", "style", "style.qss")
    window = MainWindow(qss_path=qss_path)

    controller = AppController(pointcloud_io=load_point_cloud_any)
    controller.bind(window)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
