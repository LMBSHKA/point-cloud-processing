from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget


class Viewer3D(QWidget):
    """
    Обёртка над PyVista (QtInteractor).
    Умеет показывать облака точек, меши и управлять видимостью.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._interactor = QtInteractor(self)
        self._plotter = self._interactor
        self._actors: dict[str, object] = {}

        from PySide6.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._interactor.interactor)

        self._init_scene()

    def _init_scene(self) -> None:
        self._plotter.set_background("#0B0F1E")

        self._plotter.add_axes(
            interactive=False,
            line_width=2,
            color="white"
        )

        self._plotter.enable_parallel_projection()
        # Без позиционных аргументов:
        self._plotter.reset_camera(render=False)
        self._plotter.render()

    def clear(self) -> None:
        self._plotter.clear()
        self._actors.clear()
        self._init_scene()

    def _reset_camera_to_bounds(self, bounds, zoom: float = 1.2) -> None:
        """
        Единая безопасная функция сброса камеры без deprecated-позиционных аргументов.
        """
        # bounds может быть None/битым — на всякий
        if bounds is None:
            self._plotter.reset_camera(render=True)
            return

        # Передаём bounds именованно + render именованно (это убирает warning)
        self._plotter.reset_camera(bounds=bounds, render=True)

        if zoom and zoom != 1.0:
            try:
                self._plotter.camera.zoom(float(zoom))
            except Exception:
                pass

        self._plotter.render()

    def show_point_cloud(self, obj_id: str, points: np.ndarray) -> None:
        """
        Отображает облако точек.
        """
        # Удаляем старый актор, если был
        if obj_id in self._actors:
            try:
                self._plotter.remove_actor(self._actors[obj_id])
            except Exception:
                pass
            self._actors.pop(obj_id, None)

        pts = np.asarray(points, dtype=np.float32)
        cloud = pv.PolyData(pts)

        actor = self._plotter.add_points(
            cloud,
            point_size=2,
            render_points_as_spheres=False,
            color="white"
        )
        self._actors[obj_id] = actor

        self._reset_camera_to_bounds(cloud.bounds, zoom=1.2)

    def show_mesh(self, obj_id: str, vertices, triangles) -> None:
        """
        Отображает треугольный меш.
        """
        V = np.asarray(vertices, dtype=np.float32)
        F = np.asarray(triangles, dtype=np.int64)

        if V.size == 0 or F.size == 0:
            return

        faces = np.hstack([np.full((F.shape[0], 1), 3, dtype=np.int64), F]).ravel()
        mesh = pv.PolyData(V, faces)

        # Удаляем старый актор
        if obj_id in self._actors:
            try:
                self._plotter.remove_actor(self._actors[obj_id])
            except Exception:
                pass
            self._actors.pop(obj_id, None)

        actor = self._plotter.add_mesh(mesh, smooth_shading=True)
        self._actors[obj_id] = actor

        self._reset_camera_to_bounds(mesh.bounds, zoom=1.2)

    def set_visible(self, obj_id: str, visible: bool) -> None:
        actor = self._actors.get(obj_id)
        if actor is None:
            return
        try:
            actor.SetVisibility(1 if visible else 0)
            self._plotter.render()
        except Exception:
            pass
