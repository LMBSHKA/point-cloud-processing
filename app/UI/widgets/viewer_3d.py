from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget


class Viewer3D(QWidget):
    """
    Обёртка над PyVista (QtInteractor).
    Умеет показывать облака точек и управлять их видимостью.
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

        # Оси в углу экрана (как в CAD)
        self._plotter.add_axes(
            interactive=False,
            line_width=2,
            color="white"
        )

        # Ортографическая камера (ключевой момент)
        self._plotter.enable_parallel_projection()
        self._plotter.reset_camera()

    def clear(self) -> None:
        self._plotter.clear()
        self._actors.clear()
        self._init_scene()

    def show_point_cloud(self, obj_id: str, points: np.ndarray) -> None:
        """
        Отображает облако точек в центре сцены.
        """
        # Удаляем старый актор, если был
        if obj_id in self._actors:
            try:
                self._plotter.remove_actor(self._actors[obj_id])
            except Exception:
                pass
            self._actors.pop(obj_id, None)

        cloud = pv.PolyData(points)

        actor = self._plotter.add_points(
            cloud,
            point_size=2,
            render_points_as_spheres=False,
            color="white"
        )

        self._actors[obj_id] = actor

        # Фокусируем камеру по границам облака
        bounds = cloud.bounds
        self._plotter.reset_camera(bounds)
        self._plotter.camera.zoom(1.2)

    def set_visible(self, obj_id: str, visible: bool) -> None:
        actor = self._actors.get(obj_id)
        if actor is None:
            return
        try:
            actor.SetVisibility(1 if visible else 0)
            self._plotter.render()
        except Exception:
            pass
    
    def show_mesh(self, obj_id: str, vertices, triangles) -> None:
        V = np.asarray(vertices)
        F = np.asarray(triangles)

        faces = np.hstack([np.full((F.shape[0], 1), 3), F]).astype(np.int64).ravel()
        mesh = pv.PolyData(V, faces)

        # Удаляем старый актор, если он уже был
        if obj_id in self._actors:
            try:
                self._plotter.remove_actor(self._actors[obj_id])
            except Exception:
                pass
            self._actors.pop(obj_id, None)

        actor = self._plotter.add_mesh(mesh, smooth_shading=True)
        self._actors[obj_id] = actor

        # Фокус камеры по модели
        #self._plotter.reset_camera(mesh.bounds)
        self._plotter.reset_camera(render=True)
        self._plotter.render()

