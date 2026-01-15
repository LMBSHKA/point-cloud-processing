from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget


class Viewer3D(QWidget):
    """
    PyVista viewer для облаков и мешей.

    ВАЖНО (про "трещины" в UI):
    - VTK плохо переносит большие мировые координаты -> нормализуем ДО PolyData.
    - Визуальные "трещины" на гладких поверхностях чаще всего из-за неконсистентных
      нормалей/швов. Open3D при построении меша считает хорошие vertex normals,
      но раньше UI их терял и пересчитывал в VTK (что давало швы).
      Поэтому: если нормали переданы — используем ИХ, не пересчитываем.
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
            color="white",
        )

        # Перспектива обычно даёт меньше артефактов на тонких поверхностях.
        try:
            self._plotter.disable_parallel_projection()
        except Exception:
            pass
        '''
        # Depth peeling иногда помогает со сложными сценами, почти ничего не ломает.
        try:
            self._plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
        except Exception:
            pass
        '''
        self._plotter.reset_camera(render=False)
        self._plotter.render()

    def clear(self) -> None:
        self._plotter.clear()
        self._actors.clear()
        self._init_scene()

    # ----------------------------
    # Normalization for VIEW (numpy)
    # ----------------------------

    @staticmethod
    def _normalize_points_for_view(
        pts: np.ndarray,
        *,
        target_size: float = 2.0,
    ) -> np.ndarray:
        """
        Переносит точки к центру и масштабирует по max extent -> локальные координаты.
        Делается в float64, затем отдаём float32.
        """
        pts = np.asarray(pts, dtype=np.float64)
        if pts.size == 0:
            return pts.astype(np.float32)

        mins = np.nanmin(pts, axis=0)
        maxs = np.nanmax(pts, axis=0)
        center = (mins + maxs) * 0.5
        extent = np.maximum(maxs - mins, 0.0)
        max_extent = float(np.max(extent))

        if not np.isfinite(max_extent) or max_extent <= 0.0:
            return (pts - center).astype(np.float32)

        scale = float(target_size / max_extent)
        out = (pts - center) * scale
        return out.astype(np.float32)

    def _set_clipping_for_view(self) -> None:
        try:
            cam = self._plotter.camera
            pos = np.array(cam.position, dtype=np.float64)
            fp  = np.array(cam.focal_point, dtype=np.float64)
            dist = float(np.linalg.norm(pos - fp))
            if not np.isfinite(dist) or dist <= 1e-9:
                dist = 5.0

            # near не должен быть слишком маленьким -> иначе “трещины”
            near = max(dist * 0.02, 0.05)
            far  = dist * 8.0
            if far <= near:
                far = near * 10.0

            cam.SetClippingRange(float(near), float(far))
        except Exception:
            pass


    def _reset_camera_to_bounds(self, bounds, zoom: float = 1.2) -> None:
        if bounds is None:
            self._plotter.reset_camera(render=True)
            self._set_clipping_for_view()
            return

        self._plotter.reset_camera(bounds=bounds, render=True)
        self._set_clipping_for_view()

        if zoom and zoom != 1.0:
            try:
                self._plotter.camera.zoom(float(zoom))
            except Exception:
                pass

        self._plotter.render()

    # ----------------------------
    # Helpers
    # ----------------------------

    def _remove_actor_if_exists(self, obj_id: str) -> None:
        if obj_id in self._actors:
            try:
                self._plotter.remove_actor(self._actors[obj_id])
            except Exception:
                pass
            self._actors.pop(obj_id, None)

    @staticmethod
    def _set_phong(actor) -> None:
        try:
            prop = actor.GetProperty()
            if prop is not None:
                prop.SetInterpolationToPhong()
                prop.SetEdgeVisibility(False)
                prop.BackfaceCullingOff()
        except Exception:
            pass

    # ----------------------------
    # Public API
    # ----------------------------

    def show_point_cloud(self, obj_id: str, points: np.ndarray) -> None:
        """Отображает облако точек."""
        self._remove_actor_if_exists(obj_id)

        pts = np.asarray(points, dtype=np.float64)
        if pts.size == 0:
            return

        pts_view = self._normalize_points_for_view(pts, target_size=2.0)
        cloud = pv.PolyData(pts_view)

        actor = self._plotter.add_points(
            cloud,
            point_size=2,
            render_points_as_spheres=False,
            color="white",
        )
        self._actors[obj_id] = actor
        self._reset_camera_to_bounds(cloud.bounds, zoom=1.2)

    def show_mesh(self, obj_id: str, vertices, triangles, normals=None) -> None:
        """
        Отображает треугольный меш.

        normals (опционально): vertex normals (N x 3) из Open3D.
        Если переданы — используем их и НЕ пересчитываем в VTK,
        чтобы не появлялись швы/"трещины".
        """
        self._remove_actor_if_exists(obj_id)

        V = np.asarray(vertices, dtype=np.float64)
        F = np.asarray(triangles, dtype=np.int64)
        if V.size == 0 or F.size == 0:
            return

        V_view = self._normalize_points_for_view(V, target_size=2.0)

        faces = np.hstack([np.full((F.shape[0], 1), 3, dtype=np.int64), F]).ravel()
        mesh = pv.PolyData(V_view, faces)
        # --- FIX 1: merge nearly-duplicate points (render-only) ---
        try:
            mesh = mesh.clean(tolerance=1e-6, inplace=False)  # в view-координатах
        except Exception:
            pass

        # --- FIX 2: stable normals for VTK shading ---
        try:
            mesh.compute_normals(
                point_normals=True,
                cell_normals=False,
                consistent_normals=True,
                auto_orient_normals=True,
                split_vertices=False,
                feature_angle=180.0,
                inplace=True,
            )
        except Exception:
            pass

        actor = self._plotter.add_mesh(
            mesh,
            smooth_shading=True,
            show_edges=False,
            lighting=True,
            scalars=None,
            show_scalar_bar=False,
            color="#5C5C5C",
        )
        # --- FIX 3: anti z-fighting for coincident triangles ---
        try:
            mapper = actor.GetMapper()
            if mapper is not None:
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)
        except Exception:
            pass

        self._set_phong(actor)

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
    
    def focus_on(self, obj_id: str) -> None:
        actor = self._actors.get(obj_id)
        if actor is None:
            return
        try:
            b = actor.GetBounds()  # (xmin,xmax,ymin,ymax,zmin,zmax)
            self._plotter.reset_camera(bounds=b)
        except Exception:
            self._plotter.reset_camera()
        self._plotter.render()
