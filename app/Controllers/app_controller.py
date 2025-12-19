from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
import open3d as o3d
from pathlib import Path
from typing import Callable, Optional
from types import SimpleNamespace
from app.Controllers.reconstruction_controller import run_reconstruction

from PySide6.QtWidgets import QFileDialog, QMessageBox

from app.UI.widgets.scene_tree import SceneItem
from app.Infrastructure.pointcloud_io import load_point_cloud_any
#from services.io_service import IOService


@dataclass
class ImportedObject:
    obj_id: str
    name: str
    kind: str
    path: str


class AppController:
    """
    Клей: UI -> сервисы -> UI.
    """
    def __init__(self, pointcloud_io: Callable[[Path, Optional[int]], o3d.geometry.PointCloud]) -> None:
        self.load_point_cloud_any = pointcloud_io

        self._cloud_o3d_by_id: dict[str, o3d.geometry.PointCloud] = {}
        self._cloud_preview_by_id: dict[str, object] = {}
        self._current_cloud_id: str | None = None

        self._mesh_by_id: dict[str, o3d.geometry.TriangleMesh] = {}
        self._current_mesh_id: str | None = None


    def bind(self, window) -> None:
        """
        Подписка на кнопки/сигналы UI.
        window: MainWindow (из main_window.py)
        """
        self.window = window

        window.act_import_cloud.triggered.connect(self.import_cloud)
        window.tree.item_selected.connect(self.on_tree_selected)
        window.tree.visibility_changed.connect(self.on_visibility_changed)
        window.act_build_mesh.triggered.connect(self.reconstruct)
        window.act_filter.triggered.connect(self.apply_filter)


    def import_cloud(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Импорт облака точек",
            "",
            "Cloud (*.e57 *.ply *.pcd *.las *.laz);;All files (*.*)",
        )
        if not path:
            return

        try:
            import numpy as np

            self.window.set_status("Импорт…", progress=10)

            pcd = self.load_point_cloud_any(Path(path), max_points=1_500_000)

            pts = np.asarray(pcd.points)
            if pts.shape[0] == 0:
                raise RuntimeError("Пустое облако после загрузки")

            # превью (ограничим для UI)
            max_preview = 2_000_000
            if pts.shape[0] > max_preview:
                idx = np.random.choice(pts.shape[0], size=max_preview, replace=False)
                pts_preview = pts[idx]
            else:
                pts_preview = pts

            pts_preview = pts_preview - pts_preview.mean(axis=0)
            pts_preview = pts_preview.astype(np.float32, copy=False)

            obj_id = str(uuid.uuid4())
            name = os.path.basename(path)

            self._cloud_o3d_by_id[obj_id] = pcd
            self._cloud_preview_by_id[obj_id] = pts_preview
            self._current_cloud_id = obj_id

            self.window.tree.add_cloud(SceneItem(obj_id=obj_id, name=name, kind="cloud", path=path))
            self.window.viewer.show_point_cloud(obj_id, pts_preview)

            self.window.set_status(f"Импортировано: {name} (точек: {len(pcd.points):,})", progress=100)


        except Exception as e:
            self.window.set_status("Ошибка импорта", progress=0)
            QMessageBox.critical(self.window, "Ошибка импорта", str(e))


    def on_tree_selected(self, obj_id: str) -> None:
        pts = self._cloud_preview_by_id.get(obj_id)
        if pts is not None:
            self.window.viewer.show_point_cloud(obj_id, pts)


    def on_visibility_changed(self, obj_id: str, visible: bool) -> None:
        self.window.viewer.set_visible(obj_id, visible)
    

    def reconstruct(self) -> None:
        if not self._current_cloud_id:
            QMessageBox.information(self.window, "Нет данных", "Сначала импортируйте облако точек.")
            return

        try:
            self.window.set_status("Реконструкция…", progress=20)

            # ВАЖНО: мы должны реконструировать из файла, как CLI, чтобы совпало поведение
            # Сохрани текущий pcd во временный ply (или храни путь при импорте)
            import tempfile, os
            import open3d as o3d

            item_pcd = self._cloud_o3d_by_id[self._current_cloud_id]
            tmp_dir = tempfile.mkdtemp(prefix="laser_recon_")
            tmp_ply = os.path.join(tmp_dir, "input_work.ply")
            o3d.io.write_point_cloud(tmp_ply, item_pcd)

            args = SimpleNamespace(
                input=tmp_ply,
                output=None,
                method="bpa",          # потом сделаем выбор в UI
                max_points=1_500_000,
                voxel_size=0.005,      # как в консоли
                depth=10,
                keep_ratio=0.05,
                show_steps=False,
                save_steps=False,
                show_final=False,
            )

            mesh, out_path = run_reconstruction(args, return_mesh=True)

            mesh_id = str(uuid.uuid4())
            self._mesh_by_id[mesh_id] = mesh
            self._current_mesh_id = mesh_id

            import numpy as np
            V = np.asarray(mesh.vertices)
            F = np.asarray(mesh.triangles)
            self.window.viewer.show_mesh(mesh_id, V, F)

            self.window.set_status("Готово: мэш построен", progress=100)

        except Exception as e:
            self.window.set_status("Ошибка реконструкции", progress=0)
            QMessageBox.critical(self.window, "Ошибка реконструкции", str(e))
    
    def apply_filter(self) -> None:
        if not self._current_cloud_id:
            QMessageBox.information(self.window, "Нет данных", "Сначала импортируйте облако точек.")
            return

        try:
            import numpy as np
            from app.Infrastructure.filtering_adapter import filter_pcd

            self.window.set_status("Фильтрация…", progress=20)

            pcd = self._cloud_o3d_by_id[self._current_cloud_id]

            # параметры как в твоём filtering.py
            pcd2 = filter_pcd(
                pcd,
                nb_stat=28,
                nb_radial=12,
                std=0.5,
                rad=0.08,
                vox_size=0.05,
            )

            self._cloud_o3d_by_id[self._current_cloud_id] = pcd2

            pts = np.asarray(pcd2.points)
            if pts.shape[0] == 0:
                raise RuntimeError("Пустое облако после фильтрации")

            # превью для Viewer
            max_preview = 2_000_000
            if pts.shape[0] > max_preview:
                idx = np.random.choice(pts.shape[0], size=max_preview, replace=False)
                pts_preview = pts[idx]
            else:
                pts_preview = pts

            pts_preview = pts_preview - pts_preview.mean(axis=0)
            pts_preview = pts_preview.astype(np.float32, copy=False)
            self._cloud_preview_by_id[self._current_cloud_id] = pts_preview

            # обновить показ
            self.window.viewer.show_point_cloud(self._current_cloud_id, pts_preview)

            self.window.set_status(f"Фильтрация завершена (точек: {len(pcd2.points):,})", progress=100)

        except Exception as e:
            self.window.set_status("Ошибка фильтрации", progress=0)
            QMessageBox.critical(self.window, "Ошибка фильтрации", str(e))


