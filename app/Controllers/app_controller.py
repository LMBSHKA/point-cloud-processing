from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Callable, Optional
from types import SimpleNamespace
from app.Controllers.reconstruction_controller import run_reconstruction

from app.UI.widgets.instructions import SecondaryWindow

from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QObject, Signal, Slot, QThread, Qt

from app.UI.widgets.scene_tree import SceneItem
from app.Infrastructure.pointcloud_io import load_point_cloud_any
from app.Infrastructure.filtering_adapter import FilterSettings
#from services.io_service import IOService


@dataclass
class ImportedObject:
    obj_id: str
    name: str
    kind: str
    path: str

class FilterWorker(QObject):
    """
    Фоновая фильтрация облака, чтобы UI не зависал.
    """
    finished = Signal(object, object)  # pcd_filtered (o3d), pts_preview (np.ndarray)
    failed = Signal(str)

    def __init__(self, pcd: o3d.geometry.PointCloud, settings: FilterSettings | None) -> None:
        super().__init__()
        self._pcd = pcd
        self._settings = settings

    @Slot()
    def run(self) -> None:
        try:
            import numpy as np
            from app.Infrastructure.filtering_adapter import filter_pcd

            pcd_local = o3d.geometry.PointCloud(self._pcd)

            pcd2 = filter_pcd(pcd_local, settings=self._settings)

            pts = np.asarray(pcd2.points)
            if pts.shape[0] == 0:
                raise RuntimeError("Пустое облако после фильтрации")

            # превью для Viewer (как у тебя)
            max_preview = 2_000_000
            if pts.shape[0] > max_preview:
                idx = np.random.choice(pts.shape[0], size=max_preview, replace=False)
                pts_preview = pts[idx]
            else:
                pts_preview = pts

            pts_preview = pts_preview - pts_preview.mean(axis=0)
            pts_preview = pts_preview.astype(np.float32, copy=False)

            self.finished.emit(pcd2, pts_preview)
        except Exception as e:
            self.failed.emit(str(e))


class ReconstructionWorker(QObject):
    """
    Фоновая реконструкция меша, чтобы UI не зависал.

    Важно: любые обращения к UI (Viewer/Tree/StatusBar) делать только
    в главном потоке через сигналы.
    """

    finished = Signal(object, str)  # mesh, out_path
    failed = Signal(str)

    def __init__(self, args) -> None:
        super().__init__()
        self._args = args

    @Slot()
    def run(self) -> None:
        try:
            mesh, out_path = run_reconstruction(self._args, return_mesh=True)
            self.finished.emit(mesh, str(out_path))
        except Exception as e:
            self.failed.emit(str(e))


class AppController(QObject):
    """
    Клей: UI -> сервисы -> UI.
    """
    def __init__(self, pointcloud_io: Callable[[Path, Optional[int]], o3d.geometry.PointCloud]) -> None:
        super().__init__()
        self.load_point_cloud_any = pointcloud_io

        self._cloud_o3d_by_id: dict[str, o3d.geometry.PointCloud] = {}
        self._cloud_preview_by_id: dict[str, object] = {}
        self._current_cloud_id: str | None = None

        self._mesh_by_id: dict[str, o3d.geometry.TriangleMesh] = {}
        self._current_mesh_id: str | None = None

        self._selected_id: str | None = None
        self._instructions_window = None  # чтобы окно не закрывалось сборщиком мусора
        self._cloud_path_by_id: dict[str, str] = {}
        self._filter_settings: FilterSettings | None = None  # позже UI будет сюда писать

        # держим ссылки, чтобы потоки/воркеры не уничтожались GC
        self._recon_thread: QThread | None = None
        self._recon_worker: ReconstructionWorker | None = None

        self._filter_thread: QThread | None = None
        self._filter_worker: FilterWorker | None = None
        self._filter_target_cloud_id: str | None = None

    @Slot(object, object)
    def _on_filter_done(self, pcd2, pts_preview) -> None:
        try:
            cloud_id = self._filter_target_cloud_id
            if not cloud_id:
                return

            self._cloud_o3d_by_id[cloud_id] = pcd2
            self._cloud_preview_by_id[cloud_id] = pts_preview

            # обновить показ (в GUI потоке)
            self.window.viewer.show_point_cloud(cloud_id, pts_preview)
            self._show_only(cloud_id)

            self.window.set_status(f"Фильтрация завершена (точек: {len(pcd2.points):,})", progress=100)
        except Exception as e:
            self.window.set_status("Ошибка фильтрации", progress=0)
            QMessageBox.critical(self.window, "Ошибка фильтрации", str(e))
        finally:
            self._restore_ui_after_filter()

    @Slot(str)
    def _on_filter_failed(self, message: str) -> None:
        self.window.set_status("Ошибка фильтрации", progress=0)
        QMessageBox.critical(self.window, "Ошибка фильтрации", message)
        self._restore_ui_after_filter()

    def _restore_ui_after_filter(self) -> None:
        try:
            self.window.progress.setRange(0, 100)
            self.window.progress.setValue(0)
        except Exception:
            pass
        self.window.act_filter.setEnabled(True)

    @Slot()
    def _cleanup_filter_thread(self) -> None:
        self._filter_worker = None
        self._filter_thread = None
        self._filter_target_cloud_id = None


    def bind(self, window) -> None:
        """
        Подписка на кнопки/сигналы UI.
        window: MainWindow (из main_window.py)
        """
        self.window = window

        window.act_import_model.triggered.connect(self.import_model)
        window.act_import_cloud.triggered.connect(self.import_cloud)
        window.tree.item_selected.connect(self.on_tree_selected)
        window.tree.visibility_changed.connect(self.on_visibility_changed)
        window.act_build_mesh.triggered.connect(self.reconstruct)
        window.act_filter.triggered.connect(self.apply_filter)
        window.act_select.triggered.connect(self.segment_cloud)
        
    def import_model(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self.window,
            "Импорт модели",
            "",
            "3D files (*.pcd *.ply *.stl *.obj *.off *.glb *.gltf);;All files (*.*)"
        )
        if not path_str:
            return

        path = Path(path_str)
        ext = path.suffix.lower()
        name = path.name
        obj_id = str(uuid.uuid4())

        try:
            # 1) PCD — это point cloud
            if ext == ".pcd":
                pcd = self.load_point_cloud_any(path, max_points=None)
                pts = np.asarray(pcd.points, dtype=np.float32)
                if pts.size == 0:
                    raise RuntimeError("Пустое облако после загрузки")

                self._cloud_o3d_by_id[obj_id] = pcd
                self._cloud_preview_by_id[obj_id] = pts
                self.window.tree.add_model(SceneItem(obj_id=obj_id, name=name, kind="cloud", path=str(path)))
                self.window.viewer.show_point_cloud(obj_id, pts)
                self._selected_id = obj_id
                self._current_cloud_id = obj_id
                self._current_mesh_id = None

                self._activate_new_object(obj_id)
                return

            # 2) Сначала пробуем mesh (особенно важно для PLY)
            mesh = o3d.io.read_triangle_mesh(str(path))
            if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
                self._mesh_by_id[obj_id] = mesh

                V = np.asarray(mesh.vertices, dtype=np.float32)
                F = np.asarray(mesh.triangles, dtype=np.int32)

                # Передаём vertex normals из Open3D в Viewer3D (это убирает визуальные швы в VTK)
                if not mesh.has_vertex_normals():
                    try:
                        mesh.compute_vertex_normals()
                    except Exception:
                        pass
                N = np.asarray(mesh.vertex_normals, dtype=np.float32) if mesh.has_vertex_normals() else None

                self.window.tree.add_model(SceneItem(obj_id=obj_id, name=name, kind="mesh", path=str(path)))
                self.window.viewer.show_mesh(obj_id, V, F, N)
                self._selected_id = obj_id
                self._current_mesh_id = obj_id
                self._current_cloud_id = None

                self._activate_new_object(obj_id)
                return

            # 3) Если не mesh — читаем как облако точек
            pcd = o3d.io.read_point_cloud(str(path))
            if pcd is None or len(pcd.points) == 0:
                raise RuntimeError("Файл не похож ни на mesh, ни на point cloud")

            pts = np.asarray(pcd.points, dtype=np.float32)
            self._cloud_o3d_by_id[obj_id] = pcd
            self._cloud_preview_by_id[obj_id] = pts

            self.window.tree.add_model(SceneItem(obj_id=obj_id, name=name, kind="cloud", path=str(path)))
            self.window.viewer.show_point_cloud(obj_id, pts)

            self._selected_id = obj_id
            #self._current_cloud_id = obj_id   # только если это облако
            #self._current_mesh_id = obj_id    # только если это меш


        except Exception as e:
            QMessageBox.critical(self.window, "Ошибка импорта модели", str(e))

    def _show_only(self, obj_id: str) -> None:
        # скрыть всё, что уже есть в viewer
        for oid in list(self._cloud_preview_by_id.keys()):
            self.window.viewer.set_visible(oid, oid == obj_id)
        for oid in list(self._mesh_by_id.keys()):
            self.window.viewer.set_visible(oid, oid == obj_id)

    def segment_cloud(self) -> None:
        if not self._current_cloud_id:
            QMessageBox.information(self.window, "Нет данных", "Сначала импортируйте облако точек.")
            return

        try:
            pcd = self._cloud_o3d_by_id[self._current_cloud_id]

            instructions_window = SecondaryWindow()
            instructions_window.show()

            o3d.visualization.draw_geometries_with_editing(
                [pcd],
                window_name="Сегментация",
                width=1000,
                height=800
            )

        except Exception as e:
            self.window.set_status("Ошибка сегментации", progress=0)
            QMessageBox.critical(self.window, "Ошибка сегментации", str(e))


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
            self._cloud_path_by_id[obj_id] = path
            name = os.path.basename(path)

            self._cloud_o3d_by_id[obj_id] = pcd
            self._cloud_preview_by_id[obj_id] = pts_preview
            self._current_cloud_id = obj_id

            self.window.tree.add_cloud(SceneItem(obj_id=obj_id, name=name, kind="cloud", path=path))
            self.window.viewer.show_point_cloud(obj_id, pts_preview)

            self._selected_id = obj_id
            self._current_cloud_id = obj_id
            self._current_mesh_id = None

            self._activate_new_object(obj_id)

            self.window.set_status(f"Импортировано: {name} (точек: {len(pcd.points):,})", progress=100)


        except Exception as e:
            self.window.set_status("Ошибка импорта", progress=0)
            QMessageBox.critical(self.window, "Ошибка импорта", str(e))
    
    def _activate_new_object(self, obj_id: str) -> None:
        # 1) скрыть все остальные в viewer
        self._show_only(obj_id)

        # 2) в дереве оставить галочку только у нового и выделить его
        self.window.tree.set_only_visible_checked(obj_id)
        self.window.tree.select_object(obj_id)

        # 3) камера на новый объект
        if hasattr(self.window.viewer, "focus_on"):
            self.window.viewer.focus_on(obj_id)


    def on_tree_selected(self, obj_id: str) -> None:
        self._selected_id = obj_id

        pts = self._cloud_preview_by_id.get(obj_id)
        if pts is not None:
            self._current_cloud_id = obj_id
            self._current_mesh_id = None
            self.window.viewer.show_point_cloud(obj_id, pts)
            self._show_only(obj_id)
            return

        mesh = self._mesh_by_id.get(obj_id)
        if mesh is not None:
            self._current_mesh_id = obj_id
            self._current_cloud_id = None

            V = np.asarray(mesh.vertices)
            F = np.asarray(mesh.triangles)
            N = np.asarray(mesh.vertex_normals)  # ← ВОТ ЭТОГО НЕ ХВАТАЛО

            if V.size and F.size:
                self.window.viewer.show_mesh(obj_id, V, F, N)
                self._show_only(obj_id)
            return


    def on_visibility_changed(self, obj_id: str, visible: bool) -> None:
        self.window.viewer.set_visible(obj_id, visible)
    

    def reconstruct(self) -> None:
        if not self._current_cloud_id:
            QMessageBox.information(self.window, "Нет данных", "Сначала импортируйте облако точек.")
            return

        # не даём запускать несколько реконструкций одновременно
        if self._recon_thread is not None and self._recon_thread.isRunning():
            QMessageBox.information(self.window, "Реконструкция", "Реконструкция уже выполняется.")
            return

        src_path = self._cloud_path_by_id.get(self._current_cloud_id)
        if not src_path:
            QMessageBox.critical(self.window, "Ошибка реконструкции", "Не найден исходный путь облака (cloud_path_by_id).")
            return

        # Параметры пока по умолчанию (позже UI сможет менять)
        args = SimpleNamespace(
            input=src_path,
            output=None,
            method="bpa",
            max_points=1_500_000,
            voxel_size=0.005,
            depth=10,
            keep_ratio=0.05,
            show_steps=False,
            save_steps=False,
            show_final=False,
        )

        # UI: показываем "занято" и блокируем кнопку
        try:
            self.window.progress.setRange(0, 0)  # бесконечный прогресс
        except Exception:
            pass
        self.window.set_status("Реконструкция…", progress=0)
        self.window.act_build_mesh.setEnabled(False)

        # --- Запускаем реконструкцию в отдельном потоке ---
        self._recon_thread = QThread()
        self._recon_worker = ReconstructionWorker(args)
        self._recon_worker.moveToThread(self._recon_thread)

        self._recon_thread.started.connect(self._recon_worker.run)
        self._recon_worker.finished.connect(self._on_reconstruction_done, Qt.QueuedConnection)
        self._recon_worker.failed.connect(self._on_reconstruction_failed, Qt.QueuedConnection)

        # корректное завершение потока
        self._recon_worker.finished.connect(self._recon_thread.quit)
        self._recon_worker.failed.connect(self._recon_thread.quit)
        self._recon_thread.finished.connect(self._cleanup_reconstruction_thread)

        self._recon_thread.start()

    @Slot(object, str)
    def _on_reconstruction_done(self, mesh, out_path: str) -> None:
        try:
            mesh_id = str(uuid.uuid4())
            self._mesh_by_id[mesh_id] = mesh
            self._current_mesh_id = mesh_id
            self._selected_id = mesh_id

            self.window.tree.add_model(SceneItem(
                obj_id=mesh_id,
                name=f"mesh_{mesh_id[:8]}.ply",
                kind="mesh",
                path=str(out_path or "")
            ))

            import numpy as np
            V = np.asarray(mesh.vertices)
            F = np.asarray(mesh.triangles)
            if not mesh.has_vertex_normals():
                try:
                    mesh.compute_vertex_normals()
                except Exception:
                    pass
            N = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None

            self.window.viewer.show_mesh(mesh_id, V, F, N)
            self._activate_new_object(mesh_id)
            self._show_only(mesh_id)

            self.window.set_status("Готово: мэш построен", progress=100)
        except Exception as e:
            self.window.set_status("Ошибка реконструкции", progress=0)
            QMessageBox.critical(self.window, "Ошибка реконструкции", str(e))
        finally:
            self._restore_ui_after_reconstruction()

    @Slot(str)
    def _on_reconstruction_failed(self, message: str) -> None:
        self.window.set_status("Ошибка реконструкции", progress=0)
        QMessageBox.critical(self.window, "Ошибка реконструкции", message)
        self._restore_ui_after_reconstruction()

    def _restore_ui_after_reconstruction(self) -> None:
        # вернуть обычный прогрессбар и кнопку
        try:
            self.window.progress.setRange(0, 100)
            self.window.progress.setValue(0)
        except Exception:
            pass
        self.window.act_build_mesh.setEnabled(True)

    @Slot()
    def _cleanup_reconstruction_thread(self) -> None:
        # освобождаем ссылки после остановки потока
        self._recon_worker = None
        self._recon_thread = None

    
    def apply_filter(self) -> None:
        if not self._current_cloud_id:
            QMessageBox.information(self.window, "Нет данных", "Выберите облако точек в дереве.")
            return

        # не даём запускать несколько фильтраций одновременно
        if self._filter_thread is not None and self._filter_thread.isRunning():
            QMessageBox.information(self.window, "Фильтрация", "Фильтрация уже выполняется.")
            return

        cloud_id = self._current_cloud_id
        pcd = self._cloud_o3d_by_id.get(cloud_id)
        if pcd is None:
            QMessageBox.critical(self.window, "Ошибка фильтрации", "Не найдено облако по выбранному id.")
            return

        # UI: показываем "занято" и блокируем кнопку
        try:
            self.window.progress.setRange(0, 0)  # бесконечный прогресс
        except Exception:
            pass
        self.window.set_status("Фильтрация…", progress=0)
        self.window.act_filter.setEnabled(False)

        self._filter_target_cloud_id = cloud_id

        # --- Запускаем фильтрацию в отдельном потоке ---
        self._filter_thread = QThread()
        self._filter_worker = FilterWorker(pcd, self._filter_settings)
        self._filter_worker.moveToThread(self._filter_thread)

        self._filter_thread.started.connect(self._filter_worker.run)

        # важно: QueuedConnection, чтобы UI-слоты гарантированно были в GUI потоке
        self._filter_worker.finished.connect(self._on_filter_done, Qt.QueuedConnection)
        self._filter_worker.failed.connect(self._on_filter_failed, Qt.QueuedConnection)

        # корректное завершение
        self._filter_worker.finished.connect(self._filter_thread.quit)
        self._filter_worker.failed.connect(self._filter_thread.quit)
        self._filter_thread.finished.connect(self._cleanup_filter_thread)
        self._filter_thread.setPriority(QThread.LowPriority)
        self._filter_thread.start()
