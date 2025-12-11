from typing import Tuple

import numpy as np
import open3d as o3d

"""
Нормализовать облако точек: центрировать и масштабировать в единичный куб.

Зачем это нужно:
    Нормализация позволяет:
    - избежать численных проблем при больших координатах (миллионы);
    - сделать параметры алгоритмов (depth, радиусы, voxel_size)
            более понятными и одинаковыми для разных сцен.

Шаги:
    1. Строится axis-aligned bounding box (AABB).
    2. Вычисляется центр `center` и максимальный размер ребра `scale`.
    3. Все точки сдвигаются так, чтобы центр стал в (0, 0, 0).
    4. Если `scale > 0`, облако масштабируется так, чтобы
        максимальный размер AABB стал равен 1.

Параметры:
    pcd (PointCloud):
        Входное облако точек в исходных координатах.

Возвращает:
    tuple:
        - pcd_norm (PointCloud):
            Нормализованное облако.
        - center (np.ndarray shape (3,)):
            Центр исходного AABB. Позволяет потом вернуть объект
            обратно на место.
        - scale (float):
            Масштаб (максимальное ребро AABB). Используется для
            обратного масштабирования результата.

Важно:
    Функция НЕ модифицирует исходный объект `pcd` в вызывающем коде,
    а работает с его копией (через translate/scale, которые в Open3D
    возвращают новый объект).
"""
def normalize_point_cloud(
    pcd: o3d.geometry.PointCloud,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, float]:
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    scale = bbox.get_max_extent()

    print(f"[Normalize] bbox center = {center}, max_extent = {scale}")

    pcd_norm = pcd.translate(-center)
    if scale > 0:
        pcd_norm = pcd_norm.scale(1.0 / scale, center=(0.0, 0.0, 0.0))

    return pcd_norm, np.asarray(center), float(scale)

"""
Предобработка облака точек перед реконструкцией.

Состоит из двух основных шагов:
    1. Опциональный воксельный даунсэмплинг (уменьшение числа точек).
    2. Оценка нормалей для каждой точки.

Логика:
    - Если `voxel_size > 0`:
        облако делится на трёхмерную сетку кубиков (вокселов) с ребром
        `voxel_size` (в НОРМАЛИЗОВАННЫХ координатах). В каждом вокселе
        оставляется одна “представительная” точка.
    - Если `voxel_size <= 0`:
        даунсэмплинг пропускается и используется полное облако.
    - Затем по полученному облаку вычисляются нормали. Радиус поиска
        для нормалей выбирается как 2% от максимального размера сцены.

Параметры:
    pcd (PointCloud):
        Нормализованное облако (обычно после `normalize_point_cloud`).
    voxel_size (float):
        Размер вокселя. Если 0 или отрицателен — даунсэмплинг отключён.

Возвращает:
    o3d.geometry.PointCloud:
        Обработанное облако с (возможно) уменьшенным числом точек
        и рассчитанными нормалями.

Дополнительно:
    Функция печатает в консоль количество точек до и после даунсэмплинга,
    а также использованный радиус при оценке нормалей.
"""
def preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.02,
) -> o3d.geometry.PointCloud:
    n_before = len(pcd.points)
    if voxel_size is not None and voxel_size > 0:
        print(f"[Preprocess] Voxel downsample with size = {voxel_size}")
        pcd_proc = pcd.voxel_down_sample(voxel_size)
    else:
        print("[Preprocess] Skip voxel downsample (use full cloud)")
        pcd_proc = pcd

    print(f"[Preprocess] Points: {n_before} -> {len(pcd_proc.points)}")

    # Радиус для нормалей подбираем по размеру сцены
    bbox = pcd_proc.get_axis_aligned_bounding_box()
    extent = bbox.get_max_extent()
    radius = extent * 0.02  # 2% от размера сцены

    print(f"[Preprocess] Estimating normals (radius={radius}) ...")
    pcd_proc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=30,
        )
    )
    pcd_proc.orient_normals_consistent_tangent_plane(50)
    return pcd_proc

    """
    Мягко сглаживает поверхность (Taubin) без сильного сплющивания формы.
    iterations: 3–8 обычно достаточно для стен.
    """
def smooth_mesh_soft(
    mesh: o3d.geometry.TriangleMesh,
    iterations: int = 5,
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh

    mesh = mesh.filter_smooth_taubin(
        number_of_iterations=iterations
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh

