from typing import Optional, List

import numpy as np
import open3d as o3d

"""
Реконструкция поверхности по облаку точек методом Пуассона (Poisson Surface Reconstruction).

Шаги:
    1. Вызывается Open3D-функция:
        `TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)`,
        которая возвращает:
            - треугольный меш,
            - массив плотностей вершин `densities`.
    2. По массиву `densities` оцениваются статистики (min, max).
    3. Вычисляется порог `thresh` как квантиль (по `keep_ratio`), и
        вершины с плотностью ниже порога удаляются. Это убирает “хвосты”
        и шум на границах сцены.
    4. Запускается `remove_unreferenced_vertices()`, чтобы очистить
        индексы после удаления вершин.

Параметры:
    pcd (PointCloud):
        Облако точек с нормалями (обычно после `preprocess_point_cloud`).
    depth (int):
        Глубина октодерева в алгоритме Пуассона. Большее значение даёт
        более детализированную сетку, но потребляет больше памяти и времени.
    keep_ratio (float):
        Доля вершин, которые будут отброшены по плотности. Например,
        0.05 означает, что будут удалены примерно 5% вершин с
        наименьшей плотностью.

Возвращает:
    o3d.geometry.TriangleMesh:
        Очищенный треугольный меш после Poisson-реконструкции.

Рекомендации:
    - Для тонких объектов (стены, плиты) Poisson может давать “лишний объём”.
    - Для больших сцен имеет смысл уменьшать `depth` (8–10) и ограничивать
        количество входных точек.
"""
def reconstruct_poisson(
    pcd: o3d.geometry.PointCloud,
    depth: int = 10,
    keep_ratio: float = 0.05,
) -> o3d.geometry.TriangleMesh:
    print(f"[Poisson] Running Poisson reconstruction (depth={depth}) ...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    densities = np.asarray(densities)
    print("[Poisson] Densities stats:",
          f"min={densities.min():.4f}, max={densities.max():.4f}")

    thresh = np.quantile(densities, keep_ratio)
    print(f"[Poisson] Removing vertices with density < {thresh:.4f}")
    vertices_to_keep = densities > thresh
    mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
    mesh.remove_unreferenced_vertices()

    print(f"[Poisson] Result mesh: |V|={len(mesh.vertices)}, |F|={len(mesh.triangles)}")
    return mesh

"""
Реконструкция поверхности по облаку точек методом Ball Pivoting Algorithm (BPA).

Идея:
    Представляем, что по облаку катится сфера радиуса R. Каждый раз,
    когда сфера “опирается” на три точки, образуется треугольник.
    Повторяя этот процесс с разными радиусами, можно восстановить
    поверхность по плотному облаку.

Шаги:
    1. Если `radii` не переданы:
        - вычисляются расстояния до ближайшего соседа для каждой точки;
        - берётся среднее расстояние `mean_dist`;
        - формируется список радиусов:
            [2 * mean_dist, 4 * mean_dist, 8 * mean_dist].
    2. Запускается Open3D-функция
        `TriangleMesh.create_from_point_cloud_ball_pivoting(...)`.
    3. После реконструкции выполняется очистка меша:
        - удаление дубликатов вершин;
        - удаление дубликатов треугольников;
        - удаление неиспользуемых вершин;
        - удаление вырожденных треугольников.

Параметры:
    pcd (PointCloud):
        Облако точек с нормалями. Желательно достаточно плотное и
        без сильных дыр.
    radii (list[float] | None):
        Список радиусов шаров для BPA. Если None — подбираются автоматически
        по средней дистанции между точками.

Возвращает:
    o3d.geometry.TriangleMesh:
        Реконструированный и очищенный треугольный меш.

Ограничения:
    - Плохо работает на очень разреженных облаках и больших сценах
        с малым числом точек.
    - Чувствителен к шуму и дыркам; иногда требует предварительной
        фильтрации и сегментации.
"""
def reconstruct_bpa(
    pcd: o3d.geometry.PointCloud,
    radii: Optional[List[float]] = None,
) -> o3d.geometry.TriangleMesh:
    if radii is None:
        print("[BPA] Estimating radii from nearest neighbor distances ...")
        dists = np.asarray(pcd.compute_nearest_neighbor_distance())
        mean_dist = dists.mean()
        print(f"[BPA] mean NN distance = {mean_dist}")
        # было: [2,4,8] * mean_dist
        base = mean_dist * 1.5
        radii = [base, base * 2.0, base * 3.0]

    print(f"[BPA] Running Ball Pivoting with radii={radii} ...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh

def clean_mesh_o3d(mesh: o3d.geometry.TriangleMesh, *, smooth_iters: int = 1) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    # критично для углов и "наложений"
    mesh.remove_non_manifold_edges()

    mesh.compute_vertex_normals()

    if smooth_iters > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=int(smooth_iters))
        mesh.compute_vertex_normals()

    return mesh

