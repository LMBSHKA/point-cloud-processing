from typing import Optional
from collections import defaultdict

import numpy as np
import open3d as o3d

"""
    Выпрямляет торцы только для "тонких" (slab-like) объектов.

    - Ищет ось минимального размера (толщины плиты).
    - Если толщина / макс. размер < thickness_ratio_threshold — считаем, что это плита
      (стена, перекрытие и т.п.) и прижимаем вершины к двум плоскостям.
    - Для "объёмных" объектов (дом, комната), где все размеры сопоставимы,
      функция НИЧЕГО не делает и просто возвращает исходный меш.

    Параметры:
        snap_ratio:         доля толщины, в пределах которой вершины
                            прилипают к плоскости (0.1–0.25 обычно).
        thickness_ratio_threshold:
                            порог "тонкости". Если min_extent / max_extent
                            > этого значения, выпрямление не применяется.
        force_axis:         0,1,2 — принудительно выбрать ось X/Y/Z как "толщину",
                            игнорируя автоматический анализ. Если None — авторежим.
"""
def straighten_slab_sides(
    mesh: o3d.geometry.TriangleMesh,
    snap_ratio: float = 0.15,
    thickness_ratio_threshold: float = 0.25,
    force_axis: Optional[int] = None,
) -> o3d.geometry.TriangleMesh:
    if len(mesh.vertices) == 0:
        return mesh

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()        # размеры по X,Y,Z
    min_extent = float(np.min(extent))
    max_extent = float(np.max(extent))

    # Выбираем ось "толщины"
    if force_axis is None:
        thin_axis = int(np.argmin(extent))
    else:
        thin_axis = int(force_axis)

    # Если не принудительный режим и объект не выглядит тонким — выходим
    if force_axis is None:
        ratio = min_extent / max_extent if max_extent > 0 else 1.0
        print(f"[Straighten] slab check: min/max = {ratio:.3f}")
        if ratio > thickness_ratio_threshold:
            print("[Straighten] Object is not slab-like, skipping straightening.")
            return mesh
    else:
        print(f"[Straighten] Forcing axis {thin_axis} as thickness.")

    thickness = extent[thin_axis]
    if thickness <= 0:
        return mesh

    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()

    # Плоскости-торцы
    min_plane = min_bound[thin_axis]
    max_plane = max_bound[thin_axis]

    tol = thickness * snap_ratio

    verts = np.asarray(mesh.vertices)
    coord = verts[:, thin_axis]

    mask_min = coord < (min_plane + tol)
    mask_max = coord > (max_plane - tol)

    print(
        f"[Straighten] axis={thin_axis}, thickness={thickness:.4f}, "
        f"snap_ratio={snap_ratio}, tol={tol:.4f}, "
        f"min_snap={mask_min.sum()}, max_snap={mask_max.sum()}"
    )

    verts[mask_min, thin_axis] = min_plane
    verts[mask_max, thin_axis] = max_plane

    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.compute_vertex_normals()
    return mesh

    """
    Зашивает дырки в меше:
    - находит граничные рёбра (у которых только один треугольник),
    - собирает по ним замкнутые петли,
    - для каждой маленькой петли строит "веер" треугольников к центру.

    max_hole_vertices:
        Если None — ограничение по количеству вершин не используется.
        Если число — зашиваем только петли с числом вершин <= max_hole_vertices.

    max_area_ratio:
        Доп. фильтр по площади петли.
        Площадь петли в проекции должна быть < max_area_ratio * (max_extent^2),
        где max_extent — максимальный размер bbox модели.
        Например: 0.0005 означает "дырка меньше 0.05% от площади модели".
    """
def fill_small_holes(
    mesh: o3d.geometry.TriangleMesh,
    max_hole_vertices: Optional[int] = None,
    max_area_ratio: Optional[float] = None,
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles, dtype=np.int32)

    # bbox для нормировки площади
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    max_extent = float(np.max(extent))
    if max_extent <= 0:
        max_extent = 1.0

    # 1. считаем, сколько раз встречается каждое ребро
    edge_count = {}
    for tri in tris:
        for i in range(3):
            a = int(tri[i])
            b = int(tri[(i + 1) % 3])
            if a > b:
                a, b = b, a
            edge = (a, b)
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # 2. граничные рёбра (те, что встречаются 1 раз)
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    if not boundary_edges:
        return mesh

    # 3. собираем петли по границе
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    visited = set()
    loops = []

    for start in adj.keys():
        if start in visited:
            continue

        loop = [start]
        visited.add(start)
        prev = None
        cur = start

        while True:
            neigh = [n for n in adj[cur] if n != prev]
            if not neigh:
                break
            nxt = neigh[0]
            if nxt == start:
                break
            if nxt in visited:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt

        if len(loop) >= 3:
            loops.append(loop)

    if not loops:
        return mesh

    new_vertices = []
    new_tris = []

    for loop in loops:
        if len(loop) < 3:
            continue

        # 3.1. Фильтр по количеству вершин
        if max_hole_vertices is not None and len(loop) > max_hole_vertices:
            continue

        loop_pts = verts[loop]

        # 3.2. Фильтр по площади (если задан)
        if max_area_ratio is not None:
            centroid = loop_pts.mean(axis=0)
            P = loop_pts - centroid
            # PCA-базис
            U, S, Vt = np.linalg.svd(P, full_matrices=False)
            u = Vt[0, :]
            v = Vt[1, :]

            pts2d = np.stack([P @ u, P @ v], axis=1)
            x = pts2d[:, 0]
            y = pts2d[:, 1]
            area = 0.5 * abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

            if area > (max_extent * max_extent * max_area_ratio):
                # дырка слишком большая для этого "мелкого" заполнителя
                continue

        # 4. создаём центр и веер треугольников
        center = loop_pts.mean(axis=0)
        center_index = len(verts) + len(new_vertices)
        new_vertices.append(center)

        for i in range(len(loop)):
            a = loop[i]
            b = loop[(i + 1) % len(loop)]
            new_tris.append([a, b, center_index])

    if not new_tris:
        print("[FillHoles] no small holes passed filters")
        return mesh

    verts_out = np.vstack([verts, np.asarray(new_vertices, dtype=np.float64)])
    tris_out = np.vstack([tris, np.asarray(new_tris, dtype=np.int32)])

    mesh.vertices = o3d.utility.Vector3dVector(verts_out)
    mesh.triangles = o3d.utility.Vector3iVector(tris_out)

    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    print(f"[FillHoles] Filled {len(new_tris)} triangles in {len(loops)} loops "
          f"(new verts: {len(new_vertices)})")
    return mesh

    """
    Находит все open-boundary петли по рёбрам и зашивает их веером треугольников.
    Не использует mesh.get_boundaries (подходит для старых версий Open3D).
    """
def fill_all_holes(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)

    if tris.size == 0:
        print("[fill_all_holes] no triangles")
        return mesh

    # --- 1. Собираем рёбра и считаем, сколько раз каждое встречается ---
    edges = np.vstack([
        tris[:, [0, 1]],
        tris[:, [1, 2]],
        tris[:, [2, 0]],
    ])
    edges_sorted = np.sort(edges, axis=1)  # чтобы (i,j) и (j,i) были одинаковыми
    uniq_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)

    # Рёбра, принадлежащие только одному треугольнику → границы
    boundary_edges = uniq_edges[counts == 1]
    print(f"[fill_all_holes] boundary edges: {len(boundary_edges)}")

    if len(boundary_edges) == 0:
        print("[fill_all_holes] no open boundaries found")
        return mesh

    # --- 2. Строим граф смежности по граничным рёбрам ---
    adj = defaultdict(list)
    for u, v in boundary_edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))

    def edge_key(a, b):
        return (min(a, b), max(a, b))

    visited_edges = set()
    loops = []

    # --- 3. Выделяем петли (каждый раз идём по не посещённым рёбрам) ---
    for u, v in boundary_edges:
        u = int(u)
        v = int(v)
        e0 = edge_key(u, v)
        if e0 in visited_edges:
            continue

        start = u
        loop = [start]
        cur = start
        prev = None

        while True:
            # ищем соседей по ещё НЕ посещённым рёбрам
            next_vertex = None
            for nb in adj[cur]:
                e = edge_key(cur, nb)
                if e not in visited_edges:
                    next_vertex = nb
                    break

            if next_vertex is None:
                # не смогли продолжить — разорванная граница
                break

            visited_edges.add(edge_key(cur, next_vertex))
            cur = next_vertex

            if cur == start:
                # замкнули петлю
                break
            else:
                loop.append(cur)

        if len(loop) >= 3:
            loops.append(loop)

    print(f"[fill_all_holes] loops found: {len(loops)}")

    if not loops:
        print("[fill_all_holes] no valid loops -> nothing to fill")
        return mesh

    # --- 4. Для каждой петли добавляем центр и веер треугольников ---
    tris_old = tris
    new_vertices = []
    new_tris = []

    for loop in loops:
        if len(loop) < 3:
            continue

        loop_idx = np.array(loop, dtype=int)
        loop_pts = verts[loop_idx]

        center = loop_pts.mean(axis=0)
        center_idx = len(verts) + len(new_vertices)
        new_vertices.append(center)

        for i in range(len(loop_idx)):
            a = int(loop_idx[i])
            b = int(loop_idx[(i + 1) % len(loop_idx)])
            new_tris.append([center_idx, a, b])

    if new_vertices:
        verts_new = np.vstack([verts, np.asarray(new_vertices, dtype=np.float64)])
        tris_new = np.vstack([tris_old, np.asarray(new_tris, dtype=np.int32)])

        mesh.vertices = o3d.utility.Vector3dVector(verts_new)
        mesh.triangles = o3d.utility.Vector3iVector(tris_new)

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        print(f"[fill_all_holes] added {len(new_vertices)} center verts, "
              f"{len(new_tris)} triangles")

    return mesh

    """
    Удаляет треугольники, у которых есть ОЧЕНЬ длинное ребро
    (длина > max_edge_ratio * диагонали bbox).
    Это как раз те лучи, которые тянут плоскость через пол-модели.
    """
def remove_long_triangles(
    mesh: o3d.geometry.TriangleMesh,
    max_edge_ratio: float = 0.3
) -> o3d.geometry.TriangleMesh:
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    if tris.size == 0:
        return mesh

    # Диагональ bounding box в нормализованных координатах (у тебя ~порядка 1)
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_diag = np.linalg.norm(bbox_max - bbox_min)
    if bbox_diag <= 0:
        return mesh

    threshold = bbox_diag * max_edge_ratio

    # Маска "плохих" треугольников
    bad_mask = np.zeros(len(tris), dtype=bool)

    for i, (i0, i1, i2) in enumerate(tris):
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        e01 = np.linalg.norm(v0 - v1)
        e12 = np.linalg.norm(v1 - v2)
        e20 = np.linalg.norm(v2 - v0)
        if max(e01, e12, e20) > threshold:
            bad_mask[i] = True

    if not bad_mask.any():
        return mesh

    mesh.remove_triangles_by_mask(bad_mask)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh

def _find_boundary_loops(mesh: o3d.geometry.TriangleMesh):
    """Возвращает список граничных петель (список списков индексов вершин)."""
    tris = np.asarray(mesh.triangles)
    if tris.size == 0:
        return []

    edges = np.vstack([
        tris[:, [0, 1]],
        tris[:, [1, 2]],
        tris[:, [2, 0]],
    ])
    edges_sorted = np.sort(edges, axis=1)
    uniq_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = uniq_edges[counts == 1]

    if len(boundary_edges) == 0:
        return []

    adj = defaultdict(list)
    for u, v in boundary_edges:
        u = int(u); v = int(v)
        adj[u].append(v)
        adj[v].append(u)

    def edge_key(a, b):
        return (min(a, b), max(a, b))

    visited_edges = set()
    loops = []

    for u, v in boundary_edges:
        u = int(u); v = int(v)
        e0 = edge_key(u, v)
        if e0 in visited_edges:
            continue

        start = u
        loop = [start]
        cur = start
        prev = None

        while True:
            next_vertex = None
            for nb in adj[cur]:
                e = edge_key(cur, nb)
                if e not in visited_edges:
                    next_vertex = nb
                    break

            if next_vertex is None:
                break

            visited_edges.add(edge_key(cur, next_vertex))
            cur = next_vertex

            if cur == start:
                break
            else:
                loop.append(cur)

        if len(loop) >= 3:
            loops.append(loop)

    return loops


def _point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    """
    Тест "точка в полигоне" (ray casting) для 2D.
    pt: (2,), poly: (N,2) в порядке обхода.
    """
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # пересечение с горизонтальным лучом вправо
        if (y1 > y) != (y2 > y):
            t = (y - y1) / ((y2 - y1) + 1e-12)
            x_int = x1 + t * (x2 - x1)
            if x_int >= x:
                inside = not inside
    return inside

def fill_two_largest_holes_with_grid(
    mesh: o3d.geometry.TriangleMesh,
    grid_step_ratio: float = 0.03,
    min_area_ratio: float = 0.001,
) -> o3d.geometry.TriangleMesh:
    """
    Заполняет ДВЕ самые большие дырки (по площади в проекции) регулярной сеткой.
    Без учёта осей/нормалей — максимально простой вариант
    специально под "крышки" для стенки.
    """
    verts = np.asarray(mesh.vertices)
    tris_old = np.asarray(mesh.triangles, dtype=np.int32)
    if tris_old.size == 0:
        return mesh

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    max_extent = float(np.max(extent))
    if max_extent <= 0:
        return mesh

    loops = _find_boundary_loops(mesh)
    print(f"[fill_holes_grid_simple] boundary loops: {len(loops)}")
    if not loops:
        print("[fill_holes_grid_simple] no open boundaries")
        return mesh

    candidates = []
    # --------- собираем все петли с их площадью ---------
    for loop in loops:
        if len(loop) < 3:
            continue

        loop_idx = np.array(loop, dtype=int)
        loop_pts = verts[loop_idx]

        # Плоскость по PCA
        centroid = loop_pts.mean(axis=0)
        P = loop_pts - centroid
        U, S, Vt = np.linalg.svd(P, full_matrices=False)

        u = Vt[0, :]
        v = Vt[1, :]

        pts2d = np.stack([P @ u, P @ v], axis=1)

        x = pts2d[:, 0]
        y = pts2d[:, 1]
        area = 0.5 * abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

        if area < (max_extent * max_extent * min_area_ratio):
            continue

        candidates.append(
            dict(
                loop_idx=loop_idx,
                centroid=centroid,
                Vt=Vt,
                pts2d=pts2d,
                area=area,
            )
        )

    print(f"[fill_holes_grid_simple] candidates by area: {len(candidates)}")
    if not candidates:
        print("[fill_holes_grid_simple] no candidates by area")
        return mesh

    # Берём две самые большие дырки
    candidates.sort(key=lambda c: c["area"], reverse=True)
    candidates = candidates[:2]

    new_vertices = []
    new_tris = []

    for cand in candidates:
        loop_idx = cand["loop_idx"]
        centroid = cand["centroid"]
        Vt = cand["Vt"]
        pts2d = cand["pts2d"]

        u = Vt[0, :]
        v = Vt[1, :]

        poly_min = pts2d.min(axis=0)
        poly_max = pts2d.max(axis=0)
        poly_extent = poly_max - poly_min
        poly_max_side = float(np.max(poly_extent))
        if poly_max_side <= 0:
            continue

        # шаг сетки
        h = poly_max_side * grid_step_ratio
        if h <= 1e-6:
            h = poly_max_side * 0.02

        nx = int(np.ceil(poly_extent[0] / h)) + 1
        ny = int(np.ceil(poly_extent[1] / h)) + 1

        xs = poly_min[0] + np.arange(nx + 1) * h
        ys = poly_min[1] + np.arange(ny + 1) * h

        node_idx = -np.ones((nx + 1, ny + 1), dtype=int)

        # 1) Привязываем граничные вершины к ближайшим узлам сетки
        for k, p in enumerate(pts2d):
            gx = int(round((p[0] - poly_min[0]) / h))
            gy = int(round((p[1] - poly_min[1]) / h))
            gx = max(0, min(nx, gx))
            gy = max(0, min(ny, gy))
            node_idx[gx, gy] = loop_idx[k]

        # 2) Добавляем новые внутренние вершины сетки
        for ix in range(nx + 1):
            for iy in range(ny + 1):
                if node_idx[ix, iy] != -1:
                    continue
                p2 = np.array([xs[ix], ys[iy]], dtype=np.float64)
                if not _point_in_polygon(p2, pts2d):
                    continue
                # ГЛАВНОЕ ИСПРАВЛЕНИЕ: индекс = старые вершины + кол-во новых
                new_index = len(verts) + len(new_vertices)
                p3 = centroid + u * p2[0] + v * p2[1]
                new_vertices.append(p3)
                node_idx[ix, iy] = new_index

        # 3) Квадраты -> два треугольника
        for ix in range(nx):
            for iy in range(ny):
                a = node_idx[ix, iy]
                b = node_idx[ix + 1, iy]
                c = node_idx[ix + 1, iy + 1]
                d = node_idx[ix, iy + 1]
                if a == -1 or b == -1 or c == -1 or d == -1:
                    continue

                cx = 0.5 * (xs[ix] + xs[ix + 1])
                cy = 0.5 * (ys[iy] + ys[iy + 1])
                if not _point_in_polygon(np.array([cx, cy]), pts2d):
                    continue

                new_tris.append([a, b, c])
                new_tris.append([a, c, d])

    if not new_tris:
        print("[fill_holes_grid_simple] no triangles generated")
        return mesh

    # добавляем новые вершины
    if new_vertices:
        verts_out = np.vstack([verts, np.asarray(new_vertices, dtype=np.float64)])
    else:
        verts_out = verts

    tris_new = np.vstack([tris_old, np.asarray(new_tris, dtype=np.int32)])

    mesh.vertices = o3d.utility.Vector3dVector(verts_out)
    mesh.triangles = o3d.utility.Vector3iVector(tris_new)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    print(
        f"[fill_holes_grid_simple] caps built: {len(candidates)}, "
        f"added {len(new_tris)} triangles, new verts: {len(new_vertices)}"
    )
    return mesh


def is_slab_like(mesh: o3d.geometry.TriangleMesh,
                 thickness_ratio_threshold: float = 0.3) -> bool:
    """
    Возвращает True, если объект похож на "плиту/стену":
    одна размерность (толщина) заметно меньше двух других.

    thickness_ratio_threshold:
        min_extent / max_extent должен быть МЕНЬШЕ этого порога.
        Например:
           - стена:   0.05..0.2  -> True
           - бассейн: 0.4..0.8   -> False
    """
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    min_extent = float(np.min(extent))
    max_extent = float(np.max(extent))
    if max_extent <= 0:
        return False
    ratio = min_extent / max_extent
    print(f"[SlabCheck] min/max extent = {ratio:.3f}")
    return ratio < thickness_ratio_threshold

def smooth_slab_boundaries(
    mesh: o3d.geometry.TriangleMesh,
    thin_axis: int,
    iterations: int = 3,
    alpha: float = 0.4,
) -> o3d.geometry.TriangleMesh:
    """
    Мягко разглаживает зубчатые границы у тонкой плиты.

    - находим граничные петли (open-boundary loop);
    - для вершин этих петель двигаем их к среднему соседей,
      но координату по оси толщины (thin_axis) фиксируем,
      чтобы торец оставался строго в плоскости.

    iterations — сколько раз повторить сглаживание;
    alpha      — насколько сильно тянуть к среднему (0..1).
    """
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    if tris.size == 0:
        return mesh

    # считаем граничные рёбра
    edges = np.vstack([
        tris[:, [0, 1]],
        tris[:, [1, 2]],
        tris[:, [2, 0]],
    ])
    edges_sorted = np.sort(edges, axis=1)
    uniq_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = uniq_edges[counts == 1]
    if len(boundary_edges) == 0:
        print("[smooth_slab_boundaries] no boundary edges")
        return mesh

    # граф смежности только по граничным рёбрам
    adj = defaultdict(list)
    boundary_verts = set()
    for u, v in boundary_edges:
        u = int(u); v = int(v)
        adj[u].append(v)
        adj[v].append(u)
        boundary_verts.add(u)
        boundary_verts.add(v)

    boundary_verts = sorted(boundary_verts)

    for it in range(iterations):
        new_positions = {}
        for v_idx in boundary_verts:
            neigh = adj[v_idx]
            if not neigh:
                continue
            p = verts[v_idx]
            neigh_pts = verts[neigh]
            mean_neigh = neigh_pts.mean(axis=0)
            q = (1.0 - alpha) * p + alpha * mean_neigh
            # фиксируем координату по оси толщины
            q[thin_axis] = p[thin_axis]
            new_positions[v_idx] = q

        # применяем
        for v_idx, q in new_positions.items():
            verts[v_idx] = q

    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    print(f"[smooth_slab_boundaries] smoothed {len(boundary_verts)} boundary verts, "
          f"iterations={iterations}")
    return mesh

def keep_largest_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Оставляет только самый большой по количеству треугольников связный компонент,
    все мелкие "островки" (оторванные куски) удаляет.
    """
    tris = np.asarray(mesh.triangles)
    if tris.size == 0:
        print("[KeepLargest] no triangles")
        return mesh

    # кластеризация по связности треугольников (есть в новых Open3D)
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if cluster_n_triangles.size == 0:
        print("[KeepLargest] no clusters info")
        return mesh

    largest_cluster = int(cluster_n_triangles.argmax())
    keep_count = int(cluster_n_triangles[largest_cluster])

    # маска "что удалить": все кластеры кроме самого большого
    remove_mask = triangle_clusters != largest_cluster

    print(f"[KeepLargest] keeping cluster {largest_cluster} "
          f"({keep_count} tris), "
          f"removing {(remove_mask).sum()} tris in other clusters")

    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh
