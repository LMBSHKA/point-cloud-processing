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
    - для каждой петли строит "веер" треугольников к центру.

    max_hole_vertices:
        Если None — зашиваем ЛЮБЫЕ петли по границе (все дырки).
        Если число — ограничиваемся только петлями не больше этого размера.
    """
def fill_small_holes(
    mesh: o3d.geometry.TriangleMesh,
    max_hole_vertices: Optional[int] = None
) -> o3d.geometry.TriangleMesh:
    import numpy as np
    from collections import defaultdict

    if len(mesh.triangles) == 0:
        return mesh

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles, dtype=np.int32)

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
        # если задан лимит на размер петли — применяем его
        if max_hole_vertices is not None and len(loop) > max_hole_vertices:
            continue

        loop_pts = verts[loop]
        center = loop_pts.mean(axis=0)
        center_index = len(verts) + len(new_vertices)
        new_vertices.append(center)

        # веер треугольников вокруг центра
        for i in range(len(loop)):
            a = loop[i]
            b = loop[(i + 1) % len(loop)]
            new_tris.append([a, b, center_index])

    if not new_tris:
        return mesh

    verts_out = np.vstack([verts, np.asarray(new_vertices, dtype=np.float64)])
    tris_out = np.vstack([tris, np.asarray(new_tris, dtype=np.int32)])

    mesh.vertices = o3d.utility.Vector3dVector(verts_out)
    mesh.triangles = o3d.utility.Vector3iVector(tris_out)

    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    print(f"[FillHoles] Filled {len(new_tris)} triangles in {len(loops)} loops.")
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
