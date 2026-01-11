from pathlib import Path
import numpy as np
import open3d as o3d

from app.Infrastructure.pointcloud_io import load_point_cloud_any
from app.Infrastructure.output import save_step_geometry
from app.UI.visualization import show_geometry
from app.Core.pointcloud_preprocess import normalize_point_cloud, preprocess_point_cloud, smooth_mesh_soft
from app.Core.mesh_reconstruction import reconstruct_poisson, reconstruct_bpa, clean_mesh_o3d
from app.Core.mesh_postprocess import (
    straighten_slab_sides,
    remove_long_triangles,
    fill_all_holes,
    fill_small_holes,
    fill_two_largest_holes_with_grid,
    is_slab_like,
    smooth_slab_boundaries,
    keep_largest_component,
)


def run_reconstruction(args, *, return_mesh: bool = False) -> None:
    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Файл не найден: {in_path}")

    if args.output is None:
        out_path = in_path.with_name(in_path.stem + "_mesh.ply")
    else:
        out_path = Path(args.output)

    print(f"[Main] Input:  {in_path}")
    print(f"[Main] Output: {out_path}")
    print(f"[Main] Method: {args.method}")

    # 1. Загрузка
    pcd = load_point_cloud_any(in_path, max_points=args.max_points)

    if args.show_steps:
        show_geometry(pcd, "Step 1 - Raw point cloud")
    if args.save_steps:
        save_step_geometry(pcd, out_path, "step1_raw")

    # 2. Нормализация
    pcd_norm, center, scale = normalize_point_cloud(pcd)

    if args.show_steps:
        show_geometry(pcd_norm, "Step 2 - Normalized point cloud")
    if args.save_steps:
        save_step_geometry(pcd_norm, out_path, "step2_normalized")

    # 3. Предобработка (voxel + нормали)
    pcd_proc = preprocess_point_cloud(pcd_norm, voxel_size=args.voxel_size)

    if args.show_steps:
        show_geometry(pcd_proc, "Step 3 - Preprocessed point cloud")
    if args.save_steps:
        save_step_geometry(pcd_proc, out_path, "step3_preprocessed")

    # 4. Реконструкция
    if args.method == "poisson":
        mesh_norm = reconstruct_poisson(
            pcd_proc,
            depth=args.depth,
            keep_ratio=args.keep_ratio,
        )

    else:  # BPA
        mesh_norm = reconstruct_bpa(pcd_proc)
        mesh_norm = mesh_norm.filter_smooth_taubin(number_of_iterations=1)
        mesh_norm.remove_degenerate_triangles()
        mesh_norm.compute_vertex_normals()

        # выпрямляем торцы плиты
        mesh_norm = straighten_slab_sides(
            mesh_norm,
            snap_ratio=0.25,
            thickness_ratio_threshold=0.25,
        )

        # если действительно похоже на плиту — сгладим границы и поставим крышки
        if is_slab_like(mesh_norm, thickness_ratio_threshold=0.6):
            bbox = mesh_norm.get_axis_aligned_bounding_box()
            thin_axis = int(np.argmin(bbox.get_extent()))

            mesh_norm = smooth_slab_boundaries(
                mesh_norm,
                thin_axis=thin_axis,
                iterations=3,
                alpha=0.4,
            )

            mesh_norm = fill_two_largest_holes_with_grid(
                mesh_norm,
                grid_step_ratio=0.03,
                min_area_ratio=0.0005,
            )

    # 4.1. Удаляем длинные лучи (для обоих методов)
    mesh_norm = remove_long_triangles(mesh_norm, max_edge_ratio=0.3)

    # 4.2. Зашиваем только локальные маленькие петли (в т.ч. углы),
    #      но не трогаем большие отверстия и периметр крышек
    mesh_norm = fill_small_holes(
        mesh_norm,
        max_hole_vertices=None,   # не ограничиваем по числу вершин
        max_area_ratio=0.01,      # 1% от "квадрата" модели, можно поиграть 0.005–0.02
    )

    # Удаляем все мелкие оторванные куски, оставляем только основной объект
    mesh_norm = keep_largest_component(mesh_norm)

    mesh_norm = clean_mesh_o3d(mesh_norm, smooth_iters=1)

    # 4.3. Финальное мягкое сглаживание
    mesh_norm = smooth_mesh_soft(mesh_norm, iterations=5)

    if args.show_steps:
        show_geometry(mesh_norm, "Step 4 - Mesh (normalized coords)") 
    if args.save_steps:
        save_step_geometry(mesh_norm, out_path, "step4_mesh_norm")

    # 5. Возвращаем масштаб и позицию
    print("[Rescale] Restoring original scale and position...")
    if scale > 0:
        mesh_norm.scale(scale, center=(0.0, 0.0, 0.0))
    mesh_norm.translate(center)
    mesh = mesh_norm

    # Поворот на +90° вокруг X
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0.0, 0.0))
    mesh.rotate(R, center=(0.0, 0.0, 0.0))

    # 6. Сохраняем
    print(f"[IO] Writing mesh to {out_path} ...")
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False)
    print("[Done] Finished.")

    if getattr(args, "show_final", True):
        show_geometry(mesh, "FINAL RESULT")

    if return_mesh:
        return mesh, out_path
