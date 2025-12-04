from pathlib import Path

import open3d as o3d

"""
    Сохранить промежуточный результат обработки в файл.

    В зависимости от типа `geom`:
        - если это PointCloud — сохраняется как облако точек (.ply);
        - если TriangleMesh — как треугольный меш (.ply).

    Имя файла формируется по схеме:
        <basename>_<suffix>.ply

    где:
        <basename> — имя итогового файла без расширения,
        <suffix>   — строка, описывающая этап (например, "step2_normalized").

    Параметры:
        geom (PointCloud | TriangleMesh):
            Объект, который нужно сохранить.
        base_path (Path):
            Базовый путь к итоговому файлу (обычно `out_path` из main).
        suffix (str):
            Суффикс, добавляемый к имени файла для обозначения шага.

    Возвращает:
        None. Файлы записываются на диск.
"""
def save_step_geometry(geom, base_path: Path, suffix: str) -> None:
    if geom is None:
        return

    out_path = base_path.with_name(base_path.stem + f"_{suffix}.ply")

    if isinstance(geom, o3d.geometry.PointCloud):
        print(f"[SaveStep] Writing point cloud: {out_path}")
        o3d.io.write_point_cloud(str(out_path), geom, write_ascii=False)
    elif isinstance(geom, o3d.geometry.TriangleMesh):
        print(f"[SaveStep] Writing mesh: {out_path}")
        o3d.io.write_triangle_mesh(str(out_path), geom, write_ascii=False)
