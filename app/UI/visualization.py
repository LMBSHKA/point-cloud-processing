import numpy as np
import open3d as o3d

"""
Визуализировать облако точек или меш в отдельном окне Open3D.

Особенности реализации:
    - Для стабильной визуализации сначала создаётся копия геометрии,
        чтобы не изменять оригинальные данные.
    - Затем объект центрируется в начале координат (вычитается центр AABB).
    - Если сцена очень большая (extent > 10), она дополнительно
        масштабируется так, чтобы вписаться в компактный диапазон.
        Это уменьшает визуальные артефакты, связанные с точностью float
        и глубинным буфером в рендерере.
    - Для TriangleMesh включается `mesh_show_back_face=True`, чтобы
        задние стороны треугольников тоже были видимы и меш не казался
        “прозрачным”.

Параметры:
    geom (PointCloud | TriangleMesh):
        Объект Open3D для отображения.
    window_name (str):
        Заголовок окна.

Возвращает:
    None. Функция блокирует выполнение до закрытия окна Open3D.
"""
def show_geometry(geom, window_name: str = "Open3D") -> None:
    if geom is None:
        return

    print(f"[Viz] Showing: {window_name}")

    # Делаем копию, чтобы не портить исходную геометрию
    if isinstance(geom, o3d.geometry.PointCloud):
        geom_vis = o3d.geometry.PointCloud(geom)  # копирующий конструктор
    elif isinstance(geom, o3d.geometry.TriangleMesh):
        geom_vis = o3d.geometry.TriangleMesh(geom)
    else:
        geom_vis = geom

    # Центрируем и, если нужно, уменьшаем для стабильного рендера
    if isinstance(geom_vis, (o3d.geometry.PointCloud, o3d.geometry.TriangleMesh)):
        bbox = geom_vis.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_max_extent()

        # сдвиг в начало координат
        geom_vis.translate(-center)

        # если объект ОЧЕНЬ большой — уменьшим
        if extent > 10.0:
            scale = 1.0 / extent
            geom_vis.scale(scale, center=(0.0, 0.0, 0.0))
            print(f"[Viz] Applied centering and scaling (extent={extent:.3f})")

    kwargs = dict(
        window_name=window_name,
        width=1200,
        height=800,
    )

    if isinstance(geom_vis, o3d.geometry.TriangleMesh):
        kwargs["mesh_show_back_face"] = True  # чтобы не было "прозрачности"

    o3d.visualization.draw_geometries([geom_vis], **kwargs)