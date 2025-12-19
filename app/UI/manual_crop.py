import open3d as o3d

def open3d_edit_crop(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    # Open3D editing: пользователь выделяет/сохраняет selection
    o3d.visualization.draw_geometries_with_editing([pcd])
    # Этот режим сохраняет файлы selection в рабочую папку (selected.ply и т.п.)
    # Самый простой вариант: после закрытия попросить пользователя выбрать сохранённый ply.
    return pcd