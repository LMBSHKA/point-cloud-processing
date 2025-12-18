# infrastructure/pointcloud_io.py
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import laspy
from pye57 import E57

"""
Загрузить облако точек из LAS/LAZ-файла и конвертировать его в Open3D PointCloud.

Логика работы:
1. Считывает файл формата .las/.laz с помощью библиотеки `laspy`.
2. Определяет общее количество точек в файле.
3. Если задан параметр `max_points` и точек больше этого значения —
    случайным образом выбирает подмножество индексов заданного размера
    (равномерное случайное подвыборочное облако).
4. Извлекает координаты X, Y, Z только для выбранных индексов и приводит
    их к типу float32 (экономия памяти), затем конвертирует в float64
    при передаче в Open3D.
5. Если в формате файла присутствуют каналы цвета (red, green, blue) —
    извлекает цвета только для выбранных точек, нормализует их в диапазон
    [0, 1] и записывает в поле `colors` облака.

Параметры:
    path (Path):
        Путь к входному LAS/LAZ-файлу.
    max_points (int | None):
        Максимальное число точек, которое нужно оставить.
        Если None или значение больше общего количества точек — загружаются все.

Возвращает:
    o3d.geometry.PointCloud:
        Объект облака точек Open3D с заполненными полями `points`
        и (опционально) `colors`.
"""
def load_laz(path: Path, max_points: Optional[int] = None) -> o3d.geometry.PointCloud:
    print(f"[LAZ] Reading {path} ...")
    las = laspy.read(str(path))

    total = len(las.x)
    print(f"[LAZ] Total points: {total}")

    # выбираем индексы, с которыми будем работать
    if max_points is not None and total > max_points:
        idx = np.random.choice(total, max_points, replace=False)
        print(f"[LAZ] Downsampled to {max_points} points")
    else:
        idx = np.arange(total)

    # вытаскиваем только нужные точки и сразу приводим к float32
    x = np.asarray(las.x[idx], dtype=np.float32)
    y = np.asarray(las.y[idx], dtype=np.float32)
    z = np.asarray(las.z[idx], dtype=np.float32)

    points = np.column_stack((x, y, z))
    pcd = o3d.geometry.PointCloud()
    # Open3D любит float64
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # цвета, если есть в формате
    if "red" in las.point_format.dimension_names:
        r = np.asarray(las.red[idx], dtype=np.float32)
        g = np.asarray(las.green[idx], dtype=np.float32)
        b = np.asarray(las.blue[idx], dtype=np.float32)

        colors = np.column_stack((r, g, b))
        max_val = colors.max() if colors.max() > 1.0 else 1.0
        colors = colors / max_val

        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    return pcd

"""
Загрузить облако точек из E57-файла и сконвертировать его в Open3D PointCloud.

Логика работы:
1. Открывает .e57 файл с помощью `pye57.E57`.
2. Пытается прочитать скан с параметром `ignore_missing_fields=True`,
    чтобы игнорировать отсутствующие поля.`.
3. Выбирает массивы координат X, Y, Z из доступных полей (поддерживаются
    разные варианты имени поля: cartesianX/X/x и т.п.).
4. Формирует массив точек размера (N, 3).
5. При необходимости выполняет случайное сэмплирование до `max_points`.
6. Создаёт облако Open3D и заполняет `points`.
7. Если в файле есть цветовые каналы (colorRed, colorGreen, colorBlue),
    нормализует их в [0, 1] и добавляет в `colors`.

Параметры:
    path (Path):
        Путь к .e57 файлу.
                    max_points (int | None):
        Максимальное количество точек, которое нужно оставить.
        Если None — используются все точки.

Возвращает:
    o3d.geometry.PointCloud:
        Облако точек Open3D с координатами и (опционально) цветами.
"""
def load_e57(path: Path, max_points: Optional[int] = None) -> o3d.geometry.PointCloud:
    print(f"[E57] Reading {path} ...")
    e57 = E57(str(path))
    data = e57.read_scan(
        0,
        ignore_missing_fields=True,
        intensity=True,
        colors=True,
    )

    print(f"[E57] Available fields: {list(data.keys())}")

    def find_field(possible_names):
        for name in possible_names:
            if name in data:
                return np.asarray(data[name], dtype=np.float64)
        return None

    x = find_field(["cartesianX", "X", "x"])
    y = find_field(["cartesianY", "Y", "y"])
    z = find_field(["cartesianZ", "Z", "z"])

    if x is None or y is None or z is None:
        raise ValueError(
            "Не удалось найти координаты X/Y/Z в e57. "
            f"Доступные поля: {list(data.keys())}"
        )

    points = np.vstack((x, y, z)).T
    print(f"[E57] Total points: {points.shape[0]}")

    idx = None
    if max_points is not None and points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]
        print(f"[E57] Downsampled to {points.shape[0]} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Цвета, если есть
    if all(k in data for k in ["colorRed", "colorGreen", "colorBlue"]):
        r = np.asarray(data["colorRed"], dtype=np.float64)
        g = np.asarray(data["colorGreen"], dtype=np.float64)
        b = np.asarray(data["colorBlue"], dtype=np.float64)

        colors = np.vstack((r, g, b)).T
        max_val = colors.max() if colors.max() > 1.0 else 1.0
        colors = colors / max_val

        if idx is not None:
            colors = colors[idx]

        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

"""
Универсальная функция загрузки облака точек по расширению файла.

Поддерживаемые форматы:
    - .laz  — через `load_laz`
    - .e57  — через `load_e57`
    ".ply", ".pcd"

Параметры:
    path (Path):
        Путь к входному файлу с облаком.
    max_points (int | None):
        Максимальное количество точек для последующей обработки.

Возвращает:
    o3d.geometry.PointCloud:
        Облако точек Open3D.
"""
def load_point_cloud_any(path: Path, max_points: Optional[int] = None) -> o3d.geometry.PointCloud:
    ext = path.suffix.lower()

    if ext in [".las", ".laz"]:
        return load_laz(path, max_points=max_points)

    if ext == ".e57":
        return load_e57(path, max_points=max_points)

    if ext in [".ply", ".pcd"]:
        print(f"[O3D] Reading {path} ...")
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd is None or len(pcd.points) == 0:
            raise ValueError(f"Не удалось прочитать облако точек из файла: {path}")

        # если нужно ограничить max_points — случайно выберем подмножество
        if max_points is not None and len(pcd.points) > max_points:
            pts = np.asarray(pcd.points)
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(pts[idx])

            if pcd.has_colors():
                cols = np.asarray(pcd.colors)
                pcd2.colors = o3d.utility.Vector3dVector(cols[idx])

            if pcd.has_normals():
                nrm = np.asarray(pcd.normals)
                pcd2.normals = o3d.utility.Vector3dVector(nrm[idx])

            pcd = pcd2
            print(f"[O3D] Downsampled to {max_points} points")

        return pcd

    raise ValueError(f"Неподдерживаемое расширение файла: {ext} (нужно .las/.laz/.e57/.ply/.pcd)")

