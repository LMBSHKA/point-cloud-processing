# main.py
import argparse

from Controllers import run_reconstruction

def main():
    parser = argparse.ArgumentParser(
        description="Реконструкция 3D-сетки из облака точек (LAZ/E57) с помощью Open3D."
    )
    parser.add_argument("input", type=str, help="Входной файл .laz или .e57")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Выходной файл сетки (.ply, .obj, ...). По умолчанию input_name_mesh.ply"
    )
    parser.add_argument(
        "--max-points", type=int, default=2_000_000,
        help="Максимальное количество точек для реконструкции (default: 2_000_000)"
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.02,
        help="Размер вокселя для даунсэмплинга в НОРМАЛИЗОВАННЫХ координатах (default: 0.02)"
    )
    parser.add_argument(
        "--depth", type=int, default=10,
        help="Параметр depth для Poisson (8–11 обычно нормально, default: 10)"
    )
    parser.add_argument(
        "--keep-ratio", type=float, default=0.05,
        help="Доля низкоплотных вершин, которые будут удалены в Poisson (default: 0.05)"
    )
    parser.add_argument(
        "--method", type=str, default="bpa",
        choices=["bpa", "poisson"],
        help="Алгоритм реконструкции: bpa или poisson"
    )
    parser.add_argument(
        "--show-steps", action="store_true",
        help="Показывать окна визуализации на каждом этапе обработки"
    )
    parser.add_argument(
        "--save-steps", action="store_true",
        help="Сохранять промежуточные результаты (pcd/mesh) в файлы"
    )

    args = parser.parse_args()
    run_reconstruction(args)


if __name__ == "__main__":
    main()
