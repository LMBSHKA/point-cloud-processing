from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d


@dataclass
class FilterSettings:
    # шаги
    use_stat: bool = True
    use_radius: bool = True
    use_voxel: bool = False  # по умолчанию выключаем (важно для реконструкции)

    # stat outlier
    nb_stat: int = 30
    std_ratio: float = 3.0

    # radius outlier (адаптивно от плотности)
    nb_radius: int = 4
    radius_mult: float = 5.0          # radius = radius_mult * nn_med
    radius_abs: Optional[float] = None # если задано — используем абсолютный радиус

    # voxel downsample (адаптивно)
    voxel_mult: float = 1.5
    voxel_abs: Optional[float] = None

    # safety
    max_drop: float = 0.30
    relax_iters: int = 4
    adaptive: bool = True
    debug: bool = True


def _estimate_nn_median(pcd: o3d.geometry.PointCloud, sample: int = 20000) -> float:
    pts = np.asarray(pcd.points)
    if pts.shape[0] < 10:
        return 0.0

    if pts.shape[0] > sample:
        idx = np.random.choice(pts.shape[0], size=sample, replace=False)
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts[idx])
    else:
        p = pcd

    d = np.asarray(p.compute_nearest_neighbor_distance(), dtype=np.float64)
    if d.size == 0:
        return 0.0
    return float(np.median(d))


def filter_pcd(pcd: o3d.geometry.PointCloud, settings: Optional[FilterSettings] = None) -> o3d.geometry.PointCloud:
    """
    Универсальный фильтр с безопасными дефолтами.
    Сейчас UI настроек нет -> settings=None -> берём FilterSettings().

    Позже UI сможет передать FilterSettings с любыми значениями.
    """
    s = settings or FilterSettings()

    n0 = len(pcd.points)
    if n0 == 0:
        return pcd

    nn_med = _estimate_nn_median(pcd) if s.adaptive else 0.0
    if nn_med <= 0:
        nn_med = 0.01

    radius = float(s.radius_abs) if (s.radius_abs is not None) else float(s.radius_mult * nn_med)
    voxel = None
    if s.use_voxel:
        voxel = float(s.voxel_abs) if (s.voxel_abs is not None) else float(s.voxel_mult * nn_med)

    def run_pipeline(
        base: o3d.geometry.PointCloud,
        *,
        stat_sr: float,
        rad_nb: int,
        rad_r: float,
        allow_radius: bool,
    ) -> o3d.geometry.PointCloud:
        cl = base

        if s.use_voxel and voxel and voxel > 0:
            cl = cl.voxel_down_sample(voxel_size=voxel)

        if s.use_stat:
            cl, _ = cl.remove_statistical_outlier(nb_neighbors=int(s.nb_stat), std_ratio=float(stat_sr))

        if allow_radius and s.use_radius:
            cl, _ = cl.remove_radius_outlier(nb_points=int(rad_nb), radius=float(rad_r))

        return cl

    def drop_frac(cloud: o3d.geometry.PointCloud) -> float:
        n = len(cloud.points)
        return 1.0 - (n / n0) if n0 > 0 else 0.0

    # 1) базовый прогон
    allow_radius = True
    cl = run_pipeline(pcd, stat_sr=s.std_ratio, rad_nb=s.nb_radius, rad_r=radius, allow_radius=True)
    drop = drop_frac(cl)

    # 2) если слишком много удалили — ослабляем
    it = 0
    while drop > s.max_drop and it < s.relax_iters:
        it += 1

        # статистика мягче
        sr = s.std_ratio + 0.8 * it

        # радиальный мягче
        rad_r = radius * (1.35 ** it)
        rad_nb = max(2, s.nb_radius - it)

        cl_try = run_pipeline(pcd, stat_sr=sr, rad_nb=rad_nb, rad_r=rad_r, allow_radius=True)
        drop_try = drop_frac(cl_try)

        if drop_try < drop:
            cl, drop = cl_try, drop_try

    # 3) если всё равно “жрёт” — выключаем радиальный (часто виновник на стенах/краях)
    if drop > s.max_drop and s.use_radius:
        cl2 = run_pipeline(pcd, stat_sr=max(s.std_ratio, 3.5), rad_nb=s.nb_radius, rad_r=radius, allow_radius=False)
        drop2 = drop_frac(cl2)
        if drop2 <= drop:
            cl, drop = cl2, drop2
            allow_radius = False

    if s.debug:
        print(
            f"[Filter] n0={n0} -> n={len(cl.points)} (drop={drop*100:.1f}%), "
            f"nn_med={nn_med:.5f}, "
            f"stat={'on' if s.use_stat else 'off'}(nb={s.nb_stat}, sr={s.std_ratio}), "
            f"radius={'on' if (s.use_radius and allow_radius) else 'off'}(nb={s.nb_radius}, mult={s.radius_mult}, r={radius:.5f}), "
            f"voxel={'on' if s.use_voxel else 'off'}({voxel if voxel else 'n/a'}), "
            f"max_drop={s.max_drop}"
        )

    return cl
