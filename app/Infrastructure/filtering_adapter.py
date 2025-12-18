# Радиальная фильтрация
# nb - количество соседних точек
# rad - радиус сферы, центром которого является выбранная точка
def radial_outlier_remover(cl, nb, rad):
    new_cl, ind = cl.remove_radius_outlier(nb_points=nb, radius=rad)
    return new_cl

# Статистической фильтрация
# nb - количество соседних точек
# std - коэффициент отклонения
def statistical_outlier_remover(cl, nb, std):
    new_cl, ind = cl.remove_statistical_outlier(nb_neighbors=nb, std_ratio=(std))
    return new_cl

# Препроцессинг облака точек с воксельным даунсемплингом
def filter_pcd(pcd, nb_stat, nb_radial, std, rad, vox_size):
    cl = statistical_outlier_remover(pcd, nb_stat, std)
    cl = radial_outlier_remover(cl, nb_radial, rad)
    cl = cl.voxel_down_sample(voxel_size=vox_size)
    print("voxing done")
    return cl