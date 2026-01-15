# main.py (project root)
import argparse
from app.Controllers.reconstruction_controller import run_reconstruction


def main():
    p = argparse.ArgumentParser(description="Laser: reconstruction CLI")
    p.add_argument("input", type=str, help="Input point cloud (.e57/.las/.laz/.ply/.pcd/...)")
    p.add_argument("-o", "--output", type=str, default=None, help="Output mesh path (.ply by default)")

    p.add_argument("--method", type=str, default="bpa", choices=["bpa", "poisson", "alpha", "auto"], help="Reconstruction method")
    p.add_argument("--max-points", type=int, default=2_000_000, help="Max points to load (random sampling)")
    p.add_argument("--voxel-size", type=float, default=0.005, help="Voxel downsample size (normalized space)")
    p.add_argument("--nb-neighbors", type=int, default=20, help="SOR neighbors")
    p.add_argument("--std-ratio", type=float, default=2.0, help="SOR std ratio")

    # Poisson
    p.add_argument("--poisson-depth", type=int, default=9)
    p.add_argument("--poisson-keep-ratio", type=float, default=0.02)

    # Alpha shapes
    p.add_argument("--alpha", type=float, default=0.0, help="Alpha parameter (0=auto)")

    # Postprocess
    p.add_argument("--smooth-iterations", type=int, default=5)
    p.add_argument("--max-edge-ratio", type=float, default=6.0)
    p.add_argument("--max-hole-vertices", type=int, default=120)
    p.add_argument("--max-area-ratio", type=float, default=0.0005)
    p.add_argument("--cap-grid-res", type=int, default=45)
    p.add_argument("--cap-max-area-ratio", type=float, default=0.03)
    p.add_argument("--cap-min-area-ratio", type=float, default=0.0005)
    p.add_argument("--slab-ratio", type=float, default=0.25)
    p.add_argument("--snap-ratio", type=float, default=0.15)
    p.add_argument("--slab-smooth-iterations", type=int, default=3)

    # Plane snap (fix ribbing on planar segments)
    p.add_argument("--plane-snap", action="store_true", help="Enable plane snap postprocess")
    p.add_argument("--no-plane-snap", action="store_false", dest="plane_snap", help="Disable plane snap")
    p.set_defaults(plane_snap=True)
    p.add_argument("--plane-inlier-ratio", type=float, default=0.40)
    p.add_argument("--plane-snap-slab-ratio", type=float, default=0.25)
    p.add_argument("--plane-rmse-max", type=float, default=-1.0, help="If <0 uses 1.5*thr")
    p.add_argument("--plane-snap-dist", type=float, default=-1.0, help="If <0 uses 3*thr")

    # Debug
    p.add_argument("--show-steps", action="store_true", help="Show visualization at each step")
    p.add_argument("--save-steps", action="store_true", help="Save intermediate geometries")
    p.add_argument("--show-final", action="store_true", help="Show final mesh (requires show-steps)")

    args = p.parse_args()

    # derived defaults for plane snap params
    thr = args.voxel_size * 1.3 if args.voxel_size > 0 else 0.006
    if args.plane_rmse_max < 0:
        args.plane_rmse_max = thr * 1.5
    if args.plane_snap_dist < 0:
        args.plane_snap_dist = thr * 3.0

    run_reconstruction(args)


if __name__ == "__main__":
    main()