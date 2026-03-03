#!/usr/bin/env python
import argparse

import numpy as np
from tcup_paper.data.io import write_dataset

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fit a model to data")
    parser.add_argument("--seed", type=int, default=0)
    distribution = parser.add_mutually_exclusive_group(required=True)
    distribution.add_argument("--t-dist", action="store_true")
    distribution.add_argument("--outlier", action="store_true")
    distribution.add_argument("--gaussian-mix", action="store_true")
    distribution.add_argument("--laplace", action="store_true")
    distribution.add_argument("--lognormal", action="store_true")
    args = parser.parse_args()

    if args.t_dist:
        from tcup_paper.data.fixed.t import gen_dataset

        x, y, dx, dy, info = gen_dataset(args.seed)
        data = {
            "x": x.tolist(),
            "y": y.tolist(),
            "dx": dx.tolist(),
            "dy": dy.tolist(),
        }
        write_dataset(f"fixed/t/{args.seed}", data, info)

    if args.outlier:
        from tcup_paper.data.fixed.outlier import gen_dataset

        x, y, dx, dy, info = gen_dataset(args.seed)
        data = {
            "x": x.tolist(),
            "y": y.tolist(),
            "dx": dx.tolist(),
            "dy": dy.tolist(),
        }
        write_dataset(f"fixed/outlier/{args.seed}", data, info)

        # Exclude outlier and save
        idx = info["outlier_idx"]
        mask = np.ones(shape=y.shape, dtype=bool)
        mask[idx] = False
        no_outlier_data = {
            "x": x[mask].tolist(),
            "y": y[mask].tolist(),
            "dx": dx[mask].tolist(),
            "dy": dy[mask].tolist(),
        }
        write_dataset(f"fixed/normal/{args.seed}", no_outlier_data, info)

    if args.gaussian_mix:
        from tcup_paper.data.fixed.gaussian_mix import gen_dataset

        x, y, dx, dy, info = gen_dataset(args.seed)
        data = {
            "x": x.tolist(),
            "y": y.tolist(),
            "cov_x": dx.tolist(),
            "dy": dy.tolist(),
        }
        write_dataset(f"fixed/gaussian_mix/{args.seed}", data, info)

    if args.laplace:
        from tcup_paper.data.fixed.laplace import gen_dataset

        x, y, dx, dy, info = gen_dataset(args.seed)
        data = {
            "x": x.tolist(),
            "y": y.tolist(),
            "dx": dx.tolist(),
            "dy": dy.tolist(),
        }
        write_dataset(f"fixed/laplace/{args.seed}", data, info)

    if args.lognormal:
        from tcup_paper.data.fixed.lognormal import gen_dataset

        x, y, dx, dy, info = gen_dataset(args.seed)
        data = {
            "x": x.tolist(),
            "y": y.tolist(),
            "dx": dx.tolist(),
            "dy": dy.tolist(),
        }
        write_dataset(f"fixed/lognormal/{args.seed}", data, info)
