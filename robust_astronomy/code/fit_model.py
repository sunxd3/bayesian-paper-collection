#!/usr/bin/env python
import argparse

import arviz as az
from tcup.numpyro import tcup
from tcup_paper.data.io import load_dataset

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fit a model to data")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-f", "--fixed", type=int)
    group.add_argument("-n", "--normal", action="store_true")
    parser.add_argument("-s", "--num-samples", type=int, default=5000)
    parser.add_argument("-r", "--random-seed", type=int)
    parser.add_argument("dataset")
    parser.add_argument("outfile")
    args = parser.parse_args()

    # Load dataset
    data, params = load_dataset(args.dataset)

    if args.random_seed:
        # Fit model
        if args.normal:
            mcmc = tcup(
                **data,
                seed=args.random_seed,
                model="ncup",
                num_samples=args.num_samples,
            )
        elif args.fixed:
            mcmc = tcup(
                **data,
                seed=args.random_seed,
                model="fixed",
                shape_param=args.fixed,
                num_samples=args.num_samples,
            )
        else:
            mcmc = tcup(
                **data, seed=args.random_seed, num_samples=args.num_samples
            )

        # Save chains
        mcmc.to_netcdf(args.outfile)
    else:
        # No random seed provided - loop until r_hat < 1.1
        # (Maximum of 20 loops)
        for seed in range(20):
            # Fit model
            if args.normal:
                mcmc = tcup(
                    **data,
                    seed=seed,
                    model="ncup",
                    num_samples=args.num_samples,
                )
            elif args.fixed:
                mcmc = tcup(
                    **data,
                    seed=seed,
                    model="fixed",
                    shape_param=args.fixed,
                    num_samples=args.num_samples,
                )
            else:
                mcmc = tcup(
                    **data,
                    seed=seed,
                    model="tcup",
                    num_samples=args.num_samples,
                )

            sample_stats = az.summary(mcmc)

            if sample_stats["r_hat"].max() < 1.1:
                # Successful run - save output
                mcmc.to_netcdf(args.outfile)
                break
            else:
                # Failed run - save for diagnostic purposes
                filename = "".join(args.outfile.split(".")[:-1])
                filename += f"_run_{seed}.nc"
                mcmc.to_netcdf(filename)
                del mcmc
