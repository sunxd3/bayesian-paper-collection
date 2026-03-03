import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The following are pulled by digging around in the Postscript file
ORIGIN = (3330, 3080)
AXES = (19690, 14260)

# These are from the plot itself
XLIMS = (-3, 1)
YLIMS = (-1, 4)

if __name__ == "__main__":
    with open("data/real/kelly/f10a.ps", "r") as f:
        filetext = f.read()

    matches = re.findall(
        r"(\d+) \d+ M 0 (\d+) R D \d+ (\d+) M (\d+) 0 R D\s\d+ \d+ M 24[12] 0 "
        r"R D\s\d+ \d+ M 24[12] 0 R D\s\d+ \d+ M 0 24[12] R D\s\d+ \d+ M 0 24["
        r"12] R D",
        filetext,
    )

    df = (
        pd.DataFrame(matches, columns=["x", "dy", "y", "dx"])
        .astype(int)
        .drop_duplicates()
    )

    xscale = (XLIMS[1] - XLIMS[0]) / AXES[0]
    df["log_Lbol"] = (df["x"] - ORIGIN[0]) * xscale + XLIMS[0]
    df["dlog_Lbol"] = df["dx"] * xscale / 2

    yscale = (YLIMS[1] - YLIMS[0]) / AXES[1]
    df["Gamma_X"] = (df["y"] - ORIGIN[1]) * yscale + YLIMS[0]
    df["dGamma_X"] = df["dy"] * yscale / 2

    # Reproduce Kelly:2007 Figure 10a
    plt.errorbar(
        df["log_Lbol"],
        df["Gamma_X"],
        df["dGamma_X"],
        df["dlog_Lbol"],
        fmt="r+",
    )
    plt.xlim(-3, 1)
    plt.ylim(-1, 4)
    plt.savefig("plots/Kelly_10a.pdf")
    plt.close()

    # Reproduce Kelly:2007 Figure 10b
    plt.plot(
        df["log_Lbol"],
        df["Gamma_X"],
        "r+",
    )
    plt.xlim(-2.5, 0.5)
    plt.ylim(0, 4)
    plt.savefig("plots/Kelly_10b.pdf")
    plt.close()

    data = {
        "x": df["log_Lbol"].values.tolist(),
        "dx": df["dlog_Lbol"].values.tolist(),
        "y": df["Gamma_X"].values.tolist(),
        "dy": df["dGamma_X"].values.tolist(),
    }

    with open("data/real/kelly.json", "w") as f:
        json.dump(
            {
                "data": data,
                "params": {
                    "outliers": np.zeros(df["y"].shape, dtype=bool).tolist()
                },
            },
            f,
        )
