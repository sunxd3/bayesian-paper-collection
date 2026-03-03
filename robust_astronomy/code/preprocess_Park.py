import json

import numpy as np
import pandas as pd


def write_dataset(filename, df, line_width):
    y = df["M_BH"].astype(float).values
    dy = df["dM_BH"].astype(float).values
    log_L = df["log_L"].astype(float).values
    dlog_L = df["dlog_L"].astype(float).values
    lw = df[line_width].astype(float).values
    dlw = df[f"d{line_width}"].astype(float).values
    corr = df[f"corr_{line_width}"].astype(float).values

    log_lw = np.log10(lw)
    dlog_lw = dlw / (lw * np.log(10))

    x = np.vstack(
        [
            log_L - 44,
            log_lw - 3,
        ]
    )

    cov = [
        [
            [dx_1 * dx_1, dx_1 * dx_2 * corr_12],
            [dx_1 * dx_2 * corr_12, dx_2 * dx_2],
        ]
        for dx_1, dx_2, corr_12 in zip(dlog_L, dlog_lw, corr)
    ]

    with open(filename, "w") as f:
        json.dump(
            {
                "data": {
                    "x": x.T.tolist(),
                    "cov_x": cov,
                    "y": y.tolist(),
                    "dy": dy.tolist(),
                },
                "params": {},
            },
            f,
        )


if __name__ == "__main__":
    rm_masses = pd.read_table(
        "data/real/park/table1.dat",
        skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 36, 37, 38, 45, 46],
        on_bad_lines="warn",
        skip_blank_lines=True,
        names=[
            "object",
            "z",
            "time_delay",
            "rms_disp",
            "M_BH",
            "reference",
            "unnamed",
        ],
    ).dropna(axis=1, how="all")

    rm_masses[["time_delay", "dtime_delay-", "dtime_delay+"]] = rm_masses[
        "time_delay"
    ].str.extract(r"\${([0-9.]+)}_{-([0-9.]+)}\^{\+([0-9.]+)}\$", expand=True)
    rm_masses[["rms_disp", "drms_disp"]] = rm_masses["rms_disp"].str.split(
        r" \+or- ", expand=True
    )
    rm_masses[["M_BH", "dM_BH"]] = rm_masses["M_BH"].str.split(
        r" \+or- ", expand=True
    )

    spectra = pd.read_table(
        "data/real/park/table3.dat",
        skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 37, 38, 39, 46, 47],
        on_bad_lines="warn",
        skip_blank_lines=True,
        names=[
            "object",
            "telescope/instrument",
            "obs_date",
            "snr",
            "E_BV",
            "log_L",
            "FWHM",
            "line_disp",
            "MAD",
            "corr_FWHM",
            "corr_line_disp",
            "corr_MAD",
            "unnamed",
        ],
    ).dropna(axis=1, how="all")

    spectra[["telescope", "instrument"]] = spectra[
        "telescope/instrument"
    ].str.split("/", expand=True)
    spectra[["log_L", "dlog_L"]] = spectra["log_L"].str.split(
        r" \+or- ", expand=True
    )
    spectra[["FWHM", "dFWHM"]] = spectra["FWHM"].str.split(
        r" \+or- ", expand=True
    )
    spectra[["line_disp", "dline_disp"]] = spectra["line_disp"].str.split(
        r" \+or- ", expand=True
    )
    spectra[["MAD", "dMAD"]] = spectra["MAD"].str.split(
        r" \+or- ", expand=True
    )

    df = rm_masses.merge(spectra, on="object")

    for line_width in ["FWHM", "line_disp", "MAD"]:
        write_dataset(f"data/real/park_{line_width}.json", df, line_width)
