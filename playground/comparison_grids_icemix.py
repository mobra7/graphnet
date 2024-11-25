from graphnet.training.utils import make_dataloader
from graphnet.models import Model
from graphnet.models.graphs import GraphDefinition, KNNGraph
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models.detector.icecube import IceCube86
from graphnet.training.labels import Direction
from graphnet.models.graphs.nodes import IceMixNodes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import os


model_path = "./icemix_10_23"
freedom = pd.read_pickle(f"{model_path}/performance_3.pkl")
print(freedom.head())
# selection = list(pd.read_pickle(f'{model_path}/performance_events.pkl')['event_no'])


df = pd.DataFrame()


def angle_model(az_true, zen_true, az_pred, zen_pred):
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)

    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)

    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    scalar_prod = np.clip(scalar_prod, -1, 1)

    return np.abs(np.arccos(scalar_prod))


df["SBI"] = angle_model(
    freedom["truth_az"],
    freedom["truth_ze"],
    freedom["max_llh_az"],
    freedom["max_llh_ze"],
)
df["spline"] = angle_model(
    freedom["truth_az"],
    freedom["truth_ze"],
    freedom["spline_az"],
    freedom["spline_ze"],
)
df["truth grid vs. non truth grid"] = angle_model(
    freedom["grid_max_az"],
    freedom["grid_max_ze"],
    freedom["truth_grid_max_az"],
    freedom["truth_grid_max_ze"],
)

bin_edges = np.logspace(2, 7, 15)
bin_centers = [
    0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(len(bin_edges) - 1)
]

df["energy_bin"] = pd.cut(freedom["energy"], bins=bin_edges)
df["from_true_grid"] = freedom["from_true_grid"]

percentiles = [50]


def bootstrap_percentiles(group, n_bootstrap=1000):
    if len(group) < 2:
        return pd.Series(
            [np.nan] * (len(percentiles) * 3),
            index=[
                f"{p}_{stat}"
                for p in percentiles
                for stat in ["median", "lower", "upper"]
            ],
        )

    bootstrap_results = []
    for _ in range(n_bootstrap):
        bootstrap_sample = group.sample(n=len(group), replace=True)
        bootstrap_results.append(np.percentile(bootstrap_sample, percentiles))

    bootstrap_results = np.array(bootstrap_results)
    medians = np.median(bootstrap_results, axis=0)
    lower = np.percentile(bootstrap_results, 2.5, axis=0)
    upper = np.percentile(bootstrap_results, 97.5, axis=0)

    result = {}
    for i, p in enumerate(percentiles):
        result[f"{p}_median"] = medians[i]
        result[f"{p}_lower"] = lower[i]
        result[f"{p}_upper"] = upper[i]

    return pd.Series(result)


def plot_percentiles_comparison_with_bootstrap(df, columns, title, filename):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = list(mcolors.TABLEAU_COLORS.values())
    linestyles = ["-", "--", "-."]  # Different line styles for percentiles

    for i, column in enumerate(columns):
        base_color = colors[i % len(colors)]

        percentiles_df = (
            df[df["from_true_grid"] == 0]
            .groupby("energy_bin")[column]
            .apply(bootstrap_percentiles)
            .reset_index()
        )
        percentiles_df = percentiles_df.pivot(
            index="energy_bin", columns="level_1", values=column
        ).reset_index()

        for j, p in enumerate(percentiles):
            median = np.degrees(percentiles_df[f"{p}_median"])
            lower = np.degrees(percentiles_df[f"{p}_lower"])
            upper = np.degrees(percentiles_df[f"{p}_upper"])

            ax1.plot(
                bin_centers,
                median,
                linestyle=linestyles[j],
                linewidth=2,
                color=base_color,
                label=f"{column} Non truth grid {p}th Percentile",
            )
            ax1.fill_between(
                bin_centers, lower, upper, alpha=0.3, color=base_color
            )

        base_color = colors[i + 3 % len(colors)]

        percentiles_df = (
            df[df["from_true_grid"] == 1]
            .groupby("energy_bin")[column]
            .apply(bootstrap_percentiles)
            .reset_index()
        )
        percentiles_df = percentiles_df.pivot(
            index="energy_bin", columns="level_1", values=column
        ).reset_index()

        for j, p in enumerate(percentiles):
            median = np.degrees(percentiles_df[f"{p}_median"])
            lower = np.degrees(percentiles_df[f"{p}_lower"])
            upper = np.degrees(percentiles_df[f"{p}_upper"])

            ax1.plot(
                bin_centers,
                median,
                linestyle=linestyles[j],
                linewidth=2,
                color=base_color,
                label=f"{column} Truth grid {p}th Percentile",
            )
            ax1.fill_between(
                bin_centers, lower, upper, alpha=0.3, color=base_color
            )

    ax1.set_xlabel("Energy [GeV]")
    ax1.set_ylabel("Opening Angle [deg]")
    ax1.set_title(title)
    ax1.set_xlim(bin_edges[0], bin_edges[-1])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2 = ax1.twinx()
    hist_data, hist_bins, _ = ax2.hist(
        freedom["energy"],
        bins=bin_edges,
        alpha=0.3,
        color="gray",
        edgecolor="black",
    )
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")

    # ax1.set_xticks(bin_edges[::2])
    # ax1.set_xticklabels([f'{int(edge):,}' for edge in bin_edges[::2]])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Call the function with both columns
plot_percentiles_comparison_with_bootstrap(
    df,
    ["SBI", "truth grid vs. non truth grid"],
    f"Comparison of truth grid and non truth grid points",
    f"{model_path}/grid_comparison_2.pdf",
)

print("Model mean opening angle: ", np.degrees(np.mean(df["SBI"])))
