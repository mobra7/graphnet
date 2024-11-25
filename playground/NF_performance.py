from graphnet.models import Model
from graphnet.models.graphs import GraphDefinition, KNNGraph
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models.detector.icecube import IceCube86
from graphnet.training.utils import make_dataloader
from graphnet.training.labels import Direction, Label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from torch_geometric.data import Data

import os


class x(Label):
    """Class for producing particle direction/pointing label."""

    def __init__(
        self,
        key: str = "x_direction",
        azimuth_key: str = "azimuth",
        zenith_key: str = "zenith",
    ):
        """Construct `Direction`.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `graph[key] = label`.
            azimuth_key: The name of the pre-existing key in `graph` that will
                be used to access the azimiuth angle, used when calculating
                the direction.
            zenith_key: The name of the pre-existing key in `graph` that will
                be used to access the zenith angle, used when calculating the
                direction.
        """
        self._azimuth_key = azimuth_key
        self._zenith_key = zenith_key

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        x = torch.cos(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        y = torch.sin(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        z = torch.cos(graph[self._zenith_key]).reshape(-1, 1)
        return x


class y(Label):
    """Class for producing particle direction/pointing label."""

    def __init__(
        self,
        key: str = "y_direction",
        azimuth_key: str = "azimuth",
        zenith_key: str = "zenith",
    ):
        """Construct `Direction`.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `graph[key] = label`.
            azimuth_key: The name of the pre-existing key in `graph` that will
                be used to access the azimiuth angle, used when calculating
                the direction.
            zenith_key: The name of the pre-existing key in `graph` that will
                be used to access the zenith angle, used when calculating the
                direction.
        """
        self._azimuth_key = azimuth_key
        self._zenith_key = zenith_key

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        x = torch.cos(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        y = torch.sin(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        z = torch.cos(graph[self._zenith_key]).reshape(-1, 1)
        return y


class z(Label):
    """Class for producing particle direction/pointing label."""

    def __init__(
        self,
        key: str = "z_direction",
        azimuth_key: str = "azimuth",
        zenith_key: str = "zenith",
    ):
        """Construct `Direction`.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `graph[key] = label`.
            azimuth_key: The name of the pre-existing key in `graph` that will
                be used to access the azimiuth angle, used when calculating
                the direction.
            zenith_key: The name of the pre-existing key in `graph` that will
                be used to access the zenith angle, used when calculating the
                direction.
        """
        self._azimuth_key = azimuth_key
        self._zenith_key = zenith_key

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        x = torch.cos(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        y = torch.sin(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        z = torch.cos(graph[self._zenith_key]).reshape(-1, 1)
        return z


model_path = "./NF_11_13"
path = "/scratch/users/allorana/northern_sqlite/files_no_hlc/dev_northern_tracks_full_part_2.db"
pulsemap = "InIcePulses"
target = "scrambled_class"
truth_table = "truth"
gpus = [0]
max_epochs = 30
early_stopping_patience = 5
batch_size = 2
num_workers = 30
wandb = False

features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86

graph_definition = KNNGraph(detector=IceCube86())

selection = pd.read_pickle(f"./vMF_IS_10_15/performance.pkl")[
    "event_no"
].to_list()


dataloader = make_dataloader(
    db=path,
    graph_definition=graph_definition,
    pulsemaps=pulsemap,
    features=features,
    truth=truth,
    batch_size=batch_size,
    num_workers=num_workers,
    truth_table=truth_table,
    selection=selection[:10],
    labels={"x_direction": x(), "y_direction": y(), "z_direction": z()},
    shuffle=True,
)

model = Model.load(f"{model_path}/model.pth")

df = model.PredictGrid(dataloader, 50, 0.1, 0.1)


def extract_min_info(row):
    # Find minimum value in predictions
    min_value = np.min(row["predictions"])
    # Find the index of the minimum value
    min_index = np.argmin(row["predictions"])
    # Find the corresponding direction grid value
    x = row["direction_grid"][min_index][0]
    y = row["direction_grid"][min_index][1]
    z = row["direction_grid"][min_index][2]
    return pd.Series(
        [min_value, min_index, x, y, z],
        index=["min_prediction", "min_index", "x_pred", "y_pred", "z_pred"],
    )


# Apply the function to each row
df[["min_prediction", "min_index", "x_pred", "y_pred", "z_pred"]] = df.apply(
    extract_min_info, axis=1
)

print(df.head())
print(df.columns)


def angle(az_true, zen_true, x_pred, y_pred, z_pred):
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)

    zen_pred = np.arccos(z_pred)

    sa2 = y_pred / np.sin(zen_pred)
    ca2 = x_pred / np.sin(zen_pred)
    sz2 = np.sin(zen_pred)
    cz2 = z_pred

    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    scalar_prod = np.clip(scalar_prod, -1, 1)

    return np.abs(np.arccos(scalar_prod))


df["NF-truth"] = angle(
    df["azimuth"], df["zenith"], df["x_pred"], df["y_pred"], df["z_pred"]
)

bin_edges = np.logspace(2, 7, 15)
bin_centers = [
    0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(len(bin_edges) - 1)
]

df["energy_bin"] = pd.cut(df["energy"], bins=bin_edges)
# df['azimuth'] = pd.cut(df['azimuth'], bins = bin_edges)

percentiles = [16, 50, 84]


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
            df.groupby("energy_bin")[column]
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
                label=f"{column} {p}th Percentile",
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
        df["energy"],
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


df.to_pickle(f"{model_path}/performance.pkl")
plot_percentiles_comparison_with_bootstrap(
    df, ["NF-truth"], "NF to Truth", f"{model_path}/performance.pdf"
)
print(np.degrees(np.mean(df["NF-truth"])))
