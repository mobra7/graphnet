from tqdm import tqdm
import torch
import os
from typing import Any, Dict, List, Optional
from torch.optim.adam import Adam
from torch_geometric.data import Data

from graphnet.models import NormalizingFlow
from graphnet.models.graphs import EdgelessGraph
from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.utilities.sqlite_utilities import query_database
from sklearn.model_selection import train_test_split
from graphnet.constants import ICECUBE_GEOMETRY_TABLE_DIR
from graphnet.models.detector.detector import Detector
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from graphnet.data.dataset.sqlite import SQLiteDataset
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.graphs import KNNGraph
from graphnet.models import Model
from graphnet.training.labels import Label


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
        ).reshape(-1)
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
        y = torch.sin(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1)
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
        z = torch.cos(graph[self._zenith_key]).reshape(-1)
        return z


class GridDatasetAroundPoint(SQLiteDataset):
    def __init__(
        self,
        ticks,
        azimuth_delta,
        zenith_delta,
        azimuth_center,
        zenith_center,
        selection,
        path,
        **kwargs,
    ):
        self._azimuth_centers = len(selection) * [azimuth_center]
        self._zenith_centers = len(selection) * [zenith_center]
        self._ticks = ticks
        self._azimuth_delta = azimuth_delta
        self._zenith_delta = zenith_delta
        self._pre_selection = selection
        major_list, indices = self._make_major_list(selection)
        self._major_list = major_list
        super().__init__(selection=indices, path=path, **kwargs)

    def __len__(self) -> int:
        """Return number of graphs in `Dataset`."""
        return len(self._pre_selection) * self._ticks**2

    def _make_major_list(self, selection):
        major_list = []
        new_index = []
        for seq_index, index in tqdm(
            enumerate(selection), desc="Creating grid"
        ):
            az = self._azimuth_centers[seq_index]
            ze = self._zenith_centers[seq_index]
            azimuth_range = np.linspace(
                az - self._azimuth_delta, az + self._azimuth_delta, self._ticks
            )
            zenith_range = np.linspace(
                ze - self._zenith_delta, ze + self._zenith_delta, self._ticks
            )
            for azimuth in azimuth_range:
                for zenith in zenith_range:
                    major_list.append([index, azimuth, zenith])
                    new_index.append(index)
        return major_list, new_index

    def __getitem__(self, sequential_index: int):
        """Return graph `Data` object at `index`."""
        if not (0 <= sequential_index < len(self)):
            raise IndexError(
                f"Index {sequential_index} not in range [0, {len(self) - 1}]"
            )
        features, truth, node_truth, loss_weight = self._query(
            sequential_index
        )
        graph = self._create_graph(features, truth, node_truth, loss_weight)

        # Apply grid
        _, azimuth, zenith = self._major_list[sequential_index]
        # graph["x_direction"] = torch.cos(torch.tensor(azimuth)) * torch.sin(
        #     torch.tensor(zenith)
        # ).reshape(-1)
        # graph["y_direction"] = torch.sin(torch.tensor(azimuth)) * torch.sin(
        #     torch.tensor(zenith)
        # ).reshape(-1)
        # graph["z_direction"] = torch.cos(torch.tensor(zenith)).reshape(-1)
        graph["max_llh_ze"] = torch.tensor(zenith).reshape(-1)
        graph["max_llh_az"] = torch.tensor(azimuth).reshape(-1)

        return graph


class GridDatasetBestPoint(SQLiteDataset):
    def __init__(
        self,
        ticks,
        azimuth_delta,
        zenith_delta,
        results,
        selection,
        path,
        **kwargs,
    ):
        self._azimuth_centers = []
        self._zenith_centers = []

        for event_no in selection:
            if float(event_no) in results["event_no"].values:
                min_index = results[results["event_no"] == float(event_no)][
                    "nllh"
                ].idxmin()
                self._azimuth_centers.append(results.loc[min_index, "azimuth"])
                self._zenith_centers.append(results.loc[min_index, "zenith"])
            else:
                self._azimuth_centers.append(0)
                self._zenith_centers.append(0)

        self._ticks = ticks
        self._azimuth_delta = azimuth_delta
        self._zenith_delta = zenith_delta
        self._pre_selection = selection
        major_list, indices = self._make_major_list(selection)
        self._major_list = major_list
        super().__init__(selection=indices, path=path, **kwargs)

    def __len__(self) -> int:
        """Return number of graphs in `Dataset`."""
        return len(self._pre_selection) * self._ticks**2

    def _make_major_list(self, selection):
        major_list = []
        new_index = []
        for seq_index, index in tqdm(
            enumerate(selection), desc="Creating grid"
        ):
            az = self._azimuth_centers[seq_index - 1]
            ze = self._zenith_centers[seq_index - 1]
            azimuth_range = np.linspace(
                az - self._azimuth_delta, az + self._azimuth_delta, self._ticks
            )
            zenith_range = np.linspace(
                ze - self._zenith_delta, ze + self._zenith_delta, self._ticks
            )
            for azimuth in azimuth_range:
                for zenith in zenith_range:
                    major_list.append([index, azimuth, zenith])
                    new_index.append(index)
        return major_list, new_index

    def __getitem__(self, sequential_index: int):
        """Return graph `Data` object at `index`."""
        if not (0 <= sequential_index < len(self)):
            raise IndexError(
                f"Index {sequential_index} not in range [0, {len(self) - 1}]"
            )
        features, truth, node_truth, loss_weight = self._query(
            sequential_index
        )
        graph = self._create_graph(features, truth, node_truth, loss_weight)

        # Apply grid
        _, azimuth, zenith = self._major_list[sequential_index]
        # graph["x_direction"] = torch.cos(torch.tensor(azimuth)) * torch.sin(
        #     torch.tensor(zenith)
        # ).reshape(-1)
        # graph["y_direction"] = torch.sin(torch.tensor(azimuth)) * torch.sin(
        #     torch.tensor(zenith)
        # ).reshape(-1)
        # graph["z_direction"] = torch.cos(torch.tensor(zenith)).reshape(-1)

        graph["max_llh_ze"] = torch.tensor(zenith).reshape(-1)
        graph["max_llh_az"] = torch.tensor(azimuth).reshape(-1)

        return graph


if __name__ == "__main__":

    # data_path = "/scratch/users/allorana/northern_sqlite/files_no_hlc/dev_northern_tracks_full_part_2.db"
    data_path = "/scratch/users/mbranden/sim_files/no_hlc_dev_northern_tracks_full_part_2.db"
    model_path = "./NF_11_13"
    gpus = [3]
    batch_size = 250
    num_workers = 30

    features = FEATURES.ICECUBE86
    truth = TRUTH.ICECUBE86

    graph_definition = KNNGraph(detector=IceCube86())

    events = query_database(
        database=data_path, query="select event_no from truth"
    )
    train_val_selection, test_selection = train_test_split(
        events["event_no"].tolist(), random_state=42, test_size=0.10
    )
    test_selection = test_selection[:10]


    dm = GraphNeTDataModule(
        dataset_reference=GridDatasetAroundPoint,
        dataset_args={
            "truth": truth,
            "features": features,
            "graph_definition": graph_definition,
            "pulsemaps": ["InIcePulses"],
            "path": data_path,
            "ticks": 100,
            "azimuth_delta": np.pi,
            "zenith_delta": np.pi / 2,
            "azimuth_center": np.pi,
            "zenith_center": np.pi / 2,
            "labels": {
                "x_direction": x(),
                "y_direction": y(),
                "z_direction": z(),
            },
        },
        selection=train_val_selection[:3],
        test_selection=test_selection,
        test_dataloader_kwargs={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": False,
        },
    )

    dataloader = dm.test_dataloader

    model = Model.load(f"{model_path}/model.pth")

    # Get predictions
    additional_attributes = [
        "event_no",
        "azimuth",
        "zenith",
        "energy",
        # "x_direction",
        # "y_direction",
        # "z_direction",
        "max_llh_az",
        "max_llh_ze",
    ]
    assert isinstance(additional_attributes, list)  # mypy

    results = model.predict_as_dataframe(
        dataloader=dataloader,
        additional_attributes=additional_attributes,  # + ["event_no"],
        gpus=gpus,
        precision="64-true",
    )
    print(results.head(10))

    dm = GraphNeTDataModule(
        dataset_reference=GridDatasetBestPoint,
        dataset_args={
            "truth": truth,
            "features": features,
            "graph_definition": graph_definition,
            "pulsemaps": ["InIcePulses"],
            "path": data_path,
            "ticks": 100,
            "azimuth_delta": 0.1,
            "zenith_delta": 0.1,
            "results": results,
            "labels": {
                "x_direction": x(),
                "y_direction": y(),
                "z_direction": z(),
            },
        },
        selection=train_val_selection[:3],
        test_selection=test_selection,
        test_dataloader_kwargs={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": False,
        },
    )

    dataloader = dm.test_dataloader

    results = model.predict_as_dataframe(
        dataloader=dataloader,
        additional_attributes=additional_attributes,
        gpus=gpus,
        precision="64-true",
    )

    results = results.loc[results.groupby("event_no")["nllh"].idxmin()]

    # Save results as .csv
    results.to_csv(f"{model_path}/results.csv", index=False)
    results.to_pickle(f"{model_path}/results.pkl")
    print(results.head(10))
