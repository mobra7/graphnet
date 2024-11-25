import os
from typing import Any, Dict, List, Optional
import sqlite3
import pandas as pd

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch.optim.adam import Adam
from torch_geometric.data import Data

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import Model, StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.reconstruction import (
    DirectionReconstructionWithKappa,
)
from graphnet.training.callbacks import ProgressBar
from graphnet.training.utils import make_train_validation_dataloader
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.labels import Direction, Label
from graphnet.utilities.logging import Logger
from graphnet.utilities.imports import has_jammy_flows_package


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


# torch.set_float32_matmul_precision('medium')

# Make sure the jammy flows is installed
try:
    assert has_jammy_flows_package()
    from graphnet.models import NormalizingFlow
except AssertionError:
    raise AssertionError(
        "This example requires the package`jammy_flow` "
        " to be installed. It appears that the package is "
        " not installed. Please install the package."
    )

features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86


def main(
    path: str,
    save_path: str,
    pulsemap: str,
    target: str,
    truth_table: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    wandb: bool = False,
) -> None:
    """Run example."""
    # Construct Logger
    logger = Logger()
    os.makedirs(save_path, exist_ok=True)

    # Initialise Weights & Biases (W&B) run
    if wandb:
        # Make sure W&B output directory exists
        wandb_dir = f"{save_path}/wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="NF",
            entity="mobra-technical-university-of-munich",
            save_dir=wandb_dir,
            log_model=True,
        )

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    # Configuration
    config: Dict[str, Any] = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
            "distribution_strategy": "auto",
            "precision": "64-true",
        },
    }

    if wandb:
        # Log configuration to W&B
        wandb_logger.experiment.config.update(config)

    # Define graph representation
    graph_definition = KNNGraph(detector=IceCube86())

    # filtering pulse count
    print("filtering pulse count")
    conn = sqlite3.connect(path)
    indices = pd.read_sql_query(f"SELECT event_no FROM {truth_table}", conn)
    query = f"SELECT event_no FROM {pulsemap} WHERE event_no IN ({','.join(map(str, indices))}) GROUP BY event_no HAVING COUNT(*) BETWEEN ? AND ?"

    min_count = 1
    max_count = 1024
    selection = [
        event_no
        for event_no, in conn.execute(query, (min_count, max_count)).fetchall()
    ]

    (
        training_dataloader,
        validation_dataloader,
    ) = make_train_validation_dataloader(
        db=config["path"],
        graph_definition=graph_definition,
        pulsemaps=config["pulsemap"],
        features=features,
        truth=truth,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        truth_table=truth_table,
        selection=selection,
        labels={"x_direction": x(), "y_direction": y(), "z_direction": z()},
        test_size=0.1,
    )

    # Building model

    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = DirectionReconstructionWithKappa(
        hidden_size=backbone.nb_outputs,
        target_labels=config["target"],
        loss_function=VonMisesFisher3DLoss(),
    )
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={"patience": 3},
        scheduler_config={"frequency": 1, "monitor": "val_loss"},
    ).double()
    model.load_state_dict("./dynedge_baseline/state_dict.pth")
    backbone = model.backbone.double()
    for i, param in enumerate(backbone.parameters()):
        param.requires_grad = False
        if i == len(list(backbone.parameters())) - 3:
            break

    model = NormalizingFlow(
        graph_definition=graph_definition,
        flow_layers="vvvvvv",
        target_norm=1.,
        backbone=backbone,
        optimizer_class=Adam,
        target_labels=config["target"],
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={"patience": 3, "factor": 0.1},
        scheduler_config={"frequency": 1, "monitor": "val_loss"},
    ).double()

    # Training model
    model.fit(
        training_dataloader,
        validation_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
        logger=wandb_logger if wandb else None,
        callbacks=[
            ProgressBar(),
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
            ),
            ModelCheckpoint(
                save_top_k=-1,
                every_n_epochs=5,
                dirpath=f"{save_path}/checkpoints/",
                filename=f"{model.backbone.__class__.__name__}"
                + "-{epoch}-{val_loss:.2f}-{train_loss:.2f}",
            ),
            ModelCheckpoint(
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                dirpath=f"{save_path}/checkpoints/",
                filename=f"best" + "-{epoch}-{val_loss:.2f}-{train_loss:.2f}",
            ),
        ],
        **config["fit"],
    )
    # Get predictions
    additional_attributes = model.target_labels
    assert isinstance(additional_attributes, list)  # mypy

    # Save predictions and model to file

    logger.info(f"Writing results to {save_path}")

    model.save(f"{save_path}/model.pth")

    # Save model config and state dict - Version safe save method.
    # This method of saving models is the safest way.
    model.save_state_dict(f"{save_path}/state_dict.pth")
    model.save_config(f"{save_path}/model_config.yml")

    results = model.predict_as_dataframe(
        validation_dataloader,
        additional_attributes=additional_attributes + ["event_no"],
        gpus=config["fit"]["gpus"],
        precision=config["fit"]["precision"],
    )

    # Save results as .csv
    results.to_csv(f"{save_path}/results.csv")


if __name__ == "__main__":

    # settings
    path = "/scratch/users/mbranden/sim_files/no_hlc_dev_northern_tracks_full_part_1.db"
    # path = "/scratch/users/allorana/northern_sqlite/files_no_hlc/dev_northern_tracks_full_part_1.db"
    save_path = "/ptmp/mpp/mbranden/graphnet/playground/NF_11_18"
    pulsemap = "InIcePulses"
    target = ["azimuth", "zenith"]
    truth_table = "truth"
    gpus = [2]
    max_epochs = 200
    early_stopping_patience = 8
    batch_size = 500
    num_workers = 30
    wandb = True

    main(
        path=path,
        save_path=save_path,
        pulsemap=pulsemap,
        target=target,
        truth_table=truth_table,
        gpus=gpus,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        batch_size=batch_size,
        num_workers=num_workers,
        wandb=wandb,
    )
