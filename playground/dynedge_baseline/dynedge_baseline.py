"""Example of training Model."""

import os
from typing import Any, Dict, List, Optional

from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import sqlite3
import numpy as np

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.utils import make_train_validation_dataloader
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.training.labels import Direction
os.environ["CUDA_VISIBLE_DEVICES"]="3,2,1,0"

# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86



def main(
    path: str,
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

    # Initialise Weights & Biases (W&B) run
    if wandb:
        # Make sure W&B output directory exists
        wandb_dir = "./wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="example-script",
            entity="graphnet-team",
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
        },
    }

    archive = os.path.join(EXAMPLE_OUTPUT_DIR, "train_model_without_configs")
    run_name = "dynedge_{}_example".format(config["target"])
    if wandb:
        # Log configuration to W&B
        wandb_logger.experiment.config.update(config)

    # Define graph representation
    graph_definition = KNNGraph(detector=IceCube86())

    # filtering pulse count
    print('filtering pulse count')
    conn = sqlite3.connect(path)
    indices = pd.read_sql_query(
            f"SELECT event_no FROM {truth_table}", conn
        )
    # Filter based on pulse count
    query = f"SELECT event_no FROM {pulsemap} WHERE event_no IN ({','.join(map(str, indices))}) GROUP BY event_no HAVING COUNT(*) BETWEEN ? AND ?"

    min_count = 1
    max_count = 1000
    selection = [event_no for event_no, in conn.execute(query, (min_count, max_count)).fetchall()]
    #selection = np.random.choice(selection, size = 10000 ,replace = False).tolist()

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
        labels = {'direction': Direction()},
        test_size=0.1
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
        scheduler_kwargs={'patience': 4},
        scheduler_config={'frequency': 1, 'monitor': 'val_loss'},
    )


    # Training model
    model.fit(
        training_dataloader,
        validation_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
        logger=wandb_logger if wandb else None,
        **config["fit"],
    )

    # Get predictions
    additional_attributes = model.target_labels
    assert isinstance(additional_attributes, list)  # mypy
    print(model.target_labels)
    print(model.prediction_labels)

    # Save predictions and model to file
    db_name = path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)

    model.save(f"{path}/model.pth")

    # Save model config and state dict - Version safe save method.
    # This method of saving models is the safest way.
    model.save_state_dict(f"{path}/state_dict.pth")
    model.save_config(f"{path}/model_config.yml")

    results = model.predict_as_dataframe(
        validation_dataloader,
        additional_attributes=additional_attributes + ["event_no"],
        gpus=config["fit"]["gpus"],
    )

    # Save results as .csv
    results.to_csv(f"{path}/results.csv")




if __name__ == "__main__":

    # settings
    path = "/scratch/users/allorana/northern_sqlite/files_no_hlc/dev_northern_tracks_full_part_1.db"
    pulsemap = 'InIcePulses'
    target = 'direction'
    truth_table = 'truth'
    gpus = [0]
    max_epochs = 150
    early_stopping_patience = 10
    batch_size = 500
    num_workers = 30
    wandb =  False

    main(
            path=path,
            pulsemap = pulsemap,
            target = target,
            truth_table = truth_table,
            gpus = gpus,
            max_epochs = max_epochs,
            early_stopping_patience = early_stopping_patience,
            batch_size = batch_size,
            num_workers = num_workers,
            wandb = wandb,
        )