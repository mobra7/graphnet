import os
from typing import Any, Dict, List, Optional

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import sqlite3
import numpy as np

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data import GraphNeTDataModule
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.dataset import ParquetDataset
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DeepIce
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import IceMixNodes
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.callbacks import PiecewiseLinearLR, ProgressBar
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.utils import make_train_validation_dataloader
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.training.labels import Direction
os.environ["CUDA_VISIBLE_DEVICES"]="2,3,1,0"

# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86
torch.set_float32_matmul_precision('medium')


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
            project="IceMix",
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
            "distribution_strategy": "ddp_find_unused_parameters_true",
            "precision": "16-mixed", 
        },
        "dataset_reference": SQLiteDataset
        if path.endswith(".db")
        else ParquetDataset,
    }

    if wandb:
        # Log configuration to W&B
        wandb_logger.experiment.config.update(config)

    # Define graph representation
    graph_definition = KNNGraph(
        detector=IceCube86(),
        node_definition=IceMixNodes(
            input_feature_names=features,
            max_pulses= 1024,
            z_name="dom_z",
            hlc_name='hlc',
            add_ice_properties=False,
        ),
        input_feature_names=features,
        columns=[0, 1, 2, 3],
    )


    dm = GraphNeTDataModule(
        dataset_reference=config["dataset_reference"],
        dataset_args={
            "truth": truth,
            "truth_table": truth_table,
            "features": features,
            "graph_definition": graph_definition,
            "pulsemaps": [config["pulsemap"]],
            "path": config["path"],
            "index_column": "event_no",
            "labels": {
                "direction": Direction(
                    azimuth_key="azimuth",
                    zenith_key="zenith",
                )
            },
        },
        train_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
        },
        test_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
        },
    )

    training_dataloader = dm.train_dataloader
    validation_dataloader = dm.val_dataloader

    # Building model

    backbone = DeepIce(
        hidden_dim=768,
        seq_length=1088,
        depth=12,
        head_size=64,
        n_rel=4,
        scaled_emb=True,
        include_dynedge=True,
        dynedge_args={
            "nb_inputs": graph_definition._node_definition.n_features,
            "nb_neighbours": 9,
            "post_processing_layer_sizes": [336, 384],
            "activation_layer": "gelu",
            "add_norm_layer": True,
            "skip_readout": True,
        },
        n_features=len(features),
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
        scheduler_kwargs={'patience': 2, 'factor': 0.1},
        scheduler_config={'frequency': 1, 'monitor': 'val_loss'},
    )


    # Training model
    model.fit(
        training_dataloader,
        validation_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
        logger=wandb_logger if wandb else None,
        callbacks= [ProgressBar(),
                    EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    ),
                    ModelCheckpoint(
                        save_top_k=-1,
                        every_n_epochs = 5,
                        dirpath=f"{save_path}/checkpoints/",
                        filename=f"{model.backbone.__class__.__name__}"
                        + "-{epoch}-{val_loss:.2f}-{train_loss:.2f}"
                    ),
                    ModelCheckpoint(
                        save_top_k=1,
                        monitor="val_loss",
                        mode="min",
                        dirpath=f"{save_path}/checkpoints/",
                        filename=f"best"
                        + "-{epoch}-{val_loss:.2f}-{train_loss:.2f}",
                    )],
        ckpt_path = './icemix_pretrain/checkpoints/best-epoch=21-val_loss=-2.66-train_loss=-2.67.ckpt',
        **config["fit"]
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
    )

    # Save results as .csv
    results.to_csv(f"{save_path}/results.csv")


if __name__ == "__main__":

    # settings
    #path = "/scratch/users/mbranden/sim_files/dev_northern_tracks_muon_labels_v3_part_1.db" 
    path = "/scratch/users/allorana/northern_sqlite/old_files/dev_northern_tracks_muon_labels_v3_part_1.db"
    save_path = "/ptmp/mpp/mbranden/graphnet/playground/icemix_pretrain"
    pulsemap = 'InIcePulses'
    target = 'direction'
    truth_table = 'truth'
    gpus = [3]
    max_epochs = 50
    early_stopping_patience = 5
    batch_size = 30
    num_workers = 30
    wandb =  False

    main(
            path=path,
            save_path=save_path,
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